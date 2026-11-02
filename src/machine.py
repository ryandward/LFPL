"""The hardware target with its runtime: a handler, the turn of src/turn.lfpl, and byte streams in and out.

    python3 -m src.machine examples/ipv4.lfpl -o DIR --n-max 217

writes DIR/machine.v, DIR/serve_core.v and DIR/cells.mem, and says what one turn can take before anything runs.

The turn is LFPL, appended to the handler's program by src/turn.py, so admission, ADMIT, DONE and the output queue are
typed and certified with the handler, and src/rtverilog.py lowers all of it to one state machine, serve_core. What is
here is what LFPL cannot be: the cell store's first contents, and the module around serve_core that runs turns forever.

    ingress   a frame arrives as bytes, valid and ready, and goes into the scalar fields of the record ingress holds,
              then 7 bytes to a chunk of its chain. It writes a used count, the length field, and zeros after the last
              byte of the last chunk. It never writes a link, so whatever it is given, no cell can come to point at
              another: those are the turn's, and the turn is checked. It stores no byte past what N_max cells hold:
              the byte after them ends the frame, the turn refuses it as TooLarge, and the rest of it is read and
              thrown away between turns.
    egress    the queue's head, once a turn has put an output there: the port, then the record's scalars in order and
              the chain's bytes, read out of the cells, and the next turn releases it. It follows no more of a chain
              than N_out cells hold, and a frame with more is cut there, ends, and is counted.
    turns     one after another, forever, each started with what the last one returned. A turn is the checked unit:
              rtcheck certified its steps for any frame, and src/rtverilog.py turned that into a number of cycles.

Nothing outside serve_core moves a cell from one owner to another, so conservation is the turn's, and the tests check it
by walking the store afterwards. Ingress and egress take a number of cycles that grows with the frame, at most what
N_max and N_out cells hold, and they are not part of a turn's bound.

What would hang is stopped here, in the design, whatever the turn does. A turn that runs past its bound is stopped,
and one whose results put an output at the head of the queue that no outcome accounts for is not taken, because
otherwise egress sends that head again after every release, each turn within its bound and none of them ending it.
Either way the machine counts it and writes every cell back as cells.mem has it, as a reset does. So no more frames go
out than turns said they queued: once frames stop coming in, egress sends at most what is queued, a release follows
each, and the machine is idle.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from dataclasses import dataclass

from .certify import certify_turn
from .rtcheck import Bound, Checker, RTError, TypeDecl
from .rtlower import rt_lines
from .rtverilog import CELL_BYTES, Emitter, Timing, _bits_for
from .runtime import BODY_PAYLOAD
from .turn import OUTCOMES, TURN, Shape, TurnRefused, turn_source

LINK = "$next"


# What the machine says of a turn it would not take the results of, in place of the turn's own outcome.
STOPPED, FAULTED = 255, 254
HALTS = {STOPPED: "stopped", FAULTED: "faulted"}


@dataclass(frozen=True)
class Machine:
    shape: Shape
    bound: Bound              # the turn's: W = bound.d steps, for any frame
    timing: Timing
    cells: int
    pointer_bits: int
    core: str                 # serve_core.v
    top: str                  # machine.v
    image: str                # cells.mem
    turn_text: str            # the LFPL the turn was compiled from
    start: tuple              # the pool, the ingress slot and the queue after its head, before the first turn
    queue_layout: tuple       # each cell of the queue's record, as its fields' offsets and sizes by name
    entry_layout: dict        # where an output's port and record are in its place's cell

    @property
    def turn_cycles(self) -> int:
        """The most clock cycles a turn can take, from the edge that starts it to the edge that ends it."""
        return self.timing.k * self.bound.d + self.timing.c

    def report(self) -> str:
        return (f"turn {TURN} for handler {self.shape.handler}: W = {self.bound.d} steps for any frame of up to "
                f"{self.shape.n_max} cells in and {self.shape.n_out} out, with {self.shape.k_out} places in the queue; "
                f"cycles <= {self.timing.k}*W + {self.timing.c} = {self.turn_cycles} a turn; "
                f"{self.cells} cells of {8 * CELL_BYTES} bits, {self.pointer_bits}-bit pointers")


def build(program: bytes, *, n_max: int, k_out: int = 4, cells: int | None = None) -> Machine:
    text, shape = turn_source(program, n_max=n_max, k_out=k_out)
    lines = list(rt_lines(io.BytesIO(text.encode())))
    checker = Checker()
    for line in lines:
        checker.feed(line)
    checker.finish()
    queue = _record(checker, "Queue")
    queue_cells = len(queue)
    # The turn walks two things it is given: the frame in ingress, and the output at the head of the queue, which is
    # its own cell and a record of at most N_out cells. The rest of the queue and the pool are charged nothing.
    bound = certify_turn(checker, TURN, {1: n_max, 2: 1 + shape.n_out})
    needed = n_max + queue_cells + k_out * (1 + n_max) + (n_max + 1)
    count = needed if cells is None else cells
    first_pool = n_max + queue_cells + 1
    if count < first_pool - 1:
        raise RTError(f"section 3.4: {count} cells cannot hold ingress and the queue, which take {first_pool - 1}")
    pointer_bits = _bits_for(count + 1)
    image, start = _image(checker, shape, queue, count)
    emitter = Emitter(pointer_bits)
    for line in lines:
        emitter.feed(line)
    turn = next(f for f in emitter.functions.values() if f.turn)
    core = emitter.render(turn, "serve_core", bound, count, {})
    entry = _offsets(_ctor(checker, "Pending"))
    top = _top(shape, pointer_bits, start, entry, core.timing.k * bound.d + core.timing.c, count)
    return Machine(shape, bound, core.timing, count, pointer_bits, core.verilog, top, image, text, start,
                   tuple(queue), entry)


# --- the cell store's first contents

def _decl(checker: Checker, name: str) -> TypeDecl:
    return next(decl for decl in checker.types.values() if decl.name == name)


def _ctor(checker: Checker, name: str):
    return next(ctor for ctor in checker.ctors.values() if ctor.name == name)


def _offsets(ctor) -> dict:
    return {name: (off, size) for off, size, _ty, name in ctor.fields}


def _record(checker: Checker, name: str) -> list[dict]:
    """A record's cells, in order, each as its fields' offsets and sizes by name, `$next` among them."""
    parts, decl = [], _decl(checker, name)
    while True:
        fields = _offsets(decl.ctors[0])
        parts.append(fields)
        link = next((ty for _off, _size, ty, field_name in decl.ctors[0].fields if field_name == LINK), None)
        if link is None:
            return parts
        decl = checker._decl(checker.norm(link))


def _image(checker: Checker, shape: Shape, queue: list[dict], count: int) -> tuple[str, tuple]:
    """Ingress first: the record in cells 1 and on, then its chain. Then the queue, empty, and then the pool, every
    other cell, threaded through offset 0. Word 0 is null and never a cell."""
    words = [0] * (count + 1)

    def put(cell: int, off: int, value: int) -> None:
        words[cell] |= value << (8 * off)

    walk = shape.item_walk
    record = list(range(1, shape.parts + 1))
    chunks = list(range(shape.parts + 1, shape.n_max + 1))
    for cell, off, inner in walk.links:
        put(record[cell], off, record[inner])
    assert walk.chain is not None
    put(record[walk.chain[0]], walk.chain[1], chunks[0] if chunks else 0)
    for position, chunk in enumerate(chunks):  # Chunk(used, b0 .. b6, next): next is the second 8 bytes
        put(chunk, 8, chunks[position + 1] if position + 1 < len(chunks) else 0)
    places = list(range(shape.n_max + 1, shape.n_max + len(queue) + 1))
    for position, fields in enumerate(queue):
        if LINK in fields:
            put(places[position], fields[LINK][0], places[position + 1])
    pool = list(range(shape.n_max + len(queue) + 1, count + 1))
    for position, cell in enumerate(pool):  # Cell(next), at offset 0
        put(cell, 0, pool[position + 1] if position + 1 < len(pool) else 0)
    text = "\n".join(f"{word:032x}" for word in words) + "\n"
    return text, (pool[0] if pool else 0, record[0], places[0])


# --- the module around the turn

def _top(shape: Shape, p: int, start: tuple, entry: dict, turn_cycles: int, cells: int) -> str:
    walk = shape.item_walk
    links = list(walk.links)
    assert walk.chain is not None and walk.length is not None
    wire_cells, wire_lanes = [], []
    for cell, off, size in walk.scalars:
        for byte in range(size):
            wire_cells.append(cell)
            wire_lanes.append(off + byte)
    parts = shape.parts
    cell_bits = max(1, _bits_for(parts))
    steps = len(links) + 1
    held_bits = (shape.k_out + 1).bit_length() + 1  # room for one past K_out, and for one below none
    pool0, slot0, queue0 = start
    states = ("IDLE LINK LINK_TAKE IN NEXT NEXT_TAKE FIN_USED FIN_LEN TURN TURN_WAIT ENTRY ENTRY_TAKE COUNT "
              "COUNT_TAKE WORD WORD_TAKE BYTE CHUNK CHUNK_TAKE CHUNK_BYTE SENT WIPE").split()
    wipe = "begin wipe <= {P{1'b0}}; first_word <= first[0]; state <= S_WIPE; end"
    out: list[str] = []
    add = out.append
    add("// Generated by src/machine.py: the hardware around the turn serve_core runs. Do not edit.")
    add(f"// A turn takes at most {turn_cycles} clock cycles: serve_core.v says how that follows from its W. A turn")
    add("// that runs past that is stopped, and so is one whose results put an output at the head of the queue that")
    add("// no outcome accounts for, and every cell starts again as cells.mem has it. Ingress stores no byte past what")
    add("// N_max cells hold, and egress follows no more of a chain than N_out cells hold. So nothing here runs on")
    add("// whatever a turn does, and no more frames go out than turns queued.")
    add("`timescale 1ns/1ns")
    add("`default_nettype none")
    add("module machine (")
    add("    input  wire        clk,")
    add("    input  wire        rst,")
    add("    input  wire        in_valid,      // a frame, a byte at a time, the last with in_last")
    add("    input  wire [7:0]  in_data,")
    add("    input  wire        in_last,")
    add("    output wire        in_ready,")
    add("    output reg         out_valid,     // a frame the queue held, a byte at a time, to out_port")
    add("    output reg  [7:0]  out_data,")
    add("    output reg         out_last,")
    add("    output reg  [7:0]  out_port,")
    add("    input  wire        out_ready,")
    add("    output reg         turn_done,     // a clock, when a turn has finished or been stopped")
    add("    output reg  [31:0] turn_cycles,   // from the edge that started it to the edge that ended it")
    add(f"    output reg  [7:0]  turn_outcome,  // its tag: {', '.join(f'{tag} {name}' for tag, name in enumerate(OUTCOMES))};")
    add(f"                                      // or the machine's: {FAULTED} faulted, {STOPPED} stopped")
    add("    output reg  [31:0] overruns,      // turns stopped at the bound")
    add("    output reg  [31:0] faults,        // turns whose head of the queue no outcome accounts for")
    add("    output reg  [31:0] cuts           // frames egress ended at N_out cells, with more of their chain left")
    add(");")
    add(f"    localparam P = {p};")
    add(f"    localparam CELLS = {cells};")
    add(f"    localparam WIRE = 16'd{shape.wire};")
    add(f"    localparam CAPACITY = 16'd{shape.capacity};  // what N_max cells hold")
    add(f"    localparam CHUNKS_OUT = 16'd{shape.n_out - parts};  // a chain's share of N_out cells")
    add(f"    localparam [31:0] TURN_LIMIT = 32'd{turn_cycles};")
    add(f"    localparam [7:0] QUEUED = 8'd{OUTCOMES.index('queued')};")
    add(f"    localparam [7:0] FAULTED = 8'd{FAULTED};")
    add(f"    localparam [7:0] STOPPED = 8'd{STOPPED};")
    for position, name in enumerate(states):
        add(f"    localparam [4:0] S_{name} = 5'd{position};")
    add("    reg [4:0] state;")
    add("    reg reading;                      // the links being followed are egress's, not ingress's")
    add("")
    add("    // What the turn holds from one turn to the next, and what it is told.")
    add("    reg [P-1:0] pool, slot, head, queue;")
    add("    reg [15:0] length;")
    add("    reg sent;")
    add("    reg start;")
    add("    reg [31:0] cycles;")
    add("    wire done, busy;")
    add("    wire [P-1:0] r_pool, r_slot, r_head, r_queue;")
    add("    wire [63:0] r_outcome;")
    add("    // Outputs the queue holds, as the outcomes of the turns tell it: Queued adds one, a release takes one. It")
    add("    // decides nothing a turn does. It is how the machine knows a turn's results cannot be taken.")
    add(f"    reg [{held_bits - 1}:0] held;")
    add(f"    wire [{held_bits - 1}:0] held_after = sent ? held - 1'b1 : r_outcome[7:0] == QUEUED ? held + 1'b1 : held;")
    add("    wire stop = state == S_TURN_WAIT && !done && cycles == TURN_LIMIT;")
    add("    reg host_en;")
    add(f"    reg [{CELL_BYTES - 1}:0] host_we;")
    add("    reg [P-1:0] host_addr;")
    add(f"    reg [{8 * CELL_BYTES - 1}:0] host_din;")
    add(f"    wire [{8 * CELL_BYTES - 1}:0] host_dout;")
    add("    serve_core core (.clk(clk), .rst(rst || stop), .start(start), .arg0(pool), .arg1(slot), .arg2(head), .arg3(queue),")
    add("                     .arg4(length), .arg5(sent), .done(done), .busy(busy), .res0(r_pool), .res1(r_slot),")
    add("                     .res2(r_head), .res3(r_queue), .res4(r_outcome), .host_en(host_en), .host_we(host_we),")
    add("                     .host_addr(host_addr), .host_din(host_din), .host_dout(host_dout));")
    add("")
    add("    // Every cell as it is before the first turn. A reset writes them back, and so does a turn that is stopped.")
    add(f"    reg [{8 * CELL_BYTES - 1}:0] first [0:CELLS];")
    add(f"    reg [{8 * CELL_BYTES - 1}:0] first_word;   // first[wipe], read the clock before it is written")
    add("    reg [P-1:0] wipe;")
    add('    initial $readmemh("cells.mem", first);')
    add("")
    add("    // Ingress and egress, between turns.")
    add(f"    reg [P-1:0] rec [0:{parts - 1}];   // the cells of the record being written or read")
    add("    reg [P-1:0] chunk, cur;")
    add("    reg [2:0] k;                      // bytes in the chunk being filled, or emitted from it")
    add("    reg [2:0] used;")
    add("    reg [15:0] count, total, emitted, index, walked;")
    add("    reg draining;                     // the rest of a frame past CAPACITY, read and thrown away")
    add("    reg over;                         // the chain going out goes on past CHUNKS_OUT")
    add(f"    reg [{max(1, _bits_for(steps + 1)) - 1}:0] step;")
    add(f"    reg [{8 * CELL_BYTES - 1}:0] word;")
    add(f"    reg [{cell_bits}:0] loaded;          // which of rec[] word holds, or none")
    add("    assign in_ready = state == S_IN || (state == S_IDLE && draining);")
    add("")
    add("    // Where each byte of the record goes on the wire: which of its cells, and which byte of that cell.")
    add(f"    function [{cell_bits - 1}:0] wire_cell(input [15:0] i);")
    add("        case (i)")
    for index, cell in enumerate(wire_cells):
        add(f"            16'd{index}: wire_cell = {cell_bits}'d{cell};")
    add(f"            default: wire_cell = {cell_bits}'d0;")
    add("        endcase")
    add("    endfunction")
    add("    function [3:0] wire_lane(input [15:0] i);")
    add("        case (i)")
    for index, lane in enumerate(wire_lanes):
        add(f"            16'd{index}: wire_lane = 4'd{lane};")
    add("            default: wire_lane = 4'd0;")
    add("        endcase")
    add("    endfunction")
    add("    // The link step s follows: from which cell, and where in it. The last step reads the chain.")
    add("    function [P-1:0] link_from(input [3:0] s);")
    add("        case (s)")
    for index, (cell, _off, _inner) in enumerate(links):
        add(f"            4'd{index}: link_from = rec[{cell}];")
    add(f"            default: link_from = rec[{walk.chain[0]}];")
    add("        endcase")
    add("    endfunction")
    add("")
    add("    // What drives the cell store's host port, in the clock the state is the state.")
    add("    always @* begin")
    add("        host_en = 1'b0;")
    add(f"        host_we = {CELL_BYTES}'d0;")
    add("        host_addr = {P{1'b0}};")
    add(f"        host_din = {8 * CELL_BYTES}'d0;")
    add("        case (state)")
    add(f"            S_WIPE: begin host_we = {{{CELL_BYTES}{{1'b1}}}}; host_addr = wipe; host_din = first_word; end")
    add("            S_LINK: begin host_en = 1'b1; host_addr = link_from(step); end")
    add("            S_IN: if (in_valid) begin")
    add("                if (count < WIRE) begin")
    add("                    host_addr = rec[wire_cell(count)];")
    add(f"                    host_we = {CELL_BYTES}'d1 << wire_lane(count);")
    add(f"                    host_din = {{{8 * CELL_BYTES - 8}'d0, in_data}} << (8 * wire_lane(count));")
    add("                end else if (count < CAPACITY && chunk != {P{1'b0}}) begin")
    add("                    host_addr = chunk;  // byte 1 + k, and the used count with the seventh")
    add(f"                    host_we = ({CELL_BYTES}'d2 << k) | (k == 3'd6 ? {CELL_BYTES}'d1 : {CELL_BYTES}'d0);")
    add(f"                    host_din = ({{{8 * CELL_BYTES - 8}'d0, in_data}} << (8 + 8 * k)) | "
        f"(k == 3'd6 ? {8 * CELL_BYTES}'d{BODY_PAYLOAD} : {8 * CELL_BYTES}'d0);")
    add("                end")
    add("            end")
    add("            S_NEXT: begin host_en = 1'b1; host_addr = chunk; end")
    add("            S_FIN_USED: if (k != 3'd0 && chunk != {P{1'b0}}) begin")
    add("                host_addr = chunk;  // the used count, and zeros after the last byte: an old frame's are not read")
    add(f"                host_we = {CELL_BYTES}'h00ff & ~(({CELL_BYTES}'d1 << (k + 3'd1)) - {CELL_BYTES}'d2);")
    add(f"                host_din = {{{8 * CELL_BYTES - 3}'d0, k}};")
    add("            end")
    add("            S_FIN_LEN: begin")
    add(f"                host_addr = rec[{walk.length[0]}];")
    add(f"                host_we = {CELL_BYTES}'d3 << {walk.length[1]};")
    add(f"                host_din = {{{8 * CELL_BYTES - 16}'d0, count}} << {8 * walk.length[1]};")
    add("            end")
    add("            S_ENTRY: begin host_en = 1'b1; host_addr = head; end")
    add("            S_COUNT: if (cur != {P{1'b0}}) begin host_en = 1'b1; host_addr = cur; end")
    add("            S_WORD: begin host_en = 1'b1; host_addr = rec[wire_cell(index)]; end")
    add("            S_CHUNK: if (cur != {P{1'b0}}) begin host_en = 1'b1; host_addr = cur; end")
    add("            default: ;")
    add("        endcase")
    add("    end")
    add("")
    add("    always @(posedge clk) begin")
    add("        start <= 1'b0;")
    add("        turn_done <= 1'b0;")
    add("        if (out_valid && out_ready) out_valid <= 1'b0;")
    add("        if (rst) begin")
    add("            wipe <= {P{1'b0}};")
    add("            first_word <= first[0];")
    add("            state <= S_WIPE;")
    add("            draining <= 1'b0;")
    add("            overruns <= 32'd0;")
    add("            faults <= 32'd0;")
    add("            cuts <= 32'd0;")
    add("            out_valid <= 1'b0;")
    add("            out_last <= 1'b0;")
    add("        end else case (state)")
    add("            S_WIPE: begin  // a cell a clock, and after the last, the registers as they were at the start")
    add("                wipe <= wipe + 1'b1;")
    add("                if (wipe != CELLS) first_word <= first[wipe + 1'b1];")
    add("                else begin")
    add(f"                    pool <= {p}'d{pool0};")
    add(f"                    slot <= {p}'d{slot0};")
    add("                    head <= {P{1'b0}};")
    add(f"                    queue <= {p}'d{queue0};")
    add("                    held <= 0;")
    add("                    sent <= 1'b0;")
    add("                    state <= S_IDLE;")
    add("                end")
    add("            end")
    add("            S_IDLE: begin")
    add("                if (in_valid && draining && in_last) draining <= 1'b0;")
    add("                if (head != {P{1'b0}} && out_ready) state <= S_ENTRY;")
    add("                else if (in_valid && !draining) begin")
    add("                    rec[0] <= slot;")
    add("                    reading <= 1'b0;")
    add("                    step <= 0;")
    add("                    count <= 16'd0;")
    add("                    k <= 3'd0;")
    add("                    state <= S_LINK;")
    add("                end")
    add("            end")
    add("            S_LINK: state <= S_LINK_TAKE;")
    add("            S_LINK_TAKE: begin")
    add("                case (step)")
    for index, (_cell, off, inner) in enumerate(links):
        add(f"                    {index}: rec[{inner}] <= host_dout[{8 * off} +: P];")
    add(f"                    default: begin chunk <= host_dout[{8 * walk.chain[1]} +: P]; "
        f"cur <= host_dout[{8 * walk.chain[1]} +: P]; end")
    add("                endcase")
    add(f"                state <= step == {steps - 1} ? (reading ? S_COUNT : S_IN) : S_LINK;")
    add("                step <= step + 1;")
    add("            end")
    add("            S_IN: if (in_valid) begin")
    add("                count <= count + 16'd1;")
    add("                if (count == CAPACITY) begin  // a byte past what ingress holds ends the frame, too large")
    add("                    draining <= !in_last;")
    add("                    state <= S_FIN_USED;")
    add("                end else begin")
    add("                    if (count >= WIRE && chunk != {P{1'b0}}) k <= k == 3'd6 ? 3'd0 : k + 3'd1;")
    add("                    if (in_last) state <= S_FIN_USED;")
    add("                    else if (count >= WIRE && chunk != {P{1'b0}} && k == 3'd6) state <= S_NEXT;")
    add("                end")
    add("            end")
    add("            S_NEXT: state <= S_NEXT_TAKE;")
    add("            S_NEXT_TAKE: begin chunk <= host_dout[64 +: P]; state <= S_IN; end")
    add("            S_FIN_USED: state <= S_FIN_LEN;")
    add("            S_FIN_LEN: begin length <= count; sent <= 1'b0; state <= S_TURN; end")
    add("            S_TURN: begin start <= 1'b1; cycles <= 32'hffffffff; state <= S_TURN_WAIT; end")
    add("            S_TURN_WAIT: if (done) begin")
    add("                turn_cycles <= cycles;")
    add("                turn_done <= 1'b1;")
    add("                if ((r_head != {P{1'b0}}) != (held_after != 0)) begin")
    add("                    faults <= faults + 32'd1;")
    add("                    turn_outcome <= FAULTED;")
    add(f"                    {wipe}")
    add("                end else begin")
    add("                    pool <= r_pool;")
    add("                    slot <= r_slot;")
    add("                    head <= r_head;")
    add("                    queue <= r_queue;")
    add("                    held <= held_after;")
    add("                    turn_outcome <= r_outcome[7:0];")
    add("                    state <= S_IDLE;")
    add("                end")
    add("            end else if (stop) begin  // and the core's own reset is held this clock")
    add("                overruns <= overruns + 32'd1;")
    add("                turn_cycles <= cycles + 32'd1;")
    add("                turn_outcome <= STOPPED;")
    add("                turn_done <= 1'b1;")
    add(f"                {wipe}")
    add("            end else cycles <= cycles + 32'd1;")
    add("            S_ENTRY: state <= S_ENTRY_TAKE;")
    add("            S_ENTRY_TAKE: begin")
    add(f"                out_port <= host_dout[{8 * entry['port'][0]} +: 8];")
    add(f"                rec[0] <= host_dout[{8 * entry['frame'][0]} +: P];")
    add("                reading <= 1'b1;")
    add("                step <= 0;")
    add("                total <= WIRE;")
    add("                walked <= 16'd0;")
    add("                state <= S_LINK;")
    add("            end")
    add("            S_COUNT: if (cur == {P{1'b0}} || walked == CHUNKS_OUT) begin")
    add("                over <= cur != {P{1'b0}};")
    add("                walked <= 16'd0;")
    add("                index <= 16'd0;")
    add("                emitted <= 16'd0;")
    add(f"                loaded <= {cell_bits + 1}'d{parts};")
    add("                state <= S_WORD;")
    add("            end else state <= S_COUNT_TAKE;")
    add("            S_COUNT_TAKE: begin  // a program can write a used count, so egress clamps it to 7")
    add(f"                total <= total + (host_dout[7:0] > 8'd{BODY_PAYLOAD} ? 16'd{BODY_PAYLOAD} : {{8'd0, host_dout[7:0]}});")
    add("                cur <= host_dout[64 +: P];")
    add("                walked <= walked + 16'd1;")
    add("                state <= S_COUNT;")
    add("            end")
    add("            S_WORD: state <= S_WORD_TAKE;")
    add(f"            S_WORD_TAKE: begin word <= host_dout; loaded <= {{1'b0, wire_cell(index)}}; state <= S_BYTE; end")
    add("            S_BYTE: if (!out_valid || out_ready) begin")
    add("                if (loaded != {1'b0, wire_cell(index)}) state <= S_WORD;")
    add("                else begin")
    add("                    out_valid <= 1'b1;")
    add("                    out_data <= word[8 * wire_lane(index) +: 8];")
    add("                    out_last <= emitted + 16'd1 == total;")
    add("                    emitted <= emitted + 16'd1;")
    add("                    index <= index + 16'd1;")
    add("                    if (index + 16'd1 == WIRE) begin cur <= chunk; state <= S_CHUNK; end")
    add("                end")
    add("            end")
    add("            S_CHUNK: state <= cur == {P{1'b0}} || walked == CHUNKS_OUT ? S_SENT : S_CHUNK_TAKE;")
    add("            S_CHUNK_TAKE: begin")
    add("                word <= host_dout;")
    add(f"                used <= host_dout[7:0] > 8'd{BODY_PAYLOAD} ? 3'd{BODY_PAYLOAD} : host_dout[2:0];")
    add("                k <= 3'd0;")
    add("                walked <= walked + 16'd1;")
    add("                state <= S_CHUNK_BYTE;")
    add("            end")
    add("            S_CHUNK_BYTE: if (k == used) begin cur <= word[64 +: P]; state <= S_CHUNK; end")
    add("            else if (!out_valid || out_ready) begin")
    add("                out_valid <= 1'b1;")
    add("                out_data <= word[8 + 8 * k +: 8];")
    add("                out_last <= emitted + 16'd1 == total;")
    add("                emitted <= emitted + 16'd1;")
    add("                k <= k + 3'd1;")
    add("            end")
    add("            S_SENT: if (!out_valid) begin")
    add("                if (over) cuts <= cuts + 32'd1;")
    add("                length <= 16'd0;")
    add("                sent <= 1'b1;")
    add("                state <= S_TURN;")
    add("            end")
    add("            default: state <= S_IDLE;")
    add("        endcase")
    add("    end")
    add("endmodule")
    add("`default_nettype wire")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="machine", description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("-o", dest="output", required=True, help="a directory for the Verilog and cells.mem")
    parser.add_argument("--n-max", type=int, default=217, help="the largest frame, in cells")
    parser.add_argument("--k-out", type=int, default=4, help="places in the output queue")
    parser.add_argument("--cells", type=int, help="cells in the store, instead of what the turn needs")
    args = parser.parse_args(argv)
    try:
        with open(args.source, "rb") as handle:
            machine = build(handle.read(), n_max=args.n_max, k_out=args.k_out, cells=args.cells)
    except (TurnRefused, RTError, OSError) as failure:
        print(f"{args.source}: refused: {failure}", file=sys.stderr)
        return 1
    os.makedirs(args.output, exist_ok=True)
    for name, text in (("machine.v", machine.top), ("serve_core.v", machine.core), ("cells.mem", machine.image),
                       ("turn.lfpl", machine.turn_text)):
        with open(os.path.join(args.output, name), "w") as handle:
            handle.write(text)
    print(machine.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
