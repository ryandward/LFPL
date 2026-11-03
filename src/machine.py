"""The hardware target with its runtime: a handler, the turn of src/turn.lfpl, and byte streams in and out.

    python3 -m src.machine examples/ipv4.lfpl -o DIR

writes DIR/machine.v, DIR/serve_core.v and DIR/cells.mem, and says what one turn can take before anything runs.

The turn is LFPL, appended to the handler's program by src/turn.py, so admission, ADMIT, DONE and the output queue are
typed and certified with the handler, and src/rtverilog.py lowers all of it to one state machine, serve_core. What is
here is what LFPL cannot be: the cell store's first contents, and the module around serve_core that runs turns forever.

    ingress   a frame arrives as bytes, valid and ready, and goes into the scalar fields of the record ingress holds,
              then into the slot that record holds, a byte to its place. It writes the length field and the slot's
              length, and nothing else: no link and no slot number, so whatever it is given, no cell can come to point
              at another. It stores no byte past the slot: the byte after it ends the frame, the turn refuses it as
              TooLarge, and the rest of it is read and thrown away between turns.
    egress    the queue's head, once a turn has put an output there: the port, then the record's scalars in order and
              the bytes its slot says it holds, read out of the slot's cells, and the next turn releases it. It sends
              no more than a slot holds, whatever the record says.
    turns     one after another, forever, each started with what the last one returned. A turn is the checked unit:
              rtcheck certified its clocks for each frame, charging each instruction what serve_core spends on it.

Slots are after the cells in the store, slot k at cell 128 k: ingress's, and one for each place in the queue, which the
turn keeps on a list of spares until a frame takes one. Nothing outside serve_core moves a cell or a slot from one owner
to another, so conservation is the turn's, and the tests check it by walking the store afterwards. Ingress and egress
take a number of cycles that grows with the frame, and they are not part of a turn's bound.

What would hang is stopped here, in the design, whatever the turn does. A turn that runs past its frame's bound is
stopped, and one whose results put an output at the head of the queue that no outcome accounts for is not taken,
because otherwise egress sends that head again after every release, each turn within its bound and none of them ending
it. Either way the machine counts it and writes every cell back as cells.mem has it, as a reset does. So no more frames
go out than turns said they queued: once frames stop coming in, egress sends at most what is queued, a release follows
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
from .turn import DECIDED, OUTCOMES, TURN, Shape, TurnRefused, turn_source

LINK = "$next"
INGRESS = 2  # the turn's parameter that is ingress's record, whose size each turn says: its cells and its slot's bytes


# What the machine says of a turn it would not take the results of, in place of the turn's own outcome.
STOPPED, FAULTED = 255, 254
HALTS = {STOPPED: "stopped", FAULTED: "faulted"}


@dataclass(frozen=True)
class Machine:
    shape: Shape
    bound: Bound              # the turn's: W = bound.c n + bound.d steps, n being a frame's cells and slot bytes
    timing: Timing
    cells: int
    slots: int
    first_slot: int
    pointer_bits: int
    core: str                 # serve_core.v
    top: str                  # machine.v
    image: str                # cells.mem
    turn_text: str            # the LFPL the turn was compiled from
    start: tuple              # the pool, the spare slots, ingress's record and the queue after its head
    queue_layout: tuple       # each cell of the queue's record, as its fields' offsets and sizes by name
    entry_layout: dict        # where an output's port and record are in its place's cell
    spare_layout: dict        # where a spare's slot and the next spare are in its cell

    def n_of(self, length: int) -> int:
        """The n a turn is charged for: ingress's record's cells, and the bytes its slot holds. A turn with no frame
        is charged for a full slot, since what the slot says it holds is the last frame's."""
        payload = self.shape.slot_bytes if length == 0 else min(max(length - self.shape.wire, 0), self.shape.slot_bytes)
        return self.shape.parts + payload

    def turn_cycles(self, length: int) -> int:
        """The most clock cycles a turn can take, from the edge that starts it to the edge that ends it, for a frame of
        this many bytes, or 0 for none: rtcheck's certificate with each instruction charged the clocks serve_core
        spends on it, d + c n."""
        clocks = self.timing.clocks
        return clocks.d + clocks.c * self.n_of(length)

    def report(self) -> str:
        bound, clocks, shape = self.bound, self.timing.clocks, self.shape
        steps = f"W = {_linear(bound.c, bound.d)} steps"
        turns = f"clocks <= {_linear(clocks.c, clocks.d)} a turn, certified"
        size = (f", n being a frame's {shape.parts} cells and the bytes of its slot" if bound.c or clocks.c
                else f" for any frame of up to {shape.capacity} bytes")
        return (f"turn {TURN} for handler {shape.handler}: {steps}{size}, with {shape.k_out} places in the queue; "
                f"{turns}; {self.cells} cells of {8 * CELL_BYTES} bits and {self.slots} slots, "
                f"{self.pointer_bits}-bit pointers")


def _linear(slope: int, constant: int) -> str:
    if not slope:
        return str(constant)
    return f"{slope} n + {constant}" if constant else f"{slope} n"


def build(program: bytes, *, k_out: int = 4, cells: int | None = None) -> Machine:
    text, shape = turn_source(program, k_out=k_out)
    lines = list(rt_lines(io.BytesIO(text.encode())))
    checker = Checker()
    for line in lines:
        checker.feed(line)
    checker.finish()
    queue = _record(checker, "Queue")
    # The turn holds two things that grow with frames: the frame ingress holds, whose size each turn says, and the
    # head of the queue, which is its own cell and a record with a slot. The rest of the queue, the pool and the spares
    # are charged nothing.
    sizes = {INGRESS: shape.n_max, INGRESS + 1: 1 + shape.n_max}
    bound = certify_turn(checker, TURN, sizes, per_turn=INGRESS)
    slots = k_out + 1  # ingress's, and one for each output the queue can hold
    needed = shape.parts + len(queue) + k_out + k_out * (1 + shape.parts) + (shape.parts + 1)
    count = needed if cells is None else cells
    if count < shape.parts + len(queue) + k_out:
        raise RTError(f"section 3.4: {count} cells cannot hold ingress, the queue and the spares, which take "
                      f"{shape.parts + len(queue) + k_out}")
    per_slot = shape.slot_bytes // CELL_BYTES
    first_slot = (count + 1 + per_slot - 1) // per_slot
    pointer_bits = _bits_for((first_slot + slots) * per_slot)
    image, start = _image(checker, shape, queue, count, first_slot, slots)
    emitter = Emitter(pointer_bits, header=start[2])  # ingress's record: registers, at the cells _image gave it
    for line in lines:
        emitter.feed(line)
    turn = next(f for f in emitter.functions.values() if f.turn)
    core = emitter.render(turn, "serve_core", bound, count, {}, sizes=sizes, slots=slots, per_turn=INGRESS)
    assert core.first_slot == first_slot
    entry, spare = _offsets(_ctor(checker, "Pending")), _offsets(_ctor(checker, "Spare"))
    top = _top(shape, pointer_bits, start, entry, core.timing.clocks, count)
    return Machine(shape, bound, core.timing, count, slots, first_slot, pointer_bits, core.verilog, top, image, text,
                   start, tuple(queue), entry, spare)


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


def _image(checker: Checker, shape: Shape, queue: list[dict], count: int, first_slot: int,
           slots: int) -> tuple[str, tuple]:
    """Ingress first: the record in cells 1 and on, holding the first slot. Then the queue, empty, then a cell for each
    other slot, the spares, and then the pool, every other cell, threaded through offset 0. Word 0 is null and never a
    cell, and the slots come after the cells."""
    words = [0] * (count + 1)

    def put(cell: int, off: int, value: int) -> None:
        words[cell] |= value << (8 * off)

    walk = shape.item_walk
    record = list(range(1, shape.parts + 1))
    for cell, off, inner in walk.links:
        put(record[cell], off, record[inner])
    assert walk.slot is not None
    put(record[walk.slot[0]], walk.slot[1], first_slot)  # it holds no bytes yet
    places = list(range(shape.parts + 1, shape.parts + len(queue) + 1))
    for position, fields in enumerate(queue):
        if LINK in fields:
            put(places[position], fields[LINK][0], places[position + 1])
    spare = _offsets(_ctor(checker, "Spare"))
    spares = list(range(places[-1] + 1, places[-1] + slots))
    for position, cell in enumerate(spares):
        put(cell, spare["spare"][0], first_slot + 1 + position)
        put(cell, spare["rest"][0], spares[position + 1] if position + 1 < len(spares) else 0)
    pool = list(range(spares[-1] + 1, count + 1))
    for position, cell in enumerate(pool):  # Cell(next), at offset 0
        put(cell, 0, pool[position + 1] if position + 1 < len(pool) else 0)
    text = "\n".join(f"{word:032x}" for word in words) + "\n"
    return text, (pool[0] if pool else 0, spares[0], record[0], places[0])


# --- the module around the turn

def _top(shape: Shape, p: int, start: tuple, entry: dict, clocks: Bound, cells: int) -> str:
    walk = shape.item_walk
    links = list(walk.links)
    assert walk.slot is not None and walk.length is not None
    wire_cells, wire_lanes = [], []
    for cell, off, size in walk.scalars:
        for byte in range(size):
            wire_cells.append(cell)
            wire_lanes.append(off + byte)
    parts = shape.parts
    cell_bits = max(1, _bits_for(parts))
    steps = len(links) + 1
    held_bits = (shape.k_out + 1).bit_length() + 1  # room for one past K_out, and for one below none
    slot_cell, slot_off = walk.slot
    index_bits = _bits_for(shape.slot_bytes)  # of a byte in a slot
    number_bits = p - (index_bits - 4)        # of a slot's number, in an address
    pool0, spares0, ingress0, queue0 = start
    verdicts = len(shape.verdicts)
    states = ("IDLE LINK LINK_TAKE IN FIN_SLOT FIN_LEN TURN TURN_WAIT ENTRY ENTRY_TAKE WORD WORD_TAKE BYTE SWORD "
              "SWORD_TAKE SBYTE SENT WIPE").split()
    wipe = "begin wipe <= {P{1'b0}}; first_word <= first[0]; state <= S_WIPE; end"
    out: list[str] = []
    add = out.append
    add("// Generated by src/machine.py: the hardware around the turn serve_core runs. Do not edit.")
    add(f"// A turn takes at most {_linear(clocks.c, clocks.d)} clock cycles, n being ingress's record's {parts} cells and")
    add("// the bytes its slot holds: serve_core.v says how that follows from its W. A turn that runs past that is stopped,")
    add("// and so is one whose results put an output at the head of the queue that no outcome accounts for, and every")
    add("// cell starts again as cells.mem has it. Ingress stores no byte past its slot, and egress sends no more than a")
    add("// slot holds. So nothing here runs on whatever a turn does, and no more frames go out than turns queued.")
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
    add("    output reg  [7:0]  turn_verdict,  // and for a frame the handler decided, its verdict's tag")
    add("    output reg  [31:0] overruns,      // turns stopped at the bound")
    add("    output reg  [31:0] faults,        // turns whose head of the queue no outcome accounts for")
    add(f"    output reg  [{32 * verdicts - 1}:0] verdicts  // turns that decided each value of the verdict, 32 bits each, "
        f"by tag")
    add(");")
    add(f"    localparam P = {p};")
    add(f"    localparam CELLS = {cells};")
    add(f"    localparam WIRE = 16'd{shape.wire};")
    add(f"    localparam SLOT_BYTES = 16'd{shape.slot_bytes};")
    add(f"    localparam CAPACITY = 16'd{shape.capacity};  // the record's bytes on the wire and a full slot")
    add(f"    localparam PARTS = 16'd{parts};")
    add(f"    localparam [31:0] LIMIT_BASE = 32'd{clocks.d};  // a turn's bound, d + c n")
    add(f"    localparam [31:0] LIMIT_PER = 32'd{clocks.c};")
    for name in ("queued", "dropped"):
        add(f"    localparam [7:0] {name.upper()} = 8'd{OUTCOMES.index(name)};")
    add(f"    localparam K_OUT = {shape.k_out};")
    add(f"    localparam [7:0] FAULTED = 8'd{FAULTED};")
    add(f"    localparam [7:0] STOPPED = 8'd{STOPPED};")
    for position, name in enumerate(states):
        add(f"    localparam [4:0] S_{name} = 5'd{position};")
    add("    reg [4:0] state;")
    add("    reg reading;                      // the links being followed are egress's, not ingress's")
    add("")
    add("    // What the turn holds from one turn to the next, and what it is told.")
    add("    reg [P-1:0] pool, spares, ingress, head, queue;")
    add("    reg [15:0] length;")
    add("    reg sent;")
    add("    reg start;")
    add("    reg [31:0] cycles;")
    add("    wire done, busy;")
    add("    wire [P-1:0] r_pool, r_spares, r_ingress, r_head, r_queue;")
    add("    wire [63:0] r_outcome;")
    add("    // Outputs the queue holds, as the outcomes of the turns tell it: Queued adds one, a release takes one. It")
    add("    // decides nothing a turn does. It is how the machine knows a turn's results cannot be taken.")
    add(f"    reg [{held_bits - 1}:0] held;")
    add(f"    wire [{held_bits - 1}:0] held_after = sent ? held - 1'b1 : r_outcome[7:0] == QUEUED ? held + 1'b1 : held;")
    add("    // This turn's bound: for the n of the frame ingress holds, or of a full slot, with none, since its slot says what")
    add("    // the last frame held.")
    add("    wire [15:0] payload = length == 16'd0 ? SLOT_BYTES : length < WIRE ? 16'd0 :")
    add("                          length - WIRE > SLOT_BYTES ? SLOT_BYTES : length - WIRE;")
    add("    wire [31:0] turn_limit = LIMIT_BASE + LIMIT_PER * {16'd0, PARTS + payload};")
    add("    wire stop = state == S_TURN_WAIT && !done && cycles == turn_limit;")
    add("    reg host_en;")
    add(f"    reg [{CELL_BYTES - 1}:0] host_we;")
    add("    reg [P-1:0] host_addr;")
    add(f"    reg [{8 * CELL_BYTES - 1}:0] host_din;")
    add(f"    wire [{8 * CELL_BYTES - 1}:0] host_dout;")
    add("    serve_core core (.clk(clk), .rst(rst || stop), .start(start), .arg0(pool), .arg1(spares), .arg2(ingress),")
    add("                     .arg3(head), .arg4(queue), .arg5({sent, length[14:0]}), .done(done), .busy(busy),")
    add("                     .res0(r_pool), .res1(r_spares), .res2(r_ingress), .res3(r_head), .res4(r_queue),")
    add("                     .res5(r_outcome), .host_en(host_en), .host_we(host_we), .host_addr(host_addr),")
    add("                     .host_din(host_din), .host_dout(host_dout));")
    add("")
    add("    // Every cell as it is before the first turn. A reset writes them back, and so does a turn that is stopped.")
    add(f"    reg [{8 * CELL_BYTES - 1}:0] first [0:CELLS];")
    add(f"    reg [{8 * CELL_BYTES - 1}:0] first_word;   // first[wipe], read the clock before it is written")
    add("    reg [P-1:0] wipe;")
    add('    initial $readmemh("cells.mem", first);')
    add("")
    add("    // Ingress and egress, between turns.")
    add(f"    reg [P-1:0] rec [0:{parts - 1}];   // the cells of the record being written or read")
    add(f"    reg [{number_bits - 1}:0] in_slot, out_slot;  // the number of ingress's slot, and of the one going out")
    add("    reg [15:0] count, total, emitted, index, j;  // total: the bytes going out, no more than a slot's past WIRE")
    add("    reg draining;                     // the rest of a frame past CAPACITY, read and thrown away")
    add(f"    reg [{max(1, _bits_for(steps + 1)) - 1}:0] step;")
    add(f"    reg [{8 * CELL_BYTES - 1}:0] word;")
    add(f"    reg [{cell_bits}:0] loaded;          // which of rec[] word holds, or none")
    add("    wire [15:0] byte_index = count - WIRE;     // of the byte ingress is writing, in its slot")
    add(f"    wire [15:0] said = host_dout[{8 * (slot_off + 4)} +: 16];  // what a slot field says its slot holds")
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
    add("    // The link step s follows: from which cell, and where in it. The last step reads the slot's field.")
    add("    function [P-1:0] link_from(input [3:0] s);")
    add("        case (s)")
    for index, (cell, _off, _inner) in enumerate(links):
        add(f"            4'd{index}: link_from = rec[{cell}];")
    add(f"            default: link_from = rec[{slot_cell}];")
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
    add("                end else if (count < CAPACITY) begin  // byte i of the slot: its cell, and its lane")
    add(f"                    host_addr = {{in_slot, byte_index[{index_bits - 1}:4]}};")
    add(f"                    host_we = {CELL_BYTES}'d1 << byte_index[3:0];")
    add(f"                    host_din = {{{8 * CELL_BYTES - 8}'d0, in_data}} << (8 * byte_index[3:0]);")
    add("                end")
    add("            end")
    add("            S_FIN_SLOT: begin  // what the slot holds, in the record's field: never its number")
    add(f"                host_addr = rec[{slot_cell}];")
    add(f"                host_we = {CELL_BYTES}'d3 << {slot_off + 4};")
    add(f"                host_din = {{{8 * CELL_BYTES - 16}'d0, count < WIRE ? 16'd0 : count - WIRE > SLOT_BYTES ? SLOT_BYTES "
        f": count - WIRE}} << {8 * (slot_off + 4)};")
    add("            end")
    add("            S_FIN_LEN: begin")
    add(f"                host_addr = rec[{walk.length[0]}];")
    add(f"                host_we = {CELL_BYTES}'d3 << {walk.length[1]};")
    add(f"                host_din = {{{8 * CELL_BYTES - 16}'d0, count}} << {8 * walk.length[1]};")
    add("            end")
    add("            S_ENTRY: begin host_en = 1'b1; host_addr = head; end")
    add("            S_WORD: begin host_en = 1'b1; host_addr = rec[wire_cell(index)]; end")
    add(f"            S_SWORD: begin host_en = 1'b1; host_addr = {{out_slot, j[{index_bits - 1}:4]}}; end")
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
    add(f"            verdicts <= {32 * verdicts}'d0;")
    add("            out_valid <= 1'b0;")
    add("            out_last <= 1'b0;")
    add("        end else case (state)")
    add("            S_WIPE: begin  // a cell a clock, and after the last, the registers as they were at the start")
    add("                wipe <= wipe + 1'b1;")
    add("                if (wipe != CELLS) first_word <= first[wipe + 1'b1];")
    add("                else begin")
    add(f"                    pool <= {p}'d{pool0};")
    add(f"                    spares <= {p}'d{spares0};")
    add(f"                    ingress <= {p}'d{ingress0};")
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
    add("                    rec[0] <= ingress;")
    add("                    reading <= 1'b0;")
    add("                    step <= 0;")
    add("                    count <= 16'd0;")
    add("                    state <= S_LINK;")
    add("                end")
    add("            end")
    add("            S_LINK: state <= S_LINK_TAKE;")
    add("            S_LINK_TAKE: begin")
    add("                case (step)")
    for index, (_cell, off, inner) in enumerate(links):
        add(f"                    {index}: rec[{inner}] <= host_dout[{8 * off} +: P];")
    add(f"                    default: if (reading) out_slot <= host_dout[{8 * slot_off} +: {number_bits}];")
    add(f"                    else in_slot <= host_dout[{8 * slot_off} +: {number_bits}];")
    add("                endcase")
    add(f"                if (step == {steps - 1}) begin")
    add("                    if (reading) begin")
    add("                        total <= WIRE + (said > SLOT_BYTES ? SLOT_BYTES : said);")
    add("                        index <= 16'd0;")
    add("                        emitted <= 16'd0;")
    add(f"                        loaded <= {cell_bits + 1}'d{parts};")
    add("                        state <= S_WORD;")
    add("                    end else state <= S_IN;")
    add("                end else state <= S_LINK;")
    add("                step <= step + 1;")
    add("            end")
    add("            S_IN: if (in_valid) begin")
    add("                count <= count + 16'd1;")
    add("                if (count == CAPACITY) begin  // a byte past what ingress holds ends the frame, too large")
    add("                    draining <= !in_last;")
    add("                    state <= S_FIN_SLOT;")
    add("                end else if (in_last) state <= S_FIN_SLOT;")
    add("            end")
    add("            S_FIN_SLOT: state <= S_FIN_LEN;")
    add("            S_FIN_LEN: begin length <= count; sent <= 1'b0; state <= S_TURN; end")
    add("            S_TURN: begin start <= 1'b1; cycles <= 32'hffffffff; state <= S_TURN_WAIT; end")
    add("            S_TURN_WAIT: if (done) begin")
    add("                turn_cycles <= cycles;")
    add("                turn_done <= 1'b1;")
    add("                if ((r_head != {P{1'b0}}) != (held_after != 0) || held_after > K_OUT) begin")
    add("                    faults <= faults + 32'd1;")
    add("                    turn_outcome <= FAULTED;")
    add(f"                    {wipe}")
    add("                end else begin")
    add("                    pool <= r_pool;")
    add("                    spares <= r_spares;")
    add("                    ingress <= r_ingress;")
    add("                    head <= r_head;")
    add("                    queue <= r_queue;")
    add("                    held <= held_after;")
    add("                    turn_outcome <= r_outcome[7:0];")
    add("                    turn_verdict <= r_outcome[15:8];")
    add("                    if (r_outcome[7:0] == QUEUED || r_outcome[7:0] == DROPPED) case (r_outcome[15:8])")
    for tag in range(verdicts):
        add(f"                        8'd{tag}: verdicts[{32 * tag + 31}:{32 * tag}] <= verdicts[{32 * tag + 31}:{32 * tag}] "
            f"+ 32'd1;")
    add("                        default: ;")
    add("                    endcase")
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
    add("                state <= S_LINK;")
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
    add("                    if (index + 16'd1 == WIRE) begin j <= 16'd0; state <= total == WIRE ? S_SENT : S_SWORD; end")
    add("                end")
    add("            end")
    add("            S_SWORD: state <= S_SWORD_TAKE;")
    add("            S_SWORD_TAKE: begin word <= host_dout; state <= S_SBYTE; end")
    add("            S_SBYTE: if (!out_valid || out_ready) begin")
    add("                out_valid <= 1'b1;")
    add("                out_data <= word[8 * j[3:0] +: 8];")
    add("                out_last <= emitted + 16'd1 == total;")
    add("                emitted <= emitted + 16'd1;")
    add("                j <= j + 16'd1;")
    add("                if (emitted + 16'd1 == total) state <= S_SENT;")
    add("                else if (j[3:0] == 4'd15) state <= S_SWORD;")
    add("            end")
    add("            S_SENT: if (!out_valid) begin")
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
    parser.add_argument("--k-out", type=int, default=4, help="places in the output queue")
    parser.add_argument("--cells", type=int, help="cells in the store, instead of what the turn needs")
    args = parser.parse_args(argv)
    try:
        with open(args.source, "rb") as handle:
            machine = build(handle.read(), k_out=args.k_out, cells=args.cells)
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
