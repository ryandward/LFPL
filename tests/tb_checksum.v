// Drives a generated handler with datagrams, the way src/runtime.py does on x86: it scatters each
// datagram into a chain of cells, starts the handler, and gathers what comes back. The scatter and the
// gather are test code, through the host port between items, because ingress and egress are not built
// in hardware yet. Neither counts towards the cycles. Every byte comes from a file, so that this runs
// the same bytes as the live UDP tests, from the same generators, and nobody retypes them.
//
//   iverilog -g2005 -DTOP=checksum -DPOINTER_BITS=14 -DSTREAM_BYTES=... -o sim tests/tb_checksum.v checksum.v
//   vvp sim +vectors=vectors.hex
//
// vectors.hex has one byte per line. For each item: the datagram's length, low byte first, the number of
// cells to give it as a reserve, low byte first, and then the datagram. A length of ffff ends the file.
//
// Cycles are counted from the edge that samples start to the edge that raises done. Reset, and loading
// cells, are not in the count.
`timescale 1ns/1ns
`default_nettype none

`ifndef TOP
 `define TOP checksum
`endif
`ifndef POINTER_BITS
 `define POINTER_BITS 14
`endif
`ifndef STREAM_BYTES
 `define STREAM_BYTES 16
`endif
// checksum's item is [port][sum, 2 bytes][chain], and test_walks.py's is [port][chain]. A sum is at
// offset 2 of the header cell, and leaves in front of the chain's bytes.
`ifndef SUM_BYTES
 `define SUM_BYTES 2
`endif

module tb;
    localparam P = `POINTER_BITS;
    localparam PAYLOAD = 7;           // the bytes of one chain cell, as the type declares it
    localparam LIMIT = 10000000;      // cycles: more than this for one item is a hang

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst = 1'b1;
    reg start = 1'b0;
    reg [P-1:0] item_in = 0;
    reg [P-1:0] free_in = 0;
    wire done;
    wire busy;
    wire [P-1:0] out_item;
    wire [P-1:0] free_out;
    reg host_en = 1'b0;
    reg [15:0] host_we = 16'd0;
    reg [P-1:0] host_addr = 0;
    reg [127:0] host_din = 128'd0;
    wire [127:0] host_dout;

    `TOP dut (.clk(clk), .rst(rst), .start(start), .item_in(item_in), .free_in(free_in), .done(done),
              .busy(busy), .out_item(out_item), .free_out(free_out), .host_en(host_en),
              .host_we(host_we), .host_addr(host_addr), .host_din(host_din), .host_dout(host_dout));

    // What the block RAM's ports are asked to do, at the edge that does it.
    integer reads = 0;
    integer writes = 0;
    integer both_ports = 0;       // one clock enabling a read and a write
    integer same_address = 0;     // and at one address: the case whose answer depends on the RAM
    always @(posedge clk) begin
        if (dut.a_en && |dut.b_we) begin
            both_ports = both_ports + 1;
            if (dut.a_addr == dut.b_addr) same_address = same_address + 1;
        end
        if (busy) begin
            if (dut.a_en) reads = reads + 1;
            if (|dut.b_we) writes = writes + 1;
        end
    end

    reg [7:0] stream [0:`STREAM_BYTES - 1];

    task host_write(input [P-1:0] address, input [15:0] enables, input [127:0] data);
        begin
            @(negedge clk);
            host_addr = address;
            host_we = enables;
            host_din = data;
        end
    endtask

    task host_read(input [P-1:0] address, output [127:0] data);
        begin
            @(negedge clk);
            host_we = 16'd0;
            host_addr = address;
            host_en = 1'b1;
            @(negedge clk);
            host_en = 1'b0;
            data = host_dout;
        end
    endtask

    reg [1023:0] path;
    integer at, length, reserve, body, body_cells, each, position, used;
    integer items, cycles, n, back;
    reg [127:0] word;
    reg [P-1:0] pointer;
    reg more;

    initial begin
        if (!$value$plusargs("vectors=%s", path)) begin
            $display("FAIL: no +vectors=file");
            $finish;
        end
        $readmemh(path, stream);
        repeat (4) @(negedge clk);    // reset: once, and not in any count
        rst = 1'b0;
        at = 0;
        items = 0;
        more = 1'b1;
        while (more) begin
            length = stream[at] + 256 * stream[at + 1];
            if (length == 65535) begin
                more = 1'b0;
            end else begin
                reserve = stream[at + 2] + 256 * stream[at + 3];
                at = at + 4;
                body = length - 1 - `SUM_BYTES;
                body_cells = (body + PAYLOAD - 1) / PAYLOAD;
                n = 1 + body_cells;

                // The chain, in the order the bytes came: a used count, 7 bytes, and the next cell.
                // Bytes past the used count are zero, as admission leaves them.
                for (each = 0; each < body_cells; each = each + 1) begin
                    used = (each == body_cells - 1) ? body - each * PAYLOAD : PAYLOAD;
                    word = 128'd0;
                    word[7:0] = used;
                    for (position = 0; position < used; position = position + 1)
                        word[8 * (position + 1) +: 8] = stream[at + 1 + `SUM_BYTES + each * PAYLOAD + position];
                    word[64 +: P] = (each == body_cells - 1) ? 0 : 3 + each;
                    host_write(2 + each, 16'hffff, word);
                end
                // The header writes its own fields and no others, so that a handler that read the
                // padding would read x, and print it.
                word = 128'd0;
                word[7:0] = stream[at];
                word[64 +: P] = body_cells ? 2 : 0;
                if (`SUM_BYTES == 2) begin
                    word[23:16] = stream[at + 1];
                    word[31:24] = stream[at + 2];
                    host_write(1, 16'hff0d, word);
                end else begin
                    host_write(1, 16'hff01, word);
                end
                // The reserve: a chain through offset 0.
                for (each = 0; each < reserve; each = each + 1) begin
                    word = 128'd0;
                    word[0 +: P] = (each == reserve - 1) ? 0 : 2 + body_cells + each + 1;
                    host_write(2 + body_cells + each, 16'h00ff, word);
                end
                @(negedge clk);
                host_we = 16'd0;
                item_in = 1;
                free_in = reserve ? 2 + body_cells : 0;
                at = at + length;

                reads = 0;
                writes = 0;
                start = 1'b1;
                @(posedge clk);       // this edge samples start, and the count starts from it
                #1 cycles = 0;
                @(negedge clk);
                start = 1'b0;
                while (!done) begin
                    @(posedge clk);
                    #1 cycles = cycles + 1;
                    if (cycles > LIMIT) begin
                        $display("FAIL: item %0d did not finish in %0d cycles, state %0d", items, cycles, dut.state);
                        $finish;
                    end
                end
                $write("ITEM %0d length=%0d n=%0d reserve=%0d cycles=%0d reads=%0d writes=%0d out=",
                       items, length, n, reserve, cycles, reads, writes);

                // What egress would send: the scalars, and each chain cell's used bytes. The count
                // is clamped as egress clamps it, since a program can write it.
                host_read(out_item, word);
                if (`SUM_BYTES == 2) $write("%02x%02x", word[23:16], word[31:24]);
                pointer = word[64 +: P];
                back = 1;
                while (pointer != 0) begin
                    host_read(pointer, word);
                    used = (word[7:0] > PAYLOAD) ? PAYLOAD : word[7:0];
                    for (position = 0; position < used; position = position + 1)
                        $write("%02x", word[8 * (position + 1) +: 8]);
                    pointer = word[64 +: P];
                    back = back + 1;
                end
                pointer = free_out;
                while (pointer != 0) begin
                    host_read(pointer, word);
                    pointer = word[0 +: P];
                    back = back + 1;
                end
                $display(" back=%0d", back);
                items = items + 1;
            end
        end
        $display("DONE items=%0d both_ports=%0d same_address=%0d", items, both_ports, same_address);
        $finish(0);
    end
endmodule
`default_nettype wire
