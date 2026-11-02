// Drives src/machine.py's machine: frames in as bytes, and what comes out printed as it leaves.
//
//   iverilog -g2005 -DCELLS=1310 -o sim tests/tb_machine.v machine.v serve_core.v
//   vvp sim +frames=frames.txt          (in the directory that holds cells.mem)
//
// frames.txt has a frame to a line: `hold length byte byte ...`, in hex bytes. With hold set, egress is held still
// while the frame goes in, which is how a queue fills. Afterwards egress is let go, and once the machine has been
// idle with nothing queued for a while, it prints where the pool, ingress and the queue are, and every cell.
`timescale 1ns/1ns
`default_nettype none

`ifndef CELLS
 `define CELLS 1310
`endif

module tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst = 1'b1;
    reg in_valid = 1'b0;
    reg [7:0] in_data = 8'd0;
    reg in_last = 1'b0;
    wire in_ready;
    wire out_valid;
    wire [7:0] out_data;
    wire out_last;
    wire [7:0] out_port;
    reg out_ready = 1'b1;
    wire turn_done;
    wire [31:0] turn_cycles;
    wire [7:0] turn_outcome;

    machine dut (.clk(clk), .rst(rst), .in_valid(in_valid), .in_data(in_data), .in_last(in_last),
                 .in_ready(in_ready), .out_valid(out_valid), .out_data(out_data), .out_last(out_last),
                 .out_port(out_port), .out_ready(out_ready), .turn_done(turn_done), .turn_cycles(turn_cycles),
                 .turn_outcome(turn_outcome));

    integer out_bytes = 0;
    always @(posedge clk) begin
        if (out_valid && out_ready) begin
            if (out_bytes == 0) $write("OUT %0d ", out_port);
            $write("%02x", out_data);
            out_bytes = out_bytes + 1;
            if (out_last) begin
                $write("\n");
                out_bytes = 0;
            end
        end
        if (turn_done) $display("TURN %0d %0d", turn_outcome, turn_cycles);
    end

    reg [1023:0] path;
    integer file, got, hold, length, index, value, idle;
    initial begin
        if (!$value$plusargs("frames=%s", path)) begin
            $display("FAIL: no +frames=file");
            $finish(0);
        end
        file = $fopen(path, "r");
        repeat (4) @(negedge clk);
        rst = 1'b0;
        while ($fscanf(file, "%d %d", hold, length) == 2) begin
            out_ready = hold == 0;
            for (index = 0; index < length; index = index + 1) begin
                got = $fscanf(file, "%h", value);
                in_valid = 1'b1;
                in_data = value[7:0];
                in_last = index == length - 1;
                @(negedge clk);
                while (!in_ready) @(negedge clk);
                @(posedge clk);
                #1;
            end
            in_valid = 1'b0;
            in_last = 1'b0;
        end
        out_ready = 1'b1;
        idle = 0;
        while (idle < 50) begin
            @(negedge clk);
            idle = (dut.state == 5'd0 && dut.head == 0 && !out_valid) ? idle + 1 : 0;
        end
        $display("FINAL %0d %0d %0d", dut.pool, dut.slot, dut.queue);
        for (index = 1; index <= `CELLS; index = index + 1)
            $display("WORD %0d %032x", index, dut.core.cells[index]);
        $display("DONE");
        $finish(0);
    end
endmodule
`default_nettype wire
