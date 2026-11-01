// Runs a generated handler over cell images, for handlers whose items and outputs have any layout.
// tests/test_rtverilog.py scatters each datagram into cells as src/runtime.py would, this loads them
// and runs the handler, and it prints the words back for the test to gather what egress would send.
//
//   iverilog -g2005 -DTOP=h -DPOINTER_BITS=14 -o sim tests/tb_image.v design.v
//   vvp sim +images=images.txt
//
// images.txt, in hex: for each item, `item_in free_in loads words`, then that many `address enables
// data`. Then `words` is how many cells to print back. An item_in of 0 ends the file.
`timescale 1ns/1ns
`default_nettype none

`ifndef TOP
 `define TOP h
`endif
`ifndef POINTER_BITS
 `define POINTER_BITS 14
`endif

module tb;
    localparam P = `POINTER_BITS;
    localparam LIMIT = 100000000;

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

    integer both_ports = 0;
    integer same_address = 0;
    always @(posedge clk)
        if (dut.a_en && |dut.b_we) begin
            both_ports = both_ports + 1;
            if (dut.a_addr == dut.b_addr) same_address = same_address + 1;
        end

    reg [1023:0] path;
    integer file, got, loads, words, each, cycles, items;
    reg [31:0] item_word, free_word, address;
    reg [15:0] enables;
    reg [127:0] data;
    reg more;

    initial begin
        if (!$value$plusargs("images=%s", path)) begin
            $display("FAIL: no +images=file");
            $finish(0);
        end
        file = $fopen(path, "r");
        repeat (4) @(negedge clk);
        rst = 1'b0;
        items = 0;
        more = 1'b1;
        while (more) begin
            got = $fscanf(file, "%h %h %h %h", item_word, free_word, loads, words);
            if (got != 4 || item_word == 0) begin
                more = 1'b0;
            end else begin
                for (each = 0; each < loads; each = each + 1) begin
                    got = $fscanf(file, "%h %h %h", address, enables, data);
                    @(negedge clk);
                    host_addr = address[P-1:0];
                    host_we = enables;
                    host_din = data;
                end
                @(negedge clk);
                host_we = 16'd0;
                item_in = item_word[P-1:0];
                free_in = free_word[P-1:0];
                start = 1'b1;
                @(posedge clk);
                #1 cycles = 0;
                @(negedge clk);
                start = 1'b0;
                while (!done) begin
                    @(posedge clk);
                    #1 cycles = cycles + 1;
                    if (cycles > LIMIT) begin
                        $display("FAIL: item %0d did not finish, state %0d", items, dut.state);
                        $finish(0);
                    end
                end
                $display("ITEM %0d cycles=%0d out=%0d free=%0d", items, cycles, out_item, free_out);
                for (each = 1; each <= words; each = each + 1) begin
                    @(negedge clk);
                    host_addr = each[P-1:0];
                    host_en = 1'b1;
                    @(negedge clk);
                    host_en = 1'b0;
                    $display("WORD %0d %032x", each, host_dout);
                end
                items = items + 1;
            end
        end
        $display("DONE items=%0d both_ports=%0d same_address=%0d", items, both_ports, same_address);
        $finish(0);
    end
endmodule
`default_nettype wire
