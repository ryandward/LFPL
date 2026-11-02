"""Place and route a handler on an iCE40, and turn its bound in cycles into a bound in seconds.

    python3 tools/place_and_route.py [examples/checksum.lfpl] [--cells 432] [--seeds 5]
    python3 tools/place_and_route.py --machine DIR [--seeds 5]

With --machine, DIR holds what `python3 -m src.machine ... -o DIR` wrote, and that machine is placed instead, its
ports behind shift registers the same way. It reads cells.mem from DIR, as the machine does.

Needs yosys and nextpnr-ice40, which NEXTPNR_ICE40 can name if it is not on the path. The handler's
ports are far more than a package has pins, so a wrapper puts them all behind shift registers, and
every timing path that counts is between flip-flops. fmax is nextpnr's static timing analysis of the
routed design, at the HX8K's speed grade. It has not been measured on a chip.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.rtlower import rt_lines  # noqa: E402
from src.rtverilog import CELL_BITS, CELL_BYTES, translate  # noqa: E402

WRAPPER = """`default_nettype none
module wrapper (input wire clk, input wire serial_in, input wire load, output reg serial_out);
    localparam P = {pointer};
    localparam INPUTS = 1 + 1 + P + P + 1 + {enables} + P + {bits};
    localparam OUTPUTS = 1 + 1 + P + P + {bits};
    reg [INPUTS - 1:0] shifting, held;
    reg [OUTPUTS - 1:0] leaving;
    wire rst = held[0];
    wire start = held[1];
    wire [P - 1:0] item_in = held[2 +: P];
    wire [P - 1:0] free_in = held[2 + P +: P];
    wire host_en = held[2 + 2 * P];
    wire [{enables} - 1:0] host_we = held[3 + 2 * P +: {enables}];
    wire [P - 1:0] host_addr = held[3 + 2 * P + {enables} +: P];
    wire [{bits} - 1:0] host_din = held[3 + 3 * P + {enables} +: {bits}];
    wire done, busy;
    wire [P - 1:0] out_item, free_out;
    wire [{bits} - 1:0] host_dout;
    {module} handler (.clk(clk), .rst(rst), .start(start), .item_in(item_in), .free_in(free_in),
        .done(done), .busy(busy), .out_item(out_item), .free_out(free_out), .host_en(host_en),
        .host_we(host_we), .host_addr(host_addr), .host_din(host_din), .host_dout(host_dout));
    always @(posedge clk) begin
        shifting <= {{shifting[INPUTS - 2:0], serial_in}};
        if (load) held <= shifting;
        leaving <= load ? {{done, busy, out_item, free_out, host_dout}} : {{leaving[OUTPUTS - 2:0], 1'b0}};
        serial_out <= leaving[OUTPUTS - 1];
    end
endmodule
`default_nettype wire
"""

MACHINE_WRAPPER = """`default_nettype none
module wrapper (input wire clk, input wire serial_in, input wire load, output reg serial_out);
    localparam INPUTS = 12;   // rst, in_valid, in_data, in_last, out_ready
    localparam OUTPUTS = 156; // in_ready, out_valid, out_data, out_last, out_port, turn_done, and the counts
    reg [INPUTS - 1:0] shifting, held;
    reg [OUTPUTS - 1:0] leaving;
    wire in_ready, out_valid, out_last, turn_done;
    wire [7:0] out_data, out_port, turn_outcome;
    wire [31:0] turn_cycles, overruns, faults, cuts;
    machine dut (.clk(clk), .rst(held[0]), .in_valid(held[1]), .in_data(held[2 +: 8]), .in_last(held[10]),
        .in_ready(in_ready), .out_valid(out_valid), .out_data(out_data), .out_last(out_last), .out_port(out_port),
        .out_ready(held[11]), .turn_done(turn_done), .turn_cycles(turn_cycles), .turn_outcome(turn_outcome),
        .overruns(overruns), .faults(faults), .cuts(cuts));
    always @(posedge clk) begin
        shifting <= {shifting[INPUTS - 2:0], serial_in};
        if (load) held <= shifting;
        leaving <= load ? {in_ready, out_valid, out_data, out_last, out_port, turn_done, turn_cycles, turn_outcome,
                           overruns, faults, cuts} : {leaving[OUTPUTS - 2:0], 1'b0};
        serial_out <= leaving[OUTPUTS - 1];
    end
endmodule
`default_nettype wire
"""


def place(nextpnr: str, workdir: str, sources: list, wrapper: str, args) -> list:
    """Synthesize in `workdir`, where the sources read their memories from, then route `args.seeds` times."""
    with open(os.path.join(workdir, "wrapper.v"), "w") as handle:
        handle.write(wrapper)
    netlist = os.path.join(workdir, "out.json")
    files = " ".join(sources + [os.path.join(workdir, "wrapper.v")])
    done = subprocess.run(["yosys", "-q", "-p", f"read_verilog -sv {files}; synth_ice40 -top wrapper -json {netlist}; "
                           f"tee -o {workdir}/stat.txt stat"], capture_output=True, text=True, cwd=workdir)
    if done.returncode:
        print(done.stdout + done.stderr, file=sys.stderr)
        return []
    with open(f"{workdir}/stat.txt") as handle:
        for line in handle:
            if re.search(r"SB_RAM40_4K|SB_DFF|SB_LUT4|SB_CARRY", line):
                print(line.rstrip())
    speeds = []
    for seed in range(1, args.seeds + 1):
        done = subprocess.run([nextpnr, f"--{args.device}", "--package", args.package, "--json", netlist,
                               "--freq", str(args.target), "--seed", str(seed), "--pcf-allow-unconstrained",
                               "--timing-allow-fail"],  # what it reaches is the answer, whatever was asked
                              capture_output=True, text=True)
        found = re.findall(r"Max frequency for clock\s+'[^']*':\s*([\d.]+) MHz", done.stderr)
        if done.returncode or not found:
            print(done.stderr[-2000:], file=sys.stderr)
            return []
        speeds.append(float(found[-1]))  # the last report is the routed one
        used = re.findall(r"ICESTORM_LC:\s*(\d+)/\s*(\d+)", done.stderr)
        rams = re.findall(r"ICESTORM_RAM:\s*(\d+)/\s*(\d+)", done.stderr)
        print(f"seed {seed}: fmax {speeds[-1]:.2f} MHz, {used[-1][0]} of {used[-1][1]} logic cells, "
              f"{rams[-1][0]} of {rams[-1][1]} block RAMs")
    return speeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", nargs="?", default=os.path.join(ROOT, "examples", "checksum.lfpl"))
    parser.add_argument("--cells", type=int, default=432, help="an image that fits the part: ingress and one item")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", default="hx8k")
    parser.add_argument("--package", default="ct256")
    parser.add_argument("--target", type=float, default=60.0,
                        help="the MHz to ask nextpnr for: far past what it can do, and routing takes minutes")
    parser.add_argument("--machine", metavar="DIR", help="the machine src/machine.py wrote into DIR, not a handler")
    args = parser.parse_args()
    nextpnr = os.environ.get("NEXTPNR_ICE40") or shutil.which("nextpnr-ice40")
    if not nextpnr:
        print("nextpnr-ice40 is not on the path, and NEXTPNR_ICE40 does not name it", file=sys.stderr)
        return 2
    if args.machine:
        directory = os.path.abspath(args.machine)
        speeds = place(nextpnr, directory, [os.path.join(directory, name) for name in ("machine.v", "serve_core.v")],
                       MACHINE_WRAPPER, args)
        if not speeds:
            return 1
        print(f"the slowest of {args.seeds} seeds: {min(speeds):.2f} MHz")
        return 0
    with open(args.source, "rb") as handle:
        translation = translate(rt_lines(handle), cells=args.cells)
    print(translation.report())
    with tempfile.TemporaryDirectory() as workdir:
        design = os.path.join(workdir, "design.v")
        with open(design, "w") as handle:
            handle.write(translation.verilog)
        speeds = place(nextpnr, workdir, [design], WRAPPER.format(pointer=translation.pointer_bits, enables=CELL_BYTES,
                                                                  bits=CELL_BITS, module=translation.module), args)
    if not speeds:
        return 1
    fmax = min(speeds)
    print(f"the slowest of {args.seeds} seeds: {fmax:.2f} MHz")
    print("    n  clocks(n)  bound in seconds")
    for n in (1, 2, 3, 144, 216):
        clocks = translation.timing.bound(n)
        print(f"{n:>5} {clocks:>10} {clocks / (fmax * 1e6) * 1e6:>14.2f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
