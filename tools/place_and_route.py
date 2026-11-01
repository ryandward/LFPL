"""Place and route a handler on an iCE40, and turn its bound in cycles into a bound in seconds.

    python3 tools/place_and_route.py [examples/checksum.lfpl] [--cells 432] [--seeds 5]

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", nargs="?", default=os.path.join(ROOT, "examples", "checksum.lfpl"))
    parser.add_argument("--cells", type=int, default=432, help="an image that fits the part: ingress and one item")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--device", default="hx8k")
    parser.add_argument("--package", default="ct256")
    parser.add_argument("--target", type=float, default=60.0,
                        help="the MHz to ask nextpnr for: far past what it can do, and routing takes minutes")
    args = parser.parse_args()
    nextpnr = os.environ.get("NEXTPNR_ICE40") or shutil.which("nextpnr-ice40")
    if not nextpnr:
        print("nextpnr-ice40 is not on the path, and NEXTPNR_ICE40 does not name it", file=sys.stderr)
        return 2
    with open(args.source, "rb") as handle:
        translation = translate(rt_lines(handle), cells=args.cells)
    print(translation.report())
    with tempfile.TemporaryDirectory() as workdir:
        design, wrapper, netlist = (os.path.join(workdir, name) for name in ("design.v", "wrapper.v", "out.json"))
        with open(design, "w") as handle:
            handle.write(translation.verilog)
        with open(wrapper, "w") as handle:
            handle.write(WRAPPER.format(pointer=translation.pointer_bits, enables=CELL_BYTES, bits=CELL_BITS,
                                        module=translation.module))
        done = subprocess.run(["yosys", "-q", "-p", f"read_verilog -sv {design} {wrapper}; "
                               f"synth_ice40 -top wrapper -json {netlist}; tee -o {workdir}/stat.txt stat"],
                              capture_output=True, text=True)
        if done.returncode:
            print(done.stdout + done.stderr, file=sys.stderr)
            return 1
        with open(f"{workdir}/stat.txt") as handle:
            for line in handle:
                if re.search(r"SB_RAM40_4K|SB_DFF|SB_LUT4|SB_CARRY", line):
                    print(line.rstrip())
        speeds = []
        for seed in range(1, args.seeds + 1):
            done = subprocess.run([nextpnr, f"--{args.device}", "--package", args.package, "--json", netlist,
                                   "--freq", str(args.target), "--seed", str(seed), "--pcf-allow-unconstrained"],
                                  capture_output=True, text=True)
            found = re.findall(r"Max frequency for clock\s+'[^']*':\s*([\d.]+) MHz", done.stderr)
            if done.returncode or not found:
                print(done.stderr[-2000:], file=sys.stderr)
                return 1
            speeds.append(float(found[-1]))  # the last report is the routed one
            used = re.findall(r"ICESTORM_LC:\s*(\d+)/\s*(\d+)", done.stderr)
            print(f"seed {seed}: fmax {speeds[-1]:.2f} MHz, {used[-1][0]} of {used[-1][1]} logic cells")
    fmax = min(speeds)
    timing, bound = translation.timing, translation.bound
    print(f"the slowest of {args.seeds} seeds: {fmax:.2f} MHz")
    print("    n  k*W(n)+c cycles  bound in seconds")
    for n in (1, 2, 3, 144, 216):
        cycles = timing.bound(n, bound)
        print(f"{n:>5} {cycles:>15} {cycles / (fmax * 1e6) * 1e6:>14.2f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
