"""Build a program for the AF_XDP runtime: what `python3 -m src.lfplo` does for the UDP one.

    python3 -m experimental.build SOURCE -o EXE --xdp INTERFACE --map PINNED --ctl-port PORT
                                  [--frames F] [--k-out K] [--queue Q] [--state NAME=FILE]
                                  [--asm FILE] [--count-steps]

The runtime lives here and not in src/, so this is the one place the compiler's parts are put together
for it. Every check of src/lfplo.py runs: the type check on every line, the certificate, the symbolic check
and the stack, and a failure in any of them leaves no binary.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

from experimental import runtime_xdp
from experimental.runtime_xdp import XdpImage, XdpOptions
from src.certify import certify
from src.lfplo import Report, Refused, _link, _remove_stale, _validate
from src.ring import FStarved, Reject
from src.rtcheck import Checker, RTError
from src.rtlower import Limit, rt_lines
from src.runtime import BindingError
from src.x86 import Backend, Starved


def _generate(lines: Iterable[str], asm_path: str, options: XdpOptions, initial: dict[str, str],
              *, count_steps: bool = False) -> XdpImage:
    checker = Checker()
    with open(asm_path, "w") as handle:
        handle.write("default abs\nsection .text\nglobal _start\n")
        backend = Backend(lambda line: handle.write(line + "\n"), count_steps=count_steps)
        for line in lines:
            checker.feed(line)
            backend.feed(line)
        checker.finish()
        handlers = [f for f in checker.functions.values() if f.handler]
        if len(handlers) != 1:
            raise Refused(f"section 3.1: this runtime serves exactly one handler, and the program "
                          f"declares {len(handlers)}")
        name = handlers[0].name
        depth = next(f.depth for f in backend.functions.values() if f.name == name)
        certify(checker, name, runtime_xdp.item_cells(checker))
        image = runtime_xdp.bind(checker, name, depth, options, dict(initial))
        for state in image.states:
            if state.initial is None:
                continue
            size = os.path.getsize(state.initial)
            if size != state.length * state.element_size:
                raise Refused(f"state {state.name} needs {state.length * state.element_size} "
                              f"bytes, and {state.initial} has {size}")
        handle.write(runtime_xdp.assembly(image))
    return image


def build(source_path: str, exe_path: str, options: XdpOptions, initial: dict[str, str],
          *, keep_asm: str = None, count_steps: bool = False) -> Report:
    _remove_stale(exe_path)
    initial = {name: os.path.abspath(path) for name, path in initial.items()}
    with tempfile.TemporaryDirectory() as work:
        asm_path = os.path.join(work, "program.asm")
        try:
            with open(source_path, "rb") as handle:
                source = handle.read()
                handle.seek(0)
                image = _generate(rt_lines(handle), asm_path, options, initial)
        except (RTError, BindingError, Starved, Limit, FStarved) as failure:
            raise Refused(str(failure)) from failure
        if keep_asm is not None:
            shutil.copy(asm_path, keep_asm)
        _validate(source, asm_path, image)
        if count_steps:
            with open(source_path, "rb") as handle:
                _generate(rt_lines(handle), asm_path, options, initial, count_steps=True)
        _link(asm_path, exe_path)
    return Report(image.handler, image.bound, image.depth, image.cell_count, image.bytes)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="experimental.build", description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("-o", dest="output", required=True)
    parser.add_argument("--xdp", metavar="INTERFACE", required=True)
    parser.add_argument("--map", required=True, help="the socket map experimental/xdp_load.py pinned")
    parser.add_argument("--ctl-port", type=int, required=True)
    parser.add_argument("--frames", type=int, default=4096)
    parser.add_argument("--k-out", type=int, default=2048)
    parser.add_argument("--queue", type=int, default=0)
    parser.add_argument("--state", action="append", default=[], metavar="NAME=FILE")
    parser.add_argument("--asm")
    parser.add_argument("--count-steps", action="store_true")
    args = parser.parse_args(argv)
    initial = {}
    for entry in args.state:
        name, _, path = entry.partition("=")
        initial[name] = path
    try:
        _remove_stale(args.output)
        options = XdpOptions(args.xdp, args.map, args.ctl_port, args.frames, args.k_out, args.queue)
        report = build(args.source, args.output, options, initial, keep_asm=args.asm,
                       count_steps=args.count_steps)
    except (Refused, Reject, BindingError, OSError) as failure:
        print(f"{args.source}: refused: {failure}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as failure:
        print(f"{args.source}: the assembler or linker failed:\n{failure.stderr}", file=sys.stderr)
        return 1
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
