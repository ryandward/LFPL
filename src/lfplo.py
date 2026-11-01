"""The LFPL-O compiler: source to a running binary, or a refusal and no binary.

    Lex -> Parse -> Check -> Lower (RT) -> rtcheck -> x86 -> certificate -> symcheck, stack -> nasm, ld

LFPL-O section 8.4 is the shape of this file. Those checks run on every build, and a failure in any
of them leaves nothing behind, so a program that compiled is one that passed. The fourth check of
that section, redundant moves, runs only in the test suite so far.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Union

from . import runtime_xdp
from .certify import certify
from .ring import FStarved, Reject, Ring
from .rtcheck import Bound, Checker, RTError
from .rtlower import Limit, rt_lines
from .runtime import BindingError, Image, Options, assembly, bind
from .symcheck import check_translation
from .validate import recompute_stack
from .x86 import Backend, Starved


class Refused(Exception):
    """The program, or this build of it, is refused. No binary exists."""


@dataclass(frozen=True)
class Report:
    handler: str
    bound: Bound
    stack: int
    cells: int
    image_bytes: int

    def __str__(self) -> str:
        return (f"handler {self.handler}: R(n) = {_linear(self.bound.a, self.bound.b)} cells, "
                f"W(n) = {_linear(self.bound.c, self.bound.d)} steps, stack {self.stack} bytes, "
                f"{self.cells} cells, image {self.image_bytes} bytes")


def _linear(slope: int, constant: int) -> str:
    if not slope:
        return str(constant)
    return f"{slope} n + {constant}" if constant else f"{slope} n"


def _generate(lines: Iterable[str], asm_path: str, options: Union[Options, runtime_xdp.XdpOptions],
              initial: dict[str, str], *, count_steps: bool = False) -> Union[Image, runtime_xdp.XdpImage]:
    """The stream goes through the checker and then the backend, one line at a time, so nothing
    is lowered that has not been typed, and nothing is held that is not in flight."""
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
        if isinstance(options, runtime_xdp.XdpOptions):
            certify(checker, name, runtime_xdp.item_cells(checker))
            image = runtime_xdp.bind(checker, name, depth, options, dict(initial))
        else:
            certify(checker, name, options.n_max)
            image = bind(checker, name, depth, options, dict(initial))
        for state in image.states:
            if state.initial is None:
                continue
            size = os.path.getsize(state.initial)
            if size != state.length * state.element_size:
                raise Refused(f"state {state.name} needs {state.length * state.element_size} "
                              f"bytes, and {state.initial} has {size}")
        handle.write(runtime_xdp.assembly(image) if isinstance(image, runtime_xdp.XdpImage)
                     else assembly(image))
    return image


def compile_program(source: bytes, *, target_cells: int = 1024,
                    on_step: Optional[Callable[[Ring], None]] = None) -> str:
    """A Phase 1 program, with `fn main` and no handler: the assembly, to run once and exit. The
    runtime is not involved, so this is what shows the compiler does what the old one did."""
    from .lower import _trailer

    out: list[str] = []
    checker = Checker()
    backend = Backend(out.append)
    for line in rt_lines(io.BytesIO(source), on_step, phase1=True):
        checker.feed(line)
        backend.feed(line)
    checker.finish()
    main = next((f for f in checker.functions.values() if f.name == "main"), None)
    mode = 0 if main is None else (2 if main.params else 1)
    return ("default abs\nsection .text\nglobal _start\n" + "\n".join(out) + "\n"
            + _trailer(mode, target_cells) + "\n")


def _validate(source: bytes, asm_path: str, image: Union[Image, runtime_xdp.XdpImage]) -> None:
    """Section 8.4. Meaning, and the stack. A build that fails either has no binary."""
    with open(asm_path) as handle:
        asm = handle.read()
    try:
        problems = check_translation(source, asm)
    except Exception as failure:  # a crash in the oracle is not a pass, and it is not the program's
        raise Refused(f"translation validation could not finish: {type(failure).__name__}: "
                      f"{failure}") from failure
    if problems:
        raise Refused("translation validation failed:\n" + "\n".join(f"  {p}" for p in problems))
    deepest = recompute_stack(asm)
    if deepest != image.depth:
        raise Refused(f"the stack recomputed from the assembly is {deepest} bytes, and the "
                      f"compiler reserved for {image.depth}")


def _link(asm_path: str, exe_path: str) -> None:
    work = os.path.dirname(asm_path)
    obj = os.path.join(work, "program.o")
    built = os.path.join(work, "program")
    # -Werror: NASM only warns when it truncates an immediate, which is a miscompile.
    subprocess.run(["nasm", "-Werror", "-f", "elf64", "-o", obj, asm_path], check=True,
                   capture_output=True, text=True)
    subprocess.run(["ld", "-o", built, obj], check=True, capture_output=True, text=True)
    shutil.move(built, exe_path)


def _remove_stale(exe_path: str) -> None:
    if os.path.exists(exe_path):
        os.unlink(exe_path)


def build(source_path: str, exe_path: str, options: Union[Options, runtime_xdp.XdpOptions],
          initial: Optional[dict[str, str]] = None, *, keep_asm: Optional[str] = None,
          count_steps: bool = False) -> Report:
    """Compile. A stale binary is removed first, so failure can never leave one behind."""
    _remove_stale(exe_path)
    initial = {name: os.path.abspath(path) for name, path in (initial or {}).items()}
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
        if count_steps:  # the code just validated, with a counter added at each step
            with open(source_path, "rb") as handle:
                _generate(rt_lines(handle), asm_path, options, initial, count_steps=True)
        _link(asm_path, exe_path)
    return Report(image.handler, image.bound, image.depth, image.cell_count, image.bytes)


def _print_rt(source_path: str) -> int:
    checker = Checker()
    try:
        with open(source_path, "rb") as handle:
            for line in rt_lines(handle):
                checker.feed(line)
                print(line)
        checker.finish()
    except (Reject, RTError, Limit, FStarved, OSError) as failure:
        print(f"{source_path}: refused: {failure}", file=sys.stderr)
        return 1
    return 0


def _given(value: Optional[int], default: int) -> int:
    """Not `value or default`: 0 is a value, and the options have something to say about it."""
    return default if value is None else value


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lfplo", description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("--rt", action="store_true", help="print the checked RT stream, and stop")
    parser.add_argument("-o", dest="output")
    parser.add_argument("--in-port", type=int)
    parser.add_argument("--out-port", type=int,
                        help="egress goes to this port plus the handler's port number")
    parser.add_argument("--ctl-port", type=int)
    parser.add_argument("--state", action="append", default=[], metavar="NAME=FILE",
                        help="initial contents of a state array, read at link time")
    parser.add_argument("--xdp", metavar="INTERFACE",
                        help="serve frames from this interface through AF_XDP, instead of datagrams")
    parser.add_argument("--map", help="with --xdp: the socket map tools/xdp_load.py pinned")
    parser.add_argument("--frames", type=int, help="with --xdp: frames in the packet memory, 4096 if not given")
    parser.add_argument("--queue", type=int, help="with --xdp: the interface's receive queue, 0 if not given")
    parser.add_argument("--n-max", type=int, help="the largest item, in cells, 216 if not given")
    parser.add_argument("--k-out", type=int, help="the output queue's capacity: 64, or 2048 with --xdp")
    parser.add_argument("--cells", type=int, help="cells in the image, instead of section 3.4's C")
    parser.add_argument("--asm", help="also write the assembly here")
    parser.add_argument("--count-steps", action="store_true",
                        help="count the steps that run, to measure W(n) against what they were")
    parser.add_argument("--measure-age", action="store_true",
                        help="keep max_age_ns, at the cost of a clock read for each admitted item")
    args = parser.parse_args(argv)
    if args.rt:
        return _print_rt(args.source)
    if args.xdp is None and None in (args.output, args.in_port, args.out_port, args.ctl_port):
        parser.error("a build needs -o, --in-port, --out-port and --ctl-port")
    if args.xdp is not None and None in (args.output, args.map, args.ctl_port):
        parser.error("a build with --xdp needs -o, --map and --ctl-port")
    # An option that means nothing for this runtime is refused, because ignoring it would leave
    # someone believing a binary has a property it does not.
    for name, value in (("--in-port", args.in_port), ("--out-port", args.out_port), ("--n-max", args.n_max),
                        ("--cells", args.cells), ("--measure-age", args.measure_age or None)):
        if args.xdp is not None and value is not None:
            parser.error(f"{name} has no meaning with --xdp")
    for name, value in (("--map", args.map), ("--frames", args.frames), ("--queue", args.queue)):
        if args.xdp is None and value is not None:
            parser.error(f"{name} has no meaning without --xdp")
    initial = {}
    for entry in args.state:
        name, _, path = entry.partition("=")
        initial[name] = path
    try:
        _remove_stale(args.output)
        if args.xdp is not None:
            options = runtime_xdp.XdpOptions(args.xdp, args.map, args.ctl_port,
                                             _given(args.frames, 4096), _given(args.k_out, 2048),
                                             _given(args.queue, 0))
        else:
            options = Options(args.in_port, args.out_port, args.ctl_port, _given(args.n_max, 216),
                              _given(args.k_out, 64), args.cells, args.measure_age)
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
