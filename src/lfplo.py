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
from typing import Callable, Iterable, Optional

from .ring import FStarved, Reject, Ring
from .rtcheck import Bound, Checker, RTError, Unbounded
from .rtlower import Limit, rt_lines
from .runtime import BindingError, Image, Options, assembly, bind
from .simplex import minimize
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


def certify(checker: Checker, name: str, n_max: int) -> Bound:
    """LFPL-O section 5.5. The checker writes the linear program, the solver proposes numbers, and
    the checker evaluates its own inequalities with them, so the solver is not trusted."""
    program = checker.linear_program(name, n_max)
    proposed = {}
    for resource in ("cells", "steps"):
        solved = minimize(program.objective(), program.constraints(resource))
        if solved is None:
            raise Unbounded(f"section 8.2: handler {name} has no linear bound on {resource}: no "
                            f"amount per cell of its item pays for what it does")
        proposed[resource] = solved
    return checker.verify(name, proposed["cells"], proposed["steps"])


def _generate(lines: Iterable[str], asm_path: str, options: Options,
              initial: dict[str, str], *, count_steps: bool = False) -> Image:
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
        certify(checker, name, options.n_max)
        image = bind(checker, name, depth, options, dict(initial))
        for state in image.states:
            if state.initial is None:
                continue
            size = os.path.getsize(state.initial)
            if size != state.length * state.element_size:
                raise Refused(f"state {state.name} needs {state.length * state.element_size} "
                              f"bytes, and {state.initial} has {size}")
        handle.write(assembly(image))
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


def _validate(source: bytes, asm_path: str, image: Image) -> None:
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


def build(source_path: str, exe_path: str, options: Options,
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
    parser.add_argument("--n-max", type=int, default=216, help="the largest item, in cells")
    parser.add_argument("--k-out", type=int, default=64, help="the output queue's capacity")
    parser.add_argument("--cells", type=int, help="cells in the image, instead of section 3.4's C")
    parser.add_argument("--asm", help="also write the assembly here")
    parser.add_argument("--count-steps", action="store_true",
                        help="count the steps that run, to measure W(n) against what they were")
    args = parser.parse_args(argv)
    if args.rt:
        return _print_rt(args.source)
    if None in (args.output, args.in_port, args.out_port, args.ctl_port):
        parser.error("a build needs -o, --in-port, --out-port and --ctl-port")
    initial = {}
    for entry in args.state:
        name, _, path = entry.partition("=")
        initial[name] = path
    try:
        _remove_stale(args.output)
        options = Options(args.in_port, args.out_port, args.ctl_port, args.n_max, args.k_out,
                          args.cells)
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
