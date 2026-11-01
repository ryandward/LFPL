"""The LFPL-O compiler: source to a running binary, or a refusal and no binary.

    Lex -> Parse -> Check -> Lower (RT) -> rtcheck -> x86 -> symcheck -> nasm, ld

LFPL-O section 8.4 is the shape of this file. Every check runs on every build, and a failure
in any of them leaves nothing behind, so a program that compiled is one that passed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable, Optional

from .rtcheck import Checker, RTError
from .runtime import BindingError, Image, Options, assembly, bind
from .x86 import Backend, Starved


class Refused(Exception):
    """The program, or this build of it, is refused. No binary exists."""


@dataclass(frozen=True)
class Report:
    handler: str
    reserve: int
    steps: int
    stack: int
    cells: int
    image_bytes: int


def _generate(lines: Iterable[str], asm_path: str, options: Options,
              initial: dict[str, str]) -> Image:
    """The stream goes through the checker and then the backend, one line at a time, so nothing
    is lowered that has not been typed, and nothing is held that is not in flight."""
    checker = Checker()
    with open(asm_path, "w") as handle:
        handle.write("default abs\nsection .text\nglobal _start\n")
        backend = Backend(lambda line: handle.write(line + "\n"))
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


def _validate(source: Optional[bytes], asm_path: str, image: Image) -> None:
    """Section 8.4. Meaning, and the stack. A build that fails either has no binary."""
    from .symcheck import check_translation
    from .validate import recompute_stack

    with open(asm_path) as handle:
        asm = handle.read()
    if source is not None:
        problems = check_translation(source, asm)
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


def build(source_path: str, exe_path: str, options: Options,
          initial: Optional[dict[str, str]] = None, *, keep_asm: Optional[str] = None) -> Report:
    """Compile. A stale binary is removed first, so failure can never leave one behind."""
    if os.path.exists(exe_path):
        os.unlink(exe_path)
    initial = {name: os.path.abspath(path) for name, path in (initial or {}).items()}
    with tempfile.TemporaryDirectory() as work:
        asm_path = os.path.join(work, "program.asm")
        try:
            with open(source_path, "rb") as handle:
                if source_path.endswith(".rt"):
                    lines: Iterable[str] = (raw.decode() for raw in handle)
                    source = None
                    image = _generate(lines, asm_path, options, initial)
                else:
                    from .rtlower import rt_lines

                    source = handle.read()
                    handle.seek(0)
                    image = _generate(rt_lines(handle), asm_path, options, initial)
        except (RTError, BindingError, Starved) as failure:
            raise Refused(str(failure)) from failure
        if keep_asm is not None:
            shutil.copy(asm_path, keep_asm)
        _validate(source, asm_path, image)
        _link(asm_path, exe_path)
    size = (image.cell_count * 16 + image.stack_bytes
            + sum(s.length * s.element_size for s in image.states))
    return Report(image.handler, image.reserve_draws, image.steps, image.depth, image.cell_count,
                  size)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lfplo", description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("-o", dest="output", required=True)
    parser.add_argument("--in-port", type=int, required=True)
    parser.add_argument("--out-port", type=int, required=True,
                        help="egress goes to this port plus the handler's port number")
    parser.add_argument("--ctl-port", type=int, required=True)
    parser.add_argument("--state", action="append", default=[], metavar="NAME=FILE",
                        help="initial contents of a state array, read at link time")
    parser.add_argument("--n-max", type=int, default=216, help="the largest item, in cells")
    parser.add_argument("--k-out", type=int, default=64, help="the output queue's capacity")
    parser.add_argument("--asm", help="also write the assembly here")
    args = parser.parse_args(argv)
    initial = {}
    for entry in args.state:
        name, _, path = entry.partition("=")
        initial[name] = path
    options = Options(args.in_port, args.out_port, args.ctl_port, args.n_max, args.k_out)
    from .ring import Reject

    errors: tuple = (Refused, Reject)
    try:
        report = build(args.source, args.output, options, initial, keep_asm=args.asm)
    except errors as failure:
        print(f"{args.source}: refused: {failure}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as failure:
        print(f"{args.source}: the assembler or linker failed:\n{failure.stderr}", file=sys.stderr)
        return 1
    print(f"handler {report.handler}: R(n) = {report.reserve} cells, W(n) = {report.steps} steps, "
          f"stack {report.stack} bytes, {report.cells} cells, image {report.image_bytes} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
