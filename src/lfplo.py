"""The LFPL-O compiler: source to a running binary, or a refusal and no binary.

    Lex -> Parse -> Right-to-left -> Check -> Lower (RT) -> rtcheck -> x86 -> certificate -> symcheck, stack -> nasm, ld

LFPL-O section 8.4 is the shape of this file. Those checks run on every build, and a failure in any
of them leaves nothing behind, so a program that compiled is one that passed. The fourth check of
that section, redundant moves, runs only in the test suite so far.

A program with `fn main` and no handler runs once and exits, with no runtime: `--nasm` prints its
assembly, and `--run` assembles, links and runs it, and prints what main returned.
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

from .certify import certify
from .ring import FStarved, Reject, Ring
from .rtcheck import Bound, Checker, RTError
from .rtlower import Limit, rt_lines
from .runtime import BindingError, Image, Options, assembly, bind, largest_n
from .symcheck import check_translation
from .validate import recompute_stack
from .x86 import Backend, NotLowered, Starved


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
        certify(checker, name, largest_n(checker, name, options))
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


def _trailer(mode: int, target_cells: int) -> str:
    """_start, for a program with a main: the single mint. It threads the cells into one free list,
    which main receives, and runs exactly once per process. 0 is no main, a library assembled but
    not run; 1 is `fn main()`, which owns no cells and so returns a scalar directly; and 2 is
    `fn main(free: Free)`, which with exactly-once linearity cannot turn cells into a bare scalar, so
    it returns Fail, which is null, or a cell whose first field is the Int64 result, and the rest of
    its fields hand every other cell back."""
    if mode == 0:
        return "section .note.GNU-stack noalloc noexec nowrite progbits"
    takes_cells = mode == 2
    lines = ["_start:",
             "    mov rdi, cells",
             *((f"    mov rcx, {target_cells - 1}",
                "    mov rax, rdi",
                ".mint:",
                "    lea rdx, [rax+16]",
                "    mov [rax], rdx",
                "    mov rax, rdx",
                "    dec rcx",
                "    jnz .mint",
                "    mov qword [rax], 0") if takes_cells else ()),
             "    call fn_main",
             *(("    test rax, rax",
                "    jz .fail",
                "    mov rax, [rax]") if takes_cells else ()),
             "    push rax",
             "    mov rax, 1",
             "    mov rdi, 1",
             "    mov rsi, rsp",
             "    mov rdx, 8",
             "    syscall",
             "    mov rax, 60",
             "    xor edi, edi",
             "    syscall",
             *((".fail:",
                "    mov rax, 60",
                "    mov edi, 1",
                "    syscall") if takes_cells else ()),
             "section .bss",
             "alignb 16",
             f"cells: resb {target_cells * 16}",
             "section .note.GNU-stack noalloc noexec nowrite progbits"]
    return "\n".join(lines)


def build_and_run(asm: str, workdir: str, *, name: str = "prog") -> int:
    """Assemble with nasm and link with ld, so no libc and no C compiler, run it on the silicon, and
    return the 64-bit result `_start` writes to stdout."""
    source = os.path.join(workdir, f"{name}.asm")
    obj = os.path.join(workdir, f"{name}.o")
    exe = os.path.join(workdir, name)
    with open(source, "w") as handle:
        handle.write(asm)
    # -Werror: NASM only warns when it truncates an immediate, which is a miscompile that still assembles.
    subprocess.run(["nasm", "-Werror", "-f", "elf64", "-o", obj, source], check=True,
                   capture_output=True, text=True)
    subprocess.run(["ld", "-o", exe, obj], check=True, capture_output=True, text=True)
    done = subprocess.run([exe], check=True, capture_output=True, timeout=10)
    return int.from_bytes(done.stdout, "little")


def compile_program(source: bytes, *, target_cells: int = 1024,
                    on_step: Optional[Callable[[Ring], None]] = None) -> str:
    """A program with `fn main` and no handler: the assembly, to run once and exit."""
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
        except (RTError, BindingError, Starved, NotLowered, Limit, FStarved) as failure:
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


def _main_program(source_path: str, run: bool) -> int:
    """`--nasm` and `--run`."""
    try:
        with open(source_path, "rb") as handle:
            asm = compile_program(handle.read())
    except (Reject, RTError, Limit, FStarved, Starved, NotLowered, OSError) as failure:
        print(f"{source_path}: refused: {failure}", file=sys.stderr)
        return 1
    if not run:
        print(asm, end="")
        return 0
    with tempfile.TemporaryDirectory() as workdir:
        print(build_and_run(asm, workdir))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="lfplo", description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("--rt", action="store_true", help="print the checked RT stream, and stop")
    parser.add_argument("--nasm", action="store_true", help="a program with a main: print its assembly, and stop")
    parser.add_argument("--run", action="store_true", help="a program with a main: run it, and print what it returned")
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
    parser.add_argument("--measure-age", action="store_true",
                        help="keep max_age_ns, at the cost of a clock read for each admitted item")
    args = parser.parse_args(argv)
    if args.rt:
        return _print_rt(args.source)
    if args.nasm or args.run:
        return _main_program(args.source, args.run)
    if None in (args.output, args.in_port, args.out_port, args.ctl_port):
        parser.error("a build needs -o, --in-port, --out-port and --ctl-port")
    initial = {}
    for entry in args.state:
        name, _, path = entry.partition("=")
        initial[name] = path
    try:
        _remove_stale(args.output)
        options = Options(args.in_port, args.out_port, args.ctl_port, args.n_max, args.k_out,
                          args.cells, args.measure_age)
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
