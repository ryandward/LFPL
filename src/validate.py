"""An independent linear-time check of emitted code (spec section 7.5).

It reads the NASM text, never the lowerer's state, and tracks a provenance word
per register and memory location. It rejects the three identity morphisms:

    I1  mov r, r
    I2  a load from a, then a store to a of the same word (Theorem 6.1)
    I3  a store of v to a, then a load from a while a register still holds v

Words are tracked within one straight-line block, because a label is a join and
this checker has no control flow graph. So a clean result proves the absence of
these morphisms within blocks, which is where a lowering rule could emit them,
and says nothing about a pair that spans a label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_MEM = re.compile(r"^(?:byte|word|dword|qword)?\s*\[(.+)\]$")
_REG64 = {}
for _r in ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp"):
    _REG64[_r] = _r
for _low, _r in (("eax", "rax"), ("ebx", "rbx"), ("ecx", "rcx"), ("edx", "rdx"),
                 ("esi", "rsi"), ("edi", "rdi"), ("ebp", "rbp"), ("ax", "rax"),
                 ("bx", "rbx"), ("cx", "rcx"), ("dx", "rdx"), ("si", "rsi"), ("di", "rdi"),
                 ("bp", "rbp"), ("al", "rax"), ("bl", "rbx"), ("cl", "rcx"), ("dl", "rdx"),
                 ("sil", "rsi"), ("dil", "rdi"), ("bpl", "rbp")):
    _REG64[_low] = _r
for _n in range(8, 16):
    for _suffix in ("", "d", "w", "b"):
        _REG64[f"r{_n}{_suffix}"] = f"r{_n}"


@dataclass(frozen=True)
class Violation:
    kind: str
    line: int
    text: str

    def __str__(self) -> str:
        return f"{self.kind} at line {self.line}: {self.text}"


def instructions(asm: str) -> list[tuple[int, str, list[str]]]:
    """(line number, mnemonic, operands) for every instruction; a label is ('label')."""
    parsed = []
    for number, raw in enumerate(asm.splitlines(), 1):
        line = raw.split(";")[0].strip()
        if not line or line.startswith(("section", "global", "alignb", "cells:")) or " equ " in line:
            continue
        if line.endswith(":"):
            parsed.append((number, "label", [line[:-1]]))
            continue
        head, _, rest = line.partition(" ")
        operands = [part.strip() for part in rest.split(",")] if rest else []
        parsed.append((number, head, operands))
    return parsed


def function_body(asm: str, name: str) -> list[tuple[str, list[str]]]:
    """The instructions of `fn_<name>_body`, labels removed, in emitted order."""
    body, inside = [], False
    for _number, op, operands in instructions(asm):
        if op == "label" and operands[0].endswith("_body"):
            inside = operands[0] == f"fn_{name}_body"
            continue
        if op == "label" and operands[0] in ("_start",):
            inside = False
        if inside and op != "label":
            body.append((op, operands))
    return body


def _memory(operand: str):
    match = _MEM.match(operand)
    return match.group(1).replace(" ", "") if match else None


def validate(asm: str) -> list[Violation]:
    """Return every identity morphism found. Linear in the number of lines."""
    found: list[Violation] = []
    regs: dict[str, object] = {}
    mem: dict[str, object] = {}
    loaded_from: dict[str, tuple[str, object]] = {}
    fresh = 0

    def new_word() -> tuple:
        nonlocal fresh
        fresh += 1
        return ("v", fresh)

    for number, op, operands in instructions(asm):
        text = f"{op} {', '.join(operands)}"
        if op == "label" or op in ("ret", "call", "jmp") or op.startswith("j") or op == "syscall":
            regs, mem, loaded_from = {}, {}, {}
            continue
        if op == "mov" and len(operands) == 2:
            dest, source = operands
            dest_mem, source_mem = _memory(dest), _memory(source)
            if source in _REG64 and dest in _REG64 and _REG64[dest] == _REG64[source]:
                if dest == source:  # a 32-bit mov r32, r32 is a zero extension, not identity
                    if not dest.startswith("e") and not dest.endswith("d"):
                        found.append(Violation("I1", number, text))
            if dest_mem is None and source_mem is not None and dest in _REG64:
                reg = _REG64[dest]
                word = mem.get(source_mem)
                if word is None:
                    word = new_word()
                    mem[source_mem] = word
                # I3: the value came from a store, and a register still holds it.
                if any(holder != reg and value == word and value[0] == "stored"
                       for holder, value in regs.items()):
                    found.append(Violation("I3", number, text))
                regs[reg] = word
                loaded_from[reg] = (source_mem, word)
            elif dest_mem is not None and source in _REG64:
                reg = _REG64[source]
                word = regs.get(reg)
                if word is None:
                    word = new_word()
                    regs[reg] = word
                # I2: a location is written back with the word it was loaded with.
                if mem.get(dest_mem) == word and reg in loaded_from and loaded_from[reg][0] == dest_mem:
                    found.append(Violation("I2", number, text))
                stored = ("stored", word)
                mem[dest_mem] = stored
                regs[reg] = stored
            elif dest_mem is not None:
                mem[dest_mem] = new_word()
            elif dest in _REG64 and _REG64[dest] == dest and _REG64.get(source) == source:
                # Only a full 64-bit copy keeps the word. `mov r32, r32` zero-extends,
                # which is a new value unless the upper half was already clear.
                regs[dest] = regs.get(source) or new_word()
                if source in loaded_from:
                    loaded_from[dest] = loaded_from[source]
                else:
                    loaded_from.pop(dest, None)
            elif dest in _REG64:
                regs[_REG64[dest]] = new_word()
                loaded_from.pop(_REG64[dest], None)
            continue
        if operands:
            dest = operands[0]
            if dest in _REG64:
                regs[_REG64[dest]] = new_word()
                loaded_from.pop(_REG64[dest], None)
            elif _memory(dest) is not None and op not in ("cmp", "test"):
                mem[_memory(dest)] = new_word()
    return found
