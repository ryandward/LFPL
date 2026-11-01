"""Lower: the destination-passing translation D[[e]](delta), as a fold.

The stage reads the certified event word and writes one instruction per cell.
Each event is handled against the frame of the construct that encloses it, so
the destination is fixed when a child opens, exactly as the calculus requires.

Four things differ from the spec's text because a forward fold cannot see the
future (the module docstring of dyck.py gives the fifth, prefix syntax):

* Operands are evaluated in source order. Sethi-Ullman's "needier first" needs
  need(e2) before e1's code is written. A leaf e1 is still deferred, since no
  code is owed for it, so register demand only grows on complex left operands.
* The spill victim is the oldest unpinned binder, not Belady's furthest next
  use. Hints come only from what a destination already names.
* Overwriting a location erases the scalar homed there, since a read cannot
  know whether it was the last. A later use is FailFast, never a silent load.
* `fn f(x: T) : T @x` declares the result word, so RESTORE on a recursive call
  is known before the call is lowered. Every exit is checked against it.

Register starvation, and any state the rules below do not cover, raises
FailFast carrying the exact context (Delta, H, delta).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable, Optional

from .check import P_ARM, P_BINDER, P_CALL, P_DIAMOND, P_FIELD, P_LET, P_PARAM, P_VAR, P_WITH
from .dyck import (CC_NAMES, CLOSE, K_AND, K_ARM, K_BINDER, K_BINOP, K_BOOL, K_CALL,
                   K_CMP, K_CTOR, K_DIAMOND, K_EOF, K_FIELD, K_FN, K_IF, K_INT, K_LET,
                   K_MATCH, K_NOT, K_NULLCTOR, K_OR, K_PACK, K_PARAM, K_RESWORD, K_VAR,
                   K_WITH, LEAF, OP_NAMES, OPEN, SEP, ev_class, ev_kind)
from .ring import Reject, Ring, Stage
from .tokens import (R8, R9, R10, R11, R12, R13, R14, R15, RAX, RBP, RBX, RCX, RDI, RDX,
                     REG_COUNT, REGS, RSI, Flags, Full, Hole, LImm, LMem, LReg, LSlot, RegTok,
                     Silicon, check_homes, reg_name)

# ------------------------------------------------------------ instructions

OPS = ("mov movzx lea xor add sub imul and or shl shr cmp test not setcc jcc "
       "jmp ret label alias func_alias func_label frame_label frame_enter frame_call "
       "frame_leave call_name jmp_name "
       "prelude trailer").split()
OP = {name: index for index, name in enumerate(OPS)}

# Operand encodings are at most 9 bytes and an instruction has at most two
# operands, one of them never an immediate, so one instruction fits one cell:
# the 15-byte payload the spec reserves for an encoded instruction.
O_NONE, O_REG, O_MEM, O_SLOT, O_IMM, O_LABEL = range(6)


def encode_operand(operand: tuple) -> bytes:
    kind = operand[0]
    if kind == O_REG:     # (O_REG, r, bytes wide)
        return bytes((O_REG, operand[1], operand[2]))
    if kind == O_MEM:     # (O_MEM, base r, offset, bytes wide)
        return bytes((O_MEM, operand[1], operand[2], operand[3]))
    if kind == O_SLOT:    # (O_SLOT, k, bytes wide)
        return bytes((O_SLOT, operand[1], operand[2]))
    if kind == O_IMM:     # (O_IMM, value)
        return struct.pack("<BQ", O_IMM, operand[1] & (1 << 64) - 1)
    if kind == O_LABEL:   # (O_LABEL, id)
        return struct.pack("<BI", O_LABEL, operand[1])
    raise ValueError(f"bad operand {operand}")


def decode_operand(data: bytes, at: int) -> tuple[tuple, int]:
    kind = data[at]
    if kind == O_REG:
        return (O_REG, data[at + 1], data[at + 2]), at + 3
    if kind == O_MEM:
        return (O_MEM, data[at + 1], data[at + 2], data[at + 3]), at + 4
    if kind == O_SLOT:
        return (O_SLOT, data[at + 1], data[at + 2]), at + 3
    if kind == O_IMM:
        return (O_IMM, struct.unpack_from("<Q", data, at + 1)[0]), at + 9
    if kind == O_LABEL:
        return (O_LABEL, struct.unpack_from("<I", data, at + 1)[0]), at + 5
    return (O_NONE,), at + 1


_WIDTH = {1: "byte", 2: "word", 4: "dword", 8: "qword"}


def render_operand(operand: tuple) -> str:
    kind = operand[0]
    if kind == O_REG:
        return reg_name(operand[1], operand[2])
    if kind == O_MEM:
        disp = f"+{operand[2]}" if operand[2] else ""
        return f"{_WIDTH[operand[3]]} [{REGS[operand[1]]}{disp}]"
    if kind == O_SLOT:
        # The body runs one call below its frame thunk: 8 for the return
        # address, then 8 per slot coordinate.
        return f"{_WIDTH[operand[2]]} [rsp+{8 + 8 * operand[1]}]"
    if kind == O_IMM:
        value = operand[1]
        return str(value - (1 << 64) if value >= 1 << 63 else value)
    return f"L{operand[1]}"


PRELUDE = "section .text\nglobal _start"


def render(tag: int, payload: bytes, *, target_cells: int) -> str:
    """Egress's side of the `write` transformer: one cell, one line of NASM."""
    name = OPS[tag]
    if name == "prelude":
        return PRELUDE
    if name == "func_label":
        return f"fn_{payload.rstrip(bytes(1)).decode()}_body:"
    if name == "func_alias":
        text = payload.rstrip(bytes(1)).decode()
        return f"fn_{text} equ fn_{text}_body"
    if name == "frame_label":
        return f"fn_{payload.rstrip(bytes(1)).decode()}:"
    if name == "frame_enter":
        return f"    sub rsp, {struct.unpack_from('<I', payload)[0]}"
    if name == "frame_call":
        return f"    call fn_{payload.rstrip(bytes(1)).decode()}_body"
    if name == "frame_leave":
        return f"    add rsp, {struct.unpack_from('<I', payload)[0]}\n    ret"
    if name == "trailer":
        return _trailer(payload[0], target_cells)
    if name in ("call_name", "jmp_name"):
        return f"    {name[:-5]} fn_{payload.rstrip(bytes(1)).decode()}"
    first, at = decode_operand(payload, 0)
    second, _ = decode_operand(payload, at)
    if name == "label":
        return f"{render_operand(first)}:"
    if name == "alias":
        return f"{render_operand(first)} equ {render_operand(second)}"
    if name == "setcc":
        return f"    set{CC[second[1]]} {render_operand(first)}"
    if name == "jcc":
        return f"    j{CC[second[1]]} {render_operand(first)}"
    parts = [render_operand(o) for o in (first, second) if o[0] != O_NONE]
    return f"    {name} {', '.join(parts)}".rstrip()


CC = ("b", "be", "a", "ae", "e", "ne")
NEGATED = {"b": "ae", "be": "a", "a": "be", "ae": "b", "e": "ne", "ne": "e"}


def _trailer(mode: int, target_cells: int) -> str:
    """_start: the single mint. It threads Lambda_cell into one free list, which
    is the token set Delta_0 the program receives, and runs outside the
    operational semantics, exactly once per process.

    With exactly-once linearity and no drop, main cannot turn cells into a bare
    scalar: `fn main(free: Free)` returns Fail (null) or a cell whose first field
    is the Int64 result, and the rest of its fields hand every other cell back.
    `fn main()` owns no cells, so it returns a scalar directly."""
    if mode == 0:  # no main: a library of functions, assembled but not run
        return "section .note.GNU-stack noalloc noexec nowrite progbits"
    pass_free = mode == 2
    lines = ["_start:",
             "    mov rdi, cells",
             f"    mov rcx, {target_cells - 1}" if pass_free else "",
             "    mov rax, rdi" if pass_free else "",
             ".mint:" if pass_free else "",
             "    lea rdx, [rax+16]" if pass_free else "",
             "    mov [rax], rdx" if pass_free else "",
             "    mov rax, rdx" if pass_free else "",
             "    dec rcx" if pass_free else "",
             "    jnz .mint" if pass_free else "",
             "    mov qword [rax], 0" if pass_free else "",
             "    call fn_main",
             "    test rax, rax" if pass_free else "",
             "    jz .fail" if pass_free else "",
             "    mov rax, [rax]" if pass_free else "",
             "    push rax",
             "    mov rax, 1",
             "    mov rdi, 1",
             "    mov rsi, rsp",
             "    mov rdx, 8",
             "    syscall",
             "    mov rax, 60",
             "    xor edi, edi",
             "    syscall",
             ".fail:" if pass_free else "",
             "    mov rax, 60" if pass_free else "",
             "    mov edi, 1" if pass_free else "",
             "    syscall" if pass_free else "",
             "section .bss",
             "alignb 16",
             f"cells: resb {target_cells * 16}",
             "section .note.GNU-stack noalloc noexec nowrite progbits"]
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------- failures


# The rejections the language declares. Each is a limit a program can run into and
# fix. Any other FailFast means the lowerer reached a state it has no rule for,
# which is a bug in the lowerer, not a property of the program.
DECLARED_LIMITS = ("register starvation", "section 1:", "section 4.1 against 4.2",
                   "section 4.4:", "section 5.3:", "section 5.6:")


class FailFast(Exception):
    """The directive's stop rule, raised by the compiler at the conflict itself."""

    @property
    def limit(self) -> Optional[str]:
        """Which declared limit this is, or None if it is not one."""
        return next((key for key in DECLARED_LIMITS if self.rule.startswith(key)), None)

    def __init__(self, rule: str, delta: str, homes: str, dest: str):
        super().__init__(f"{rule}\n  Delta = {delta}\n  H     = {homes}\n  delta = {dest}")
        self.rule = rule
        self.delta = delta
        self.homes = homes
        self.dest = dest


# ------------------------------------------------------------ destinations

D_NONE, D_REG, D_MEM, D_COND, D_FIELDS, D_EFF, D_RET, D_ANY, D_PMEM, D_TRMC = range(10)
_DEST_NAMES = ("none", "reg", "mem", "cond", "fields", "eff", "ret", "any", "pmem", "trmc")


@dataclass(frozen=True)
class Dest:
    """delta. x and y hold reg r, (cell binder, off | size << 8) or (L_t, L_f)."""
    kind: int
    x: int = 0
    y: int = 0

    def __str__(self) -> str:
        if self.kind == D_REG:
            return f"reg {REGS[self.x]}"
        if self.kind in (D_MEM, D_PMEM):
            return f"mem[b{self.x}+{self.y & 255}]:{self.y >> 8}"
        if self.kind == D_COND:
            return f"cond(L{self.x}, L{self.y})"
        return _DEST_NAMES[self.kind]


# ----------------------------------------------------------------- binders

L_NONE, L_REG, L_FIELD, L_SLOT, L_IMM, L_VIRT, L_ALIAS = range(7)
F_LINEAR, F_CONSUMED, F_ERASED, F_PINNED, F_TEMP, F_NULLARM, F_EVACUATED = (
    1, 2, 4, 8, 16, 32, 64)
_BINDER = struct.Struct("<BBHBBBHI")  # loc, r, base, off, size, flags, aux, word
MAX_BINDERS = 128  # a language limit on one function, so arm snapshots are constant


@dataclass
class Binder:
    """H(x). `word` is the provenance word nu; `r` is a register or slot number.
    A field binder stays on the stack after its use because a hole remembers
    its last evacuee: that record is what makes RESTORE decidable."""
    loc: int = L_NONE
    r: int = 0
    base: int = 0
    off: int = 0
    size: int = 8
    flags: int = 0
    aux: int = 0
    word: int = 0

    def pack(self) -> bytes:
        return _BINDER.pack(self.loc, self.r, self.base, self.off, self.size,
                            self.flags, self.aux, self.word)

    @staticmethod
    def unpack(data: bytes) -> "Binder":
        return Binder(*_BINDER.unpack_from(data))


# Registers in the order a fresh value takes them: argument registers last, so
# that a let-bound value rarely sits where a later call wants its arguments.
ALLOC_ORDER = (R10, R11, RBX, RBP, R12, R13, R14, R15, R9, R8, RCX, RDX, RSI, RDI, RAX)
PARAM_REGS = (RDI, RSI, RDX, RCX, R8, R9)  # the internal convention


# ------------------------------------------------------------------ frames

_HDR = struct.Struct("<BBIIHHBBBBBIIIQI")
_HDR_FIELDS = ("st", "dk", "dx", "dy", "mark", "srcmark", "b1", "b2", "b3", "b4", "w",
               "l1", "l2", "l3", "q", "word")
_SNAPSHOT_BYTES = 16 + 16 + 16 + REG_COUNT + 8 + 16
_ENTRY = struct.Struct("<BBBBQ")  # kind, op, off, size, value: one pending move
E_EMPTY, E_BIND, E_IMM, E_RMW_BIND, E_RMW_IMM = range(5)
_NO_REG = 255


class Fr:
    """A decoded frame. It exists only while one event is being handled, and
    goes back into cells through `_save`, so nothing outlives a step in Python."""

    __slots__ = _HDR_FIELDS + ("kind", "tail")

    def dest(self) -> Dest:
        return Dest(self.dk, self.dx, self.dy)


_LEAVES = frozenset((K_VAR, K_INT, K_BOOL, K_NULLCTOR))
_RMW_OPS = frozenset(("add", "sub", "and", "or", "xor"))
_NEEDS_NORMAL = frozenset(("add", "sub", "imul", "shl"))
_IMM32_MAX = (1 << 31) - 1


class Lower(Stage):
    """D[[e]](delta) as a fold: one event in, up to E instructions out."""

    name = "lower"

    def __init__(self) -> None:
        super().__init__()
        self.sil = Silicon()
        self.next_label = 1
        self.next_word = 2
        self.pend: Optional[tuple] = None
        self.out: list[tuple[int, bytes]] = []
        self.srcmap: list[int] = []
        self.skip = 0
        self.res: Optional[tuple] = None
        self.res_word = 0
        self.dead = False
        self.cond_active = False
        self.cond_jumped = False
        self.tmp: list[int] = []
        self.locked = 0
        self.touched = 0
        self.fn_name = b""
        self.fn_loop = 0
        self.root = -1           # the @-result binder, or -1
        self.res_src = -1
        self.has_main = 0
        self.started = False
        self.max_expansion = 0
        self.max_frame_cells = 0
        self.stack_bound = 0
        self.fn_stack = 0
        self.home_checks = False
        self.arm_level = 0
        self.lock_base = 0
        self.last_width = 8
        self.close_payload = b""
        # A register freed by SPILL is a hole that remembers its evacuee: entry r
        # is slot + 1 while r still physically holds that slot's value. It is the
        # stale-read window of section 6 applied to rho_r, and it is what keeps
        # the I3 round trip of Theorem 7.5 out of emitted code.
        self.reg_slot = [0] * REG_COUNT

    def attach(self, ring: Ring) -> None:
        super().attach(ring)
        self.K = self.stack()
        self.B = self.stack()
        self.table = self.stack()  # per function: name, kappa, result parameter

    def depth(self) -> int:
        return len(self.K)

    # --- instructions

    def _raw(self, op: str, a: Optional[tuple] = None, b: Optional[tuple] = None) -> None:
        payload = b""
        if a is not None:
            payload += encode_operand(a)
            if a[0] == O_REG:
                self.touched |= 1 << a[1]
                self.reg_slot[a[1]] = 0
        if b is not None:
            payload += encode_operand(b)
        self.out.append((OP[op], payload))

    def i(self, op: str, a: Optional[tuple] = None, b: Optional[tuple] = None) -> None:
        """Emit one instruction. A held phi_cc is consumed by its jcc first, so
        no flag-clobbering instruction can land between cmp and jcc (section 4.3)."""
        self._flush()
        self._raw(op, a, b)

    def _label(self) -> int:
        label = self.next_label
        self.next_label += 1
        return label

    def _hold_flags(self, cc: int, on_true: int, on_false: int) -> None:
        """phi -o phi_cc. The branch waits for the next event, so it can fall
        through to whichever label is placed next: section 5.1's `jmp bullet`."""
        self.pend = ("jcc", cc, on_true, on_false)
        self.sil.flags = Flags(CC[cc])
        self.cond_jumped = True

    def _flush(self, next_label: Optional[int] = None) -> None:
        pending = self.pend
        if pending is None:
            return
        self.pend = None
        self.sil.flags = Flags()  # jcc: phi_cc -o phi
        self.cond_jumped = True
        if pending[0] == "jmp":
            if pending[1] != next_label:
                self._raw("jmp", (O_LABEL, pending[1]))
            return
        _, cc, on_true, on_false = pending
        if next_label == on_true:
            self._raw("jcc", (O_LABEL, on_false), (O_IMM, CC.index(NEGATED[CC[cc]])))
        elif next_label == on_false:
            self._raw("jcc", (O_LABEL, on_true), (O_IMM, cc))
        else:
            self._raw("jcc", (O_LABEL, on_true), (O_IMM, cc))
            self._raw("jmp", (O_LABEL, on_false))

    def place(self, label: int) -> None:
        """Bind a label. If the pending branch targets it, the jump is the
        spec's `jmp bullet`: it falls through and emits nothing."""
        self._flush(next_label=label)
        self._raw("label", (O_LABEL, label))
        self._forget_registers()  # a join: other paths may hold something else

    def _forget_registers(self) -> None:
        self.reg_slot = [0] * REG_COUNT

    def _forget_slot(self, slot: int) -> None:
        self.reg_slot = [0 if held == slot + 1 else held for held in self.reg_slot]

    def _alias(self, label: int, target: int) -> None:
        """entry(true) = L_t: a label that needs no code of its own. Jumps to it
        were already written, so the assembler resolves them, with no trampoline."""
        if self.pend is not None:
            first = 1 if self.pend[0] == "jmp" else 2
            self.pend = tuple(target if i >= first and x == label else x
                              for i, x in enumerate(self.pend))
        self._raw("alias", (O_LABEL, label), (O_LABEL, target))

    def _goto(self, label: int) -> None:
        self._flush()
        self.pend = ("jmp", label)
        self.cond_jumped = True

    # --- failure context

    def _fail(self, rule: str, dest: object = "") -> FailFast:
        """Report the exact conflicting rule and context (Delta, H, delta)."""
        delta = [REGS[r] for r in range(REG_COUNT) if self.sil.free >> r & 1]
        pinned = [REGS[r] for r in range(REG_COUNT) if self.sil.pinned >> r & 1]
        homes = [f"{t.a}" + ("!" if isinstance(t.a, LReg) and self._pinned_partial(t.a.r) else "")
                 for t in self.tokens() if isinstance(t, Full)]
        return FailFast(rule, f"free={delta} pinned={pinned}",
                        "{" + ", ".join(homes) + "}", str(dest) or "(none)")

    def _pinned_partial(self, r: int) -> bool:
        for index in range(len(self.B)):
            b = self._b(index)
            if b.loc == L_REG and b.r == r and b.flags & F_PINNED and not b.flags & F_CONSUMED:
                return True
        return False

    # --- binders

    def _b(self, index: int) -> Binder:
        return Binder.unpack(self.B.at(index)[1])

    def _set_b(self, index: int, b: Binder) -> None:
        self.B.rewrite(index, 0, b.pack())

    def _new_b(self, b: Binder) -> int:
        if len(self.B) >= MAX_BINDERS:
            raise Reject(f"more than {MAX_BINDERS} live binders in a function")
        self.B.push(0, b.pack())
        return len(self.B) - 1

    def _resolve(self, index: int) -> int:
        b = self._b(index)
        while b.loc == L_ALIAS:
            index = b.base
            b = self._b(index)
        return index

    def _word_of(self, index: int) -> int:
        """nu(x). An imm home keeps its literal in `word`, which is not a word:
        comparing it with a hole's mu could make RESTORE drop a real store."""
        b = self._b(self._resolve(index))
        return 0 if b.loc == L_IMM else b.word

    def _word(self) -> int:
        word = self.next_word
        self.next_word += 1
        return word

    def _free_binder_token(self, b: Binder) -> None:
        if b.flags & F_CONSUMED:
            return
        if b.loc == L_REG:
            self.sil.give(b.r)
        elif b.loc == L_SLOT:
            self._forget_slot(b.r)
            self.sil.give_slot(b.r)

    def _pop_binders(self, mark: int) -> None:
        while len(self.B) > mark:
            self._free_binder_token(self._b(len(self.B) - 1))
            self.B.pop()

    def _consume(self, index: int) -> None:
        """A linear value left its home, so the home's token goes back to Delta."""
        b = self._b(index)
        if b.flags & F_CONSUMED:
            return
        self._free_binder_token(b)
        b.flags |= F_CONSUMED
        self._set_b(index, b)

    def _used(self, index: int) -> None:
        """The value was read. Scalars stay live; linear values and temps do not."""
        b = self._b(index)
        if b.flags & (F_LINEAR | F_TEMP):
            self._consume(index)

    def _check_usable(self, index: int, dest: object = "") -> None:
        b = self._b(index)
        if b.flags & F_ERASED and not b.flags & F_CONSUMED:
            raise self._fail(
                f"section 1: scalar b{index} is erased, because its cell was rebuilt or "
                f"given to a callee before this use; copy it first, with +(x, 0)", dest)

    # --- registers

    def _home_changing(self, dest: object = "") -> None:
        """Rule 5.3: all jumps to a join must agree on homes, and a jump already
        emitted under this cond cannot be revised."""
        if self.cond_active and self.cond_jumped:
            raise self._fail(
                "section 5.3: a home would change after a jump to this cond's labels; "
                "bind the operand with let before the condition", dest)

    def _take(self, avoid: int = 0, prefer: Optional[int] = None) -> int:
        """Take a free register, spilling only if Delta_reg has none (section 4.1)."""
        free = self.sil.free & ~avoid
        if prefer is not None and free >> prefer & 1:
            self.sil.take(prefer)
            return prefer
        for r in ALLOC_ORDER:
            if free >> r & 1:
                self.sil.take(r)
                return r
        victim = self._spill_victim(avoid)
        r = self._b(victim).r
        self._spill(victim)
        self.sil.take(r)
        return r

    def _spill_victim(self, avoid: int) -> int:
        for index in range(len(self.B)):
            b = self._b(index)
            if (b.loc == L_REG and not b.flags & (F_CONSUMED | F_PINNED)
                    and not avoid >> b.r & 1 and not self.locked >> b.r & 1):
                return index
        raise self._fail("register starvation: every register is pinned, Delta_reg is "
                         "empty and SPILL has no unpinned victim (section 4.1)")

    def _spill(self, index: int) -> None:
        self._home_changing()
        b = self._b(index)
        slot = self.sil.take_slot()
        self.i("mov", (O_SLOT, slot, 8), (O_REG, b.r, 8))
        self.reg_slot[b.r] = slot + 1
        self.sil.give(b.r)
        b.loc, b.r = L_SLOT, slot
        self._set_b(index, b)

    def _scratch(self, avoid: int = 0) -> int:
        r = self._take(avoid=avoid | self.locked)
        self.tmp.append(r)
        self.locked |= 1 << r
        return r

    def _end_scratch(self) -> None:
        for r in self.tmp:
            self.sil.give(r)
        self.tmp = []
        self.locked = self.lock_base

    def _pin_new(self, prefer: Optional[int] = None) -> int:
        r = self._take(prefer=prefer)
        self.sil.pin(r)
        return r

    def _unpin_free(self, r: int) -> None:
        self.sil.unpin(r)
        self.sil.give(r)

    # --- operands and values

    def _base(self, index: int) -> int:
        """The register holding a cell's pointer, reloading through a scratch
        token if the pointer was spilled."""
        b = self._b(self._resolve(index))
        if b.loc == L_REG:
            self.locked |= 1 << b.r
            return b.r
        if b.loc == L_SLOT:
            holder = next((q for q, held in enumerate(self.reg_slot)
                           if held == b.r + 1), None)
            if holder is not None:  # that register still holds the pointer
                self.locked |= 1 << holder
                return holder
            r = self._scratch()
            self._load_opnd((O_SLOT, b.r, 8), r)
            return r
        raise self._fail(f"cell pointer b{index} has no home")

    def _opnd(self, index: int) -> tuple:
        index = self._resolve(index)
        b = self._b(index)
        if b.loc == L_VIRT:
            r, _word = self._materialize(index)
            return (O_REG, r, 8)
        if b.loc == L_REG:
            # Locked until its instruction is emitted. Getting the instruction's
            # other operand may need a scratch register, and a spill must not take
            # this one out from under an operand already computed.
            self.locked |= 1 << b.r
            return (O_REG, b.r, 8)
        if b.loc == L_FIELD:
            return (O_MEM, self._base(b.base), b.off, b.size)
        if b.loc == L_SLOT:
            return (O_SLOT, b.r, 8)
        if b.loc == L_IMM:
            return (O_IMM, b.word)
        raise self._fail(f"b{index} has no home")

    def _mov_imm(self, r: int, n: int) -> None:
        if n == 0:
            self.i("xor", (O_REG, r, 4), (O_REG, r, 4))
        elif n < 1 << 32:
            self.i("mov", (O_REG, r, 4), (O_IMM, n))
        else:
            self.i("mov", (O_REG, r, 8), (O_IMM, n))

    def _load_opnd(self, o: tuple, r: int) -> None:
        """Bring an operand into r. Section 5.2: never `mov r, r`."""
        if o[0] == O_REG:
            if o[1] != r:
                self.i("mov", (O_REG, r, 8), (O_REG, o[1], 8))
        elif o[0] == O_IMM:
            self._mov_imm(r, o[1])
        elif o[0] == O_SLOT:
            if self.reg_slot[r] == o[1] + 1:
                return  # r still holds this value: RESTORE, on a register
            holder = next((q for q, held in enumerate(self.reg_slot)
                           if held == o[1] + 1), None)
            self.i("mov", (O_REG, r, 8), (O_REG, holder, 8) if holder is not None else o)
            self.reg_slot[r] = o[1] + 1
        elif o[3] == 8:
            self.i("mov", (O_REG, r, 8), o)
        elif o[3] == 4:
            self.i("mov", (O_REG, r, 4), o)
        else:
            self.i("movzx", (O_REG, r, 4), o)

    def _value_opnd(self, value: tuple, width: int) -> tuple:
        """An instruction operand for a leaf value at an operation's width."""
        wide = 8 if width == 8 else 4
        if value[0] == "imm":
            if value[1] <= _IMM32_MAX:
                return (O_IMM, value[1])
            r = self._scratch()
            self._mov_imm(r, value[1])
            return (O_REG, r, wide)
        o = self._opnd(value[1])
        if o[0] == O_REG:
            return (O_REG, o[1], wide)
        if o[0] == O_IMM:
            return o if o[1] <= _IMM32_MAX else self._value_opnd(("imm", o[1]), width)
        if o[0] == O_SLOT and width == 8:
            return o
        if o[0] == O_MEM and o[3] == width and width >= 4:
            return o
        r = self._scratch()
        self._load_opnd(o, r)
        return (O_REG, r, wide)

    def _consume_value(self, value: Optional[tuple]) -> None:
        if value is not None and value[0] == "bind":
            self._used(self._resolve(value[1]))

    # --- the location rules of section 6.2

    def _key(self, index: int) -> Optional[tuple]:
        index = self._resolve(index)
        b = self._b(index)
        if b.loc == L_REG:
            return ("r", b.r)
        if b.loc == L_SLOT:
            return ("s", b.r)
        if b.loc == L_FIELD:
            return ("m", self._resolve(b.base), b.off, b.size)
        return None

    @staticmethod
    def _overlap(a: Optional[tuple], b: Optional[tuple]) -> bool:
        if a is None or b is None or a[0] != b[0]:
            return False
        if a[0] != "m":
            return a[1] == b[1]
        return a[1] == b[1] and a[2] < b[2] + b[3] and b[2] < a[2] + a[3]

    def _word_at(self, cell: int, off: int, size: int) -> int:
        """mu: the provenance word location a still physically holds (Lemma 6.2)."""
        for index in range(len(self.B)):
            b = self._b(index)
            if (b.loc == L_FIELD and self._resolve(b.base) == cell
                    and b.off == off and b.size == size):
                return 0 if b.flags & F_ERASED else b.word
        return 0

    def _stored(self, cell: int, off: int, size: int, word: int) -> None:
        """STORE overwrote a location: its old scalar is erased, and the hole's
        provenance is now the stored word."""
        for index in range(len(self.B)):
            b = self._b(index)
            if b.loc != L_FIELD or self._resolve(b.base) != cell:
                continue
            if not self._overlap(("m", cell, off, size), ("m", cell, b.off, b.size)):
                continue
            b.flags |= F_ERASED  # consumed or not: the old word is gone
            self._set_b(index, b)

    def _deferred_reads(self) -> frozenset:
        """Binders an open frame has promised to read later: a deferred left
        operand, a pending move, or a field of an unbuilt constructor."""
        reads = set()
        for index in range(len(self.K)):
            fr = self._fr_at(index)
            if fr.kind in (K_BINOP, K_CMP) and fr.b4 == 1:
                reads.add(self._resolve(fr.q))
            elif fr.kind in (K_CALL, K_CTOR):
                for at in range(0, len(fr.tail) - _ENTRY.size + 1, _ENTRY.size):
                    kind, _op, _off, _size, value = _ENTRY.unpack_from(fr.tail, at)
                    if kind in (E_BIND, E_RMW_BIND):
                        reads.add(self._resolve(value))
        for index in range(len(self.B)):
            b = self._b(index)
            if b.loc == L_ALIAS:
                reads.add(self._resolve(b.base))
        return frozenset(reads)

    def _evacuate_linear(self, cell: int, off: int, size: int, keep: frozenset) -> None:
        """EVAC before a store. A linear value still owes its one use, and a scalar
        may have a read deferred by an enclosing frame: both are real relocations,
        not copies. A scalar nobody is waiting for is simply erased, with no load."""
        written = ("m", cell, off, size)
        deferred: Optional[frozenset] = None
        for index in range(len(self.B)):
            b = self._b(index)
            if (b.loc != L_FIELD or b.flags & (F_CONSUMED | F_ERASED) or index in keep
                    or self._resolve(b.base) != cell
                    or not self._overlap(written, ("m", cell, b.off, b.size))):
                continue
            if not b.flags & F_LINEAR:
                if deferred is None:
                    deferred = self._deferred_reads()
                if index not in deferred:
                    continue
            self._home_changing()
            r = self._take(avoid=self.locked)
            self._load_opnd(self._opnd(index), r)
            self._end_scratch_keep()
            b.loc, b.r = L_REG, r
            b.flags |= F_EVACUATED
            self._set_b(index, b)

    def _end_scratch_keep(self) -> None:
        """Between moves of one parallel move, operand locks do not carry over."""
        self.locked = self.lock_base
        for r in self.tmp:
            self.locked |= 1 << r

    def _pmove(self, moves: list[list]) -> None:
        """Parallel move, section 6.3. Each move is [dest, kind, op, value]; dest
        is ("r", r), ("s", k) or ("m", cell, off, size). Step 1 drops the trivial
        moves, which are STAY and RESTORE, so nothing is emitted for them."""
        sources = frozenset(self._resolve(m[3]) for m in moves
                            if m[1] in (E_BIND, E_RMW_BIND))
        kept: list[list] = []
        for move in moves:
            dest, kind, _op, value = move
            if kind == E_BIND and (self._key(value) == dest or self._restores(dest, value)):
                self._used(self._resolve(value))
                continue
            kept.append(move)
        for dest, *_rest in kept:
            if dest[0] == "m":
                self._evacuate_linear(dest[1], dest[2], dest[3], sources)
        self._run_moves(kept)
        for _dest, kind, _op, value in kept:
            if kind in (E_BIND, E_RMW_BIND):
                self._used(self._resolve(value))

    def _run_moves(self, moves: list[list]) -> None:
        """Steps 2 and 3: every source is read before its location is written, and
        each cycle is broken with a scratch token."""
        saved: dict[int, tuple] = {}
        pending = list(moves)
        while pending:
            ready = None
            for move in pending:
                if not any(other is not move and self._reads(other, move[0], saved)
                           for other in pending):
                    ready = move
                    break
            if ready is None:
                ready = self._break_cycle(pending, saved)
            pending = [m for m in pending if m is not ready]
            self._emit_move(ready, saved)
            self._end_scratch_keep()
        self._end_scratch()

    def _restores(self, dest: tuple, value: int) -> bool:
        if dest[0] != "m":
            return False
        word = self._word_of(value)
        return word != 0 and word == self._word_at(dest[1], dest[2], dest[3])

    def _reads(self, move: list, dest: tuple, saved: dict) -> bool:
        _d, kind, _op, value = move
        if kind in (E_BIND, E_RMW_BIND):
            source = self._resolve(value)
            if source not in saved:
                if self._overlap(self._key(source), dest):
                    return True
                b = self._b(source)
                if b.loc == L_FIELD and self._overlap(self._key(b.base), dest):
                    return True  # reading [p+off] reads p
        return kind in (E_RMW_BIND, E_RMW_IMM) and move[0] != dest and self._overlap(move[0], dest)

    def _break_cycle(self, pending: list[list], saved: dict) -> list:
        """A cycle of k locations: k loads and k stores through two scratch tokens."""
        move = pending[0]
        dest = move[0]
        for other in pending:
            if other is move or other[1] not in (E_BIND, E_RMW_BIND):
                continue
            source = self._resolve(other[3])
            if self._overlap(self._key(source), dest) and source not in saved:
                r = self._scratch()
                self._load_opnd(self._opnd(source), r)
                saved[source] = (O_REG, r, 8)
        return move

    def _dest_opnd(self, dest: tuple) -> tuple:
        return (O_MEM, self._base(dest[1]), dest[2], dest[3])

    def _emit_move(self, move: list, saved: dict) -> None:
        dest, kind, op, value = move
        if kind in (E_RMW_BIND, E_RMW_IMM):
            self._emit_rmw(dest, OP_NAMES[op], kind, value, saved)
            return
        source: tuple
        if kind == E_IMM:
            source = (O_IMM, value)
        else:
            resolved = self._resolve(value)
            source = saved.get(resolved) or self._opnd(resolved)
        if dest[0] == "r":
            self._load_opnd(source, dest[1])
            return
        if dest[0] == "s":
            self.i("mov", (O_SLOT, dest[1], 8), source)
            return
        word = 0 if kind == E_IMM else self._word_of(value)
        self._store_opnd(source, dest, word)

    def _store_opnd(self, source: tuple, dest: tuple, word: int) -> None:
        """STORE: x86-64 has no memory-to-memory mov, so memory needs a scratch."""
        size = dest[3]
        if source[0] == O_IMM and source[1] <= _IMM32_MAX:
            self.i("mov", self._dest_opnd(dest), source)
        else:
            if source[0] != O_REG:
                r = self._scratch()
                self._load_opnd(source, r)
                source = (O_REG, r, 8)
            self.i("mov", self._dest_opnd(dest), (O_REG, source[1], size))
        self._stored(dest[1], dest[2], dest[3], word)

    def _emit_rmw(self, dest: tuple, op: str, kind: int, value: int, saved: dict) -> None:
        """op [a], o: read-modify-write in place, with no load and no store-back."""
        if kind == E_RMW_IMM:
            source = (O_IMM, value)
        else:
            resolved = self._resolve(value)
            source = saved.get(resolved) or self._opnd(resolved)
        width = 8 if dest[0] == "r" else dest[3]
        if source[0] == O_REG:
            source = (O_REG, source[1], width)
        elif source[0] != O_IMM or source[1] > _IMM32_MAX:
            # x86-64 has no 64-bit immediate here, and imm32 sign-extends.
            r = self._scratch()
            self._load_opnd(source, r)
            source = (O_REG, r, width)
        if dest[0] == "r":
            self.i(op, (O_REG, dest[1], 8), source)
        else:
            self.i(op, self._dest_opnd(dest), source)
            self._stored(dest[1], dest[2], dest[3], self._word())

    # --- frames

    def _push(self, kind: int, dest: Dest = Dest(D_NONE), tail: int = 0, **fields: int) -> None:
        values = {name: 0 for name in _HDR_FIELDS}
        values.update(dk=dest.kind, dx=dest.x, dy=dest.y, mark=len(self.B),
                      srcmark=len(self.srcmap))
        values.update(fields)
        data = _HDR.pack(*(values[name] for name in _HDR_FIELDS)) + bytes(tail)
        self.K.push(kind, data)
        cells = (len(data) + 14) // 15
        self.max_frame_cells = max(self.max_frame_cells, cells)

    def _fr_at(self, index: int) -> Fr:
        kind, data = self.K.at(index)
        fr = Fr()
        for name, value in zip(_HDR_FIELDS, _HDR.unpack_from(data)):
            setattr(fr, name, value)
        fr.kind = kind
        start, stop = self.K._span(index)
        fr.tail = bytearray(data[_HDR.size:(stop - start) * 15])
        return fr

    def _top(self) -> Fr:
        return self._fr_at(len(self.K) - 1)

    def _save_at(self, index: int, fr: Fr) -> None:
        data = _HDR.pack(*(getattr(fr, name) for name in _HDR_FIELDS)) + bytes(fr.tail)
        self.K.rewrite(index, fr.kind, data)

    def _save(self, fr: Fr) -> None:
        self._save_at(len(self.K) - 1, fr)

    # --- arms: snapshot, restore, normalize. Constant size: no value ever
    # reloads from a slot, so at most 15 homes can differ at a join.

    def _snapshot(self) -> bytes:
        consumed = erased = evacuated = 0
        held = 0
        regmap = bytearray(b"\xff" * REG_COUNT)
        for index in range(len(self.B)):
            b = self._b(index)
            if b.flags & F_CONSUMED:
                consumed |= 1 << index
            if b.flags & F_ERASED:
                erased |= 1 << index
            if b.flags & F_EVACUATED:
                evacuated |= 1 << index
            if b.flags & F_CONSUMED:
                continue
            if b.loc == L_REG:
                regmap[b.r] = index
            elif b.loc == L_SLOT:
                held |= 1 << b.r
        return (consumed.to_bytes(16, "little") + erased.to_bytes(16, "little")
                + evacuated.to_bytes(16, "little")
                + bytes(regmap) + held.to_bytes(8, "little") + bytes(16))

    def _restore(self, snap: bytes) -> None:
        consumed = int.from_bytes(snap[:16], "little")
        erased = int.from_bytes(snap[16:32], "little")
        evacuated = int.from_bytes(snap[32:48], "little")
        regmap = snap[48:48 + REG_COUNT]
        held = int.from_bytes(snap[48 + REG_COUNT:56 + REG_COUNT], "little")
        for index in range(len(self.B)):
            b = self._b(index)
            if b.flags & F_EVACUATED and not evacuated >> index & 1:
                b.loc = L_FIELD  # on this path the evacuation never happened
                b.flags &= ~F_EVACUATED
            b.flags &= ~(F_CONSUMED | F_ERASED)
            if consumed >> index & 1:
                b.flags |= F_CONSUMED
            if erased >> index & 1:
                b.flags |= F_ERASED
            self._set_b(index, b)
        occupied = 0
        for r, index in enumerate(regmap):
            if index != 0xff:
                b = self._b(index)
                b.loc, b.r = L_REG, r
                self._set_b(index, b)
                occupied |= 1 << r
        all_regs = (1 << REG_COUNT) - 1
        self.sil.free = all_regs & ~(occupied | self.sil.pinned)
        minted = (1 << self.sil.slots_minted) - 1
        self.sil.free_slots = minted & ~held
        self.dead = False
        self._forget_registers()

    def _normalize(self, snap: bytes) -> None:
        """A join needs one set of homes, so every value that moved inside the
        arm goes back. These are the genuine join relocations of section 5.3."""
        evacuated = int.from_bytes(snap[32:48], "little")
        regmap = snap[48:48 + REG_COUNT]
        for index in range(len(self.B)):
            b = self._b(index)
            if (b.flags & F_EVACUATED and not evacuated >> index & 1
                    and not b.flags & F_CONSUMED):
                raise self._fail(
                    f"section 5.3: b{index} was evacuated from a cell rebuilt inside this "
                    f"arm and is still live at the join, so the arms disagree on its "
                    f"home; use it before the conditional")
        moves = []
        movers = []
        for r, index in enumerate(regmap):
            if index == 0xff:
                continue
            b = self._b(index)
            if b.flags & F_CONSUMED or (b.loc == L_REG and b.r == r):
                continue
            moves.append([("r", r), E_BIND, 0, index])
            movers.append((index, r))
        if not moves:
            return
        self._run_moves(moves)
        for index, r in movers:
            b = self._b(index)
            if b.loc == L_SLOT:
                self._forget_slot(b.r)
                self.sil.give_slot(b.r)
            b.loc, b.r = L_REG, r
            self._set_b(index, b)
        occupied = 0
        for index in range(len(self.B)):
            b = self._b(index)
            if b.loc == L_REG and not b.flags & F_CONSUMED:
                occupied |= 1 << b.r
        self.sil.free = ((1 << REG_COUNT) - 1) & ~(occupied | self.sil.pinned)

    def _merge_erased(self, fr: Fr, snap_offset: int = 0) -> None:
        """A location overwritten on either path is unreadable after the join."""
        extra = int.from_bytes(fr.tail[snap_offset + _SNAPSHOT_BYTES - 16:
                                       snap_offset + _SNAPSHOT_BYTES], "little")
        for index in range(len(self.B)):
            if extra >> index & 1:
                b = self._b(index)
                b.flags |= F_ERASED
                self._set_b(index, b)

    def _remember_erased(self, fr: Fr, snap_offset: int = 0) -> None:
        erased = int.from_bytes(fr.tail[snap_offset + _SNAPSHOT_BYTES - 16:
                                        snap_offset + _SNAPSHOT_BYTES], "little")
        for index in range(min(len(self.B), MAX_BINDERS)):
            if self._b(index).flags & F_ERASED:
                erased |= 1 << index
        fr.tail[snap_offset + _SNAPSHOT_BYTES - 16:snap_offset + _SNAPSHOT_BYTES] = (
            erased.to_bytes(16, "little"))

    # --- delivering a value to a destination

    def _ret(self) -> None:
        self.i("ret")
        self.dead = True

    def _check_result_word(self, word: int) -> None:
        """Every exit of `fn f(x) @x` must return x's own word (the spec's RESTORE
        premise for the recursive call). A fold verifies it exit by exit."""
        if self.root < 0:
            return
        root_word = self._b(self.root).word
        if word == root_word:
            return
        if word == 1:  # NULL: legal only in the null arm of the same cell
            for index in range(len(self.B)):
                b = self._b(index)
                if b.flags & F_NULLARM and b.word == root_word:
                    return
        raise Reject("the result is not the word declared by '@': an exit returns a "
                     "different cell, so RESTORE on the recursive call would be unsound")

    def _ret_root(self) -> None:
        """@-function exit: the result is the root cell, which is already known."""
        self._load_opnd(self._opnd(self.root), RAX)
        self._ret()

    def _deliver_reg(self, r: int, word: int, dest: Dest) -> None:
        """A value in register r goes to dest. The caller keeps r's token."""
        self.res = None
        self.res_word = word
        kind = dest.kind
        if kind == D_REG:
            if dest.x != r:
                self.i("mov", (O_REG, dest.x, 8), (O_REG, r, 8))
        elif kind in (D_MEM, D_PMEM):
            cell, off, size = self._resolve(dest.x), dest.y & 255, dest.y >> 8
            if word and word == self._word_at(cell, off, size):
                return  # RESTORE: the location still holds this word
            self._evacuate_linear(cell, off, size, frozenset())
            self.locked |= 1 << r
            self.i("mov", self._dest_opnd(("m", cell, off, size)), (O_REG, r, size))
            self._stored(cell, off, size, word)
            self._end_scratch()
        elif kind == D_COND:
            self.i("test", (O_REG, r, 8), (O_REG, r, 8))
            self._hold_flags(CC_NAMES.index("ne"), dest.x, dest.y)
        elif kind == D_RET:
            self._check_result_word(word)
            if self.root >= 0:
                self._ret_root()
            else:
                if r != RAX:
                    self.i("mov", (O_REG, RAX, 8), (O_REG, r, 8))
                self._ret()
        elif kind != D_EFF:
            raise self._fail("a register value has no rule for this destination", dest)

    def _deliver_imm(self, n: int, dest: Dest, word: int = 0) -> None:
        self.res_word = word
        kind = dest.kind
        if kind == D_ANY:
            self.res = ("imm", n)
        elif kind == D_REG:
            self.res = None
            self._mov_imm(dest.x, n)
        elif kind in (D_MEM, D_PMEM):
            self.res = None
            target = ("m", self._resolve(dest.x), dest.y & 255, dest.y >> 8)
            self._pmove([[target, E_IMM, 0, n]])
        elif kind == D_COND:
            self.res = None
            self._goto(dest.x if n else dest.y)
        elif kind == D_RET:
            self.res = None
            self._check_result_word(word)
            if self.root >= 0:
                self._ret_root()
            else:
                self._mov_imm(RAX, n)
                self._ret()
        elif kind == D_EFF:
            self.res = None
        else:
            raise self._fail("a literal has no rule for this destination", dest)

    def _deliver_var(self, index: int, dest: Dest) -> None:
        """D[[x]](delta), section 5.2."""
        index = self._resolve(index)
        self._check_usable(index, dest)
        b = self._b(index)
        word = self._word_of(index)
        kind = dest.kind
        self.res_word = word
        if kind == D_ANY:
            self.res = ("bind", index)
            return
        self.res = None
        if b.loc == L_VIRT:
            r, word = self._materialize(index)
            self._deliver_reg(r, word, dest)
            return
        if b.flags & F_LINEAR:
            # A cell's capability going anywhere but a rebuild, say through a let into
            # an argument: from here this function cannot tell who writes it next.
            self._close_stale_window(index, frozenset((index,)))
        if kind == D_REG:
            self._load_opnd(self._opnd(index), dest.x)
            self._end_scratch()
            self._used(index)
        elif kind in (D_MEM, D_PMEM):
            target = ("m", self._resolve(dest.x), dest.y & 255, dest.y >> 8)
            self._pmove([[target, E_BIND, 0, index]])
        elif kind == D_COND:
            o = self._opnd(index)
            if o[0] == O_REG:
                self.i("test", o, o)
            elif o[0] == O_IMM:
                self._goto(dest.x if o[1] else dest.y)
                return
            else:
                self.i("cmp", o, (O_IMM, 0))
            self._end_scratch()
            self._hold_flags(CC_NAMES.index("ne"), dest.x, dest.y)
        elif kind == D_RET:
            self._check_result_word(word)
            if self.root >= 0:
                self._ret_root()
            else:
                self._load_opnd(self._opnd(index), RAX)
                self._ret()
        elif kind == D_EFF:
            self._used(index)
        else:
            raise self._fail("a variable has no rule for this destination", dest)

    def _deliver_pointer(self, cell: int, dest: Dest) -> None:
        """The value of a constructor built in place is addr(c), whatever its
        fields are: section 6.1."""
        b = self._b(cell)
        r = self._base(cell)
        self._deliver_reg(r, b.word, dest)
        self._end_scratch()

    def _write_tag(self, cell: int, tag: int, tagged: int) -> None:
        """tag(p, C), or STAY when the cell already holds this tag."""
        if tagged and self._b(cell).aux != tag + 1:
            # aux is the tag dispatch proved, and it never changes after that. The
            # diamond is linear, so this cell is rebuilt at most once on any path, and
            # recording this write could only mislead the other arm of an `if`.
            self.i("mov", (O_MEM, self._base(cell), 0, 1), (O_IMM, tag))

    def _materialize(self, virt: int) -> tuple[int, int]:
        """virt(C, u) reached a use that is not a match: build it now, in place."""
        v = self._b(virt)
        cell = self._resolve(v.base)
        moves = []
        for j in range(v.off):
            h = v.aux + j
            hb = self._b(h)
            target = ("m", cell, hb.off, hb.size)
            if hb.loc == L_IMM:
                moves.append([target, E_IMM, 0, hb.word])
            elif hb.loc == L_ALIAS:
                moves.append([target, E_BIND, 0, hb.base])
            else:
                moves.append([target, E_BIND, 0, h])
        self._pmove(moves)
        self._close_stale_window(cell)
        self._write_tag(cell, v.r, v.size)
        v.flags |= F_CONSUMED
        self._set_b(virt, v)
        word = self._b(cell).word
        return self._base(cell), word

    # --- the fold

    def fire(self, cell: Optional[int]) -> None:
        assert cell is not None
        store = self.ring.store
        tag, payload = store.tag(cell), store.payload(cell)
        self.out = []
        self.locked = self.lock_base = 0  # operand locks last one instruction at most
        if not self.started:
            self.started = True
            self._raw("prelude")
        self._event(ev_class(tag), ev_kind(tag), payload)
        self.max_expansion = max(self.max_expansion, len(self.out))
        if len(self.out) > E_MAX:
            raise self._fail(f"one event expanded to {len(self.out)} instructions, "
                             f"past E = {E_MAX} (Theorem 7.2)")
        self.ring.expand(self, cell, self.out)
        if self.home_checks:
            self.check_invariants()

    def tokens(self) -> list:
        """Delta as the spec writes it: free registers, full locations, holes with
        their last evacuee, and the flags. Built from the binder frames, so the
        tokens are a view of the machine's state and cannot drift away from it."""
        view: list = [RegTok(r) for r in range(REG_COUNT) if self.sil.free >> r & 1]
        view.append(self.sil.flags)
        for index in range(len(self.B)):
            b = self._b(index)
            if b.loc == L_FIELD:
                location = LMem(self._resolve(b.base), b.off, b.size)
                if b.flags & F_ERASED:
                    view.append(Hole(location, 0))  # overwritten: the evacuee is gone
                elif b.flags & F_CONSUMED:
                    view.append(Hole(location, b.word))  # stale-read window, Lemma 6.2
                else:
                    view.append(Full(location, b.word))
            elif b.flags & F_CONSUMED:
                continue
            elif b.loc == L_REG:
                view.append(Full(LReg(b.r), b.word))
            elif b.loc == L_SLOT:
                view.append(Full(LSlot(b.r), b.word))
            elif b.loc == L_IMM:
                view.append(Full(LImm(b.word), b.word))
        return view

    def check_invariants(self) -> None:
        """Theorem 7.1's register partition and Invariant H, over those tokens."""
        view = self.tokens()
        self.sil.check_partition(t.a.r for t in view
                                 if isinstance(t, Full) and isinstance(t.a, LReg))
        check_homes({i: t.a for i, t in enumerate(view) if isinstance(t, Full)})

    def _event(self, cls: int, kind: int, payload: bytes) -> None:
        if self.skip:
            self.skip += (cls == OPEN) - (cls == CLOSE)
            return
        if cls == OPEN:
            if kind == K_FN:
                self._open_fn(payload)
            elif kind == K_ARM:
                self._open_arm(payload)
            else:
                self.OPENERS[kind](self, self._child_dest(kind, payload), payload)
        elif cls == LEAF:
            if kind in _LEAVES:
                self._leaf_value(kind, self._child_dest(kind, payload), payload)
            else:
                self._leaf_structural(kind, payload)
        elif cls == SEP:
            self.SEPS[kind](self, payload)
        else:
            self.close_payload = payload
            self.CLOSERS[kind](self)

    def _fresh_child(self, fr: Fr, prefer: Optional[int] = None) -> Dest:
        r = self._pin_new(prefer=prefer)
        fr.l2 = r + 1
        self._save(fr)
        return Dest(D_REG, r)

    def _work(self, fr: Fr, partial: bool = False) -> int:
        """The accumulator: the destination register when there is one (pinned).
        A partial result avoids rax, which every call in the right operand clobbers."""
        if fr.b2:
            return fr.b2 - 1
        if fr.dk == D_REG:
            r = fr.dx
        else:
            r = self._pin_new(prefer=RAX if fr.dk == D_RET and not partial else None)
            fr.b3 = 1
        fr.b2 = r + 1
        return r

    def _child_dest(self, kind: int, payload: bytes) -> Dest:
        fr = self._top()
        parent = fr.kind
        leaf = kind in _LEAVES
        any_dest = Dest(D_ANY)
        if parent == K_FN:
            return Dest(D_RET)
        if parent == K_ARM:
            return fr.dest()
        if parent == K_LET:
            if fr.st:
                return fr.dest()
            if leaf:
                return any_dest
            return Dest(D_FIELDS) if kind == K_CTOR else self._fresh_child(fr)
        if parent == K_IF:
            if fr.st == 0:
                return Dest(D_COND, fr.l1, fr.l2)
            label = fr.l1 if fr.st == 1 else fr.l2
            dest = fr.dest()
            if dest.kind == D_COND and kind == K_BOOL:
                self._alias(label, dest.x if payload[0] else dest.y)
                return Dest(D_EFF)
            self.place(label)
            return dest
        if parent in (K_MATCH, K_WITH):
            if fr.st == 2:
                return Dest(D_MEM, fr.l1 - 1, fr.b1 | fr.b2 << 8)
            if leaf:
                return any_dest
            if kind == K_CTOR and parent == K_MATCH:
                return Dest(D_FIELDS)
            return self._fresh_child(fr)
        if parent in (K_BINOP, K_CMP):
            if fr.dk == D_EFF:
                return Dest(D_EFF)
            if leaf:
                return any_dest
            if fr.st == 0:
                w = self._work(fr, partial=True)
                self._save(fr)
                return Dest(D_REG, w)
            return self._fresh_child(fr)
        if parent == K_AND:
            return Dest(D_COND, fr.l1 if fr.st == 0 else fr.l2, fr.l3)
        if parent == K_OR:
            return Dest(D_COND, fr.l2, fr.l1 if fr.st == 0 else fr.l3)
        if parent == K_NOT:
            return Dest(D_COND, fr.l3, fr.l2)
        if parent == K_CALL:
            if leaf or kind == K_CTOR:
                return any_dest  # a cell built in place is already where it is
            return self._fresh_child(fr, PARAM_REGS[fr.st])
        if parent == K_PACK:
            if fr.dk == D_EFF:
                return Dest(D_EFF)
            return any_dest if leaf else self._fresh_child(fr)
        if parent == K_CTOR:
            if leaf or kind == K_CTOR:
                return any_dest
            if fr.dk == D_FIELDS:
                return self._fresh_child(fr)
            if kind == K_BINOP:
                return Dest(D_PMEM, fr.l1 - 1, fr.b4 | fr.w << 8)
            if (kind == K_CALL and P_CALL.unpack_from(payload)[1] and fr.dk == D_RET
                    and fr.st == fr.b3 - 1):
                return Dest(D_TRMC)
            return self._fresh_child(fr)
        raise self._fail(f"an expression cannot appear inside construct {parent}")

    def _leaf_value(self, kind: int, dest: Dest, payload: bytes) -> None:
        if kind == K_VAR:
            src, _linear, size = P_VAR.unpack_from(payload)
            self.last_width = size
            self._deliver_var(self.srcmap[src], dest)
        elif kind == K_INT:
            self.last_width = 0
            self._deliver_imm(struct.unpack_from("<Q", payload)[0], dest)
        elif kind == K_BOOL:
            self.last_width = 1
            self._deliver_imm(payload[0], dest)
        else:
            self.last_width = 8
            self._deliver_imm(0, dest, word=1)

    def _leaf_structural(self, kind: int, payload: bytes) -> None:
        if kind == K_EOF:
            self._flush()
            self.out.append((OP["trailer"], bytes((self.has_main,))))
            return
        fr = self._top()
        if kind == K_PARAM:
            _src, size, linear = P_PARAM.unpack_from(payload)
            r = PARAM_REGS[len(self.srcmap)]
            self.sil.take(r)
            flags = F_LINEAR if linear else 0
            self.srcmap.append(self._new_b(Binder(L_REG, r, size=size, flags=flags,
                                                  word=self._word())))
            fr.b2 = len(self.srcmap)
            self._save(fr)
        elif kind == K_RESWORD:
            self.res_src = struct.unpack_from("<H", payload)[0]
        elif kind == K_BINDER:
            self._leaf_binder(fr, payload)
        elif kind == K_FIELD:
            off, size, linear = P_WITH.unpack_from(payload)
            cell = fr.l1 - 1
            self.srcmap.append(self._new_b(Binder(
                L_FIELD, 0, cell, off, size, F_LINEAR if linear else 0,
                word=self._word())))
            fr.b1, fr.b2 = off, size
            self._save(fr)
        elif kind == K_DIAMOND:
            src, tag, tagged, count, off, size, _linear = P_DIAMOND.unpack_from(payload)
            fr.l1 = self._resolve(self.srcmap[src]) + 1
            fr.b1, fr.b2, fr.b3, fr.b4, fr.w = tag, tagged, count, off, size
            self._save(fr)

    def _leaf_binder(self, fr: Fr, payload: bytes) -> None:
        is_field, off, size, linear = P_BINDER.unpack_from(payload)
        if fr.b1:  # phantom: the pattern names the homes of the unbuilt fields
            v = self._b(fr.l1 - 1)
            if is_field:
                self.srcmap.append(v.aux + fr.b2)
                fr.b2 += 1
                self._save(fr)
            else:
                self.srcmap.append(v.base)
            return
        cell = fr.l3 - 1
        if is_field:
            self.srcmap.append(self._new_b(Binder(
                L_FIELD, 0, cell, off, size, F_LINEAR if linear else 0,
                word=self._word())))
        else:
            self.srcmap.append(cell)

    # --- let

    def _open_let(self, dest: Dest, _payload: bytes) -> None:
        self._push(K_LET, dest)

    def _sep_let(self, payload: bytes) -> None:
        _sep, linear, size = P_LET.unpack_from(payload)
        fr = self._top()
        flags = F_LINEAR if linear else 0
        res = self.res
        if fr.l2:
            r = fr.l2 - 1
            self.sil.unpin(r)
            index = self._new_b(Binder(L_REG, r, size=size, flags=flags,
                                       word=self.res_word or self._word()))
        elif res is not None and res[0] == "bind":
            index = res[1]  # let is renaming: H(x) := H(y), no instructions
        else:
            assert res is not None
            n = res[1]
            if n <= _IMM32_MAX:
                index = self._new_b(Binder(L_IMM, size=size, flags=flags, word=n))
            else:
                r = self._take()
                self._mov_imm(r, n)
                index = self._new_b(Binder(L_REG, r, size=size, flags=flags, word=self._word()))
        self.srcmap.append(index)
        fr.st, fr.l2 = 1, 0
        self._save(fr)

    def _close_scope(self, fr: Fr) -> None:
        self._pop_binders(fr.mark)
        del self.srcmap[fr.srcmark:]

    def _close_let(self) -> None:
        self._close_scope(self._top())
        self.K.pop()

    # --- if

    def _save_cond(self) -> int:
        saved = self.cond_active | self.cond_jumped << 1
        self.cond_active = True
        self.cond_jumped = False
        return saved

    def _restore_cond(self, saved: int) -> None:
        outer_active = bool(saved & 1)
        self.cond_jumped = outer_active and (bool(saved & 2) or self.cond_jumped)
        self.cond_active = outer_active

    def _open_if(self, dest: Dest, _payload: bytes) -> None:
        labels = (self._label(), self._label(), self._label())
        self._push(K_IF, dest, tail=_SNAPSHOT_BYTES, b3=self._save_cond(),
                   l1=labels[0], l2=labels[1], l3=labels[2])

    def _sep_if(self, payload: bytes) -> None:
        fr = self._top()
        if payload[0] == 1:
            self._restore_cond(fr.b3)
            fr.tail[:_SNAPSHOT_BYTES] = self._snapshot()
            self.arm_level += 1
            fr.st = 1
            self._save(fr)
            return
        self._remember_erased(fr)
        fr.word = self.res_word
        if fr.dk != D_COND and not self.dead:
            self._normalize(bytes(fr.tail))
            self._goto(fr.l3)
        self._restore(bytes(fr.tail))
        fr.st = 2
        self._save(fr)

    def _close_if(self) -> None:
        fr = self._top()
        self.arm_level -= 1
        if fr.dk != D_COND and not self.dead:
            self._normalize(bytes(fr.tail))
            self.place(fr.l3)
        self._merge_erased(fr)
        self.res_word = self._join_word(fr.word, self.res_word)
        self.K.pop()
        self.res = None

    @staticmethod
    def _join_word(first: int, second: int) -> int:
        """The word of a join. A value has a provenance word only if every arm
        delivered that same word. Trusting one arm would let RESTORE drop a store
        that the other arm needs, which is a miscompile, not a lost optimization."""
        return first if first == second else 0

    # --- match

    def _open_match(self, dest: Dest, _payload: bytes) -> None:
        self._push(K_MATCH, dest, tail=_SNAPSHOT_BYTES)

    def _scrutinee(self, fr: Fr) -> Optional[int]:
        """Bring the scrutinee to a register, or report a known constructor."""
        res = self.res
        if fr.l2:
            r = fr.l2 - 1
            self.sil.unpin(r)
            fr.l2 = 0
            return self._new_b(Binder(L_REG, r, flags=F_LINEAR,
                                      word=self.res_word or self._word()))
        if res is None or res[0] != "bind":
            return None
        index = self._resolve(res[1])
        b = self._b(index)
        if b.loc == L_VIRT:
            return None
        if b.loc == L_REG:
            return index
        r = self._take()
        self._load_opnd(self._opnd(index), r)
        self._end_scratch()
        self._used(index)
        return self._new_b(Binder(L_REG, r, flags=F_LINEAR, word=b.word))

    def _sep_match(self, _payload: bytes) -> None:
        fr = self._top()
        cell = self._scrutinee(fr)
        if cell is None:
            res = self.res
            fr.b4 = 2 if res is not None and res[0] == "imm" else 1  # phantom
            if fr.b4 == 1:
                assert res is not None
                fr.l3 = self._resolve(res[1]) + 1
        else:
            fr.l3 = cell + 1
            fr.tail[:_SNAPSHOT_BYTES] = self._snapshot()
            fr.l1, fr.l2 = self._label(), self._label()
            fr.q = self._label()
        fr.st = 1
        self.res = None
        self._save(fr)

    def _open_arm(self, payload: bytes) -> None:
        tag, tagged, is_null, arm_index, nctors = P_ARM.unpack_from(payload)
        fr = self._top()
        if fr.b4:  # phantom: only the known constructor's arm exists
            if fr.b4 == 2:
                taken = is_null
            else:
                taken = self._b(fr.l3 - 1).r == tag
            if not taken:
                self.skip = 1
                return
            self._push(K_ARM, fr.dest(), b1=1, l1=fr.l3)
            return
        cell = fr.l3 - 1
        if tagged:
            if arm_index:
                self.place(fr.l1)
            if arm_index < nctors - 1:
                fr.l1 = self._label()
                self.i("cmp", (O_MEM, self._base(cell), 0, 1), (O_IMM, tag))
                self.i("jcc", (O_LABEL, fr.l1), (O_IMM, CC_NAMES.index("ne")))
                self._end_scratch()
        elif nctors == 2:  # tagless: dispatch is `test p, p`; a record has no dispatch
            if arm_index == 0:
                r = self._base(cell)
                self.i("test", (O_REG, r, 8), (O_REG, r, 8))
                self._end_scratch()
                self._hold_flags(CC_NAMES.index("e"), fr.l1, fr.l2)
            self.place(fr.l1 if is_null else fr.l2)
        b = self._b(cell)
        b.flags = b.flags | F_NULLARM if is_null else b.flags & ~F_NULLARM
        b.aux = tag + 1 if tagged else 0
        self._set_b(cell, b)
        self.arm_level += 1
        self._save(fr)
        self._push(K_ARM, fr.dest(), l3=fr.l3, b4=arm_index == nctors - 1)

    def _close_arm(self) -> None:
        arm = self._top()
        self._close_scope(arm)
        self.K.pop()
        if arm.b1:
            return
        fr = self._top()
        self.arm_level -= 1
        cell = fr.l3 - 1
        word = self.res_word
        if not self.dead:
            self._consume(cell)
        self._remember_erased(fr)
        fr.word = word if not fr.b3 else self._join_word(fr.word, word)
        fr.b3 = 1
        self.res_word = fr.word
        if arm.b4:
            self._save(fr)
            return
        if fr.dk != D_COND and not self.dead:
            self._normalize(bytes(fr.tail))
            self._goto(fr.q)
        self._restore(bytes(fr.tail))
        self._save(fr)

    def _close_match(self) -> None:
        fr = self._top()
        if not fr.b4:
            if fr.dk != D_COND and not self.dead:
                self._normalize(bytes(fr.tail))
                self.place(fr.q)
            self._merge_erased(fr)
            b = self._b(fr.l3 - 1)
            b.flags &= ~F_NULLARM
            self._set_b(fr.l3 - 1, b)
        self._close_scope(fr)
        self.K.pop()
        self.res = None

    # --- record update, section 5.5: only field k emits code

    def _open_with(self, dest: Dest, _payload: bytes) -> None:
        self._push(K_WITH, dest)

    def _sep_with(self, payload: bytes) -> None:
        fr = self._top()
        if payload[0] == 2:
            fr.st = 2
            self._save(fr)
            return
        cell = self._scrutinee(fr)
        if cell is None:  # let-bound and unbuilt: this is its use, so it is built now
            virt = self._resolve(self.res[1])
            self._materialize(virt)
            cell = self._resolve(self._b(virt).base)
        fr.l1 = cell + 1
        fr.st = 1
        self.res = None
        self._save(fr)

    def _close_with(self) -> None:
        fr = self._top()
        cell = fr.l1 - 1
        self._deliver_pointer(cell, fr.dest())
        word_result = self.res_word
        if not self.dead:
            self._consume(cell)
        self._close_scope(fr)
        self.K.pop()
        self.res = None
        self.res_word = word_result

    # --- constructors, section 5.4

    def _open_ctor(self, dest: Dest, payload: bytes) -> None:
        self._push(K_CTOR, dest, tail=_ENTRY.size * payload[0])

    def _entry(self, fr: Fr, position: int, off: int, size: int) -> None:
        """Record field or argument `position` as a pending move."""
        res = self.res
        if res is not None and res[0] == "trmc":
            fr.q = 1
            return
        if fr.l2:
            r = fr.l2 - 1
            self.sil.unpin(r)
            index = self._new_b(Binder(L_REG, r, flags=F_TEMP,
                                       word=self.res_word or self._word()))
            fr.l2 = 0
            entry = (E_BIND, 0, off, size, index)
        elif res is None:
            raise self._fail("a pending move has no value")
        elif res[0] == "bind":
            entry = (E_BIND, 0, off, size, res[1])
        elif res[0] == "imm":
            entry = (E_IMM, 0, off, size, res[1])
        else:
            _rmw, op, value = res
            kind = E_RMW_IMM if value[0] == "imm" else E_RMW_BIND
            entry = (kind, op, off, size, value[1])
        _ENTRY.pack_into(fr.tail, _ENTRY.size * position, *entry)
        self.res = None

    def _sep_ctor(self, payload: bytes) -> None:
        index, off, size, _linear = P_FIELD.unpack_from(payload)
        fr = self._top()
        self._entry(fr, index - 1, fr.b4, fr.w)
        fr.st, fr.b4, fr.w = index, off, size
        self._save(fr)

    def _ctor_flush(self, fr: Fr, index: int, keep: frozenset = frozenset()) -> None:
        cell = fr.l1 - 1
        moves = []
        for position in range(fr.b3):
            kind, op, off, size, value = _ENTRY.unpack_from(fr.tail, _ENTRY.size * position)
            if kind != E_EMPTY:
                moves.append([("m", cell, off, size), kind, op, value])
            _ENTRY.pack_into(fr.tail, _ENTRY.size * position, E_EMPTY, 0, 0, 0, 0)
        # These moves are being carried out, so they are no longer deferred reads.
        self._save_at(index, fr)
        self._pmove(moves)
        # The tag byte is a store like any other: whatever still lives at offset 0 has
        # to leave before it lands, so the window closes first.
        self._close_stale_window(cell, keep)
        self._write_tag(cell, fr.b1, fr.b2)

    def _close_stale_window(self, cell: int, keep: frozenset = frozenset()) -> None:
        """Lemma 6.2's window ends when the cell is rebuilt: the cell now belongs to
        the value just built, which may go anywhere. A linear field still unused
        leaves first, and a scalar field is erased. `keep` names values the very
        next move reads, whose locations no store in this rebuild has touched."""
        self._evacuate_linear(cell, 0, 16, keep)
        self._end_scratch()
        self._stored(cell, 0, 16, 0)

    def _close_ctor(self) -> None:
        fr = self._top()
        cell = fr.l1 - 1
        if fr.b3:
            self._entry(fr, fr.st, fr.b4, fr.w)
        dest = fr.dest()
        if fr.q:  # the last field was a recursive call: control already left
            self._pop_binders(fr.mark)
            self.K.pop()
            self.res = None
            return
        if dest.kind == D_FIELDS:
            self._virtualize(fr)
            self.K.pop()
            return
        self._ctor_flush(fr, len(self.K) - 1)
        self._pop_binders(fr.mark)
        self.K.pop()
        self._hand_over(cell, dest)

    def _hand_over(self, cell: int, dest: Dest) -> None:
        """The built value leaves. Under D_ANY the cell's own home is the operand,
        so a constructor passed to a call costs no move at all."""
        b = self._b(cell)
        if dest.kind == D_ANY:
            b.flags |= F_LINEAR
            self._set_b(cell, b)
            self.res = ("bind", cell)
            self.res_word = b.word
            return
        self._deliver_pointer(cell, dest)
        word = self.res_word
        if not self.dead:
            self._consume(cell)
        self.res_word = word

    def _virtualize(self, fr: Fr) -> None:
        """Phantom construction: the fields keep their homes and nothing is
        written, because the one consumer of this value is linear (Theorem 7.4)."""
        cell = fr.l1 - 1
        first = len(self.B)
        for position in range(fr.b3):
            kind, _op, off, size, value = _ENTRY.unpack_from(fr.tail, _ENTRY.size * position)
            if kind == E_IMM and value <= _IMM32_MAX:
                self._new_b(Binder(L_IMM, off=off, size=size, word=value))
            elif kind == E_IMM:
                r = self._take()
                self._mov_imm(r, value)
                self._new_b(Binder(L_REG, r, off=off, size=size, flags=F_TEMP,
                                   word=self._word()))
            else:
                self._new_b(Binder(L_ALIAS, base=value, off=off, size=size))
        virt = self._new_b(Binder(L_VIRT, fr.b1, cell, fr.b3, fr.b2, flags=F_LINEAR,
                                  aux=first, word=self._b(cell).word))
        self.res = ("bind", virt)
        self.res_word = self._b(cell).word

    # --- calls, section 5.7

    def _open_call(self, dest: Dest, payload: bytes) -> None:
        ordinal, is_self, nargs = P_CALL.unpack_from(payload)
        if is_self and dest.kind not in (D_RET, D_TRMC):
            raise self._fail("section 4.4: a recursive call must be a tail call or the "
                             "last field of a returned constructor; structural descent "
                             "is not implemented in Phase 1", dest)
        self._push(K_CALL, dest, tail=_ENTRY.size * nargs, b1=is_self, b2=nargs, q=ordinal)

    def _sep_call(self, payload: bytes) -> None:
        fr = self._top()
        self._entry(fr, payload[0] - 1, 0, 8)
        fr.st = payload[0]
        self._save(fr)

    def _call_moves(self, fr: Fr) -> list[list]:
        moves = []
        for position in range(fr.b2):
            kind, _op, _off, _size, value = _ENTRY.unpack_from(fr.tail, _ENTRY.size * position)
            moves.append([("r", PARAM_REGS[position]), kind, 0, value])
        return moves

    def _arg_word(self, moves: list[list], position: int) -> int:
        move = moves[position]
        if move[1] == E_BIND:
            return self._word_of(move[3])
        return 1 if move[1] == E_IMM and move[3] == 0 else 0

    def _param_mask(self, moves: list[list]) -> int:
        mask = 0
        for move in moves:
            mask |= 1 << move[0][1]
        return mask

    def _close_call(self) -> None:
        fr = self._top()
        if fr.b2:
            self._entry(fr, fr.st, 0, 8)
        moves = self._call_moves(fr)
        dest = fr.dest()
        if dest.kind == D_TRMC:
            self._close_trmc(fr, moves)
        elif dest.kind == D_RET and (fr.b1 or self.root < 0):
            self._close_tail(fr, moves)
        else:
            self._close_plain(fr, moves, dest)

    def _jump_after(self, fr: Fr, moves: list[list]) -> None:
        """The loop head was lowered with the root in rax and each parameter in its
        register, so that is where a back-edge must leave them."""
        if fr.b1 and self.root >= 0:
            moves = moves + [[("r", RAX), E_BIND, 0, self.root]]
        self.lock_base = self._param_mask(moves)
        self.locked = self.lock_base
        self._pmove(moves)
        self.lock_base = self.locked = 0
        if fr.b1:
            self.i("jmp", (O_LABEL, self.fn_loop))
        else:
            self._raw_name("jmp_name", self._fn_entry(fr.q)[0])
        self.dead = True
        self._pop_binders(fr.mark)
        self.K.pop()

    def _raw_name(self, op: str, name: bytes) -> None:
        self._flush()
        self.out.append((OP[op], name))

    def _close_tail(self, fr: Fr, moves: list[list]) -> None:
        """D[[f(e)]]_tail = pmove{param <- e} . jmp f: the frame is reused."""
        if self.root >= 0 and self._arg_word(moves, self.res_src) != self._b(self.root).word:
            raise Reject("a tail call passes a different cell than the one '@' declares")
        if not fr.b1:
            entry = self._fn_entry(fr.q)
            self.fn_stack = max(self.fn_stack, entry[3])
            self.touched |= entry[1]
        self._jump_after(fr, moves)
        self.res = None

    def _close_trmc(self, fr: Fr, moves: list[list]) -> None:
        """Section 5.6. The destination hole has the argument's provenance, so
        RESTORE applies and the location reverts with zero instructions."""
        parent_index = len(self.K) - 2
        parent = self._fr_at(parent_index)
        cell = parent.l1 - 1
        hole = self._word_at(cell, parent.b4, parent.w)
        word = self._arg_word(moves, self.res_src) if self.res_src >= 0 else 0
        if self.root < 0 or word == 0 or word != hole or (
                self._b(cell).word != self._b(self.root).word):
            raise self._fail(
                "section 5.6: this recursive call is not in tail position modulo cons, "
                "because RESTORE does not apply; declare '@' on the parameter whose "
                "own cell is rebuilt, and pass that field to the recursive call")
        arguments = frozenset(self._resolve(m[3]) for m in moves if m[1] == E_BIND)
        self._ctor_flush(parent, parent_index, keep=arguments)
        self._jump_after(fr, moves)
        self.res = ("trmc",)

    def _evacuation_moves(self, mask: int, consumed: frozenset) -> list[tuple[int, tuple]]:
        """Choose a new home for every live value in the registers a call consumes
        (section 4.2), taking the new home's token now."""
        moved = []
        for index in range(len(self.B)):
            b = self._b(index)
            if (b.loc != L_REG or b.flags & F_CONSUMED or not mask >> b.r & 1
                    or index in consumed):
                continue
            if b.flags & F_PINNED:
                raise self._fail(
                    f"section 4.1 against 4.2: pinned partial result in {REGS[b.r]} "
                    f"crosses a call that clobbers it; bind the call with let first")
            self._home_changing()
            free = self.sil.free & ~mask
            target = next((r for r in ALLOC_ORDER if free >> r & 1), None)
            if target is None:
                moved.append((index, ("s", self.sil.take_slot())))
            else:
                self.sil.take(target)
                moved.append((index, ("r", target)))
        return moved

    def _fn_entry(self, ordinal: int) -> tuple:
        kind, data = self.table.at(ordinal)
        name = data[:15]
        kappa, res, stack = _FN_ATTR.unpack_from(data, 15)
        return name, kappa, res, stack

    def _close_plain(self, fr: Fr, moves: list[list], dest: Dest) -> None:
        """Section 5.7 writes evacuate . pmove, but a value spilled first would be
        reloaded while its register still holds it, which is the I3 round trip that
        Theorem 7.5 rules out. So both happen as one parallel move: every source is
        read from where it is now, and nothing comes back from a slot."""
        name, kappa, res, stack = self._fn_entry(fr.q)
        word = self._arg_word(moves, res) if res != 255 else self._word()
        params = self._param_mask(moves)
        mask = kappa | params | 1 << RAX
        arguments = [m for m in moves if m[1] == E_BIND]
        # The callee owns any cell passed to it and may write it, so the stale-read
        # window closes before the call, exactly as it does for a rebuild here. Tail
        # calls and returns do not need this: nothing in this function runs after them.
        passed = frozenset(self._resolve(m[3]) for m in arguments)
        for index in passed:
            self._close_stale_window(index, passed)
        leaving = frozenset(
            self._resolve(m[3]) for m in arguments
            if self._b(self._resolve(m[3])).flags & (F_LINEAR | F_TEMP))
        moved = self._evacuation_moves(mask, leaving)
        self.lock_base = params
        self.locked = params
        pending = [m for m in moves
                   if not (m[1] == E_BIND and self._key(m[3]) == m[0])]
        pending += [[home, E_BIND, 0, index] for index, home in moved]
        self._run_moves(pending)
        self.lock_base = self.locked = 0
        for index, home in moved:
            b = self._b(index)
            self.sil.give(b.r)
            b.loc, b.r = (L_REG, home[1]) if home[0] == "r" else (L_SLOT, home[1])
            self._set_b(index, b)
        for move in arguments:
            self._used(self._resolve(move[3]))
        self._raw_name("call_name", name)
        self._forget_registers()
        self.touched |= mask
        self.fn_stack = max(self.fn_stack, stack)
        self._pop_binders(fr.mark)
        self.K.pop()
        self.locked = 1 << RAX
        self.lock_base = 1 << RAX
        self._deliver_reg(RAX, word, dest)
        self.lock_base = self.locked = 0
        self.last_width = 8

    # --- arithmetic, section 4.1: source order, with leaves deferred

    def _open_binop(self, dest: Dest, payload: bytes) -> None:
        self._push(K_BINOP, dest, b1=payload[0])

    def _open_cmp(self, dest: Dest, payload: bytes) -> None:
        self._push(K_CMP, dest, b1=payload[0])

    def _sep_operand(self, _payload: bytes) -> None:
        fr = self._top()
        if fr.dk != D_EFF:
            res = self.res
            if res is None:
                w = fr.b2 - 1
                self.sil.unpin(w)
                fr.l1 = self._new_b(Binder(L_REG, w, flags=F_PINNED | F_TEMP,
                                           word=self._word())) + 1
                fr.b4, fr.w = 3, self.last_width
            elif res[0] == "imm":
                fr.b4, fr.q, fr.w = 2, res[1], 0
            else:
                fr.b4, fr.q = 1, res[1]
                fr.w = self._b(self._resolve(res[1])).size
                if (fr.kind == K_BINOP and fr.dk in (D_MEM, D_PMEM)
                        and OP_NAMES[fr.b1] in _RMW_OPS):
                    target = ("m", self._resolve(fr.dx), fr.dy & 255, fr.dy >> 8)
                    if self._key(res[1]) == target:
                        fr.l3 = 1
        fr.st = 1
        self._save(fr)

    def _release_partial(self, fr: Fr) -> None:
        """The accumulator stops being a temp and goes back to the destination."""
        if fr.l1:
            index = fr.l1 - 1
            b = self._b(index)
            b.flags |= F_CONSUMED
            self._set_b(index, b)
            self.sil.pin(b.r)

    def _second_value(self, fr: Fr) -> Optional[tuple]:
        """The right operand: a leaf, or a register the child delivered into."""
        if fr.l2:
            return ("reg", fr.l2 - 1)
        return self.res

    def _operand_of(self, value: tuple, width: int) -> tuple:
        if value[0] == "reg":
            return (O_REG, value[1], 8 if width == 8 else 4)
        return self._value_opnd(value, width)

    def _close_binop(self) -> None:
        fr = self._top()
        dest = fr.dest()
        op = OP_NAMES[fr.b1]
        second = self._second_value(fr)
        if dest.kind == D_EFF:
            self._consume_value(second if second and second[0] == "bind" else None)
            if fr.b4 == 1:
                self._used(self._resolve(fr.q))
            self.K.pop()
            self.res = None
            return
        width = self.close_payload[0]
        if fr.l3:
            self._close_rmw(fr, dest, op, second)
            return
        w = self._work(fr)
        assert second is not None
        shifting = op in ("shl", "shr")
        literal = self._literal_count(second) if shifting else None
        vanishes = literal is not None and literal >= width * 8
        if fr.b4 == 1:
            self.locked = 1 << w
            if not vanishes:
                self._load_opnd(self._opnd(fr.q), w)
            self._used(self._resolve(fr.q))
        elif fr.b4 == 2 and not vanishes:
            self._mov_imm(w, fr.q)
        self.locked = 1 << w
        wide = 8 if width == 8 else 4
        if vanishes:
            self._mov_imm(w, 0)  # section 4.4: x86 would mask the count and shift anyway
        elif shifting and literal is None:
            self._shift_by_register(op, w, wide, width * 8, second)
        else:
            self.i(op, (O_REG, w, wide), self._operand_of(second, width))
        self._end_scratch()
        if width < 4 and op in _NEEDS_NORMAL and not vanishes:
            self.i("movzx", (O_REG, w, 4), (O_REG, w, width))
        if second[0] == "bind":
            self._used(self._resolve(second[1]))
        elif second[0] == "reg":
            self._unpin_free(second[1])
        self._release_partial(fr)
        self._pop_binders(fr.mark)
        self.K.pop()
        self.last_width = width
        self._finish_work(fr, w, dest)

    def _literal_count(self, value: tuple) -> Optional[int]:
        """The count, when it is known here: a literal, or a let bound to one."""
        if value[0] == "imm":
            return value[1]
        if value[0] == "bind":
            b = self._b(self._resolve(value[1]))
            if b.loc == L_IMM:
                return b.word
        return None

    def _shift_by_register(self, op: str, w: int, wide: int, bits: int, count: tuple) -> None:
        """Section 4.4: a shift is 0 when the count reaches the width. x86 takes the count
        from cl and masks it, so this compares first and branches around the shift.

        rcx is borrowed for this one operation rather than taken, so no value changes
        home, which also keeps this legal under a condition that has already jumped.
        Whatever rcx holds is saved around it, unless it is the count itself."""
        source = (O_REG, count[1], 8) if count[0] == "reg" else self._opnd(count[1])
        holds_count = source[0] == O_REG and source[1] == RCX
        target = w
        saved = None
        if w == RCX:  # the value has to get out of the count's way, and the result come back
            target = self._scratch(avoid=1 << RCX)
            self.i("mov", (O_REG, target, 8), (O_REG, RCX, 8))
        elif not holds_count and not self.sil.free >> RCX & 1:
            saved = self._scratch(avoid=1 << RCX)
            self.i("mov", (O_REG, saved, 8), (O_REG, RCX, 8))
        if not holds_count:
            self._load_opnd(source, RCX)
        too_far, done = self._label(), self._label()
        self.i("cmp", (O_REG, RCX, 8), (O_IMM, bits))
        self.i("jcc", (O_LABEL, too_far), (O_IMM, CC_NAMES.index("ae")))
        self.i(op, (O_REG, target, wide), (O_REG, RCX, 1))
        self.i("jmp", (O_LABEL, done))
        self.place(too_far)
        self.i("xor", (O_REG, target, 4), (O_REG, target, 4))
        self.place(done)
        if w == RCX:
            self.i("mov", (O_REG, RCX, 8), (O_REG, target, 8))
        elif saved is not None:
            self.i("mov", (O_REG, RCX, 8), (O_REG, saved, 8))

    def _finish_work(self, fr: Fr, w: int, dest: Dest) -> None:
        word = self._word()
        if dest.kind == D_REG:
            self.res, self.res_word = None, word
            return
        if dest.kind == D_PMEM:
            self.sil.unpin(w)
            self.res = ("bind", self._new_b(Binder(L_REG, w, flags=F_TEMP, word=word)))
            self.res_word = word
            return
        self._deliver_reg(w, word, dest)
        self._unpin_free(w)

    def _close_rmw(self, fr: Fr, dest: Dest, op: str, second: Optional[tuple]) -> None:
        """op [a], o for x = mem a under delta = mem a (section 5.4)."""
        assert second is not None
        index = OP_NAMES.index(op)
        self.K.pop()
        if second[0] == "reg":
            self.sil.unpin(second[1])
            source = self._new_b(Binder(L_REG, second[1], flags=F_TEMP, word=self._word()))
            value = ("bind", source)
        else:
            value = second
        self.last_width = fr.w
        if dest.kind == D_PMEM:
            self.res = ("rmw", index, value)
            return
        target = ("m", self._resolve(dest.x), dest.y & 255, dest.y >> 8)
        kind = E_RMW_IMM if value[0] == "imm" else E_RMW_BIND
        self._pmove([[target, kind, index, value[1]]])
        self._pop_binders(fr.mark)
        self.res = None

    def _close_cmp(self) -> None:
        fr = self._top()
        dest = fr.dest()
        second = self._second_value(fr)
        if dest.kind == D_EFF:
            self._consume_value(second if second and second[0] == "bind" else None)
            if fr.b4 == 1:
                self._used(self._resolve(fr.q))
            self.K.pop()
            self.res = None
            return
        assert second is not None
        if fr.b4 == 1:
            first = self._opnd(fr.q)
        elif fr.b4 == 2:
            r = self._scratch()
            self._mov_imm(r, fr.q)
            first = (O_REG, r, 8)
        else:
            first = (O_REG, fr.b2 - 1, 8)
        if second[0] == "imm" and second[1] <= _IMM32_MAX:
            other = (O_IMM, second[1])
        elif second[0] == "imm":
            r = self._scratch()
            self._mov_imm(r, second[1])
            other = (O_REG, r, 8)
        elif second[0] == "reg":
            other = (O_REG, second[1], 8)
        else:
            other = self._opnd(second[1])
        if first[0] not in (O_REG, O_MEM, O_SLOT) or (
                first[0] != O_REG and other[0] not in (O_REG, O_IMM)):
            r = self._scratch()
            self._load_opnd(first, r)
            first = (O_REG, r, 8)
        if first[0] == O_MEM and other[0] == O_IMM and first[3] < 8 and (
                other[1] >= 1 << 8 * first[3]):
            # The literal does not fit the field: compare in 64 bits, as registers do.
            r = self._scratch()
            self._load_opnd(first, r)
            first = (O_REG, r, 8)
        compare = first[3] if first[0] == O_MEM else (
            other[3] if other[0] == O_MEM else 8)
        if first[0] == O_REG:
            first = (O_REG, first[1], compare)
        if other[0] == O_REG:
            other = (O_REG, other[1], compare)
        self.i("cmp", first, other)
        self._end_scratch()
        if fr.b4 == 1:
            self._used(self._resolve(fr.q))
        if second[0] == "bind":
            self._used(self._resolve(second[1]))
        elif second[0] == "reg":
            self._unpin_free(second[1])
        partial = fr.b4 == 3
        self._release_partial(fr)
        self._pop_binders(fr.mark)
        self.K.pop()
        self.last_width = 1
        cc = fr.b1
        if dest.kind == D_COND:
            self.res = None
            self._hold_flags(cc, dest.x, dest.y)
            if partial and fr.b3:
                self._unpin_free(fr.b2 - 1)
            return
        # Value position, the only rule that emits setcc (section 5.3).
        if dest.kind == D_REG:
            w = dest.x
            owned = False
        elif partial and fr.b3:
            w = fr.b2 - 1
            owned = True
        else:
            w = self._pin_new(prefer=RAX if dest.kind == D_RET else None)
            owned = True
        self.i("setcc", (O_REG, w, 1), (O_IMM, cc))
        self.i("movzx", (O_REG, w, 4), (O_REG, w, 1))
        word = self._word()
        self._deliver_reg(w, word, dest)
        if owned:
            self._unpin_free(w)

    # --- jumping code, section 5.3

    def _open_logic(self, kind: int, dest: Dest) -> None:
        mid = self._label() if kind != K_NOT else 0
        if dest.kind == D_COND:
            self._push(kind, dest, l1=mid, l2=dest.x, l3=dest.y)
            return
        # Value position: the same jumping code, materialized once at the end.
        on_true, on_false = self._label(), self._label()
        self._push(kind, dest, b1=1, b3=self._save_cond(), l1=mid, l2=on_true, l3=on_false)

    def _open_and(self, dest: Dest, _payload: bytes) -> None:
        self._open_logic(K_AND, dest)

    def _open_or(self, dest: Dest, _payload: bytes) -> None:
        self._open_logic(K_OR, dest)

    def _open_not(self, dest: Dest, _payload: bytes) -> None:
        self._open_logic(K_NOT, dest)

    def _sep_logic(self, _payload: bytes) -> None:
        fr = self._top()
        self.place(fr.l1)
        fr.st = 1
        self._save(fr)

    def _close_logic(self) -> None:
        fr = self._top()
        self.K.pop()
        self.res = None
        if not fr.b1:
            return
        self._restore_cond(fr.b3)
        dest = fr.dest()
        after = self._label()
        self.place(fr.l2)
        self._deliver_imm(1, dest)
        if not self.dead:
            self._goto(after)
        self.dead = False
        self.place(fr.l3)
        self._deliver_imm(0, dest)
        if not self.dead:
            self.place(after)

    # --- pack, section 4.5: in-place Horner accumulation in a pinned register

    def _open_pack(self, dest: Dest, payload: bytes) -> None:
        widths = payload.rstrip(bytes(1))
        self._push(K_PACK, dest, b1=len(widths), q=int.from_bytes(widths, "little"),
                   tail=8)

    def _pack_field(self, fr: Fr, position: int) -> None:
        width = fr.q >> (8 * position) & 255
        value = self._second_value(fr)
        if fr.dk == D_EFF:
            self._consume_value(value if value and value[0] == "bind" else None)
            return
        assert value is not None
        accumulated = int.from_bytes(fr.tail[:8], "little")
        if not fr.b4:  # still folding leading constants at compile time
            if value[0] == "imm":
                fr.tail[:8] = ((accumulated << width | value[1] & (1 << width) - 1)
                               & (1 << 64) - 1).to_bytes(8, "little")
                return
            w = self._work(fr)
            fr.b4 = 1
            if position == 0:
                self.locked = 1 << w
                self._pack_first(value, w, width)
                self._after_field(value)
                return
            self._mov_imm(w, accumulated << width & (1 << 64) - 1)
        else:
            w = fr.b2 - 1
            self.i("shl", (O_REG, w, 8), (O_IMM, width))
        self.locked = 1 << w
        self._pack_or(value, w, width)
        self._after_field(value)

    def _after_field(self, value: tuple) -> None:
        self._end_scratch()
        if value[0] == "bind":
            self._used(self._resolve(value[1]))
        elif value[0] == "reg":
            self._unpin_free(value[1])

    def _pack_first(self, value: tuple, w: int, width: int) -> None:
        """D[[e1]](reg r), then truncated if e1 is wider than its field."""
        if value[0] == "reg":
            size = self.last_width or 8
            self.i("mov", (O_REG, w, 8), (O_REG, value[1], 8))
        else:
            size = self._b(self._resolve(value[1])).size
            self._load_opnd(self._opnd(value[1]), w)
        if size * 8 > width:
            if width == 32:
                self.i("mov", (O_REG, w, 4), (O_REG, w, 4))
            else:
                self.i("movzx", (O_REG, w, 4), (O_REG, w, width // 8))

    def _pack_or(self, value: tuple, w: int, width: int) -> None:
        """or r_|w|, opnd(e): 8- and 16-bit writes merge without disturbing the
        accumulator. A 32-bit or would clear it, so 32 goes through the zero
        extension of a 32-bit register write (the extra load section 8 names)."""
        wide = width // 8
        if value[0] == "imm":
            bits = value[1] & (1 << width) - 1
            if bits > _IMM32_MAX:
                r = self._scratch()
                self._mov_imm(r, bits)
                self.i("or", (O_REG, w, 8), (O_REG, r, 8))
            elif bits:
                self.i("or", (O_REG, w, 8), (O_IMM, bits))
            return
        if value[0] == "reg":
            r = value[1]
            source = (O_REG, r, wide)
            size = 8
        else:
            b = self._b(self._resolve(value[1]))
            size = b.size
            source = self._opnd(value[1])
        if source[0] == O_MEM and source[3] == wide and wide < 4:
            self.i("or", (O_REG, w, wide), source)
            return
        if source[0] == O_REG and wide < 4:
            self.i("or", (O_REG, w, wide), (O_REG, source[1], wide))
            return
        if source[0] == O_REG and size <= 4:
            self.i("or", (O_REG, w, 8), (O_REG, source[1], 8))
            return
        r = self._scratch()
        if source[0] == O_REG:
            self.i("mov", (O_REG, r, min(wide, 4)), (O_REG, source[1], min(wide, 4)))
        elif source[0] == O_MEM and source[3] >= 4:
            self.i("mov", (O_REG, r, min(wide, 4)), (O_MEM, source[1], source[2], min(wide, 4)))
        else:
            self._load_opnd(source, r)
        if wide < 4:
            self.i("or", (O_REG, w, wide), (O_REG, r, wide))
        else:
            self.i("or", (O_REG, w, 8), (O_REG, r, 8))

    def _sep_pack(self, payload: bytes) -> None:
        fr = self._top()
        self._pack_field(fr, payload[0] - 1)
        fr.l2 = 0
        fr.st = payload[0]
        self._save(fr)

    def _close_pack(self) -> None:
        fr = self._top()
        self._pack_field(fr, fr.st)
        dest = fr.dest()
        self.K.pop()
        self.last_width = 8
        if dest.kind == D_EFF:
            self.res = None
            return
        if not fr.b4:
            w = self._work(fr)
            self._mov_imm(w, int.from_bytes(fr.tail[:8], "little"))
        else:
            w = fr.b2 - 1
        self._finish_work(fr, w, dest)

    # --- functions

    def _open_fn(self, name: bytes) -> None:
        self.fn_name = name
        self.sil = Silicon()
        self.touched = 0
        self.fn_stack = 0
        self.root = -1
        self.res_src = -1
        self.dead = False
        self.arm_level = 0
        self.cond_active = self.cond_jumped = False
        self._push(K_FN, Dest(D_RET))
        self._raw_name("func_label", name)

    def _sep_fn(self, _payload: bytes) -> None:
        if self.res_src >= 0:
            param = self._b(self.srcmap[self.res_src])
            self.sil.take(RAX)
            self.i("mov", (O_REG, RAX, 8), (O_REG, param.r, 8))
            self.root = self._new_b(Binder(L_REG, RAX, word=param.word))
        self.fn_loop = self._label()
        self.place(self.fn_loop)

    def _close_fn(self) -> None:
        fr = self._top()
        self._flush()
        slots = self.sil.slots_minted
        name = self.fn_name
        if slots:
            size = struct.pack("<I", 8 * slots)
            self.out.extend(((OP["frame_label"], name), (OP["frame_enter"], size),
                             (OP["frame_call"], name), (OP["frame_leave"], size)))
        else:
            self.out.append((OP["func_alias"], name))
        stack = self.fn_stack + 8 + (8 * slots + 8 if slots else 0)
        self.stack_bound = max(self.stack_bound, stack)
        res = 255 if self.res_src < 0 else self.res_src
        self.table.push(0, name.ljust(15, b"\0") + _FN_ATTR.pack(self.touched, res, stack))
        if name.rstrip(bytes(1)) == b"main":
            self.has_main = 2 if fr.b2 else 1
        self.root = -1
        self._close_scope(fr)
        self.K.pop()

    OPENERS: dict = {}
    SEPS: dict = {}
    CLOSERS: dict = {}


_FN_ATTR = struct.Struct("<HBI")  # kappa, result parameter or 255, stack bound

Lower.OPENERS = {K_LET: Lower._open_let, K_IF: Lower._open_if, K_MATCH: Lower._open_match,
                 K_WITH: Lower._open_with, K_CTOR: Lower._open_ctor, K_CALL: Lower._open_call,
                 K_BINOP: Lower._open_binop, K_CMP: Lower._open_cmp, K_AND: Lower._open_and,
                 K_OR: Lower._open_or, K_NOT: Lower._open_not, K_PACK: Lower._open_pack}
Lower.SEPS = {K_FN: Lower._sep_fn, K_LET: Lower._sep_let, K_IF: Lower._sep_if,
              K_MATCH: Lower._sep_match, K_ARM: lambda self, payload: None,
              K_WITH: Lower._sep_with, K_CTOR: Lower._sep_ctor, K_CALL: Lower._sep_call,
              K_BINOP: Lower._sep_operand, K_CMP: Lower._sep_operand,
              K_AND: Lower._sep_logic, K_OR: Lower._sep_logic, K_PACK: Lower._sep_pack}
Lower.CLOSERS = {K_FN: Lower._close_fn, K_LET: Lower._close_let, K_IF: Lower._close_if,
                 K_MATCH: Lower._close_match, K_ARM: Lower._close_arm,
                 K_WITH: Lower._close_with, K_CTOR: Lower._close_ctor,
                 K_CALL: Lower._close_call, K_BINOP: Lower._close_binop,
                 K_CMP: Lower._close_cmp, K_AND: Lower._close_logic,
                 K_OR: Lower._close_logic, K_NOT: Lower._close_logic,
                 K_PACK: Lower._close_pack}


# ---------------------------------------------------------------- occupancy

# Cells per open construct, stage by stage. Spec section 2.5 says 1 for Lower,
# but join homes and pending moves do not fit in 15 bytes, so these are the honest
# constants. They stay constants because a field needs at least one byte (at most
# 16 per cell) and a spilled value never reloads (at most 15 homes move at a join).
_MAX_FIELDS = 16
S_PARSE = 1
S_CHECK = 3 + 1 + 2 * (_MAX_FIELDS + 1)   # frame, let name, binder name and attributes
S_LOWER = ((_HDR.size + max(_SNAPSHOT_BYTES, _MAX_FIELDS * _ENTRY.size) + 14) // 15
           + 2 * (_MAX_FIELDS + 1))       # frame, then field binders and unbuilt fields
E_MAX = 128        # the largest expansion one event may produce
G_CELLS_PER_ENTRY = 2
DEFAULT_G_MAX = 2048


def occupancy_bound(depth: int, g_cells: int, l_max: int = 15, w: int = 1) -> int:
    """Theorem 7.2: W + 4 + (E - 1) + L_max + (per-stage cells) * D + s * G."""
    return w + 4 + (E_MAX - 1) + l_max + (S_PARSE + S_CHECK + S_LOWER) * depth + g_cells


def required_cells(d_max: int, g_max: int = DEFAULT_G_MAX) -> int:
    """C >= B(D_max) + 1: with this many cells, receiving from F cannot fail."""
    return occupancy_bound(d_max, 2 * G_CELLS_PER_ENTRY * g_max) + 1


# ------------------------------------------------------------------ driver


def assemble_and_run(source: str, workdir: str, *, name: str = "prog") -> int:
    """Assemble a NASM file with nasm, link with ld (no libc, no C compiler), run it
    on the silicon, and return the 64-bit result `_start` writes to stdout."""
    import os
    import subprocess

    obj = os.path.join(workdir, f"{name}.o")
    exe = os.path.join(workdir, name)
    # -Werror: NASM only warns when it truncates an immediate, and a truncated
    # immediate is a miscompile that still assembles.
    subprocess.run(["nasm", "-Werror", "-f", "elf64", "-o", obj, source], check=True,
                   capture_output=True, text=True)
    subprocess.run(["ld", "-o", exe, obj], check=True, capture_output=True, text=True)
    done = subprocess.run([exe], check=True, capture_output=True, timeout=10)
    return int.from_bytes(done.stdout, "little")


def build_and_run(asm: str, workdir: str, *, name: str = "prog") -> int:
    import os

    source = os.path.join(workdir, f"{name}.asm")
    with open(source, "w") as handle:
        handle.write(asm)
    return assemble_and_run(source, workdir, name=name)


def compile_source(source, *, cells: Optional[int] = None, d_max: int = 64,
                   target_cells: int = 1024,
                   on_line: Optional[Callable[[str], None]] = None,
                   on_step: Optional[Callable[[Ring], None]] = None,
                   checks: bool = False) -> tuple[Ring, "Lower"]:
    """The whole ring: Ingest, Lex, Parse, Check, Lower, Egress. Lines go to
    `on_line` as Egress drains them, and no stage keeps a copy."""
    from .check import CheckStage
    from .dyck import Lexer, Parser

    lower = Lower()
    lower.home_checks = checks
    if cells is None:
        cells = required_cells(d_max)
    emit = on_line if on_line is not None else lambda line: None

    def sink(tag: int, payload: bytes) -> None:
        emit(render(tag, payload, target_cells=target_cells))

    ring = Ring(cells, iter(source), [Lexer(), Parser(d_max), CheckStage(), lower], sink, on_step)
    ring.run()
    return ring, lower


def compile_text(source: bytes, **options) -> str:
    lines: list[str] = []
    compile_source(source, on_line=lines.append, **options)
    return "\n".join(lines) + "\n"


def file_bytes(handle, chunk: int = 4096):
    """Ingest from an open file: the ring sees one byte at a time, and no more than
    one small chunk of the source is ever in memory."""
    while True:
        block = handle.read(chunk)
        if not block:
            return
        yield from block


def main(argv: list[str]) -> int:
    """lfplc FILE [-r]: print NASM for FILE, or assemble it, run it, print the result."""
    import os
    import sys
    import tempfile

    if not argv:
        print("usage: python3 -m src.lower FILE.lfpl [-r]", file=sys.stderr)
        return 2
    try:
        with open(argv[0], "rb") as handle:
            if "-r" not in argv:
                compile_source(file_bytes(handle), on_line=print)
                return 0
            with tempfile.TemporaryDirectory() as workdir:
                source = os.path.join(workdir, "prog.asm")
                with open(source, "w") as out:
                    compile_source(file_bytes(handle),
                                   on_line=lambda line: out.write(line + "\n"))
                print(assemble_and_run(source, workdir))
    except (Reject, FailFast) as failure:
        print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
