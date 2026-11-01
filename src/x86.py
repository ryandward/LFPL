"""x86-64 from the register-transfer stream: docs/rt-layer.md's "Lowering to x86-64".

This is a forward fold holding one table from RT registers to homes, which are physical
registers or frame slots. All the facts about x86 live here and nowhere else: fifteen registers,
two-operand forms, `cl`, spilling, frames and clobber sets. The stream it reads has already
passed rtcheck, so this code trusts the tokens and only tracks where each value is.

r10 and r11 stay out of allocation. r11 carries a value through one move, for the forms x86 has
no instruction for, and r10 breaks cycles. Everything else can be spilled, so an instruction
pins at most its own operands, and a spill victim always exists. That is LFPL-O's T8 argument.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional, Union

Home = Union[str, int]  # a physical register, or a frame slot

ALLOC = ("rbx", "rbp", "r12", "r13", "r14", "r15", "r9", "r8", "rcx", "rdx", "rsi", "rdi", "rax")
PARAMS = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
RESULTS = ("rax", "rdx", "rcx", "rsi", "rdi", "r8")
SCRATCH, TRANSIT = "r10", "r11"
SCALAR_TYPES = frozenset(("i8", "i16", "i32", "i64", "bool", "lit"))
WIDTH = {1: "byte", 2: "word", 4: "dword", 8: "qword"}
NEGATED = {"b": "ae", "be": "a", "a": "be", "ae": "b", "e": "ne", "ne": "e"}
ROOT = -1  # the entry word of an @k function, which `ret` returns however many iterations ran

_SUB: dict[str, dict[int, str]] = {}
for _full, _d, _w, _b in (("rax", "eax", "ax", "al"), ("rbx", "ebx", "bx", "bl"),
                          ("rcx", "ecx", "cx", "cl"), ("rdx", "edx", "dx", "dl"),
                          ("rsi", "esi", "si", "sil"), ("rdi", "edi", "di", "dil"),
                          ("rbp", "ebp", "bp", "bpl")):
    _SUB[_full] = {8: _full, 4: _d, 2: _w, 1: _b}
for _n in range(8, 16):
    _SUB[f"r{_n}"] = {8: f"r{_n}", 4: f"r{_n}d", 2: f"r{_n}w", 1: f"r{_n}b"}
_FULL = {name: full for full, sizes in _SUB.items() for name in sizes.values()}


class Starved(Exception):
    """A parallel move needed more temporaries than exist. Rule text starts with the declared
    limit it is, so the driver can tell it from a bug."""


@dataclass(frozen=True)
class Reg:
    name: str


@dataclass(frozen=True)
class Slot:
    index: int


@dataclass(frozen=True)
class Imm:
    value: int


@dataclass(frozen=True)
class Mem:
    base: str
    off: int
    size: int


@dataclass(frozen=True)
class Func:
    """op on an old field and an operand: a build source."""
    op: str
    place: Mem
    operand: Union[Reg, Imm]


@dataclass
class Move:
    dest: Union[Reg, Slot, Mem]
    src: Union[Reg, Slot, Imm, Mem, Func]


def _overlap(a: Mem, b: Mem) -> bool:
    return a.base == b.base and a.off < b.off + b.size and b.off < a.off + a.size


def _reads(src: object, dest: object) -> bool:
    """Does reading `src` need the old contents of `dest`?"""
    if isinstance(src, Func):
        return (_reads(src.place, dest) or _reads(src.operand, dest))
    if isinstance(dest, Reg):
        return src == dest or (isinstance(src, Mem) and src.base == dest.name)
    if isinstance(dest, Slot):
        return src == dest
    if isinstance(dest, Mem):
        return isinstance(src, Mem) and _overlap(src, dest)
    return False


def _args(text: str) -> list[str]:
    parts, depth, current = [], 0, ""
    for char in text:
        if char in "([<":
            depth += 1
        elif char in ")]>":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts


_REG = re.compile(r"r(\d+)$")
_LOC = re.compile(r"r(\d+)\[(\d+):(\d+)\]$")
_LIT = re.compile(r"#(\d+)$")
_INNER = re.compile(r"\[(\d+):(\d+)\]$")
_FUNC = re.compile(r"(\w+)\(\s*\[(\d+):(\d+)\]\s*,\s*(.+)\)$")
_HEADER = re.compile(r"(?:fn|handler) (f\d+) (\w+)\((.*)\) : (\S+)(?: @(\d+))?$")
_LABEL = re.compile(r"(L\d+):\s*(.*)$")
_X86_OP = {"add": "add", "sub": "sub", "mul": "imul", "and": "and", "or": "or", "xor": "xor"}


@dataclass
class Function:
    name: str
    ret: str
    at: Optional[int]
    clobbers: frozenset = frozenset()
    depth: int = 8
    instructions: int = 0


class Backend:
    """Feed it the stream, one line at a time. Lines of NASM go to `emit`."""

    def __init__(self, emit: Callable[[str], None]):
        self.out = emit
        self.ctors: dict[str, tuple[str, list]] = {}
        self.tagged: dict[str, bool] = {}
        self.tuples: dict[str, list[str]] = {}
        self.fins: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.functions: dict[str, Function] = {}
        self.fn: Optional[Function] = None
        self.ordinal = ""
        self.next_label = 0
        self._reset_function()

    def _reset_function(self) -> None:
        self.home: dict[int, Home] = {}
        self.owner: dict[str, int] = {}
        self.slots: set[int] = set()
        self.peak = 0
        self.linear: set[int] = set()
        self.edges: dict[str, dict[int, Home]] = {}
        self.hold: Optional[tuple] = None
        self.live = False
        self.clobbered: set[str] = set()
        self.entry: dict[int, Home] = {}
        self.entry_label = ""
        self.calls_depth = 0
        self.tail_depth = 0
        self.count = 0

    # --- the output

    def ins(self, op: str, dest: Optional[str] = None, src: Optional[str] = None) -> None:
        if dest in _FULL and op not in ("cmp", "test"):
            self.clobbered.add(_FULL[dest])
        operands = ", ".join(o for o in (dest, src) if o is not None)
        self.out(f"    {op} {operands}".rstrip())
        self.count += 1

    def place(self, label: str) -> None:
        self.out(f"{label}:")

    def fresh_label(self) -> str:
        self.next_label += 1
        return f"X{self.next_label}"

    @staticmethod
    def sub(register: str, size: int) -> str:
        return _SUB[register][size]

    @staticmethod
    def mem(place: Mem) -> str:
        disp = f"+{place.off}" if place.off else ""
        return f"{WIDTH[place.size]} [{place.base}{disp}]"

    @staticmethod
    def slot(index: int) -> str:
        disp = f"+{8 * index}" if index else ""
        return f"qword [rsp{disp}]"

    # --- homes

    def _is_linear(self, ty: str) -> bool:
        return ty not in SCALAR_TYPES and self._resolve(ty) not in self.fins

    def _resolve(self, name: str) -> str:
        while name in self.aliases:
            name = self.aliases[name]
        return name

    def _new_slot(self) -> int:
        index = 0
        while index in self.slots:
            index += 1
        self.slots.add(index)
        self.peak = max(self.peak, index + 1)
        return index

    def _bind(self, r: int, home: Home) -> None:
        self.home[r] = home
        if isinstance(home, str):
            self.owner[home] = r
        else:
            self.slots.add(home)

    def _free(self, r: int) -> None:
        home = self.home.pop(r)
        if isinstance(home, str):
            del self.owner[home]
        else:
            self.slots.discard(home)

    def _table(self, table: dict[int, Home]) -> None:
        self.home = {}
        self.owner = {}
        self.slots = set()
        for r, home in table.items():
            self._bind(r, home)

    def _take(self, pins: frozenset, *, avoid: frozenset = frozenset()) -> str:
        """A register with nothing in it, spilling a value that is not pinned if none is."""
        banned = pins | avoid
        for register in ALLOC:
            if register not in self.owner and register not in banned:
                return register
        for register in ALLOC:
            if register not in banned:
                victim = self.owner[register]
                index = self._new_slot()
                self.ins("mov", self.slot(index), register)
                self._free(victim)
                self._bind(victim, index)
                return register
        raise Starved("register starvation: every register is pinned by one instruction")

    def _phys(self, r: int, pins: frozenset = frozenset()) -> str:
        """Bring a value into a register, and leave it there."""
        home = self.home[r]
        if isinstance(home, str):
            return home
        register = self._take(pins)
        self.ins("mov", register, self.slot(home))
        self._free(r)
        self._bind(r, register)
        return register

    @staticmethod
    def reg(text: str) -> int:
        match = _REG.match(text)
        assert match is not None, text
        return int(match.group(1))

    def _define(self, text: str, register: str, linear: bool = False) -> None:
        r = self.reg(text)
        self._bind(r, register)
        if linear:
            self.linear.add(r)
        else:
            self.linear.discard(r)

    # --- operands

    def _operand(self, text: str, pins: frozenset) -> tuple[Union[Reg, Imm, Mem], frozenset]:
        """A readable place for an operand, and the pins widened to keep it readable."""
        literal = _LIT.match(text)
        if literal:
            return Imm(int(literal.group(1))), pins
        located = _LOC.match(text)
        if located:
            base = self._phys(int(located.group(1)), pins)
            return Mem(base, int(located.group(2)), int(located.group(3))), pins | {base}
        register = self._phys(self.reg(text), pins)
        return Reg(register), pins | {register}

    def _settle(self, texts: list[str], pins: frozenset) -> frozenset:
        """Bring every register these sources must read through into a register, pinned. This
        comes before any move is planned, because bringing one in can spill another, and a
        planned move names homes. After this, planning a move allocates nothing."""
        for text in texts:
            computed = _FUNC.match(text)
            if computed:
                _place, pins = self._operand(computed.group(4), pins)
            elif _LOC.match(text):
                _place, pins = self._operand(text, pins)
        return pins

    def _load(self, register: str, place: Union[Reg, Imm, Mem, Slot]) -> None:
        """Into a register, zero extended. A value in a register is always zero extended."""
        if isinstance(place, Imm):
            if place.value == 0:
                self.ins("xor", self.sub(register, 4), self.sub(register, 4))
            elif place.value < 1 << 32:
                self.ins("mov", self.sub(register, 4), str(place.value))
            else:
                self.ins("mov", register, str(_signed(place.value)))
        elif isinstance(place, Reg):
            if place.name != register:
                self.ins("mov", register, place.name)
        elif isinstance(place, Slot):
            self.ins("mov", register, self.slot(place.index))
        elif place.size < 4:
            self.ins("movzx", self.sub(register, 4), self.mem(place))
        else:
            self.ins("mov", self.sub(register, place.size), self.mem(place))

    def _store(self, place: Mem, value: Union[Reg, Imm]) -> None:
        if isinstance(value, Reg):
            self.ins("mov", self.mem(place), self.sub(value.name, place.size))
        else:
            self.ins("mov", self.mem(place), str(_masked(value.value, place.size)))

    # --- parallel moves

    def _pmove(self, moves: list[Move]) -> None:
        pending = [m for m in moves if m.dest != m.src]
        while pending:
            ready = next((m for m in pending if not self._blocked(m, pending)), None)
            if ready is not None:
                pending.remove(ready)
                self._emit_move(ready)
                continue
            self._break_cycle(pending)

    def _blocked(self, move: Move, pending: list[Move]) -> bool:
        for other in pending:
            if other is move:
                continue
            if _reads(other.src, move.dest):
                return True
            if isinstance(move.dest, Reg) and isinstance(other.dest, Mem) and (
                    other.dest.base == move.dest.name):
                return True
        return False

    def _temporary(self, pending: list[Move]) -> str:
        def used(register: str) -> bool:
            for move in pending:
                for place in (move.src, move.dest):
                    if place == Reg(register) or (isinstance(place, Mem) and place.base == register):
                        return True
                    if isinstance(place, Func) and (place.place.base == register
                                                    or place.operand == Reg(register)):
                        return True
            return False

        if not used(SCRATCH):
            return SCRATCH
        for register in ALLOC:
            if register not in self.owner and not used(register):
                return register
        raise Starved("register starvation: a parallel move needs more than the two scratch "
                      "registers and every other register holds a live value")

    def _break_cycle(self, pending: list[Move]) -> None:
        move = pending[0]
        temp = self._temporary(pending)
        if isinstance(move.dest, Mem):
            for other in pending:
                if other is not move and _reads(other.src, move.dest):
                    self._buffer(other, temp)
                    return
            raise AssertionError("a blocked store with no reader")
        self._load(temp, move.dest)
        self.clobbered.add(temp)
        old = move.dest
        # Every reader moves to the temporary, this move's own source included. A store and a
        # read of one location must keep naming the same base, or they stop looking like they
        # overlap and the store could land first.
        for other in pending:
            other.src = _rewrite(other.src, old, temp)
            if isinstance(other.dest, Mem) and isinstance(old, Reg) and other.dest.base == old.name:
                other.dest = Mem(temp, other.dest.off, other.dest.size)

    def _buffer(self, move: Move, temp: str) -> None:
        """Read this move's source now, while the location still holds the old value."""
        if isinstance(move.src, Func):
            self._load(temp, move.src.place)
            self._apply(move.src.op, temp, move.src.place.size, move.src.operand)
        else:
            assert isinstance(move.src, Mem)
            self._load(temp, move.src)
        move.src = Reg(temp)

    def _apply(self, op: str, register: str, size: int, operand: Union[Reg, Imm]) -> None:
        wide = size == 8
        target = register if wide else self.sub(register, 4)
        if isinstance(operand, Imm):
            value = _masked(operand.value, size)
            if wide and value >= 1 << 31:
                self.ins("mov", TRANSIT, str(_signed(value)))
                self.ins(_X86_OP[op], target, TRANSIT)
            else:
                self.ins(_X86_OP[op], target, str(value))
        else:
            self.ins(_X86_OP[op], target, self.sub(operand.name, 8 if wide else 4))
        if size < 4 and op in ("add", "sub", "mul"):
            self.ins("movzx", self.sub(register, 4), self.sub(register, size))

    def _emit_move(self, move: Move) -> None:
        dest, src = move.dest, move.src
        if isinstance(src, Func):
            self._emit_computed(dest, src)
        elif isinstance(dest, Reg):
            self._load(dest.name, src)
        elif isinstance(dest, Slot):
            if isinstance(src, Reg):
                self.ins("mov", self.slot(dest.index), src.name)
            elif isinstance(src, Imm) and src.value < 1 << 31:
                self.ins("mov", self.slot(dest.index), str(src.value))
            else:
                self._load(TRANSIT, src)
                self.ins("mov", self.slot(dest.index), TRANSIT)
        elif isinstance(src, Reg):
            self._store(dest, src)
        elif isinstance(src, Imm) and (dest.size < 8 or src.value < 1 << 31):
            self._store(dest, src)
        else:
            self._load(TRANSIT, src)
            self._store(dest, Reg(TRANSIT))

    def _emit_computed(self, dest: object, src: Func) -> None:
        assert isinstance(dest, Mem)
        size = src.place.size
        if src.place == dest and src.op in ("add", "sub", "and", "or", "xor"):
            if isinstance(src.operand, Imm):
                value = _masked(src.operand.value, size)
                if size == 8 and value >= 1 << 31:
                    self._load(TRANSIT, src.operand)
                    self.ins(src.op, self.mem(dest), TRANSIT)
                else:
                    self.ins(src.op, self.mem(dest), str(value))
            else:
                self.ins(src.op, self.mem(dest), self.sub(src.operand.name, size))
            return
        self._load(TRANSIT, src.place)
        self._apply(src.op, TRANSIT, size, src.operand)
        self._store(dest, Reg(TRANSIT))

    # --- the fold

    def feed(self, line: str) -> None:
        text = line.split(";")[0].strip()
        if not text:
            return
        label = _LABEL.match(text)
        if label:
            self._flush(label.group(1))
            self._label(label.group(1))
            text = label.group(2)
            if not text:
                return
        else:
            self._flush(None)
        head, _, rest = text.partition(" ")
        if head in ("fn", "handler"):
            self._header(text)
        elif head == "end":
            self._end()
        elif head == "ctor":
            parts = rest.split()
            self.ctors[parts[0]] = (parts[2], [])
        elif head == "field":
            parts = rest.split()
            self.ctors[parts[0]][1].append((int(parts[1]), int(parts[2])))
        elif head == "type":
            parts = rest.split()
            self.tagged[parts[0]] = parts[1] == "tagged"
        elif head == "alias":
            parts = rest.split()
            self.aliases[parts[0]] = parts[1]
        elif head == "tuple":
            parts = rest.split()
            self.tuples[parts[0]] = parts[1:]
        elif head == "fin":
            self.fins.add(rest.split()[0])
        elif head in ("array", "state"):
            pass
        else:
            getattr(self, "i_" + head.replace(".", "_"))(_args(rest))

    def _header(self, text: str) -> None:
        match = _HEADER.match(text)
        assert match is not None, text
        ordinal, name, params, ret, at = match.groups()
        self._reset_function()
        self.fn = Function(name, ret, int(at) if at else None)
        self.ordinal = ordinal
        self.out(f"fn_{name} equ fn_{name}_body")
        self.place(f"fn_{name}_body")
        self.ins("sub", "rsp", f"F_{name}")
        for position, part in enumerate(_args(params)):
            _register, _, ty = part.partition(" : ")
            self._bind(position, PARAMS[position])
            if self._is_linear(ty.strip()):
                self.linear.add(position)
        if self.fn.at is not None:
            self.ins("mov", "rax", PARAMS[self.fn.at])
            self._bind(ROOT, "rax")
        self.entry = dict(self.home)
        self.entry_label = self.fresh_label()
        self.place(self.entry_label)
        self.live = True

    def _end(self) -> None:
        fn = self.fn
        assert fn is not None
        self.out(f"F_{fn.name} equ {8 * self.peak}")
        fn.clobbers = frozenset(self.clobbered)
        fn.depth = max(8 + 8 * self.peak + self.calls_depth, self.tail_depth)
        fn.instructions = self.count
        self.functions[self.ordinal] = fn
        self.fn = None

    # --- control

    def _flush(self, next_label: Optional[str]) -> None:
        hold, self.hold = self.hold, None
        if hold is None:
            return
        if hold[0] == "jump":
            if hold[1] != next_label:
                self.ins("jmp", hold[1])
            return
        _kind, cc, then, other = hold
        if other == next_label:
            self.ins(f"j{cc}", then)
        elif then == next_label:
            self.ins(f"j{NEGATED[cc]}", other)
        else:
            self.ins(f"j{cc}", then)
            self.ins("jmp", other)

    def _label(self, label: str) -> None:
        if self.live:
            self._arrive(label, dict(self.home))
        target = self.edges.pop(label, None)
        assert target is not None, f"{label} has no incoming edge"
        self._table(target)
        self.place(label)
        self.live = True

    def _needs_moves(self, label: str, table: dict[int, Home]) -> bool:
        target = self.edges.get(label)
        if target is None:
            return False
        return any(r in target and target[r] != home for r, home in table.items())

    def _arrive(self, label: str, table: dict[int, Home]) -> None:
        """An edge reaches a label. The first sets its homes, and later ones move to them.
        A value not live on every edge is not live after the label, by the merge rule."""
        self._table(table)
        target = self.edges.get(label)
        if target is None:
            self.edges[label] = dict(self.home)
            return
        for r in [r for r in target if r not in self.home]:
            del target[r]
        for r in [r for r in self.home if r not in target]:
            self._free(r)
        moves = [Move(_as_place(target[r]), _as_place(home))
                 for r, home in self.home.items() if target[r] != home]
        self._pmove(moves)
        self._table(target)

    def i_jump(self, a: list[str]) -> None:
        self._arrive(a[0], dict(self.home))
        self.hold = ("jump", a[0])
        self.live = False

    def _branch(self, cc: str, then: str, other: str, then_table: dict[int, Home],
                other_table: dict[int, Home]) -> None:
        if self._needs_moves(then, then_table) or self._needs_moves(other, other_table):
            bridge = self.fresh_label()
            self.ins(f"j{cc}", bridge)
            self._arrive(other, other_table)
            self.ins("jmp", other)
            self.place(bridge)
            self._arrive(then, then_table)
            self.ins("jmp", then)
        else:
            self._arrive(then, then_table)
            self._arrive(other, other_table)
            self.hold = ("br", cc, then, other)
        self.live = False

    def _compare(self, left: str, right: str) -> None:
        x, pins = self._operand(left, frozenset())
        y, pins = self._operand(right, pins)
        if isinstance(x, Mem) and isinstance(y, Imm) and y.value < _limit(x.size):
            self.ins("cmp", self.mem(x), str(y.value))
            return
        if isinstance(x, Mem) and isinstance(y, Reg) and x.size == 8:
            self.ins("cmp", self.mem(x), y.name)
            return
        if isinstance(x, Reg):
            first = x.name
        else:
            self._load(SCRATCH, x)
            self.clobbered.add(SCRATCH)
            first = SCRATCH
        if isinstance(y, Imm) and y.value < 1 << 31:
            self.ins("cmp", first, str(y.value))
        elif isinstance(y, Reg):
            self.ins("cmp", first, y.name)
        else:
            self._load(TRANSIT, y)
            self.clobbered.add(TRANSIT)
            self.ins("cmp", first, TRANSIT)

    def i_br(self, a: list[str]) -> None:
        self._compare(a[1], a[2])
        self._branch(a[0], a[3], a[4], dict(self.home), dict(self.home))

    def i_br_null(self, a: list[str]) -> None:
        r = self.reg(a[0])
        register = self._phys(r)
        self.ins("test", register, register)
        null_table = {k: v for k, v in self.home.items() if k != r}
        self._branch("e", a[1], a[2], null_table, dict(self.home))

    def i_br_tag(self, a: list[str]) -> None:
        register = self._phys(self.reg(a[0]))
        self.ins("cmp", f"byte [{register}]", a[1])
        self._branch("e", a[2], a[3], dict(self.home), dict(self.home))

    def i_open(self, a: list[str]) -> None:
        """Nothing: the cell's fields are named by offset, and no test is needed."""

    # --- values

    def i_const(self, a: list[str]) -> None:
        literal, _, ty = a[1].partition(" : ")
        match = _LIT.match(literal.strip())
        assert match is not None
        register = self._take(frozenset())
        self._load(register, Imm(int(match.group(1))))
        self._define(a[0], register, self._is_linear(ty.strip()))

    def i_move(self, a: list[str]) -> None:
        """A rename: the value stays where it is."""
        source = self.reg(a[1])
        home = self.home[source]
        linear = source in self.linear
        self._free(source)
        r = self.reg(a[0])
        self._bind(r, home)
        if linear:
            self.linear.add(r)
        else:
            self.linear.discard(r)

    def i_copy(self, a: list[str]) -> None:
        source = self._phys(self.reg(a[1]))
        register = self._take(frozenset((source,)))
        self.ins("mov", register, source)
        self._define(a[0], register)

    def i_drop(self, a: list[str]) -> None:
        self._free(self.reg(a[0]))

    def i_op(self, a: list[str]) -> None:
        name, _, bits_text = a[1].partition("/")
        bits = int(bits_text)
        size = bits // 8
        x, pins = self._operand(a[2], frozenset())
        y, pins = self._operand(a[3], pins)
        by_register = name in ("shl", "shr") and not isinstance(y, Imm)
        register = self._take(pins, avoid=frozenset(("rcx",)) if by_register else frozenset())
        self._load(register, x)
        target = register if size == 8 else self.sub(register, 4)
        if name in ("shl", "shr"):
            self._shift(name, register, target, bits, y)
        else:
            self._arithmetic(name, register, target, size, y)
        if size < 4 and name in ("add", "sub", "mul", "shl"):
            self.ins("movzx", self.sub(register, 4), self.sub(register, size))
        self._define(a[0], register)

    def _arithmetic(self, name: str, register: str, target: str, size: int,
                    y: Union[Reg, Imm, Mem]) -> None:
        wide = size == 8
        if isinstance(y, Imm):
            value = _masked(y.value, size)
            if wide and value >= 1 << 31:
                self._load(TRANSIT, y)
                self.ins(_X86_OP[name], target, TRANSIT)
            else:
                self.ins(_X86_OP[name], target, str(value))
        elif isinstance(y, Reg):
            self.ins(_X86_OP[name], target, y.name if wide else self.sub(y.name, 4))
        elif (wide and y.size == 8) or (size == 4 and y.size == 4):
            self.ins(_X86_OP[name], target, self.mem(y))
        else:
            self._load(TRANSIT, y)
            self.ins(_X86_OP[name], target, TRANSIT if wide else self.sub(TRANSIT, 4))

    def _shift(self, name: str, register: str, target: str, bits: int,
               count: Union[Reg, Imm, Mem]) -> None:
        """LFPL-O section 4.4: the result is 0 once the count reaches the width, and the
        hardware masks the count, so the comparison has to come first."""
        if isinstance(count, Imm):
            if count.value >= bits:
                self.ins("xor", self.sub(register, 4), self.sub(register, 4))
            else:
                self.ins(name, target, str(count.value))
            return
        saved = "rcx" in self.owner and count != Reg("rcx")
        if saved:
            self.ins("mov", TRANSIT, "rcx")
        self._load("rcx", count)
        zero, done = self.fresh_label(), self.fresh_label()
        self.ins("cmp", "rcx", str(bits))
        self.ins("jae", zero)
        self.ins(name, target, "cl")
        self.ins("jmp", done)
        self.place(zero)
        self.ins("xor", self.sub(register, 4), self.sub(register, 4))
        self.place(done)
        if saved:
            self.ins("mov", "rcx", TRANSIT)

    def i_cmp(self, a: list[str]) -> None:
        self._compare(a[2], a[3])
        register = self._take(frozenset())
        self.ins(f"set{a[1]}", self.sub(register, 1))
        self.ins("movzx", self.sub(register, 4), self.sub(register, 1))
        self._define(a[0], register)

    def i_trunc(self, a: list[str]) -> None:
        size = int(a[1]) // 8
        x, pins = self._operand(a[2], frozenset())
        register = self._take(pins)
        self._load(register, x)
        self._narrow(register, size)
        self._define(a[0], register)

    def _narrow(self, register: str, size: int) -> None:
        if size < 4:
            self.ins("movzx", self.sub(register, 4), self.sub(register, size))
        elif size == 4:
            self.ins("mov", self.sub(register, 4), self.sub(register, 4))

    def i_pack(self, a: list[str]) -> None:
        widths = [int(w) for w in a[1].strip("<>").split()]
        pins: frozenset = frozenset()
        places = []
        for operand in a[2:]:
            place, pins = self._operand(operand, pins)
            places.append(place)
        register = self._take(pins)
        for position, (bits, place) in enumerate(zip(widths, places)):
            if position == 0:
                self._load(register, place)
                self._narrow(register, bits // 8)
            else:
                self.ins("shl", register, str(bits))
                self._load(TRANSIT, place)
                self._narrow(TRANSIT, bits // 8)
                self.ins("or", register, TRANSIT)
        self._define(a[0], register)

    def i_idx(self, a: list[str]) -> None:
        mask = int(a[1]) - 1
        x, pins = self._operand(a[2], frozenset())
        register = self._take(pins)
        self._load(register, x)
        if mask >= 1 << 31:
            self._load(TRANSIT, Imm(mask))
            self.ins("and", register, TRANSIT)
        else:
            self.ins("and", register, str(mask))
        self._define(a[0], register)

    def i_aget(self, a: list[str]) -> None:
        size = int(a[3])
        base = self._phys(self.reg(a[1]))
        index = self._phys(self.reg(a[2]), frozenset((base,)))
        register = self._take(frozenset((base, index)))
        address = f"{WIDTH[size]} [{base}+{index}*{size}]"
        if size < 4:
            self.ins("movzx", self.sub(register, 4), address)
        else:
            self.ins("mov", self.sub(register, size), address)
        self._define(a[0], register)

    # --- cells

    def _read(self, a: list[str], linear: bool) -> None:
        located = _LOC.match(a[1])
        assert located is not None
        base = self._phys(int(located.group(1)))
        register = self._take(frozenset((base,)))
        self._load(register, Mem(base, int(located.group(2)), int(located.group(3))))
        self._define(a[0], register, linear)

    def i_read(self, a: list[str]) -> None:
        self._read(a, False)

    def i_take(self, a: list[str]) -> None:
        self._read(a, True)

    def _field_moves(self, base: str, ctor_id: str, tag: str, sources: list[str],
                     pins: frozenset) -> tuple[list[Move], list[int], frozenset]:
        """The stores of one constructor, and the registers its sources consume."""
        tag_text, fields = self.ctors[ctor_id]
        moves: list[Move] = []
        consumed: list[int] = []
        for source, (off, size) in zip(sources, fields):
            where = Mem(base, off, size)
            if source in ("=", "*"):
                continue
            if source.endswith("!"):
                consumed.append(self.reg(source[:-1]))
            elif _LIT.match(source):
                match = _LIT.match(source)
                assert match is not None
                moves.append(Move(where, Imm(int(match.group(1)))))
            elif _INNER.match(source):
                match = _INNER.match(source)
                assert match is not None
                moves.append(Move(where, Mem(base, int(match.group(1)), int(match.group(2)))))
            elif _FUNC.match(source):
                match = _FUNC.match(source)
                assert match is not None
                name, old_off, old_size, operand = match.groups()
                place, pins = self._operand(operand, pins)
                assert not isinstance(place, Mem), "a computed source reads a register or literal"
                moves.append(Move(where, Func(name, Mem(base, int(old_off), int(old_size)), place)))
            elif _LOC.match(source):
                place, pins = self._operand(source, pins)
                moves.append(Move(where, place))
            else:
                r = self.reg(source)
                moves.append(Move(where, _as_place(self.home[r])))
                if r in self.linear:
                    consumed.append(r)
        if tag == "write":
            moves.append(Move(Mem(base, 0, 1), Imm(int(tag_text))))
        return moves, consumed, pins

    def i_build(self, a: list[str]) -> None:
        base = self._phys(self.reg(a[0]))
        pins = self._settle(a[3:], frozenset((base,)))
        moves, consumed, _pins = self._field_moves(base, a[1], a[2], a[3:], pins)
        self._pmove(moves)
        for r in consumed:
            self._free(r)
        self.linear.add(self.reg(a[0]))

    # --- calls and exits

    def _argument_moves(self, operands: list[str], pins: frozenset
                        ) -> tuple[list[Move], list[int], frozenset]:
        moves: list[Move] = []
        consumed: list[int] = []
        for position, operand in enumerate(operands):
            target = Reg(PARAMS[position])
            if _LIT.match(operand):
                match = _LIT.match(operand)
                assert match is not None
                moves.append(Move(target, Imm(int(match.group(1)))))
            elif _LOC.match(operand):
                place, pins = self._operand(operand, pins)
                moves.append(Move(target, place))
            else:
                r = self.reg(operand)
                moves.append(Move(target, _as_place(self.home[r])))
                if r in self.linear:
                    consumed.append(r)
        return moves, consumed, pins

    def i_call(self, a: list[str]) -> None:
        callee = self.functions[a[1]]
        pins = self._settle(a[2:], frozenset())
        moves, consumed, pins = self._argument_moves(a[2:], pins)
        results = self.tuples.get(self._resolve(callee.ret))
        count = 1 if results is None else len(results)
        clobbered = set(callee.clobbers) | set(RESULTS[:count])
        wanted = {m.dest.name for m in moves if isinstance(m.dest, Reg)}
        survivors = [r for r in self.home if r not in consumed]
        taken = wanted | clobbered | pins
        for r in survivors:
            home = self.home[r]
            if isinstance(home, str) and (home in clobbered or home in wanted):
                refuge = next((g for g in ALLOC if g not in taken and g not in self.owner), None)
                if refuge is not None:
                    taken.add(refuge)
                    target: Home = refuge
                else:
                    target = self._new_slot()
                moves.append(Move(_as_place(target), Reg(home)))
                self._free(r)
                self._bind(r, target)
        self._pmove(moves)
        for r in consumed:
            self._free(r)
        self.clobbered |= clobbered
        self.ins("call", f"fn_{callee.name}")
        self.calls_depth = max(self.calls_depth, callee.depth)
        if results is None:
            self._define(a[0], RESULTS[0], self._is_linear(callee.ret))
        else:
            names = _args(a[0].strip("()"))
            for position, (name, ty) in enumerate(zip(names, results)):
                self._define(name, RESULTS[position], self._is_linear(ty))

    def _leave_to(self, ordinal: str, moves: list[Move]) -> None:
        """Control goes to a function: this one's own back-edge, or another's entry."""
        fn = self.fn
        assert fn is not None
        if ordinal == self.ordinal:
            if ROOT in self.home and self.home[ROOT] != "rax":
                moves.append(Move(Reg("rax"), _as_place(self.home[ROOT])))
            self._pmove(moves)
            self.ins("jmp", self.entry_label)
        else:
            callee = self.functions[ordinal]
            self._pmove(moves)
            self.clobbered |= callee.clobbers
            self.ins("add", "rsp", f"F_{fn.name}")
            self.ins("jmp", f"fn_{callee.name}")
            self.tail_depth = max(self.tail_depth, callee.depth)
        self.live = False

    def i_tail(self, a: list[str]) -> None:
        pins = self._settle(a[1:], frozenset())
        moves, _consumed, _pins = self._argument_moves(a[1:], pins)
        self._leave_to(a[0], moves)

    def i_tailmod(self, a: list[str]) -> None:
        _tag, fields = self.ctors[a[1]]
        sources = a[3:3 + len(fields)]
        ordinal = a[3 + len(fields)]
        operands = a[4 + len(fields):]
        base = self._phys(self.reg(a[0]))
        pins = self._settle(sources + operands, frozenset((base,)))
        field_moves, _consumed, pins = self._field_moves(base, a[1], a[2], sources, pins)
        param_moves, _consumed, _pins = self._argument_moves(operands, pins)
        self._leave_to(ordinal, field_moves + param_moves)

    def i_ret(self, a: list[str]) -> None:
        fn = self.fn
        assert fn is not None
        if fn.at is not None:
            moves = [Move(Reg("rax"), _as_place(self.home[ROOT]))]
        else:
            moves = [Move(Reg(RESULTS[position]), _as_place(self.home[self.reg(operand)]))
                     for position, operand in enumerate(a)]
        self._pmove(moves)
        self.clobbered |= {m.dest.name for m in moves if isinstance(m.dest, Reg)}
        self.ins("add", "rsp", f"F_{fn.name}")
        self.ins("ret")
        self.live = False


def _as_place(home: Home) -> Union[Reg, Slot]:
    return Reg(home) if isinstance(home, str) else Slot(home)


def _rewrite(src: object, old: object, temp: str) -> object:
    if isinstance(old, Reg):
        if src == old:
            return Reg(temp)
        if isinstance(src, Mem) and src.base == old.name:
            return Mem(temp, src.off, src.size)
        if isinstance(src, Func):
            place = _rewrite(src.place, old, temp)
            operand = _rewrite(src.operand, old, temp)
            assert isinstance(place, Mem) and isinstance(operand, (Reg, Imm))
            return Func(src.op, place, operand)
    elif src == old:
        return Reg(temp)
    return src


def _signed(value: int) -> int:
    return value - (1 << 64) if value >= 1 << 63 else value


def _masked(value: int, size: int) -> int:
    return value & ((1 << 8 * size) - 1)


def _limit(size: int) -> int:
    """Literals below this fit a comparison at that size, read as unsigned by the flags used."""
    return 1 << 31 if size == 8 else 1 << 8 * size
