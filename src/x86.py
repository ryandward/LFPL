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
SCALAR_TYPES = frozenset(("i8", "i16", "i32", "i64", "bool", "lit", "be16", "be32", "be64"))
NO_WORK = frozenset(("drop", "open", "move"))  # they write no instruction, and are not a step
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


class NotLowered(Exception):
    """Something RT has that this backend does not lower yet: a declared limit of section 12, not a bug."""


# A slot and a countdown are the Verilog backend's for now: src/runtime.py places no slot, and src/symcheck.py, which
# every x86 build has to pass, does not read either yet.
_NOT_LOWERED = {"sget": "a byte of a slot", "slen": "a slot's length", "br.down": "a match that counts a Fin down"}


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
_HEADER = re.compile(r"(?:fn|handler|turn) (f\d+) (\w+)\((.*)\) : (\S+)(?: @(\d+))?$")
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

    def __init__(self, emit: Callable[[str], None], *, count_steps: bool = False):
        self.out = emit
        self.count_steps = count_steps  # to measure W(n): every step adds one to a counter
        self.waiting: Optional[str] = None  # a label not placed yet: if it is only a jump, it is not a place
        self.aliases: dict[str, list[str]] = {}  # the labels that turned out to be each label still to come
        self.ctors: dict[str, tuple[str, list]] = {}
        self.tagged: dict[str, bool] = {}
        self.tuples: dict[str, list[str]] = {}
        self.fins: set[str] = set()
        self.enums: set[str] = set()  # types whose values are one register: scalars, whatever their name
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
        self.clean: dict[int, int] = {}  # a value in a register that still has a copy in this slot
        self.peak = 0
        self.linear: set[int] = set()
        self.edges: dict[str, tuple[dict[int, Home], set[int]]] = {}
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
        self._place_waiting()
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

    def asm_label(self, label: str) -> str:
        """RT labels are numbered per function, and the assembler has one namespace."""
        return f"{label}_{self.ordinal}"

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
        return ty not in SCALAR_TYPES and self._resolve(ty) not in self.fins and self._resolve(ty) not in self.enums

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
        self.linear.discard(r)
        copy = self.clean.pop(r, None)
        if copy is not None:
            self.slots.discard(copy)

    def _table(self, table: dict[int, Home]) -> None:
        self.home = {}
        self.owner = {}
        self.slots = set()
        self.clean = {}
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
                index = self.clean.pop(victim, None)
                if index is None:
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
        del self.home[r]
        self._bind(r, register)
        self.clean[r] = home  # values do not change, so the slot stays good until this one goes
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
                if isinstance(ready.dest, Reg) and isinstance(ready.src, (Slot, Mem)):
                    # The same value is wanted again, and it is now in a register that nothing
                    # in this move will change: h(x, x) reads x's slot once.
                    for other in pending:
                        if other.src == ready.src:
                            other.src = ready.dest
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
        """rtcheck only lets a computed source work in place, so this is one instruction."""
        assert isinstance(dest, Mem) and src.place == dest, (dest, src)
        size = src.place.size
        if isinstance(src.operand, Imm):
            value = _masked(src.operand.value, size)
            if size == 8 and value >= 1 << 31:
                self._load(TRANSIT, src.operand)
                self.ins(src.op, self.mem(dest), TRANSIT)
            else:
                self.ins(src.op, self.mem(dest), str(value))
        else:
            self.ins(src.op, self.mem(dest), self.sub(src.operand.name, size))

    # --- the fold

    def feed(self, line: str) -> None:
        text = line.split(";")[0].strip()
        if not text:
            return
        label = _LABEL.match(text)
        if label:
            self._place_waiting()
            self._label(label.group(1))
            text = label.group(2)
            if not text:
                return
        elif self.waiting is None:
            self._flush(None)
        head, _, rest = text.partition(" ")
        if self.waiting is not None:
            if self._only_a_jump(head, rest):
                return
            self._place_waiting()
        if head in ("fn", "handler", "turn"):
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
            if parts[1] == "slot":
                raise NotLowered(f"section 12: slot {parts[2]} is not lowered to x86 yet, which has no slots in its "
                                 f"image: src/rtverilog.py lowers it")
            self.tagged[parts[0]] = parts[1] == "tagged"
            if parts[1] == "enum":
                self.enums.add(parts[0])
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
            if head in _NOT_LOWERED:
                raise NotLowered(f"section 12: {_NOT_LOWERED[head]} is not lowered to x86 yet: src/rtverilog.py lowers it")
            if self.count_steps and head not in NO_WORK:
                self.ins("inc", "qword [cnt_steps]")
            getattr(self, "i_" + head.replace(".", "_"))(_args(rest))

    def _header(self, text: str) -> None:
        match = _HEADER.match(text)
        assert match is not None, text
        ordinal, name, params, ret, at = match.groups()
        self._reset_function()
        self.fn = Function(name, ret, int(at) if at else None)
        self.ordinal = ordinal
        # `$` cannot be in a name, so one function's body label cannot be another's entry.
        self.out(f"fn_{name} equ fn_{name}$body")
        self.place(f"fn_{name}$body")
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
                self.ins("jmp", self.asm_label(hold[1]))
            return
        _kind, cc, then, other = hold
        if then == other:  # both ways turned out to be only a jump to one place
            if then != next_label:
                self.ins("jmp", self.asm_label(then))
        elif other == next_label:
            self.ins(f"j{cc}", self.asm_label(then))
        elif then == next_label:
            self.ins(f"j{NEGATED[cc]}", self.asm_label(other))
        else:
            self.ins(f"j{cc}", self.asm_label(then))
            self.ins("jmp", self.asm_label(other))

    def _label(self, label: str) -> None:
        """The tables are the label's from here, and where it is in the code waits for the next line. What
        was held waits too, because a jump to a label that turns out to be only a jump goes further."""
        falls_into_it = self.live
        if falls_into_it:
            self._arrive(label, dict(self.home), set(self.linear))
        edge = self.edges.pop(label, None)
        assert edge is not None, f"{label} has no incoming edge"
        self._table(edge[0])
        self.linear = set(edge[1])
        self.live = True
        self.waiting = label
        if falls_into_it:
            self._place_waiting()

    def _place_waiting(self) -> None:
        """What was found to be another name for this label is said after it: NASM cannot settle the size of a
        jump to a name defined by `equ` from a label further on, and gives up after a thousand passes."""
        label, self.waiting = self.waiting, None
        if label is not None:
            self._flush(label)
            self.place(self.asm_label(label))
            for name in self.aliases.pop(label, ()):
                self.out(f"{self.asm_label(name)} equ {self.asm_label(label)}")

    def _only_a_jump(self, head: str, rest: str) -> bool:
        """Is the waiting label followed by a jump, or a comparison of literals, which is one, with no
        move on the way? Then it is another name for where that goes: section 5.3's zero trampolines. An
        `if` on `if c then true else false` branches on c, straight to the outer labels."""
        args = _args(rest)
        target = args[0] if head == "jump" else _literal_branch(args) if head == "br" else None
        if target is None or self.count_steps or self._needs_moves(target, self.home):
            return False
        waiting, self.waiting = self.waiting, None
        assert waiting is not None
        self.aliases.setdefault(target, []).extend((*self.aliases.pop(waiting, ()), waiting))
        if self.hold is not None:
            self.hold = (self.hold[0], *(target if part == waiting else part for part in self.hold[1:]))
        self._go_to(target, _other_way(args, target) if head == "br" else None)
        return True

    def _go_to(self, target: str, other: Optional[str]) -> None:
        """Control goes to `target`. The other way a branch names is never taken, but it has to type, so its
        label gets an edge if it has none, and no move is written for it."""
        linear = set(self.linear)
        if other is not None and other not in self.edges:
            self.edges[other] = (dict(self.home), set(linear))
        self._arrive(target, dict(self.home), linear)
        self.live = False

    def _needs_moves(self, label: str, table: dict[int, Home]) -> bool:
        edge = self.edges.get(label)
        if edge is None:
            return False
        return any(r in edge[0] and edge[0][r] != home for r, home in table.items())

    def _arrive(self, label: str, table: dict[int, Home], linear: set[int],
                clean: Optional[dict[int, int]] = None) -> None:
        """An edge reaches a label. The first sets its homes, and later ones move to them.
        A value not live on every edge is not live after the label, by the merge rule. One that was
        reloaded on this edge and whose home there is the slot it came from is there already."""
        clean = dict(self.clean) if clean is None else clean
        self._table(table)
        self.linear = {r for r in linear if r in table}
        edge = self.edges.get(label)
        if edge is None:
            self.edges[label] = (dict(self.home), set(self.linear))
            return
        target = edge[0]
        for r in [r for r in target if r not in self.home]:
            del target[r]
        for r in [r for r in self.home if r not in target]:
            self._free(r)
        moves = [Move(_as_place(target[r]), _as_place(home))
                 for r, home in self.home.items() if target[r] != home and clean.get(r) != target[r]]
        self._pmove(moves)
        self._table(target)

    def i_jump(self, a: list[str]) -> None:
        self._go_to(a[0], None)
        self.hold = ("jump", a[0])

    def _branch(self, cc: str, then: str, other: str, then_table: dict[int, Home],
                other_table: dict[int, Home]) -> None:
        linear = set(self.linear)
        clean = dict(self.clean)
        if self._needs_moves(then, then_table) or self._needs_moves(other, other_table):
            bridge = self.fresh_label()
            self.ins(f"j{cc}", bridge)
            self._arrive(other, other_table, linear, clean)
            self.ins("jmp", self.asm_label(other))
            self.place(bridge)
            self._arrive(then, then_table, linear, clean)
            self.ins("jmp", self.asm_label(then))
        else:
            self._arrive(then, then_table, linear, clean)
            self._arrive(other, other_table, linear, clean)
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
        target = _literal_branch(a)
        if target is not None:
            self._go_to(target, _other_way(a, target))
            self.hold = ("jump", target)
            return
        self._compare(a[1].rstrip("!"), a[2].rstrip("!"))
        for operand in (a[1], a[2]):
            if operand.endswith("!"):
                self._free(self.reg(operand[:-1]))
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
        copy = self.clean.pop(source, None)
        self._free(source)
        r = self.reg(a[0])
        if copy is not None:  # its slot still holds this value, under its new name
            self.slots.add(copy)
            self.clean[r] = copy
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
        """Horner. The literals in front are one literal by the time this runs, shifted over the field
        after them. Then each field is a shift and an `or`, into the subregister of the field's width,
        which takes only those bits of a wider operand, so nothing has to be narrowed first."""
        widths = [int(w) for w in a[1].strip("<>").split()]
        pins: frozenset = frozenset()
        places = []
        for operand in a[2:]:
            place, pins = self._operand(operand, pins)
            places.append(place)
        register = self._take(pins)
        known = 0
        position = 0
        while position < len(places) and isinstance(places[position], Imm):
            known = (known << widths[position] | _masked(places[position].value, widths[position] // 8)) & _MASK64
            position += 1
        if position == len(places):
            self._load(register, Imm(known))
        elif position:
            self._load(register, Imm(known << widths[position] & _MASK64))
            self._or_into(register, widths[position] // 8, places[position])
            position += 1
        else:
            self._load(register, _narrowed(places[0], widths[0] // 8))
            if isinstance(places[0], Reg):
                self._narrow(register, widths[0] // 8)
            position = 1
        for bits, place in zip(widths[position:], places[position:]):
            self.ins("shl", register, str(bits))
            self._or_into(register, bits // 8, place)
        self._define(a[0], register)

    def i_etag(self, a: list[str]) -> None:
        x, pins = self._operand(a[1], frozenset())
        assert isinstance(x, Reg), x
        register = self._take(pins)
        self.ins("movzx", self.sub(register, 4), self.sub(x.name, 1))
        self._define(a[0], register)

    def i_efield(self, a: list[str]) -> None:
        """The field's bytes, shifted down to the bottom, and nothing above them."""
        _tag, fields = self.ctors[a[2]]
        off, size = fields[int(a[3])]
        x, pins = self._operand(a[1], frozenset())
        register = self._take(pins)
        self._load(register, x)
        self.ins("shr", register, str(8 * off))
        self._narrow(register, size)
        self._define(a[0], register)

    def i_enum(self, a: list[str]) -> None:
        """The tag and every literal field are one immediate. A field in a register is zero extended already,
        and goes to its bytes by a shift, through r11."""
        tag, fields = self.ctors[a[1]]
        pins: frozenset = frozenset()
        places = []
        for operand in a[2:]:
            place, pins = self._operand(operand, pins)
            places.append(place)
        register = self._take(pins)
        known = int(tag)
        for (off, size), place in zip(fields, places):
            if isinstance(place, Imm):
                known |= _masked(place.value, size) << 8 * off
        self._load(register, Imm(known))
        for (off, _size), place in zip(fields, places):
            if isinstance(place, Imm):
                continue
            self._load(TRANSIT, place)
            self.ins("shl", TRANSIT, str(8 * off))
            self.ins("or", register, TRANSIT)
            self.clobbered.add(TRANSIT)
        self._define(a[0], register)

    def i_bswap(self, a: list[str]) -> None:
        """`rol` on the 16-bit register leaves the upper bits, which are zero, as they are, and a 32-bit `bswap`
        clears them. Nothing here reads `movbe`, which an older x86-64 does not have."""
        bits = int(a[1])
        x, pins = self._operand(a[2], frozenset())
        register = self._take(pins)
        self._load(register, x)
        if bits == 16:
            self.ins("rol", self.sub(register, 2), "8")
        else:
            self.ins("bswap", self.sub(register, bits // 8))
        self._define(a[0], register)

    def _or_into(self, register: str, size: int, place: Union[Reg, Imm, Mem]) -> None:
        """The low `size` bytes of the register are zero. A 32-bit `or` would clear the upper half."""
        if isinstance(place, Imm):
            value = _masked(place.value, size)
            if value >= 1 << 31:
                self._load(TRANSIT, Imm(value))
                self.ins("or", register, TRANSIT)
            elif value:
                self.ins("or", register, str(value))
        elif size == 4:
            if isinstance(place, Reg):
                self.ins("mov", self.sub(TRANSIT, 4), self.sub(place.name, 4))
            else:
                self._load(TRANSIT, _narrowed(place, 4))
            self.ins("or", register, TRANSIT)
        elif isinstance(place, Reg):
            self.ins("or", self.sub(register, size), self.sub(place.name, size))
        else:
            narrowed = _narrowed(place, size)
            self.ins("or", self.sub(register, narrowed.size), self.mem(narrowed))

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
        if a[2] != "write" and all(source in ("=", "*") or source.endswith("!") for source in a[3:]):
            # Every field holds what it should already, so nothing is stored, and the cell need not be in a register.
            for source in a[3:]:
                if source.endswith("!"):
                    self._free(self.reg(source[:-1]))
            self.linear.add(self.reg(a[0]))
            return
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
            moves = [Move(Reg(RESULTS[position]), Imm(0) if operand == "#0" else _as_place(self.home[self.reg(operand)]))
                     for position, operand in enumerate(a)]
        self._pmove(moves)
        self.clobbered |= {m.dest.name for m in moves if isinstance(m.dest, Reg)}
        self.ins("add", "rsp", f"F_{fn.name}")
        self.ins("ret")
        self.live = False


_MASK64 = (1 << 64) - 1
_COMPARED = {"b": lambda x, y: x < y, "be": lambda x, y: x <= y, "a": lambda x, y: x > y, "ae": lambda x, y: x >= y,
             "e": lambda x, y: x == y, "ne": lambda x, y: x != y}


def _literal_branch(a: list[str]) -> Optional[str]:
    """Where `br cc, #x, #y, Lt, Lf` goes, or None for a branch on anything else."""
    x, y = _LIT.match(a[1]), _LIT.match(a[2])
    if x is None or y is None:
        return None
    return a[3] if _COMPARED[a[0]](int(x.group(1)), int(y.group(1))) else a[4]


def _other_way(a: list[str], target: str) -> str:
    return a[4] if target == a[3] else a[3]


def _narrowed(place: Union[Reg, Imm, Mem], size: int) -> Union[Reg, Imm, Mem]:
    """A location, read at no more than this width: the low bytes are at the same address."""
    return Mem(place.base, place.off, min(place.size, size)) if isinstance(place, Mem) else place


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
