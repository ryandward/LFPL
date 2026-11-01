"""The Verilog backend: LFPL-O section 9, from the checked RT stream and nothing else.

The stream is text, and this reads it as text, as every reader of it does. It imports nothing from
the x86 backend or the x86 runtime, and it emits nothing until src/rtcheck.py has accepted every line
and the bound has passed its certificate. The whole handler becomes one synchronous state machine:

    an RT register      a flip-flop bank at its type's width, named for its function
    a cell              one word of block RAM. Port A reads, port B writes with byte enables
    a pointer           that word's address, log2(C) bits. Word 0 is null, and is never a cell
    br.null             the next state, chosen by a comparison, and the read of the cell if there is one
    take, read          a slice of port A's output, latched with <=
    build               one write on port B, whose enables cover exactly the fields it stores. A field
                        that says `=` has no enable, so an unchanged field cannot be written
    tail                <= on every parameter in one clock, then the entry state. The parallel move x86
                        has to schedule, with a scratch register to break cycles, does not exist here
    call                <= on the callee's parameters, and on its return state if it has several
    ret                 <= on the destination registers of whichever call this is, then its next state

A function's registers are static, with no stack and no frames. rtcheck refuses non-tail recursion, so
the call graph has no cycle but a function's jump to its own entry, and no function is ever active
twice.

Every state belongs to one RT instruction, so a cycle count can be derived before anything runs. A
state occupies the machine for exactly one clock. Most instructions are exactly one state, and three
things put another in front of one. A field can only be read from port A's output, so an instruction
that names a field of a cell whose word is not there has to wait for a read. An array read needs a clock
for the same reason. And locations of two cells in one instruction need the first one latched, because
there is one read port. See `Timing`.
"""
from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

from .certify import certify
from .ring import FStarved, Reject
from .rtcheck import NO_WORK, Bound, Checker, RTError
from .rtlower import Limit, rt_lines

CELL_BITS = 128
CELL_BYTES = CELL_BITS // 8
K_IN = 1
O_MAX = 1
SCALAR_BITS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "bool": 1, "lit": 64}
OPERATORS = {"add": "+", "sub": "-", "mul": "*", "and": "&", "or": "|", "xor": "^", "shl": "<<", "shr": ">>"}
COMPARISONS = {"b": "<", "be": "<=", "a": ">", "ae": ">=", "e": "==", "ne": "!="}
KEYWORDS = frozenset(("always", "and", "assign", "begin", "buf", "case", "default", "edge", "else", "end",
                      "endcase", "endmodule", "for", "function", "if", "initial", "input", "integer",
                      "module", "nand", "negedge", "nor", "not", "or", "output", "parameter", "posedge",
                      "reg", "task", "wire", "xor"))
FALL = "+"  # the state after this one, whichever that turns out to be

_REG = re.compile(r"r(\d+)$")
_LOC = re.compile(r"r(\d+)\[(\d+):(\d+)\]$")
_LIT = re.compile(r"#(\d+)$")
_INNER = re.compile(r"\[(\d+):(\d+)\]$")
_FUNC = re.compile(r"(\w+)\(\s*\[(\d+):(\d+)\]\s*,\s*(.+)\)$")
_HEADER = re.compile(r"(fn|handler) (f\d+) (\w+)\((.*)\) : (\S+)(?: @(\d+))?$")
_LABEL = re.compile(r"(L\d+):\s*(.*)$")


def _args(text: str) -> list[str]:
    """Split on commas outside brackets, so `add([0:8], #1)` stays whole."""
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


def _bits_for(values: int) -> int:
    """How many bits name this many different things."""
    return max(1, math.ceil(math.log2(values))) if values > 1 else 1


def _pad(expression: str, width: int, want: int) -> str:
    if width >= want:
        return expression
    return f"{{{want - width}'d0, {expression}}}"


def _slice(name: str, low: int, width: int, want: int) -> str:
    """Bits of a named register, zero-extended or cut to the width wanted."""
    kept = min(width, want)
    return _pad(f"{name}[{low + kept - 1}:{low}]", kept, want)


def _whole(name: str, width: int, want: int) -> str:
    """A whole register at the width wanted. Every value in one is zero-extended."""
    if width > want:
        return f"{name}[{want - 1}:0]" if want > 1 else f"{name}[0]"
    return _pad(name, width, want)


@dataclass
class Ctor:
    tag: str
    fields: list = field(default_factory=list)  # (offset, size, type text)


@dataclass
class Site:
    """A place a function returns to: the state after the call, and where its results go."""
    state: str = ""
    results: list = field(default_factory=list)  # (register name, width)
    finished: bool = False                       # the handler's: this is what raises done
    caller: str = ""                             # the ordinal of the function it is in


@dataclass
class Enter:
    """The next state is another function's entry, or this one's."""
    callee: "Function"
    site: Optional[Site] = None    # a call. A tail call has none, and returns where this one would


@dataclass
class Return:
    values: list  # (expression, width)


@dataclass
class Choice:
    condition: str
    yes: str
    no: str


Next = Union[str, Enter, Return, Choice]


@dataclass
class State:
    name: str
    comment: str
    clocked: list = field(default_factory=list)  # <= statements, at the clock edge
    ports: list = field(default_factory=list)    # what this state drives on the memory ports
    next: Next = FALL
    hardware: bool = False  # a state RT charges nothing for, in front of an instruction it does
    reads: bool = False
    writes: bool = False


@dataclass
class Function:
    ordinal: str
    name: str
    params: list
    ret: str
    at: Optional[int]
    handler: bool
    widths: dict = field(default_factory=dict)    # RT register -> bits
    types: dict = field(default_factory=dict)     # RT register -> type text, where it matters
    temporaries: list = field(default_factory=list)
    states: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)
    sites: list = field(default_factory=list)
    tail_calls: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    recursive: bool = False
    reads: bool = False
    costs: list = field(default_factory=list)     # (kind, cycles) of each instruction RT charges for

    def flop(self, register: int) -> str:
        return f"{self.name}_r{register}"


@dataclass(frozen=True)
class Span:
    """Cycles along a function's paths. `walks` says a path also runs something recursive."""
    low: int
    high: int
    walks: bool = False

    def __str__(self) -> str:
        text = str(self.low) if self.low == self.high else f"{self.low} to {self.high}"
        return text + (" and the walks it calls" if self.walks else "")


@dataclass(frozen=True)
class Timing:
    """The clock cycles this machine takes, derived from the states it was given.

    Count from the edge that samples `start` to the edge that raises `done`. Each state holds the
    machine for one clock, and none is ever entered but by a transition, so that count is the
    number of states executed. Each of those belongs to an instruction RT charges a step for, or is
    a hardware-only state in front of one, or the handler's zero-step `open` of a cell: and that
    costs nothing here until a field of it is named, when it is a hardware-only state too.

    Write cost(I) for 1 plus the hardware-only states in front of I. Split the code in two:

        in a walk   a recursive function, and all it calls. These run a number of times that depends
                    on the item. k is the largest cost(I) among them, so each execution of one of
                    them takes at most k cycles for its one step
        the rest    runs at most once per call, on a call graph with no cycle. c is the largest number
                    of hardware-only states on any path through it, found by walking its states

    So cycles <= k * steps + c, and rtcheck has certified steps <= W(n), which gives the bound.
    """
    k: int
    c: int
    kinds: tuple           # (kind, steps, fewest cycles, most cycles, how many in the machine)
    iterations: tuple      # (function name, Span of an iteration, Span of leaving)
    states: int

    def bound(self, n: int, bound: Bound) -> int:
        return self.k * (bound.c * n + bound.d) + self.c

    def per_cell(self, bound: Bound) -> int:
        return self.k * bound.c


@dataclass(frozen=True)
class Translation:
    verilog: str
    module: str
    bound: Bound
    timing: Timing
    cells: int
    pointer_bits: int
    memories: dict  # state slot -> hex text for $readmemh, to go beside the Verilog

    def report(self) -> str:
        lines = [f"handler {self.module}: R(n) = {_linear(self.bound.a, self.bound.b)} cells, "
                 f"W(n) = {_linear(self.bound.c, self.bound.d)} steps, {self.cells} cells of "
                 f"{CELL_BITS} bits, {self.pointer_bits}-bit pointers, {self.timing.states} states",
                 "the emitted machine, by RT instruction:",
                 "  kind      steps  cycles  in the machine"]
        for kind, steps, low, high, count in self.timing.kinds:
            cycles = str(low) if low == high else f"{low}-{high}"
            lines.append(f"  {kind:<9} {steps:>5}  {cycles:>6}  {count:>14}")
        if any(low != high for _kind, _steps, low, high, _count in self.timing.kinds):
            lines.append("  (the extra cycle is a state in front: a read of the cell, or a latch)")
        for name, iteration, leaving in self.timing.iterations:
            lines.append(f"walk {name}: {iteration} cycles per iteration, {leaving} to leave")
        k, c = self.timing.k, self.timing.c
        lines.append(f"k = {k}: the most cycles any one instruction takes in a walk")
        lines.append(f"c = {c}: the most states RT does not charge for, on a path outside the walks")
        slope, constant = k * self.bound.c, k * self.bound.d + c
        lines.append(f"cycles(n) <= {k}*W(n) + {c} = {_linear(slope, constant)}, from the edge that "
                     f"samples start to the edge that raises done")
        return "\n".join(lines)


def _linear(slope: int, constant: int) -> str:
    if not slope:
        return str(constant)
    return f"{slope} n + {constant}" if constant else f"{slope} n"


class Emitter:
    """Reads a stream rtcheck has accepted, and holds the machine until `render`."""

    def __init__(self, pointer_bits: int) -> None:
        self.pointer_bits = pointer_bits
        self.aliases: dict[str, str] = {}
        self.composite: dict[str, tuple] = {}
        self.tuples: dict[str, list[str]] = {}
        self.ctors: dict[str, Ctor] = {}
        self.slots: list[tuple[str, str]] = []
        self.functions: dict[str, Function] = {}
        self.current: list[str] = []
        self.fn: Optional[Function] = None
        self.loaded: Optional[int] = None  # the register whose cell's word is at port A's output
        self.pending: dict[int, int] = {}  # moves not yet made: a move costs no state of its own
        self.edges: dict[str, list[Optional[int]]] = {}
        self.waiting_labels: list[str] = []
        self.waiting_sites: list[Site] = []
        self.latched: dict[str, tuple[str, int]] = {}
        self.attached = 0
        self.open_path = False
        self.kinds: dict[str, list[int]] = {}
        self.cut: dict[str, tuple[int, int]] = {}  # registers only read in part: (kept, width)

    # --- declarations

    def feed(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        if self.current:
            self.current.append(text)
            if text.split()[0] == "end":
                self._function(self.current)
                self.current = []
            return
        head, *parts = text.split()
        if head in ("fn", "handler"):
            self.current = [text]
        elif head == "alias":
            self.aliases[parts[0]] = parts[1]
        elif head == "array":
            self.composite[parts[0]] = ("array", int(parts[1]), parts[2])
        elif head == "fin":
            self.composite[parts[0]] = ("fin", int(parts[1]))
        elif head == "tuple":
            self.tuples[parts[0]] = parts[1:]
        elif head == "ctor":
            self.ctors[parts[0]] = Ctor(parts[2])
        elif head == "field":
            self.ctors[parts[0]].fields.append((int(parts[1]), int(parts[2]), parts[3]))
        elif head == "state":
            self.slots.append((parts[0], parts[1]))

    def _resolve(self, name: str) -> str:
        while name in self.aliases:
            name = self.aliases[name]
        return name

    def _array_of(self, type_text: str) -> Optional[tuple]:
        entry = self.composite.get(self._resolve(type_text))
        return entry if entry is not None and entry[0] == "array" else None

    def _array_bits(self) -> int:
        """An array is linear, and nothing about it is stored but which one it is. With one array
        in the program, even that needs no bits."""
        return 0 if len(self.slots) < 2 else _bits_for(len(self.slots))

    def _width(self, type_text: str) -> int:
        name = self._resolve(type_text)
        if name in SCALAR_BITS:
            return SCALAR_BITS[name]
        entry = self.composite.get(name)
        if entry is None:
            return self.pointer_bits  # a capability, or any value made of cells
        return self._array_bits() if entry[0] == "array" else _bits_for(entry[1])

    def _results(self, type_text: str) -> list[str]:
        return self.tuples.get(self._resolve(type_text), [type_text])

    # --- one function: first what each register is, then what each instruction does

    def _function(self, lines: list[str]) -> None:
        match = _HEADER.match(lines[0])
        assert match is not None, lines[0]
        kind, ordinal, name, params, ret, at = match.groups()
        types = [part.partition(" : ")[2].strip() for part in _args(params)]
        fn = Function(ordinal, name, types, ret, int(at) if at else None, kind == "handler")
        self.fn = fn
        body = [(_LABEL.match(line).group(1) if _LABEL.match(line) else None,
                 _LABEL.match(line).group(2) if _LABEL.match(line) else line)
                for line in lines[1:-1]]
        self._measure_registers(fn, body)
        self.loaded = None
        self.pending = {}
        self.edges = {}
        self.attached = 0
        self.open_path = True
        for label, instruction in body:
            if label is not None:
                self._label(label)
            if instruction:
                self._instruction(instruction)
        # A callee comes before its caller, so this is all there is to know about what it reads.
        fn.reads = any(state.reads for state in fn.states) or any(callee.reads for callee in fn.calls)
        self.functions[ordinal] = fn
        self.fn = None

    def _measure_registers(self, fn: Function, body: list) -> None:
        """A register's width is the widest thing the function ever puts in it."""
        widths: dict[int, int] = {}
        types: dict[int, str] = {}

        def define(text: str, width: int, type_text: Optional[str] = None) -> None:
            r = int(_REG.match(text).group(1))
            widths[r] = max(widths.get(r, 0), width)
            if type_text is not None:
                types[r] = type_text

        for position, type_text in enumerate(fn.params):
            define(f"r{position}", self._width(type_text), type_text)
        moves = []
        for _label, text in body:
            if not text:
                continue
            head, _, rest = text.partition(" ")
            a = _args(rest)
            if head == "const":
                define(a[0], self._width(a[1].partition(" : ")[2].strip()))
            elif head in ("move", "copy"):
                moves.append((a[0], a[1]))
            elif head == "op":
                define(a[0], int(a[1].split("/")[1]))
            elif head == "cmp":
                define(a[0], 1)
            elif head == "trunc":
                define(a[0], int(a[1]))
            elif head == "pack":
                define(a[0], 64)
            elif head == "idx":
                define(a[0], _bits_for(int(a[1])))
            elif head == "aget":
                define(a[0], 8 * int(a[3]))
            elif head == "read":
                define(a[0], 8 * int(_LOC.match(a[1]).group(3)))
            elif head == "take":
                define(a[0], self.pointer_bits)
            elif head == "call":
                callee = self.functions[a[1]]
                names = _args(a[0].strip("()")) if a[0].startswith("(") else [a[0]]
                for name, member in zip(names, self._results(callee.ret)):
                    define(name, self._width(member), member)
        changed = True
        while changed:  # a move's target is as wide as its source, whichever came first in the text
            changed = False
            for target, source in moves:
                s = int(_REG.match(source).group(1))
                t = int(_REG.match(target).group(1))
                if widths.get(t, 0) < widths.get(s, 0) or (s in types and t not in types):
                    widths[t] = max(widths.get(t, 0), widths.get(s, 0))
                    if s in types:
                        types[t] = types[s]
                    changed = True
        fn.widths = widths
        fn.types = types

    def _instruction(self, text: str) -> None:
        head, _, rest = text.partition(" ")
        self.latched = {}
        getattr(self, "i_" + head.replace(".", "_"))(text, _args(rest))

    # --- states

    def _state(self, comment: str, *, clocked: Iterable[str] = (), ports: Iterable[str] = (),
               next_state: Next = FALL, hardware: bool = False, reads: bool = False,
               writes: bool = False) -> State:
        fn = self.fn
        assert fn is not None
        index = len(fn.states)
        state = State(f"{fn.name}_s{index}", comment, [], list(ports), next_state, hardware, reads, writes)
        for label in self.waiting_labels:
            fn.labels[label] = index
        self.waiting_labels = []
        for site in self.waiting_sites:
            site.state = state.name
        self.waiting_sites = []
        for target, source in self.pending.items():  # every waiting move happens in this clock
            state.clocked.append(f"{fn.flop(target)} <= "
                                 f"{_whole(fn.flop(source), fn.widths[source], fn.widths[target])};")
        self.pending = {}
        state.clocked.extend(clocked)
        fn.states.append(state)
        return state

    def _hardware(self, comment: str, **kwargs) -> None:
        """A state in front of an instruction, which RT does not charge for."""
        self._state(comment, hardware=True, **kwargs)
        self.attached += 1

    def _step(self, kind: str, comment: str, **kwargs) -> State:
        """The state of an instruction RT charges a step for. Whatever hardware-only states came
        just before it are part of what this instruction takes."""
        fn = self.fn
        assert fn is not None
        state = self._state(comment, **kwargs)
        cost = 1 + self.attached
        fn.costs.append((kind, cost))
        self._count(kind, cost)
        self.attached = 0
        return state

    def _count(self, kind: str, cost: int) -> None:
        low, high, count = self.kinds.get(kind, [cost, cost, 0])
        self.kinds[kind] = [min(low, cost), max(high, cost), count + 1]

    def _edge(self, label: str, loaded: Optional[int]) -> None:
        self.edges.setdefault(label, []).append(loaded)

    def _label(self, label: str) -> None:
        incoming = self.edges.pop(label, [])
        if self.open_path:
            if self.pending:  # a label has every register in its own flip-flop on every edge
                self._hardware("the moves before a join")
            incoming.append(self.loaded)
        else:
            self.attached = 0
        self.loaded = incoming[0] if incoming and all(i == incoming[0] for i in incoming) else None
        self.open_path = True
        self.waiting_labels.append(label)

    # --- reading operands

    @staticmethod
    def _location(text: str) -> Optional[tuple[int, int, int]]:
        match = _LOC.match(text)
        return (int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None

    def _pointer(self, register: int) -> str:
        fn = self.fn
        assert fn is not None
        return fn.flop(self.pending.get(register, register))

    def _at_port(self, register: int) -> None:
        """A field is read from port A's output, which holds the last word read until the next
        read. If that is not this cell's word, reading it takes a state."""
        if self.loaded == register:
            return
        pointer = self._pointer(register)
        self._hardware(f"read the cell in r{register}", reads=True,
                       ports=["a_en = 1'b1;", f"a_addr = {pointer};"])
        self.loaded = register

    def _locations(self, texts: Iterable[str]) -> None:
        """Everything an instruction names in memory must be readable in its one clock. With one
        read port, that is one word. A location of any other cell is latched first, starting with
        the cell already at the port, since that one needs no read."""
        fn = self.fn
        assert fn is not None
        locations = [text for text in texts if self._location(text)]
        bases = list(dict.fromkeys(self._location(text)[0] for text in locations))
        if not bases:
            return
        last = next(b for b in reversed(bases) if b != self.loaded or len(bases) == 1)
        earlier = [b for b in bases if b != last]
        for base in sorted(earlier, key=lambda b: b != self.loaded):
            self._at_port(base)
            for text in locations:
                where = self._location(text)
                if where[0] != base or text in self.latched:
                    continue
                width = 8 * where[2]
                name = f"{fn.name}_t{len(fn.temporaries)}"
                fn.temporaries.append((name, width))
                self._hardware(f"latch {text}: one read port, and two cells",
                               clocked=[f"{name} <= {_slice('a_dout', 8 * where[1], width, width)};"])
                self.latched[text] = (name, width)
        self._at_port(last)

    def _value(self, text: str, want: int) -> str:
        """An operand as an expression of exactly `want` bits."""
        fn = self.fn
        assert fn is not None
        text = text.rstrip("!")
        literal = _LIT.match(text)
        if literal:  # rtcheck has seen that it fits, but for what trunc is there to cut
            return f"{want}'d{int(literal.group(1)) & ((1 << want) - 1)}"
        if text in self.latched:
            name, width = self.latched[text]
            return _whole(name, width, want)
        where = self._location(text)
        if where is not None:
            return _slice("a_dout", 8 * where[1], 8 * where[2], want)
        register = int(_REG.match(text).group(1))
        source = self.pending.get(register, register)
        if fn.widths[source] > want:
            name = fn.flop(source)
            self.cut[name] = (min(want, self.cut.get(name, (want,))[0]), fn.widths[source])
        return _whole(fn.flop(source), fn.widths[source], want)

    def _natural_width(self, text: str) -> Optional[int]:
        fn = self.fn
        assert fn is not None
        text = text.rstrip("!")
        if _LIT.match(text):
            return None
        where = self._location(text)
        if where is not None:
            return 8 * where[2]
        register = int(_REG.match(text).group(1))
        return fn.widths[self.pending.get(register, register)]

    def _set(self, text: str, expression: str, width: int) -> str:
        """`text` <= an expression of `width` bits. This is never a cell's capability, so it cannot
        be the register `loaded` names: those come from a dispatch, a move or a call."""
        fn = self.fn
        assert fn is not None
        register = int(_REG.match(text).group(1))
        self.pending.pop(register, None)
        return f"{fn.flop(register)} <= {_pad(expression, width, fn.widths[register])};"

    # --- values

    def i_const(self, text: str, a: list[str]) -> None:
        fn = self.fn
        assert fn is not None
        literal = _LIT.match(a[1].partition(" : ")[0].strip())
        register = int(_REG.match(a[0]).group(1))
        width = fn.widths[register]
        self._step("const", text, clocked=[self._set(a[0], f"{width}'d{literal.group(1)}", width)])

    def i_move(self, _text: str, a: list[str]) -> None:
        """No state: the move happens in the next clock, whatever else does, and until then a read
        of the target reads the source."""
        target = int(_REG.match(a[0]).group(1))
        source = int(_REG.match(a[1]).group(1))
        self._count("move", 0)
        real = self.pending.pop(source, source)
        self.pending[target] = real
        if self.loaded == source:
            self.loaded = target
        elif self.loaded == target:
            self.loaded = None  # it held that cell once, and it does not now

    def i_copy(self, text: str, a: list[str]) -> None:
        fn = self.fn
        assert fn is not None
        width = fn.widths[int(_REG.match(a[0]).group(1))]
        value = self._value(a[1], width)
        self._step("copy", text, clocked=[self._set(a[0], value, width)])

    def i_drop(self, _text: str, a: list[str]) -> None:
        self._count("drop", 0)
        self.pending.pop(int(_REG.match(a[0]).group(1)), None)

    def i_op(self, text: str, a: list[str]) -> None:
        name, bits = a[1].split("/")
        width = int(bits)
        self._locations(a[2:4])
        left = self._value(a[2], width)
        if name in ("shl", "shr"):
            expression = self._shift(name, left, a[3], width)
        else:
            expression = f"{left} {OPERATORS[name]} {self._value(a[3], width)}"
        self._step("op", text, clocked=[self._set(a[0], expression, width)])

    def _shift(self, name: str, left: str, count: str, width: int) -> str:
        """Section 4.4 makes a count at or past the width give 0, and Verilog's shifts already do."""
        literal = _LIT.match(count)
        if literal:
            amount = int(literal.group(1))
            if amount >= width:
                return f"{width}'d0"
            return f"{left} {OPERATORS[name]} {_bits_for(amount + 1)}'d{amount}"
        natural = self._natural_width(count)
        assert natural is not None
        return f"{left} {OPERATORS[name]} {self._value(count, natural)}"

    def _comparison(self, cc: str, left: str, right: str) -> str:
        widths = [w for w in (self._natural_width(left), self._natural_width(right)) if w is not None]
        width = max(widths) if widths else 64
        return f"{self._value(left, width)} {COMPARISONS[cc]} {self._value(right, width)}"

    def i_cmp(self, text: str, a: list[str]) -> None:
        self._locations(a[2:4])
        expression = self._comparison(a[1], a[2], a[3])
        self._step("cmp", text, clocked=[self._set(a[0], f"({expression})", 1)])

    def i_trunc(self, text: str, a: list[str]) -> None:
        width = int(a[1])
        self._locations(a[2:3])
        self._step("trunc", text, clocked=[self._set(a[0], self._value(a[2], width), width)])

    def i_pack(self, text: str, a: list[str]) -> None:
        widths = [int(w) for w in a[1].strip("<>").split()]
        self._locations(a[2:])
        parts = [self._value(operand, bits) for bits, operand in zip(widths, a[2:])]
        padding = 64 - sum(widths)
        if padding:
            parts.insert(0, f"{padding}'d0")
        expression = parts[0] if len(parts) == 1 else "{" + ", ".join(parts) + "}"
        self._step("pack", text, clocked=[self._set(a[0], expression, 64)])

    def i_idx(self, text: str, a: list[str]) -> None:
        size = int(a[1])
        width = _bits_for(size)
        self._locations(a[2:3])
        value = "1'd0" if size == 1 else self._value(a[2], width)
        self._step("idx", text, clocked=[self._set(a[0], value, width)])

    def i_aget(self, text: str, a: list[str]) -> None:
        """An array is its own block RAM, so this reads, and then latches what the read returned."""
        fn = self.fn
        assert fn is not None
        array = int(_REG.match(a[1]).group(1))
        array_type = self._array_of(fn.types[array])
        assert array_type is not None
        self._locations(a[2:3])
        address_bits = _bits_for(array_type[1])
        index = self._value(a[2], address_bits)
        slots = [position for position, (_name, ty) in enumerate(self.slots)
                 if self._array_of(ty) == array_type]
        width = 8 * int(a[3])
        selector = self._pointer(array)
        ports = []
        result = ""
        for position in slots:
            name = self.slots[position][0]
            bits = self._array_bits()
            enable = "1'b1" if len(slots) == 1 else f"({selector} == {bits}'d{position})"
            ports += [f"{name}_en = {enable};", f"{name}_addr = {index};"]
            if len(slots) == 1 or position == slots[-1]:
                result = result + f"{name}_dout" + ")" * (len(slots) - 1)
            else:
                result += f"(({selector} == {bits}'d{position}) ? {name}_dout : "
        self._hardware(f"read array r{array}", ports=ports)
        self._step("aget", text, clocked=[self._set(a[0], result, width)])

    # --- cells

    def i_open(self, _text: str, _a: list[str]) -> None:
        """Nothing yet. If a field of it is named, that is when its word has to be at the port."""
        self._count("open", 0)

    def i_read(self, text: str, a: list[str]) -> None:
        self._locations(a[1:2])
        width = 8 * self._location(a[1])[2]
        self._step("read", text, clocked=[self._set(a[0], self._value(a[1], width), width)])

    def i_take(self, text: str, a: list[str]) -> None:
        self._locations(a[1:2])
        width = self.pointer_bits
        self._step("take", text, clocked=[self._set(a[0], self._value(a[1], width), width)])

    def _sources_in_memory(self, p: int, sources: list[str]) -> list[str]:
        """The locations a constructor's sources name, written as locations."""
        named = []
        for source in sources:
            inner = _INNER.match(source)
            computed = _FUNC.match(source)
            if inner:
                named.append(f"r{p}[{inner.group(1)}:{inner.group(2)}]")
            elif computed:
                named.append(f"r{p}[{computed.group(2)}:{computed.group(3)}]")
            elif self._location(source):
                named.append(source)
        return named

    def _write(self, p: int, ctor_id: str, tag: str, sources: list[str]) -> list[str]:
        """One write on port B. The byte enables cover exactly what this constructor stores, so a
        field that says `=` cannot be written: the first specification's Theorem 6.1, and RTL-3."""
        ctor = self.ctors[ctor_id]
        enables = 0
        data = []
        for source, (offset, size, _type) in zip(sources, ctor.fields):
            if source in ("=", "*") or source.endswith("!"):
                continue
            enables |= ((1 << size) - 1) << offset
            low, high, width = 8 * offset, 8 * (offset + size) - 1, 8 * size
            inner = _INNER.match(source)
            computed = _FUNC.match(source)
            if inner:
                value = _slice("a_dout", 8 * int(inner.group(1)), 8 * int(inner.group(2)), width)
            elif computed:
                name, old_offset, old_size, operand = computed.groups()
                old = _slice("a_dout", 8 * int(old_offset), 8 * int(old_size), width)
                value = f"{old} {OPERATORS[name]} {self._value(operand, width)}"
            else:
                value = self._value(source, width)
            data.append(f"b_din[{high}:{low}] = {value};")
        if tag == "write":
            enables |= 1
            data.append(f"b_din[7:0] = 8'd{ctor.tag};")
        if not enables:
            return []  # every field stayed: the cell does not change, and memory is not touched
        return [f"b_we = {CELL_BYTES}'h{enables:04x};", f"b_addr = {self._pointer(p)};", *data]

    def i_build(self, text: str, a: list[str]) -> None:
        p = int(_REG.match(a[0]).group(1))
        sources = a[3:]
        self._locations(self._sources_in_memory(p, sources))
        ports = self._write(p, a[1], a[2], sources)
        self._step("build", text, ports=ports, writes=bool(ports))
        if self.loaded == p:
            self.loaded = None  # what the port holds is what the cell held before this

    # --- control

    def i_jump(self, text: str, a: list[str]) -> None:
        self._edge(a[0], self.loaded)
        self._step("jump", text, next_state=a[0])
        self.open_path = False

    def i_br(self, text: str, a: list[str]) -> None:
        self._locations([a[1].rstrip("!"), a[2].rstrip("!")])
        condition = self._comparison(a[0], a[1], a[2])
        self._edge(a[3], self.loaded)
        self._edge(a[4], self.loaded)
        self._step("br", text, next_state=Choice(condition, a[3], a[4]))
        self.open_path = False

    def i_br_null(self, text: str, a: list[str]) -> None:
        """The test and the read are the same state. The read's address is the register being
        tested, and its word is at port A's output in the next state, so a walk spends no extra
        clock per cell. If the pointer is null nothing is read, so what the port held still holds."""
        p = int(_REG.match(a[0]).group(1))
        pointer = self._pointer(p)
        zero = f"{self.pointer_bits}'d0"
        self._edge(a[1], self.loaded)
        self._edge(a[2], p)
        self._step("br.null", text, reads=True, next_state=Choice(f"{pointer} == {zero}", a[1], a[2]),
                   ports=[f"a_en = ({pointer} != {zero});", f"a_addr = {pointer};"])
        self.open_path = False

    def i_br_tag(self, text: str, a: list[str]) -> None:
        """The tag is in the cell, so unlike x86, whose compare reads memory itself, this needs the
        word first."""
        p = int(_REG.match(a[0]).group(1))
        self._at_port(p)
        self._edge(a[2], p)
        self._edge(a[3], p)
        self._step("br.tag", text, next_state=Choice(f"a_dout[7:0] == 8'd{a[1]}", a[2], a[3]))
        self.open_path = False

    # --- calls and exits

    def _parameters(self, callee: Function, operands: list[str]) -> list[str]:
        """<= on every parameter, in one clock, each from what it was before the clock: so a swap,
        or any other cycle, is written out as it is and needs no scratch register."""
        fn = self.fn
        assert fn is not None
        lines = []
        for position, operand in enumerate(operands):
            width = callee.widths[position]
            if not width:
                continue
            target = callee.flop(position)
            value = self._value(operand, width)
            if value != target:
                lines.append(f"{target} <= {value};")
        if callee.at is not None and callee is not fn:
            lines.append(f"{callee.name}_root <= {self._value(operands[callee.at], self.pointer_bits)};")
        return lines

    def i_call(self, text: str, a: list[str]) -> None:
        fn = self.fn
        assert fn is not None
        callee = self.functions[a[1]]
        operands = a[2:]
        self._locations(operands)
        clocked = self._parameters(callee, operands)
        names = _args(a[0].strip("()")) if a[0].startswith("(") else [a[0]]
        site = Site(caller=fn.ordinal)
        for name in names:
            register = int(_REG.match(name).group(1))
            site.results.append((fn.flop(register), fn.widths[register]))
            if self.loaded == register:
                self.loaded = None
            self.pending.pop(register, None)
        callee.sites.append(site)
        fn.calls.append(callee)
        self._step("call", text, clocked=clocked, next_state=Enter(callee, site))
        self.waiting_sites.append(site)
        if callee.reads:  # it leaves its own word at the port
            self.loaded = None

    def _self_or_other(self, ordinal: str) -> Function:
        fn = self.fn
        assert fn is not None
        if ordinal == fn.ordinal:
            fn.recursive = True
            return fn
        fn.calls.append(self.functions[ordinal])
        fn.tail_calls.append(self.functions[ordinal])
        return self.functions[ordinal]

    def i_tail(self, text: str, a: list[str]) -> None:
        callee = self._self_or_other(a[0])
        operands = a[1:]
        self._locations(operands)
        clocked = self._parameters(callee, operands)
        self._step("tail", text, clocked=clocked, next_state=Enter(callee))
        self.open_path = False

    def i_tailmod(self, text: str, a: list[str]) -> None:
        """The operands are read from what the cell holds now, and the write and the jump are one
        clock, so this is `tail` and `build` at once, as RT says."""
        ctor = self.ctors[a[1]]
        count = len(ctor.fields)
        sources = a[3:3 + count]
        callee = self._self_or_other(a[3 + count])
        operands = a[4 + count:]
        p = int(_REG.match(a[0]).group(1))
        self._locations(self._sources_in_memory(p, sources) + operands)
        ports = self._write(p, a[1], a[2], sources)
        clocked = self._parameters(callee, operands)
        self._step("tailmod", text, clocked=clocked, ports=ports, writes=bool(ports), next_state=Enter(callee))
        self.open_path = False

    def i_ret(self, text: str, a: list[str]) -> None:
        fn = self.fn
        assert fn is not None
        if fn.at is not None:  # whatever this iteration holds, the caller gets the cell it passed
            values = [(f"{fn.name}_root", self.pointer_bits)]
        else:
            values = []
            for operand in a:
                register = int(_REG.match(operand).group(1))
                source = self.pending.get(register, register)
                values.append((fn.flop(source), fn.widths[source]))
        self._step("ret", text, next_state=Return(values))
        self.open_path = False

    # --- the whole machine

    def _reachable(self, handler: Function) -> list[Function]:
        seen = {handler.ordinal: handler}
        waiting = [handler]
        while waiting:
            for callee in waiting.pop().calls:
                if callee.ordinal not in seen:
                    seen[callee.ordinal] = callee
                    waiting.append(callee)
        return [fn for fn in self.functions.values() if fn.ordinal in seen]

    def _timing(self, handler: Function, functions: list[Function]) -> Timing:
        walking: set[str] = set()
        waiting = [fn for fn in functions if fn.recursive]
        while waiting:
            fn = waiting.pop()
            if fn.ordinal not in walking:
                walking.add(fn.ordinal)
                waiting.extend(fn.calls)
        k = max((cost for fn in functions if fn.ordinal in walking for _kind, cost in fn.costs), default=1)
        c = self._hardware_states(handler, walking, {})
        kinds = []
        for kind, (low, high, count) in sorted(self.kinds.items()):
            kinds.append((kind, 0 if kind in NO_WORK else 1, low, high, count))
        spans: dict[str, tuple[Span, Span]] = {}
        iterations = []
        for fn in functions:  # a callee comes before its caller
            spans[fn.ordinal] = self._spans(fn, spans)
            if fn.recursive:
                iterations.append((fn.name, *spans[fn.ordinal]))
        total = sum(len(fn.states) for fn in functions) + 1
        return Timing(max(1, k), c, tuple(kinds), tuple(iterations), total)

    @staticmethod
    def _successors(fn: Function, index: int) -> list[int]:
        state = fn.states[index]
        following = state.next
        if isinstance(following, Enter):
            return [index + 1] if following.site is not None else []
        if isinstance(following, Return):
            return []
        targets = [following.yes, following.no] if isinstance(following, Choice) else [following]
        return [index + 1 if target == FALL else fn.labels[target] for target in targets]

    def _hardware_states(self, fn: Function, walking: set[str], known: dict[str, int]) -> int:
        """The most hardware-only states on any path through code outside a walk. Edges only go
        forward there, so one pass from the last state back finds it."""
        if fn.ordinal in walking:
            return 0  # all of it is in k
        if fn.ordinal in known:
            return known[fn.ordinal]
        best = [0] * len(fn.states)
        for index in range(len(fn.states) - 1, -1, -1):
            state = fn.states[index]
            own = 1 if state.hardware else 0
            if isinstance(state.next, Enter):
                own += self._hardware_states(state.next.callee, walking, known)
            after = [best[s] for s in self._successors(fn, index)]
            best[index] = own + (max(after) if after else 0)
        known[fn.ordinal] = best[0]
        return best[0]

    def _spans(self, fn: Function, known: dict[str, tuple[Span, Span]]) -> tuple[Span, Span]:
        """Cycles along each kind of path from the entry: those that come back to it (an iteration),
        and those that leave. Edges within a function only go forward, so this is one pass."""
        def join(spans: list[Span]) -> Optional[Span]:
            spans = [s for s in spans if s is not None]
            if not spans:
                return None
            return Span(min(s.low for s in spans), max(s.high for s in spans), any(s.walks for s in spans))

        def plus(span: Optional[Span], other: Span) -> Optional[Span]:
            if span is None:
                return None
            return Span(span.low + other.low, span.high + other.high, span.walks or other.walks)

        count = len(fn.states)
        loop: list[Optional[Span]] = [None] * count
        leave: list[Optional[Span]] = [None] * count
        for index in range(count - 1, -1, -1):
            state = fn.states[index]
            successors = self._successors(fn, index)
            cost = Span(1, 1)
            if isinstance(state.next, Enter) and state.next.callee is fn:
                loop[index] = cost
                continue
            if isinstance(state.next, Enter):
                callee = state.next.callee
                cost = plus(cost, known[callee.ordinal][1])
                if callee.recursive:
                    cost = Span(cost.low, cost.high, True)
                if state.next.site is None:
                    leave[index] = cost
                    continue
            if isinstance(state.next, Return):
                leave[index] = cost
                continue
            loop[index] = plus(join([loop[s] for s in successors]), cost)
            leave[index] = plus(join([leave[s] for s in successors]), cost)
        return (loop[0] or Span(0, 0), leave[0] or Span(0, 0))

    def render(self, handler: Function, module: str, bound: Bound, cells: int,
               initial: dict[str, str]) -> Translation:
        functions = self._reachable(handler)
        kept = {fn.ordinal for fn in functions}
        for fn in functions:  # a function the handler never reaches is not in the machine
            fn.sites = [site for site in fn.sites if site.caller in kept]
        site = Site("IDLE", finished=True, caller=handler.ordinal)
        results = self._results(handler.ret)
        names = {len(results) - 2: "out_item", len(results) - 1: "free_out"}
        site.results = [(names.get(position, ""), self._width(member) if position in names else 0)
                        for position, member in enumerate(results)]
        handler.sites.append(site)
        for fn in reversed(functions):  # a tail call returns where its caller would
            for callee in fn.tail_calls:
                for inherited in fn.sites:
                    if inherited not in callee.sites:
                        callee.sites.append(inherited)
        timing = self._timing(handler, functions)
        for fn in functions:
            for state in fn.states:
                if state.reads and state.writes:
                    raise RTError(f"{state.name} ({state.comment}) both reads and writes the cell store")
        memories = {name: _hex_lines(data, self._array_of(ty)) for (name, ty), data in
                    ((slot, initial[slot[0]]) for slot in self.slots if slot[0] in initial)}
        text = self._text(handler, functions, module, cells, timing, memories, bound)
        return Translation(text, module, bound, timing, cells, self.pointer_bits, memories)

    def _text(self, handler: Function, functions: list[Function], module: str, cells: int,
              timing: Timing, memories: dict, bound: Bound) -> str:
        p = self.pointer_bits
        states = [state.name for fn in functions for state in fn.states]
        state_bits = _bits_for(len(states) + 1)
        out: list[str] = []
        add = out.append
        add("// Generated by src/rtverilog.py from the checked RT stream. Do not edit.")
        add(f"// {module}: {timing.states} states, one RT instruction each but for what is marked.")
        add(f"// cycles(n) <= {timing.k}*W(n) + {timing.c}, with W(n) = {_linear(bound.c, bound.d)}, counted from the edge")
        add("// that samples start to the edge that raises done.")
        add("`timescale 1ns/1ns")
        add("`default_nettype none")
        add(f"module {module} (")
        add("    input  wire clk,")
        add("    input  wire rst,")
        add("    input  wire start,          // the edge that samples this starts the count")
        add(f"    input  wire [{p - 1}:0] item_in,")
        add(f"    input  wire [{p - 1}:0] free_in,")
        add("    output reg  done,           // and the edge that raises this ends it")
        add("    output wire busy,")
        add(f"    output reg  [{p - 1}:0] out_item,")
        add(f"    output reg  [{p - 1}:0] free_out,")
        add("    // the cell store from outside, between items: ingress and egress are not built yet")
        add("    input  wire host_en,")
        add(f"    input  wire [{CELL_BYTES - 1}:0] host_we,")
        add(f"    input  wire [{p - 1}:0] host_addr,")
        add(f"    input  wire [{CELL_BITS - 1}:0] host_din,")
        add(f"    output wire [{CELL_BITS - 1}:0] host_dout")
        add(");")
        add(f"    localparam [{state_bits - 1}:0] IDLE = {state_bits}'d0;")
        for position, name in enumerate(states, 1):
            add(f"    localparam [{state_bits - 1}:0] {name} = {state_bits}'d{position};")
        add(f"    reg [{state_bits - 1}:0] state;")
        add("    assign busy = state != IDLE;")
        add("")
        add("    // Every RT register is a flip-flop bank. A function's are static: nothing recurses but")
        add("    // by jumping to its own entry, so no function is ever active twice.")
        bodies = {state.name: list(state.clocked) + self._transition(fn, index)
                  for fn in functions for index, state in enumerate(fn.states)}
        declared = []
        for fn in functions:
            for register, width in sorted(fn.widths.items()):
                if width:
                    declared.append((fn.flop(register), width))
            declared.extend(fn.temporaries)
            if fn.at is not None:
                declared.append((f"{fn.name}_root", p))
        entry = self._entry(handler)
        written_only = _written_only(declared, bodies, entry, functions)
        for fn in functions:
            for name, width in declared:
                if name not in written_only and (name.startswith(fn.name + "_r") or name.startswith(fn.name + "_t")):
                    add(f"    reg {_range(width)}{name};")
            if len(fn.sites) > 1:
                add(f"    reg [{state_bits - 1}:0] {fn.name}_ret;")
        kept = [f"{name}[{width - 1}:{low}]" for name, (low, width) in sorted(self.cut.items())
                if name not in written_only and any(name.startswith(fn.name + "_r") for fn in functions)]
        if kept:
            add("    // What a truncation throws away. RT says to compute it, and nothing reads it.")
            add(f"    wire unused_bits = &{{1'b0, {', '.join(kept)}}};")
        add("")
        add(f"    // The cell store: {cells} cells, and word 0, which is null and never a cell. One block")
        add("    // RAM: port A reads, and its output holds until the next read; port B writes, a byte")
        add("    // enable per byte. No state enables both, and a host that writes does not read, so a")
        add("    // clock never reads and writes one address. yosys proves that from this decode, and")
        add("    // needs no logic for what such a read should return.")
        add(f"    reg [{CELL_BITS - 1}:0] cells [0:{cells}];")
        add("    reg a_en;")
        add(f"    reg [{p - 1}:0] a_addr;")
        add(f"    reg [{CELL_BITS - 1}:0] a_dout;")
        add(f"    reg [{CELL_BYTES - 1}:0] b_we;")
        add(f"    reg [{p - 1}:0] b_addr;")
        add(f"    reg [{CELL_BITS - 1}:0] b_din;")
        add("    assign host_dout = a_dout;")
        add("    integer lane;")
        add("    always @(posedge clk) begin")
        add("        if (a_en) a_dout <= cells[a_addr];")
        add(f"        for (lane = 0; lane < {CELL_BYTES}; lane = lane + 1)")
        add("            if (b_we[lane]) cells[b_addr][8 * lane +: 8] <= b_din[8 * lane +: 8];")
        add("    end")
        for name, ty in self.slots:
            array = self._array_of(ty)
            assert array is not None
            width, address = 8 * _scalar_bytes(array[2]), _bits_for(array[1])
            add("")
            add(f"    // state slot {name}: its own block RAM, read only until `swap` exists")
            add(f"    reg {_range(width)}{name}_mem [0:{array[1] - 1}];")
            add(f"    reg {name}_en;")
            add(f"    reg {_range(address)}{name}_addr;")
            add(f"    reg {_range(width)}{name}_dout;")
            if name in memories:
                add(f'    initial $readmemh("{name}.mem", {name}_mem);')
            else:
                add(f"    integer {name}_word;")
                add(f"    initial for ({name}_word = 0; {name}_word < {array[1]}; {name}_word = {name}_word + 1)")
                add(f"        {name}_mem[{name}_word] = {width}'d0;")
            add(f"    always @(posedge clk) if ({name}_en) {name}_dout <= {name}_mem[{name}_addr];")
        add("")
        add("    // What each state drives on the memory ports, in the clock it is the state.")
        add("    always @* begin")
        add("        a_en = 1'b0;")
        add(f"        a_addr = {p}'d0;")
        add(f"        b_we = {CELL_BYTES}'d0;")
        add(f"        b_addr = {p}'d0;")
        add(f"        b_din = {CELL_BITS}'d0;")
        for name, ty in self.slots:
            array = self._array_of(ty)
            add(f"        {name}_en = 1'b0;")
            add(f"        {name}_addr = {_bits_for(array[1])}'d0;")
        add("        case (state)")
        add("            IDLE: begin")
        add(f"                a_en = host_en && host_we == {CELL_BYTES}'d0;")
        add("                a_addr = host_addr;")
        add("                b_we = host_we;")
        add("                b_addr = host_addr;")
        add("                b_din = host_din;")
        add("            end")
        for fn in functions:
            for state in fn.states:
                if state.ports:
                    add(f"            {state.name}: begin  // {state.comment}")
                    for line in state.ports:
                        add(f"                {line}")
                    add("            end")
        add("            default: ;")
        add("        endcase")
        add("    end")
        add("")
        add("    // The machine. Every <= in one state happens at one edge, from what held before it.")
        add("    always @(posedge clk) begin")
        add("        done <= 1'b0;")
        add("        if (rst) state <= IDLE;")
        add("        else case (state)")
        add("            IDLE: if (start) begin")
        for line in entry:
            add(f"                {line}")
        add("            end")
        for fn in functions:
            add("")
            header = f"fn {fn.name}"
            if fn.recursive:
                header += ", a loop"
            add(f"            // --- {header}")
            for state in fn.states:
                body = bodies[state.name]
                mark = " (a state RT does not charge for)" if state.hardware else ""
                add(f"            // {state.comment}{mark}")
                if len(body) == 1:
                    add(f"            {state.name}: {body[0]}")
                else:
                    add(f"            {state.name}: begin")
                    for line in body:
                        add(f"                {line}")
                    add("            end")
        add("            default: state <= IDLE;")
        add("        endcase")
        add("    end")
        add("endmodule")
        add("`default_nettype wire")
        return "\n".join(out) + "\n"

    def _entry(self, handler: Function) -> list[str]:
        """The handler's parameters, from the ports, at the edge that samples start."""
        entry = []
        for position, type_text in enumerate(handler.params):
            width = handler.widths[position]
            if self._array_of(type_text) is not None:
                if width:
                    entry.append(f"{handler.flop(position)} <= {width}'d{position};")
            else:
                port = "free_in" if position == len(handler.params) - 1 else "item_in"
                entry.append(f"{handler.flop(position)} <= {port};")
        entry.append(f"state <= {handler.states[0].name};")
        return entry

    def _state_after(self, fn: Function, index: int, target: str) -> str:
        position = index + 1 if target == FALL else fn.labels[target]
        return fn.states[position].name

    def _transition(self, fn: Function, index: int) -> list[str]:
        """The <= that choose the next state, with what a call or a return has to carry."""
        state = fn.states[index]
        following = state.next
        if isinstance(following, Choice):
            yes, no = (self._state_after(fn, index, target) for target in (following.yes, following.no))
            return [f"state <= ({following.condition}) ? {yes} : {no};"]
        if isinstance(following, Return):
            return self._returning(fn, following)
        if isinstance(following, Enter):
            callee = following.callee
            lines = []
            if len(callee.sites) > 1 and callee is not fn:
                if following.site is not None:
                    lines.append(f"{callee.name}_ret <= {following.site.state};")
                elif len(fn.sites) > 1:
                    lines.append(f"{callee.name}_ret <= {fn.name}_ret;")
                else:
                    lines.append(f"{callee.name}_ret <= {fn.sites[0].state};")
            lines.append(f"state <= {callee.states[0].name};")
            return lines
        return [f"state <= {self._state_after(fn, index, following)};"]

    def _delivery(self, site: Site, values: list) -> list[str]:
        lines = []
        for (expression, width), (target, wanted) in zip(values, site.results):
            if wanted:
                if width > wanted:
                    self.cut[expression] = (min(wanted, self.cut.get(expression, (wanted,))[0]), width)
                lines.append(f"{target} <= {_whole(expression, width, wanted)};")
        if site.finished:
            lines.append("done <= 1'b1;")
        return lines

    def _returning(self, fn: Function, following: Return) -> list[str]:
        if len(fn.sites) == 1:
            site = fn.sites[0]
            return self._delivery(site, following.values) + [f"state <= {site.state};"]
        lines = [f"case ({fn.name}_ret)"]
        for site in fn.sites:
            delivered = self._delivery(site, following.values)
            lines.append(f"    {site.state}: begin {' '.join(delivered)} end")
        lines += ["    default: ;", "endcase", f"state <= {fn.name}_ret;"]
        return lines


def _written_only(declared: list, bodies: dict, entry: list, functions: list) -> set[str]:
    """Registers RT writes and nothing reads, such as the null an `@` function's `ret` names when it
    returns its root. They need no flip-flops. Their states stay: RT charged for them."""
    dead: set[str] = set()
    while True:
        lines = [line for body in bodies.values() for line in body] + entry
        lines += [line for fn in functions for state in fn.states for line in state.ports]
        text = "\n".join(lines)
        found = {name for name, _width in declared if name not in dead
                 and len(re.findall(rf"\b{name}\b", text)) == len(re.findall(rf"\b{name} <= ", text))}
        if not found:
            return dead
        dead |= found
        for name in found:
            pattern = re.compile(rf"(^|(?<=\s)){name} <= [^;]*;\s*")
            for state_name, body in bodies.items():
                bodies[state_name] = [kept for kept in (pattern.sub("", line) for line in body) if kept.strip()]
            entry[:] = [kept for kept in (pattern.sub("", line) for line in entry) if kept.strip()]


def _range(width: int) -> str:
    return f"[{width - 1}:0] " if width > 1 else ""


def _scalar_bytes(type_text: str) -> int:
    return 1 if type_text == "bool" else SCALAR_BITS[type_text] // 8


def _hex_lines(data: bytes, array: tuple) -> str:
    size = _scalar_bytes(array[2])
    words = [int.from_bytes(data[at:at + size], "little") for at in range(0, len(data), size)]
    return "".join(f"{word:0{2 * size}x}\n" for word in words)


def translate(lines: Iterable[str], *, n_max: int = 216, k_out: int = 64, cells: Optional[int] = None,
              initial: Optional[dict[str, bytes]] = None) -> Translation:
    """The only way in. rtcheck reads every line first, and the bound has to pass its certificate, so
    nothing it refuses can come out as Verilog. RTError says what was refused."""
    stream = list(lines)
    checker = Checker()
    for line in stream:
        checker.feed(line)
    checker.finish()
    handlers = [f for f in checker.functions.values() if f.handler]
    if len(handlers) != 1:
        raise RTError(f"section 3.1: this machine serves exactly one handler, and the program "
                      f"declares {len(handlers)}")
    name = handlers[0].name
    bound = certify(checker, name, n_max)
    reserve = bound.a * n_max + bound.b
    largest_output = n_max + reserve
    count = cells if cells is not None else (K_IN * n_max + reserve + n_max + k_out * O_MAX * largest_output)
    if count < K_IN * n_max + reserve + n_max:
        raise RTError(f"section 3.4: {count} cells cannot hold ingress and one maximal transaction")
    pointer_bits = _bits_for(count + 1)  # word 0 is null
    emitter = Emitter(pointer_bits)
    for line in stream:
        emitter.feed(line)
    handler = next(f for f in emitter.functions.values() if f.handler)
    module = name + "_handler" if name in KEYWORDS else name
    return emitter.render(handler, module, bound, count, initial or {})


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rtverilog", description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("-o", dest="output", required=True)
    parser.add_argument("--n-max", type=int, default=216, help="the largest item, in cells")
    parser.add_argument("--k-out", type=int, default=64, help="the output queue's capacity")
    parser.add_argument("--cells", type=int, help="cells in the store, instead of section 3.4's C")
    parser.add_argument("--state", action="append", default=[], metavar="NAME=FILE",
                        help="initial contents of a state array, read at translation time")
    args = parser.parse_args(argv)
    if os.path.exists(args.output):
        os.unlink(args.output)  # a refusal must leave nothing to mistake for this build
    try:
        initial = {}
        for entry in args.state:
            name, _, path = entry.partition("=")
            with open(path, "rb") as handle:
                initial[name] = handle.read()
        with open(args.source, "rb") as handle:
            source = handle.read()
        translation = translate(rt_lines(io.BytesIO(source)), n_max=args.n_max, k_out=args.k_out,
                                cells=args.cells, initial=initial)
    except (RTError, Reject, Limit, FStarved, OSError) as failure:
        print(f"{args.source}: refused: {failure}", file=sys.stderr)
        return 1
    with open(args.output, "w") as handle:
        handle.write(translation.verilog)
    for name, text in translation.memories.items():
        with open(os.path.join(os.path.dirname(os.path.abspath(args.output)), f"{name}.mem"), "w") as handle:
            handle.write(text)
    print(translation.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
