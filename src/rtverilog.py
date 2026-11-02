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

A program is its own circuit: straight-line code is combinational logic that settles in one clock. The
emitter writes a state for each RT instruction first, and three things put another in front of one: a
field can only be read from port A's output, so an instruction that names a field of a cell whose word is
not there waits for a read; an array read needs a clock for the same reason; and locations of two cells in
one instruction need the first one latched, because there is one read port. Then every state that touches
no memory and does not branch is joined with the state after it, unless something else can enter that one,
and what it computed becomes a wire. So a clock ends only at a block-RAM access, a branch or jump, a call or
return, or a label another path reaches, and nothing is reordered.

Each state is a clock, charged to the first instruction in it. rtcheck runs again with those charges in
place of its steps, and the same potential method that bounds steps bounds clocks. See `Timing`.
"""
from __future__ import annotations

import argparse
import collections
import io
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

from .certify import certify, certify_turn
from .ring import FStarved, Reject
from .rtcheck import NO_WORK, Bound, Checker, RTError
from .rtlower import Limit, rt_lines

CELL_BITS = 128
CELL_BYTES = CELL_BITS // 8
K_IN = 1
O_MAX = 1
SCALAR_BITS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "bool": 1, "lit": 64, "be16": 16, "be32": 32, "be64": 64}
ENUM_BITS = 64  # an enum is one register: LFPL-O section 4.11
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
_HEADER = re.compile(r"(fn|handler|turn) (f\d+) (\w+)\((.*)\) : (\S+)(?: @(\d+))?$")
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
    clocked: list = field(default_factory=list)  # (register, expression): <= at the clock edge
    ports: list = field(default_factory=list)    # what this state drives on the memory ports
    next: Next = FALL
    hardware: bool = False  # a state in front of an instruction, which RT has no instruction for
    reads: bool = False
    writes: bool = False
    line: Optional[int] = None  # the instruction its clock is charged to, by its line in the stream


@dataclass
class Function:
    ordinal: str
    name: str
    params: list
    ret: str
    at: Optional[int]
    handler: bool
    turn: bool = False                             # the runtime's turn: a port for each parameter and result
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
    wires: list = field(default_factory=list)     # (name, width, expression): what a joined state computed

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
    """The clocks this machine takes, certified before anything runs.

    Count from the edge that samples `start` to the edge that raises `done`. Each state holds the machine
    for one clock, and none is entered but by a transition, so that count is the number of states entered.
    Every state is charged to one instruction: the first in it, or for a state in front of an instruction,
    that instruction. Every instruction of a state runs whenever the state does, since a state is
    straight-line code, so the charges add up to the clocks exactly, but for a state that only makes the
    moves before a join, which is charged to the instruction after the label, and that one may be reached
    from elsewhere too. rtcheck charges each instruction what it was charged here, in place of a step, and
    its certificate is then a bound in clocks: `clocks`, as c n + d.
    """
    clocks: Bound          # its c and d: clocks(n) <= c n + d
    iterations: tuple      # (function name, Span of an iteration, Span of leaving)
    states: int

    def bound(self, n: int) -> int:
        return self.clocks.c * n + self.clocks.d


@dataclass(frozen=True)
class Translation:
    verilog: str
    module: str
    bound: Bound
    timing: Timing
    cells: int
    pointer_bits: int
    memories: dict  # state slot -> hex text for $readmemh, to go beside the Verilog
    verdict: bool = False  # the handler returns an enum beside its record, on a port of its own

    def report(self) -> str:
        lines = [f"handler {self.module}: R(n) = {_linear(self.bound.a, self.bound.b)} cells, "
                 f"W(n) = {_linear(self.bound.c, self.bound.d)} steps, {self.cells} cells of "
                 f"{CELL_BITS} bits, {self.pointer_bits}-bit pointers, {self.timing.states} states"]
        for name, iteration, leaving in self.timing.iterations:
            lines.append(f"walk {name}: {iteration} clocks per iteration, {leaving} to leave")
        clocks = self.timing.clocks
        lines.append(f"clocks(n) <= {_linear(clocks.c, clocks.d)}, certified, from the edge that samples start to "
                     f"the edge that raises done")
        return "\n".join(lines)


def _linear(slope: int, constant: int) -> str:
    if not slope:
        return str(constant)
    return f"{slope} n + {constant}" if constant else f"{slope} n"


class Emitter:
    """Reads a stream rtcheck has accepted, and holds the machine until `render`."""

    def __init__(self, pointer_bits: int, header: Optional[int] = None) -> None:
        """`header` is where the record a turn hands its handler lives, if it does: its cells, from that address
        on, are registers, and so the handler's decision reads and writes no block RAM. See `_header_text`."""
        self.pointer_bits = pointer_bits
        self.header = header
        self.header_type: Optional[str] = None       # the record's type, once the handler names it
        self.header_cells: list[str] = []            # its constructors, a cell each, in the order its links go
        self.facts: dict[int, int] = {}              # a register that holds one of its cells: which
        self.links: dict[str, tuple[int, int, str]] = {}  # a constructor's `$next`: offset, size, the next type
        self.ctor_types: dict[str, str] = {}
        self.aliases: dict[str, str] = {}
        self.composite: dict[str, tuple] = {}
        self.tuples: dict[str, list[str]] = {}
        self.ctors: dict[str, Ctor] = {}
        self.enums: set[str] = set()  # types whose values are one register of ENUM_BITS
        self.slots: list[tuple[str, str]] = []
        self.functions: dict[str, Function] = {}
        self.current: list[str] = []
        self.fn: Optional[Function] = None
        self.loaded: Optional[int] = None  # the register whose cell's word is at port A's output
        self.pending: dict[int, int] = {}  # moves not yet made: a move costs no state of its own
        self.edges: dict[str, list[tuple[Optional[int], dict]]] = {}
        self.waiting_labels: list[str] = []
        self.waiting_sites: list[Site] = []
        self.latched: dict[str, tuple[str, int]] = {}
        self.unattributed: list[State] = []  # states in front of an instruction not yet reached
        self.open_path = False
        self.cut: dict[str, tuple[int, int]] = {}  # registers only read in part: (kept, width)
        self.lines: list[str] = []  # the stream, for rtcheck to read again with clocks for steps
        self.line = 0               # the line of the instruction being emitted

    # --- declarations

    def feed(self, line: str) -> None:
        self.lines.append(line)
        number = len(self.lines)  # as rtcheck counts them: every line fed, from 1
        text = line.strip()
        if not text:
            return
        if self.current:
            self.current.append((number, text))
            if text.split()[0] == "end":
                self._function(self.current)
                self.current = []
            return
        head, *parts = text.split()
        if head in ("fn", "handler", "turn"):
            self.current = [(number, text)]
        elif head == "alias":
            self.aliases[parts[0]] = parts[1]
        elif head == "array":
            self.composite[parts[0]] = ("array", int(parts[1]), parts[2])
        elif head == "fin":
            self.composite[parts[0]] = ("fin", int(parts[1]))
        elif head == "tuple":
            self.tuples[parts[0]] = parts[1:]
        elif head == "type" and parts[1] == "enum":
            self.enums.add(parts[0])
        elif head == "ctor":
            self.ctors[parts[0]] = Ctor(parts[2])
            self.ctor_types[parts[0]] = parts[1]
        elif head == "field":
            self.ctors[parts[0]].fields.append((int(parts[1]), int(parts[2]), parts[3]))
            if parts[4] == "$next":
                self.links[parts[0]] = (int(parts[1]), int(parts[2]), parts[3])
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
        if name in self.enums:
            return ENUM_BITS
        entry = self.composite.get(name)
        if entry is None:
            return self.pointer_bits  # a capability, or any value made of cells
        return self._array_bits() if entry[0] == "array" else _bits_for(entry[1])

    def _results(self, type_text: str) -> list[str]:
        return self.tuples.get(self._resolve(type_text), [type_text])

    # --- one function: first what each register is, then what each instruction does

    def _function(self, lines: list[tuple[int, str]]) -> None:
        match = _HEADER.match(lines[0][1])
        assert match is not None, lines[0]
        kind, ordinal, name, params, ret, at = match.groups()
        types = [part.partition(" : ")[2].strip() for part in _args(params)]
        fn = Function(ordinal, name, types, ret, int(at) if at else None, kind == "handler", kind == "turn")
        self.fn = fn
        body = [(_LABEL.match(line).group(1) if _LABEL.match(line) else None,
                 _LABEL.match(line).group(2) if _LABEL.match(line) else line, number)
                for number, line in lines[1:-1]]
        self._measure_registers(fn, [(label, instruction) for label, instruction, _number in body])
        self.loaded = None
        self.pending = {}
        self.edges = {}
        self.facts = {}
        if self.header is not None and fn.handler:
            self.header_type = self._resolve(types[-2])
            self.header_cells = self._cells_of(self.header_type)
            self.facts[len(types) - 2] = 0
        elif self.header is not None and fn.turn:
            if self.header_type is None:
                raise RTError("the turn comes before its handler, and the header is the handler's item")
            mine = [position for position, ty in enumerate(types) if self._resolve(ty) == self.header_type]
            if len(mine) != 1:
                raise RTError(f"the turn takes {len(mine)} records of the handler's item type, and the header is one")
            self.facts[mine[0]] = 0
        self.unattributed = []
        self.open_path = True
        for label, instruction, number in body:
            if label is not None:
                self._label(label)
            if instruction:
                self.line = number
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
            elif head in ("trunc", "bswap"):
                define(a[0], int(a[1]))
            elif head in ("pack", "enum"):
                define(a[0], 64)
            elif head == "etag":
                define(a[0], 8)
            elif head == "efield":
                define(a[0], 8 * self.ctors[a[2]].fields[int(a[3])][1])
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
            state.clocked.append((fn.flop(target), _whole(fn.flop(source), fn.widths[source], fn.widths[target])))
        self.pending = {}
        state.clocked.extend(clocked)
        fn.states.append(state)
        return state

    def _hardware(self, comment: str, **kwargs) -> None:
        """A state in front of an instruction, which RT has no instruction for. Its clock is that
        instruction's to pay."""
        self.unattributed.append(self._state(comment, hardware=True, **kwargs))

    def _step(self, kind: str, comment: str, **kwargs) -> State:
        """The state of an instruction, and the instruction the states just in front of it are charged to."""
        state = self._state(comment, **kwargs)
        for waiting in self.unattributed + [state]:
            waiting.line = self.line
        self.unattributed = []
        return state

    def _edge(self, label: str, loaded: Optional[int]) -> None:
        self.edges.setdefault(label, []).append((loaded, dict(self.facts)))

    def _label(self, label: str) -> None:
        incoming = self.edges.pop(label, [])
        if self.open_path:
            if self.pending:  # a label has every register in its own flip-flop on every edge
                self._hardware("the moves before a join")
            incoming.append((self.loaded, dict(self.facts)))
        words = [loaded for loaded, _facts in incoming]
        self.loaded = words[0] if words and all(i == words[0] for i in words) else None
        known = [facts for _loaded, facts in incoming]
        self.facts = {register: cell for register, cell in (known[0].items() if known else ())
                      if all(facts.get(register) == cell for facts in known)}
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

    def _cells_of(self, record: str) -> list[str]:
        """A record's constructors, a cell each, in the order its `$next` links go."""
        cells, ty = [], record
        while True:
            ctor = next(c for c, owner in self.ctor_types.items() if self._resolve(owner) == ty)
            cells.append(ctor)
            if ctor not in self.links:
                return cells
            ty = self._resolve(self.links[ctor][2])

    def _header_word(self, cell: int) -> str:
        assert self.header is not None
        return f"header_{self.header + cell}"

    def _in_header(self, text: str) -> Optional[tuple[int, int, int]]:
        """The header cell, offset and size of a location the emitter knows is in the header."""
        where = self._location(text.rstrip("!"))
        if where is None or where[0] not in self.facts:
            return None
        return self.facts[where[0]], where[1], where[2]

    def _locations(self, texts: Iterable[str]) -> None:
        """Everything an instruction names in memory must be readable in its one clock. With one
        read port, that is one word. A location of any other cell is latched first, starting with
        the cell already at the port, since that one needs no read. The header's are registers."""
        fn = self.fn
        assert fn is not None
        locations = [text for text in texts if self._location(text) and not self._in_header(text)]
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
                               clocked=[(name, _slice("a_dout", 8 * where[1], width, width))])
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
        header = self._in_header(text)
        if header is not None:
            return _slice(self._header_word(header[0]), 8 * header[1], 8 * header[2], want)
        where = self._location(text)
        if where is not None:
            return _slice("a_dout", 8 * where[1], 8 * where[2], want)
        register = int(_REG.match(text).group(1))
        if register in self.facts:  # one of the header's cells: its address, which nothing changes
            assert self.header is not None
            return _pad(f"{self.pointer_bits}'d{self.header + self.facts[register]}", self.pointer_bits, want)
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
        if register in self.facts:
            return self.pointer_bits
        return fn.widths[self.pending.get(register, register)]

    def _set(self, text: str, expression: str, width: int) -> tuple[str, str]:
        """`text` <= an expression of `width` bits. This is never a cell's capability, so it cannot
        be the register `loaded` names: those come from a dispatch, a move or a call."""
        fn = self.fn
        assert fn is not None
        register = int(_REG.match(text).group(1))
        self.pending.pop(register, None)
        self.facts.pop(register, None)
        return fn.flop(register), _pad(expression, width, fn.widths[register])

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
        real = self.pending.pop(source, source)
        self.pending[target] = real
        if source in self.facts:
            self.facts[target] = self.facts.pop(source)
        else:
            self.facts.pop(target, None)
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

    def i_bswap(self, text: str, a: list[str]) -> None:
        """Nothing but wires: the same bytes, the other way round."""
        width = int(a[1])
        self._locations(a[2:3])
        parts = [self._byte_of(a[2], index) for index in range(width // 8)]
        self._step("bswap", text, clocked=[self._set(a[0], "{" + ", ".join(parts) + "}", width)])

    def _byte_of(self, text: str, index: int) -> str:
        """One byte of an operand, counted from its least significant, as an expression of 8 bits."""
        fn = self.fn
        assert fn is not None
        text = text.rstrip("!")
        low = 8 * index
        if text in self.latched:
            return f"{self.latched[text][0]}[{low + 7}:{low}]"
        header = self._in_header(text)
        if header is not None:
            return _slice(self._header_word(header[0]), 8 * header[1] + low, 8, 8)
        where = self._location(text)
        if where is not None:
            return _slice("a_dout", 8 * where[1] + low, 8, 8)
        register = int(_REG.match(text).group(1))
        source = self.pending.get(register, register)
        name = fn.flop(source)
        if low >= fn.widths[source]:
            return "8'd0"
        self.cut[name] = (min(low + 8, self.cut.get(name, (low + 8,))[0]), fn.widths[source])
        return f"{name}[{low + 7}:{low}]"

    def i_pack(self, text: str, a: list[str]) -> None:
        widths = [int(w) for w in a[1].strip("<>").split()]
        self._locations(a[2:])
        parts = [self._value(operand, bits) for bits, operand in zip(widths, a[2:])]
        padding = 64 - sum(widths)
        if padding:
            parts.insert(0, f"{padding}'d0")
        expression = parts[0] if len(parts) == 1 else "{" + ", ".join(parts) + "}"
        self._step("pack", text, clocked=[self._set(a[0], expression, 64)])

    def i_enum(self, text: str, a: list[str]) -> None:
        """Wires again: the tag in the low byte, each field at its bytes, and zeros between."""
        ctor = self.ctors[a[1]]
        self._locations(a[2:])
        pieces = {0: (f"8'd{int(ctor.tag)}", 8)}
        for (off, size, _ty), operand in zip(ctor.fields, a[2:]):
            pieces[off] = (self._value(operand, 8 * size), 8 * size)
        parts = []
        top = ENUM_BITS
        for off in sorted(pieces, reverse=True):
            expression, width = pieces[off]
            gap = top - (8 * off + width)
            if gap:
                parts.append(f"{gap}'d0")
            parts.append(expression)
            top = 8 * off
        self._step("enum", text, clocked=[self._set(a[0], "{" + ", ".join(parts) + "}", ENUM_BITS)])

    def i_etag(self, text: str, a: list[str]) -> None:
        self._step("etag", text, clocked=[self._set(a[0], self._value(a[1], 8), 8)])

    def i_efield(self, text: str, a: list[str]) -> None:
        """A slice of the register: the field's bytes, wherever its constructor put them."""
        fn = self.fn
        assert fn is not None
        off, size, _ty = self.ctors[a[2]].fields[int(a[3])]
        register = int(_REG.match(a[1]).group(1))
        source = self.pending.get(register, register)
        expression = f"{fn.flop(source)}[{8 * (off + size) - 1}:{8 * off}]"
        self._step("efield", text, clocked=[self._set(a[0], expression, 8 * size)])

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

    def i_read(self, text: str, a: list[str]) -> None:
        self._locations(a[1:2])
        width = 8 * self._location(a[1])[2]
        self._step("read", text, clocked=[self._set(a[0], self._value(a[1], width), width)])

    def i_take(self, text: str, a: list[str]) -> None:
        """A link of the header is the next of its cells: the links between them are fixed, see `_header_write`."""
        self._locations(a[1:2])
        width = self.pointer_bits
        header = self._in_header(a[1])
        link = self.links.get(self.header_cells[header[0]]) if header is not None else None
        following = header is not None and link is not None and link[:2] == header[1:]
        self._step("take", text, clocked=[self._set(a[0], self._value(a[1], width), width)])
        if following:
            assert header is not None
            self.facts[int(_REG.match(a[0]).group(1))] = header[0] + 1

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

    def _header_write(self, cell: int, ctor_id: str, tag: str, sources: list[str]) -> list[tuple[str, str]]:
        """A build of a header cell: its word, with what the build stores in place of what was there, in the
        clock of the instructions around it. A link is never stored, so the cells stay in their order."""
        word = self._header_word(cell)
        kept, stored = (1 << CELL_BITS) - 1, []
        for source, (offset, size, _type) in zip(sources, self.ctors[ctor_id].fields):
            if source in ("=", "*") or source.endswith("!"):
                continue
            if self.links.get(ctor_id, (None, None))[:2] == (offset, size):
                raise RTError(f"a build stores a link of the header's cell {cell}, whose links are fixed")
            width = 8 * size
            inner, computed = _INNER.match(source), _FUNC.match(source)
            if inner:
                value = _slice(word, 8 * int(inner.group(1)), 8 * int(inner.group(2)), width)
            elif computed:
                name, old_offset, old_size, operand = computed.groups()
                value = (f"({_slice(word, 8 * int(old_offset), 8 * int(old_size), width)} {OPERATORS[name]} "
                         f"{self._value(operand, width)})")
            else:
                value = self._value(source, width)
            kept &= ~(((1 << width) - 1) << (8 * offset))
            stored.append(f"({_pad(value, width, CELL_BITS)} << {8 * offset})" if offset else _pad(value, width, CELL_BITS))
        if tag == "write":
            raise RTError("a header cell is a record's, and a record has no tag")
        if not stored:
            return []  # every field stayed
        return [(word, " | ".join([f"({word} & {CELL_BITS}'h{kept:032x})", *stored]))]

    def i_build(self, text: str, a: list[str]) -> None:
        p = int(_REG.match(a[0]).group(1))
        sources = a[3:]
        if p in self.facts:
            self._locations(sources)
            self._step("build", text, clocked=self._header_write(self.facts[p], a[1], a[2], sources))
            return
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
        if p in self.facts:
            raise RTError("a test of a header cell: the header is a record, and nothing tests one")
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
        if p in self.facts:
            raise RTError("a test of a header cell: the header is a record, and nothing tests one")
        self._at_port(p)
        self._edge(a[2], p)
        self._edge(a[3], p)
        self._step("br.tag", text, next_state=Choice(f"a_dout[7:0] == 8'd{a[1]}", a[2], a[3]))
        self.open_path = False

    # --- calls and exits

    def _parameters(self, callee: Function, operands: list[str]) -> list[tuple[str, str]]:
        """<= on every parameter, in one clock, each from what it was before the clock: so a swap,
        or any other cycle, is written out as it is and needs no scratch register."""
        fn = self.fn
        assert fn is not None
        writes = []
        for position, operand in enumerate(operands):
            width = callee.widths[position]
            if not width:
                continue
            target = callee.flop(position)
            value = self._value(operand, width)
            if value != target:
                writes.append((target, value))
        if callee.at is not None and callee is not fn:
            writes.append((f"{callee.name}_root", self._value(operands[callee.at], self.pointer_bits)))
        return writes

    def i_call(self, text: str, a: list[str]) -> None:
        fn = self.fn
        assert fn is not None
        callee = self.functions[a[1]]
        operands = a[2:]
        header = self.header is not None and callee.handler
        if header and self.facts.get(int(_REG.match(operands[-2]).group(1)) if _REG.match(operands[-2]) else -1) != 0:
            raise RTError("the turn gives its handler something other than the header, whose cells are registers")
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
            self.facts.pop(register, None)
        if header:  # it returns what it was given: see i_ret
            self.facts[int(_REG.match(names[-2]).group(1))] = 0
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
        if p in self.facts:
            raise RTError("tailmod of a header cell: nothing walks the header, whose cells are registers")
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
            for position, operand in enumerate(a):
                if operand == "#0":  # a null: the one pointer no cell has
                    values.append((f"{self.pointer_bits}'d0", self.pointer_bits))
                    continue
                register = int(_REG.match(operand).group(1))
                if register in self.facts:
                    values.append((self._value(operand, self.pointer_bits), self.pointer_bits))
                    continue
                source = self.pending.get(register, register)
                values.append((fn.flop(source), fn.widths[source]))
        if self.header is not None and (fn.handler or fn.turn):
            self._gives_the_header_back(fn, a)
        self._step("ret", text, next_state=Return(values))
        self.open_path = False

    def _header_text(self) -> list[str]:
        """The header's cells, as registers behind the same two ports as the block RAM. The ports decode their
        addresses, so what reaches a header cell through them, a host's write or a read of a pointer the emitter
        could not place, reads and writes the register, and nothing is unsound, only slower. What the emitter
        could place reads and writes the registers directly, in the clock of the code around it."""
        assert self.header is not None
        p, first, last = self.pointer_bits, self.header, self.header + len(self.header_cells) - 1
        words = [self._header_word(cell) for cell in range(len(self.header_cells))]
        out = [f"    // The header: cells {first} to {last}, the record a turn gives its handler, as registers. A "
               f"block-RAM word", "    // at each of those addresses is never read or written."]
        out.append(f"    wire header_a = a_addr >= {p}'d{first} && a_addr <= {p}'d{last};")
        out.append(f"    wire header_b = b_addr >= {p}'d{first} && b_addr <= {p}'d{last};")
        bits = ", ".join(f"{{8{{b_we[{lane}]}}}}" for lane in reversed(range(CELL_BYTES)))
        out.append(f"    wire [{CELL_BITS - 1}:0] b_bits = {{{bits}}};")
        out.append(f"    reg [{CELL_BITS - 1}:0] cells_dout, header_dout;")
        out.append("    reg header_hit;")
        out.append(f"    wire [{CELL_BITS - 1}:0] a_dout = header_hit ? header_dout : cells_dout;")
        out.append("    assign host_dout = a_dout;")
        out.append(f"    wire [{CELL_BITS - 1}:0] header [{first}:{last}];  // for a testbench to look at")
        for cell, word in enumerate(words):
            out.append(f"    assign header[{first + cell}] = {word};")
        out.append("    always @(posedge clk) begin")
        out.append("        if (a_en) cells_dout <= cells[a_addr];")
        out.append(f"        for (lane = 0; lane < {CELL_BYTES}; lane = lane + 1)")
        out.append("            if (b_we[lane] && !header_b) cells[b_addr][8 * lane +: 8] <= b_din[8 * lane +: 8];")
        out.append("    end")
        choice = f"{CELL_BITS}'d0"
        for cell, word in reversed(list(enumerate(words))):
            choice = f"a_addr == {p}'d{first + cell} ? {word} : {choice}"
        out.append(f"    always @(posedge clk) if (a_en) begin header_hit <= header_a; header_dout <= {choice}; end")
        return out

    def _gives_the_header_back(self, fn: Function, operands: list[str]) -> None:
        """The handler returns the record it was given, and the turn returns ingress's: so on the next turn it is
        the header again, at the same address, and what this emitter knows of it holds by induction."""
        members = [self._resolve(member) for member in self._results(fn.ret)]
        position = len(members) - 2 if fn.handler else members.index(self.header_type)
        register = _REG.match(operands[position])
        if register is None or self.facts.get(int(register.group(1))) != 0:
            raise RTError(f"{fn.name} returns something other than the header as its {position}th result, and "
                          f"the header's cells are registers that stay where they are")

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

    def _timing(self, functions: list[Function], clocks: Bound) -> Timing:
        spans: dict[str, tuple[Span, Span]] = {}
        iterations = []
        for fn in functions:  # a callee comes before its caller
            spans[fn.ordinal] = self._spans(fn, spans)
            if fn.recursive:
                iterations.append((fn.name, *spans[fn.ordinal]))
        total = sum(len(fn.states) for fn in functions) + 1
        return Timing(clocks, tuple(iterations), total)

    def _clocks(self, handler: Function, costs: collections.Counter, n_max: Optional[int],
                sizes: Optional[dict]) -> Bound:
        """The certificate again, with every instruction charged the clocks this machine spends on it in place
        of its step: the same inequalities, and other numbers in them."""
        checker = Checker(costs)
        for line in self.lines:
            checker.feed(line)
        checker.finish()
        if handler.turn:
            assert sizes is not None
            return certify_turn(checker, handler.name, sizes)
        assert n_max is not None
        return certify(checker, handler.name, n_max)

    @staticmethod
    def _goes_on(state: State) -> bool:
        """The next state can share its clock: it falls through, and it enables no read, whose word would come
        back only in the next clock. A write may: nothing after it can read that cell without a read of its own."""
        return state.next == FALL and not any(line.startswith("a_") or "_en =" in line for line in state.ports)

    def _join(self, fn: Function, entered: set) -> None:
        """Straight-line code as one clock: each state is joined with the one after it while it goes on, the
        next is entered from nowhere else, and the clock touches memory once at most. Nothing moves: the states
        are taken in the order they were written."""
        widths = {fn.flop(register): width for register, width in fn.widths.items() if width}
        widths.update(dict(fn.temporaries))
        widths.update({self._header_word(cell): CELL_BITS for cell in range(len(self.header_cells))})
        joined: list[State] = []
        renumber: dict[int, int] = {}
        index = 0
        while index < len(fn.states):
            first = index
            touched = bool(fn.states[index].ports)
            while (index + 1 < len(fn.states) and self._goes_on(fn.states[index])
                   and fn.states[index + 1].name not in entered and not (touched and fn.states[index + 1].ports)):
                index += 1
                touched = touched or bool(fn.states[index].ports)
            renumber[first] = len(joined)
            run = fn.states[first:index + 1]
            joined.append(run[0] if len(run) == 1 else self._one_clock(fn, run, widths))
            index += 1
        fn.labels = {label: renumber[position] for label, position in fn.labels.items()}
        fn.states = joined

    def _one_clock(self, fn: Function, run: list[State], widths: dict) -> State:
        """States that run one after another with at most one memory access between them, as one. A register
        gets the last value written to it, and a value one of them computes that a later one reads is a wire."""
        values: dict[str, str] = {}  # each register written so far, and the expression of its newest value
        carried: dict[str, str] = {}  # the wire carrying that value, once something later in the clock reads it

        def read(text: str) -> str:
            for target, expression in values.items():
                if target not in carried and re.search(rf"\b{re.escape(target)}\b", text):
                    carried[target] = f"{fn.name}_w{len(fn.wires)}"
                    fn.wires.append((carried[target], widths[target], expression))
                    if target in self.cut:  # what is read of it in part is read of the wire now
                        self.cut[carried[target]] = self.cut[target]
            return _renamed(text, carried)

        ports: list[str] = []
        for state in run:
            ports += [read(line) for line in state.ports]
            writes = [(target, read(expression)) for target, expression in state.clocked]
            for target, expression in writes:  # every expression of a state reads what held before it
                values[target] = expression
                carried.pop(target, None)
        following = run[-1].next
        if isinstance(following, Choice):
            following = Choice(read(following.condition), following.yes, following.no)
        elif isinstance(following, Return):
            following = Return([(read(expression), width) for expression, width in following.values])
        final = [(target, carried.get(target, expression)) for target, expression in values.items()]
        reads, writes_memory = any(s.reads for s in run), any(s.writes for s in run)
        assert not (reads and writes_memory), [s.comment for s in run]
        return State(run[0].name, "; ".join(state.comment for state in run), final, ports, following,
                     all(s.hardware for s in run), reads, writes_memory, run[0].line)

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

    def render(self, handler: Function, module: str, bound: Bound, cells: int, initial: dict[str, str], *,
               n_max: Optional[int] = None, sizes: Optional[dict] = None) -> Translation:
        """The machine, and its bound in clocks: a handler's for items of up to `n_max` cells, a turn's with
        `sizes` saying how many cells each sized parameter holds, as src/certify.py's certify_turn."""
        functions = self._reachable(handler)
        kept = {fn.ordinal for fn in functions}
        for fn in functions:  # a function the handler never reaches is not in the machine
            fn.sites = [site for site in fn.sites if site.caller in kept]
        site = Site("IDLE", finished=True, caller=handler.ordinal)
        results = self._results(handler.ret)
        names = {len(results) - 2: "out_item", len(results) - 1: "free_out"}
        self.verdict = len(results) == len(self.slots) + 3 and not handler.turn
        if self.verdict:
            names[len(results) - 3] = "verdict_out"
        if handler.turn:  # a turn's every result goes out on a port of its own
            names = {position: f"res{position}" for position in range(len(results))}
        site.results = [(names.get(position, ""), self._width(member) if position in names else 0)
                        for position, member in enumerate(results)]
        handler.sites.append(site)
        for fn in reversed(functions):  # a tail call returns where its caller would
            for callee in fn.tail_calls:
                for inherited in fn.sites:
                    if inherited not in callee.sites:
                        callee.sites.append(inherited)
        for fn in functions:
            for state in fn.states:
                if state.reads and state.writes:
                    raise RTError(f"{state.name} ({state.comment}) both reads and writes the cell store")
        entered = {f.states[index].name for f in functions for index in f.labels.values()}
        entered |= {site.state for f in functions for site in f.sites}
        for fn in functions:
            self._join(fn, entered)
        costs = collections.Counter(state.line for fn in functions for state in fn.states)
        assert None not in costs, "a state is charged to no instruction"
        timing = self._timing(functions, self._clocks(handler, costs, n_max, sizes))
        memories = {name: _hex_lines(data, self._array_of(ty)) for (name, ty), data in
                    ((slot, initial[slot[0]]) for slot in self.slots if slot[0] in initial)}
        text = self._text(handler, functions, module, cells, timing, memories, bound)
        return Translation(text, module, bound, timing, cells, self.pointer_bits, memories, self.verdict)

    def _text(self, handler: Function, functions: list[Function], module: str, cells: int,
              timing: Timing, memories: dict, bound: Bound) -> str:
        p = self.pointer_bits
        states = [state.name for fn in functions for state in fn.states]
        state_bits = _bits_for(len(states) + 1)
        out: list[str] = []
        add = out.append
        add("// Generated by src/rtverilog.py from the checked RT stream. Do not edit.")
        add(f"// {module}: {timing.states} states, a clock each. Straight-line code is one state: a clock ends only at a")
        add("// block-RAM access, a branch or jump, a call or return, or a label another path reaches.")
        add(f"// clocks(n) <= {_linear(timing.clocks.c, timing.clocks.d)}, certified as W(n) = {_linear(bound.c, bound.d)} "
            f"steps is, counted")
        add("// from the edge that samples start to the edge that raises done.")
        add("`timescale 1ns/1ns")
        add("`default_nettype none")
        add(f"module {module} (")
        add("    input  wire clk,")
        add("    input  wire rst,")
        add("    input  wire start,          // the edge that samples this starts the count")
        if handler.turn:
            for position, type_text in enumerate(handler.params):
                add(f"    input  wire {_range(max(1, self._width(type_text)))}arg{position},")
        else:
            add(f"    input  wire [{p - 1}:0] item_in,")
            add(f"    input  wire [{p - 1}:0] free_in,")
        add("    output reg  done,           // and the edge that raises this ends it")
        add("    output wire busy,")
        if handler.turn:
            for position, member in enumerate(self._results(handler.ret)):
                add(f"    output reg  {_range(max(1, self._width(member)))}res{position},")
        else:
            add(f"    output reg  [{p - 1}:0] out_item,")
            add(f"    output reg  [{p - 1}:0] free_out,")
        if self.verdict:
            add(f"    output reg  [{ENUM_BITS - 1}:0] verdict_out,  // which way out_item goes: section 4.11")
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
        bodies = {state.name: [f"{target} <= {expression};" for target, expression in state.clocked]
                  + self._transition(fn, index) for fn in functions for index, state in enumerate(fn.states)}
        wires = {name: expression for fn in functions for name, _width, expression in fn.wires}
        declared = []
        for fn in functions:
            for register, width in sorted(fn.widths.items()):
                if width:
                    declared.append((fn.flop(register), width))
            declared.extend(fn.temporaries)
            if fn.at is not None:
                declared.append((f"{fn.name}_root", p))
        entry = self._entry(handler)
        written_only, unread = _written_only(declared, bodies, entry, functions, wires)
        for fn in functions:
            for name, width in declared:
                if name not in written_only and (name.startswith(fn.name + "_r") or name.startswith(fn.name + "_t")):
                    add(f"    reg {_range(width)}{name};")
            if len(fn.sites) > 1:
                add(f"    reg [{state_bits - 1}:0] {fn.name}_ret;")
        if self.header is not None:
            add("    // The header's cells, a register each: see below where the ports reach them.")
            for cell in range(len(self.header_cells)):
                add(f"    reg [{CELL_BITS - 1}:0] {self._header_word(cell)};")
        add("")
        add(f"    // The cell store: {cells} cells, and word 0, which is null and never a cell. One block")
        add("    // RAM: port A reads, and its output holds until the next read; port B writes, a byte")
        add("    // enable per byte. No state enables both, and a host that writes does not read, so a")
        add("    // clock never reads and writes one address. yosys proves that from this decode, and")
        add("    // needs no logic for what such a read should return.")
        add(f"    reg [{CELL_BITS - 1}:0] cells [0:{cells}];")
        add("    reg a_en;")
        add(f"    reg [{p - 1}:0] a_addr;")
        add(f"    reg [{CELL_BYTES - 1}:0] b_we;")
        add(f"    reg [{p - 1}:0] b_addr;")
        add(f"    reg [{CELL_BITS - 1}:0] b_din;")
        add("    integer lane;")
        if self.header is None:
            add(f"    reg [{CELL_BITS - 1}:0] a_dout;")
            add("    assign host_dout = a_dout;")
            add("    always @(posedge clk) begin")
            add("        if (a_en) a_dout <= cells[a_addr];")
            add(f"        for (lane = 0; lane < {CELL_BYTES}; lane = lane + 1)")
            add("            if (b_we[lane]) cells[b_addr][8 * lane +: 8] <= b_din[8 * lane +: 8];")
            add("    end")
        else:
            out.extend(self._header_text())
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
        live = [(name, width, expression) for fn in functions for name, width, expression in fn.wires
                if name not in unread]
        if live:
            add("")
            add("    // What a state computes for a later instruction in the same clock: logic, settled within it.")
            for name, width, expression in live:
                add(f"    wire {_range(width)}{name} = {expression};")
        kept = [f"{name}[{width - 1}:{low}]" for name, (low, width) in sorted(self.cut.items())
                if name not in written_only and name not in unread
                and any(name.startswith((fn.name + "_r", fn.name + "_w")) for fn in functions)
                and (name in wires or any(name == declared_name for declared_name, _width in declared))]
        if kept:
            add("    // What a truncation throws away. RT says to compute it, and nothing reads it.")
            add(f"    wire unused_bits = &{{1'b0, {', '.join(kept)}}};")
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
        if self.header is not None:  # what the ports write to a header cell, the host's included
            add("        if (header_b) case (b_addr)")
            for cell in range(len(self.header_cells)):
                word = self._header_word(cell)
                add(f"            {p}'d{self.header + cell}: {word} <= ({word} & ~b_bits) | (b_din & b_bits);")
            add("            default: ;")
            add("        endcase")
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
                if handler.turn:
                    port = f"arg{position}"
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


def _written_only(declared: list, bodies: dict, entry: list, functions: list, wires: dict) -> tuple[set, set]:
    """Registers RT writes and nothing reads, such as the null an `@` function's `ret` names when it
    returns its root, or a value a joined state used in its own clock and nothing after it does. They need
    no flip-flops. And wires nothing reads, once those are gone. Their states stay: their clocks were charged."""
    dead: set[str] = set()
    unread: set[str] = set()
    while True:
        lines = [line for body in bodies.values() for line in body] + entry
        lines += [line for fn in functions for state in fn.states for line in state.ports]
        lines += [expression for name, expression in wires.items() if name not in unread]
        text = "\n".join(lines)
        # A write is a statement, so it starts a line or follows a space. `(r <= 94)` is a comparison, and a read.
        found = {name for name, _width in declared if name not in dead
                 and len(re.findall(rf"\b{name}\b", text)) == len(re.findall(rf"(?:^|(?<=\s)){name} <= ", text, re.M))}
        orphans = {name for name in wires if name not in unread and not re.search(rf"\b{name}\b", text)}
        if not found and not orphans:
            return dead, unread
        dead |= found
        unread |= orphans
        for name in found:
            pattern = re.compile(rf"(^|(?<=\s)){name} <= [^;]*;\s*")
            for state_name, body in bodies.items():
                bodies[state_name] = [kept for kept in (pattern.sub("", line) for line in body) if kept.strip()]
            entry[:] = [kept for kept in (pattern.sub("", line) for line in entry) if kept.strip()]


def _renamed(text: str, env: dict) -> str:
    """`text` with each register `env` names read as the value it was given earlier in the same clock."""
    if not env:
        return text
    names = "|".join(re.escape(name) for name in sorted(env, key=len, reverse=True))
    return re.sub(rf"\b({names})\b", lambda match: env[match.group(1)], text)


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
    return emitter.render(handler, module, bound, count, initial or {}, n_max=n_max)


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
