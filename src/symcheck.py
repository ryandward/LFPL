"""Does the assembly mean what the source means? An independent symbolic check.

`validate.py` only rejects redundant moves, so a dropped store passes it. This
module compares meaning. It executes the emitted NASM symbolically, evaluates the
source symbolically, and requires that every value leaving a function agrees: at
each `ret`, call and tail jump, each field of every constructor that was built must
hold what the source says it holds.

It shares no code with the compiler, which a test enforces by reading the imports:
it has its own source parser, its own layout rule and its own reading of x86. Two
separate mistakes would have to agree for a miscompile to get through. It runs on every
LFPL-O build, after the compiler has finished, so unlike the compiler it may build whole
syntax trees: it is an oracle over one finished program, and it is what LFPL-O section 8.4
means by Meaning.

A slot is the source's too: `slot S;` is 2048 bytes, and this has its own rule for where they are, which is the
hardware target's, slot k at 2048 k bytes into the cell store. `s[i]` is the byte there at i, or 0 at or past the
slot's length, which is bits 32 to 47 of its value. `len(s)` is those bits. A match on a Fin is its value being 0 or
one less than it, and `within(s, x)` is x being below the length or not.

A record that spans cells, and a field in network byte order, are the source's business, so this works out both
for itself. `unspan` rewrites the syntax tree into the chain of one-cell records the machine is really working on,
with a layout rule of its own, and a field of type Be16, Be32 or Be64 means a load that is swapped and a store of
what was swapped: `swap` is a term, and `swap(swap(x))` is x.

Terms and joins. A value is a canonical decision diagram: comparisons at the inner
nodes in one fixed order, and data at the leaves. Every operation distributes over
decisions, so two values that agree under every outcome of the comparisons are the
same object, however they were reached.

The merge rule. At a label, a location that holds the same term on every incoming
edge keeps it; otherwise it holds ite(edge condition, a, b). Because the form is
canonical, `if c then 7 else z` in the source compares directly against a register
that a branch filled on two paths. Within a straight-line block this is plain
symbolic execution, and nothing is merged.

What this assumes, and so cannot catch:
  * Linearity. Distinct pointer terms are treated as distinct cells, because Check
    guarantees it. A broken Check would not be noticed here.
  * Callees are uninterpreted: f(x) is checked against f's meaning when f itself is
    checked, and a recursive jump is taken to mean what the function means. That is
    partial correctness by induction, and it says nothing about termination.
  * Comparisons are independent propositions. That can only cause a false alarm on
    code relying on arithmetic facts between them, never a missed mismatch.
  * Encodings. A truncated immediate is NASM's business, not visible in the text.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

MASK64 = (1 << 64) - 1
NO_ATOM = 1 << 60
FIELD_SIZES = {"Int8": 1, "Int16": 2, "Int32": 4, "Int64": 8, "Bool": 1, "Be16": 2, "Be32": 4, "Be64": 8}
SLOT_BYTES, SLOT_SHIFT = 2048, 11  # a slot, and where slot k starts: 2048 k bytes into the cell store
WIRE_FIELDS = frozenset(("Be16", "Be32", "Be64"))
CELL_BYTES, BYTES_AHEAD_OF_THE_LINK = 16, 8


class TooLarge(Exception):
    """The diagrams outgrew the budget. Reported as inconclusive, never as a pass."""


# ------------------------------------------------------------------- terms


class Terms:
    """Hash-consed terms: two terms are equal exactly when their ids are equal.

    Leaves: ('c', n)  ('arg', i, size)  ('load', ptr, off, size)
            ('op', add|sub|imul|shl, width, x, y)  ('bit', and|or|xor, x, y)
            ('shr', x, count)  ('trunc', width, x)  ('call', f, args, size)
            ('struct', ctor, tag, cell, ((off, size, leaf), ...))
            ('setl', width, old, low)  ('undef', why)
            ('tuple', members)  ('member', leaf, i, size)  ('swap', width, x)
            ('aload', array, index, size): one element of an array, as it is at entry
    Decisions: ('ite', atom, hi, lo), where an atom is ('lt', x, y) or ('eq', x, y).
    Values are in unsigned integer terms, so a width only matters where it wraps.
    """

    def __init__(self, budget: int = 400_000):
        self.ids: dict[tuple, int] = {}
        self.nodes: list[tuple] = []
        self.budget = budget
        self.select_memo: dict[tuple, int] = {}
        self.lift_memo: dict[tuple, int] = {}
        self.ZERO = self.const(0)
        self.ONE = self.const(1)

    def intern(self, node: tuple) -> int:
        found = self.ids.get(node)
        if found is None:
            found = len(self.nodes)
            if found >= self.budget:
                raise TooLarge(f"more than {self.budget} terms")
            self.nodes.append(node)
            self.ids[node] = found
        return found

    def const(self, value: int) -> int:
        return self.intern(("c", value & MASK64))

    def undef(self, why: str) -> int:
        return self.intern(("undef", why))

    def is_const(self, term: int) -> bool:
        return self.nodes[term][0] == "c"

    def value(self, term: int) -> int:
        return self.nodes[term][1]

    # --- decisions

    def top(self, term: int) -> int:
        node = self.nodes[term]
        return node[1] if node[0] == "ite" else NO_ATOM

    def cof(self, term: int, atom: int, outcome: bool) -> int:
        node = self.nodes[term]
        if node[0] == "ite" and node[1] == atom:
            return node[2] if outcome else node[3]
        return term

    def _node(self, atom: int, hi: int, lo: int) -> int:
        return hi if hi == lo else self.intern(("ite", atom, hi, lo))

    def select(self, cond: int, a: int, b: int) -> int:
        """ite(cond, a, b), canonical. This is the whole of the merge rule."""
        if cond == self.ONE or a == b:
            return a
        if cond == self.ZERO:
            return b
        key = (cond, a, b)
        found = self.select_memo.get(key)
        if found is not None:
            return found
        atom = min(self.top(cond), self.top(a), self.top(b))
        result = self._node(
            atom,
            self.select(self.cof(cond, atom, True), self.cof(a, atom, True), self.cof(b, atom, True)),
            self.select(self.cof(cond, atom, False), self.cof(a, atom, False), self.cof(b, atom, False)))
        self.select_memo[key] = result
        return result

    def neg(self, cond: int) -> int:
        return self.select(cond, self.ZERO, self.ONE)

    def both(self, a: int, b: int) -> int:
        return self.select(a, b, self.ZERO)

    def either(self, a: int, b: int) -> int:
        return self.select(a, self.ONE, b)

    def lift(self, fn: Callable[..., int], *terms: int, key: Optional[tuple] = None) -> int:
        """Apply fn to leaves, distributing over every decision in its arguments.
        fn may itself return a decision, so results are rebuilt through select."""
        if not terms:
            return fn()
        memo_key = None
        if key is not None:
            memo_key = (key, terms)
            found = self.lift_memo.get(memo_key)
            if found is not None:
                return found
        atom = min(self.top(t) for t in terms)
        if atom == NO_ATOM:
            result = fn(*terms)
        else:
            hi = self.lift(fn, *(self.cof(t, atom, True) for t in terms), key=key)
            lo = self.lift(fn, *(self.cof(t, atom, False) for t in terms), key=key)
            result = self.select(self._node(atom, self.ONE, self.ZERO), hi, lo)
        if memo_key is not None:
            self.lift_memo[memo_key] = result
        return result

    def paths(self, term: int) -> list[tuple[int, int]]:
        """(condition, leaf) for each leaf of a value."""
        node = self.nodes[term]
        if node[0] != "ite":
            return [(self.ONE, term)]
        test = self._node(node[1], self.ONE, self.ZERO)
        found = [(self.both(test, c), leaf) for c, leaf in self.paths(node[2])]
        found += [(self.both(self.neg(test), c), leaf) for c, leaf in self.paths(node[3])]
        return found

    # --- data leaves

    def width(self, term: int) -> int:
        node = self.nodes[term]
        kind = node[0]
        if kind == "c":
            value = node[1]
            return 1 if value < 1 << 8 else 2 if value < 1 << 16 else 4 if value < 1 << 32 else 8
        if kind == "arg":
            return node[2]
        if kind == "load":
            return node[3]
        if kind == "aload":
            return node[3]
        if kind == "sload":  # a byte of a slot
            return 1
        if kind == "member":
            return node[3]
        if kind == "op":
            return node[2]
        if kind in ("trunc", "swap"):
            return node[1]
        if kind == "call":
            return node[3]
        if kind == "shr":
            return self.width(node[1])
        if kind == "bit":
            pair = (self.width(node[2]), self.width(node[3]))
            return min(pair) if node[1] == "and" else max(pair)
        return 8

    def trunc(self, width: int, term: int) -> int:
        if width >= 8 or self.width(term) <= width:
            return term
        node = self.nodes[term]
        kind = node[0]
        if kind == "c":
            return self.const(node[1] & ((1 << 8 * width) - 1))
        if kind == "op":  # the low bits of these depend only on the low bits
            return self.arith(node[1], width, node[3], node[4])
        if kind == "bit":
            return self.bitop(node[1], self.trunc(width, node[2]), self.trunc(width, node[3]))
        if kind == "setl" and width <= node[1]:
            return self.trunc(width, node[3])
        if kind == "load":  # x86-64 is little-endian: the low bytes are at the same address
            return self.load(node[1], node[2], width)
        if kind == "trunc":
            return self.trunc(width, node[2])
        return self.intern(("trunc", width, term))

    def arith(self, name: str, width: int, x: int, y: int) -> int:
        mask = (1 << 8 * width) - 1
        x = self.trunc(width, x)
        if name != "shl":
            y = self.trunc(width, y)
        if self.is_const(x) and self.is_const(y):
            a, b = self.value(x), self.value(y)
            table = {"add": a + b, "sub": a - b, "imul": a * b, "shl": a << b if b < 64 else 0}
            return self.const(table[name] & mask)
        if name in ("add", "imul") and y < x:
            x, y = y, x
        if (name in ("add", "sub", "shl") and y == self.ZERO) or (
                name == "imul" and y == self.ONE):
            return x
        if name == "add" and x == self.ZERO:
            return y
        return self.intern(("op", name, width, x, y))

    def bitop(self, name: str, x: int, y: int) -> int:
        """And, or and xor are associative and commutative, so a chain of one of them is its operands in one order:
        the same value built in any order is the same term. The constants come last, as one."""
        known: Optional[int] = None
        others: list[int] = []
        for leaf in self._chain(name, x) + self._chain(name, y):
            if self.is_const(leaf):
                v = self.value(leaf)
                known = v if known is None else {"and": known & v, "or": known | v, "xor": known ^ v}[name]
            else:
                others.append(leaf)
        if name == "xor":
            others = sorted(leaf for leaf in set(others) if others.count(leaf) % 2)
        else:
            others = sorted(set(others))
        if known == 0 and name == "and":
            return self.ZERO
        if known is not None and (known or not others):
            others.append(self.const(known))
        if not others:
            return self.ZERO
        result = others[0]
        for leaf in others[1:]:
            result = self.intern(("bit", name, result, leaf))
        return result

    def _chain(self, name: str, x: int) -> list[int]:
        node = self.nodes[x]
        if node[0] == "bit" and node[1] == name:
            return self._chain(name, node[2]) + self._chain(name, node[3])
        return [x]

    def shr(self, x: int, count: int) -> int:
        if count == self.ZERO:
            return x
        if self.is_const(x) and self.is_const(count):
            amount = self.value(count)
            return self.const(self.value(x) >> amount if amount < 64 else 0)
        return self.intern(("shr", x, count))

    def total_shift(self, name: str, width: int, x: int, count: int) -> int:
        """What a shift means in the source: 0 once the count reaches the width in bits.
        An unknown count becomes a decision, which the machine's guard has to match."""
        bits = width * 8
        if self.is_const(count) and self.value(count) >= bits:
            return self.ZERO
        shifted = self.arith("shl", width, x, count) if name == "shl" else self.shr(x, count)
        if self.is_const(count):
            return shifted
        return self.select(self.lt(count, self.const(bits)), shifted, self.ZERO)

    def setl(self, width: int, old: int, low: int) -> int:
        """A write to a subregister: the low `width` bytes change, and the rest do not. With nothing above
        them in the old value, that is the new low bytes."""
        return low if self.width(old) <= width else self.intern(("setl", width, old, low))

    def swap(self, width: int, x: int) -> int:
        """The bytes the other way round. Doing it twice is what the value was."""
        x = self.trunc(width, x)
        node = self.nodes[x]
        if node[0] == "c":
            return self.const(int.from_bytes(node[1].to_bytes(width, "little"), "big"))
        if node[0] == "swap" and node[1] == width:
            return node[2]
        return self.intern(("swap", width, x))

    def load(self, pointer: int, off: int, size: int) -> int:
        return self.intern(("load", pointer, off, size))

    def call(self, name: str, args: tuple, size: int) -> int:
        return self.intern(("call", name, args, size))

    def struct(self, ctor: str, tag: Optional[int], cell: int, fields: tuple) -> int:
        return self.intern(("struct", ctor, tag, cell, fields))

    def unchanged(self, cell: int, fields: tuple) -> bool:
        """Every field still holds what the cell held on entry, so nothing was built."""
        return all(leaf == self.load(cell, off, size) for off, size, leaf in fields)

    def lt(self, x: int, y: int) -> int:
        if x == y:
            return self.ZERO
        if self.is_const(x) and self.is_const(y):
            return self.ONE if self.value(x) < self.value(y) else self.ZERO
        return self._node(self.intern(("lt", x, y)), self.ONE, self.ZERO)

    def eq(self, x: int, y: int) -> int:
        if x == y:
            return self.ONE
        if self.is_const(x) and self.is_const(y):
            return self.ZERO
        if y < x:
            x, y = y, x
        return self._node(self.intern(("eq", x, y)), self.ONE, self.ZERO)

    # --- substitution and display

    def subst(self, term: int, sigma: dict, memo: dict) -> int:
        """Replace leaves by what a path's `x == constant` atoms say they equal."""
        if term in sigma:
            return sigma[term]
        found = memo.get(term)
        if found is not None:
            return found
        node = self.nodes[term]
        kind = node[0]
        sub = lambda t: self.subst(t, sigma, memo)  # noqa: E731
        if kind == "load":
            result = self.load(sub(node[1]), node[2], node[3])
        elif kind == "aload":
            result = self.intern(("aload", sub(node[1]), sub(node[2]), node[3]))
        elif kind == "tuple":
            result = self.intern(("tuple", tuple(sub(m) for m in node[1])))
        elif kind == "member":
            result = self.intern(("member", sub(node[1]), node[2], node[3]))
        elif kind == "op":
            result = self.arith(node[1], node[2], sub(node[3]), sub(node[4]))
        elif kind == "bit":
            result = self.bitop(node[1], sub(node[2]), sub(node[3]))
        elif kind == "shr":
            result = self.shr(sub(node[1]), sub(node[2]))
        elif kind == "trunc":
            result = self.trunc(node[1], sub(node[2]))
        elif kind == "swap":
            result = self.swap(node[1], sub(node[2]))
        elif kind == "call":
            result = self.call(node[1], tuple(sub(a) for a in node[2]), node[3])
        elif kind == "struct":
            result = self.struct(node[1], node[2], sub(node[3]),
                                 tuple((o, s, sub(f)) for o, s, f in node[4]))
        elif kind == "setl":
            result = self.setl(node[1], sub(node[2]), sub(node[3]))
        else:
            result = term
        memo[term] = result
        return result

    def show(self, term: int) -> str:
        node = self.nodes[term]
        kind = node[0]
        if kind == "c":
            return str(node[1])
        if kind == "arg":
            return f"arg{node[1]}"
        if kind == "load":
            return f"[{self.show(node[1])}+{node[2]}]"
        if kind == "aload":
            return f"{self.show(node[1])}[{self.show(node[2])}]"
        if kind == "tuple":
            return "(" + ", ".join(self.show(m) for m in node[1]) + ")"
        if kind == "member":
            return f"{self.show(node[1])}.{node[2]}"
        if kind == "op":
            return f"{node[1]}{node[2] * 8}({self.show(node[3])}, {self.show(node[4])})"
        if kind == "bit":
            return f"{node[1]}({self.show(node[2])}, {self.show(node[3])})"
        if kind == "shr":
            return f"shr({self.show(node[1])}, {self.show(node[2])})"
        if kind == "trunc":
            return f"low{node[1] * 8}({self.show(node[2])})"
        if kind == "swap":
            return f"swap{node[1] * 8}({self.show(node[2])})"
        if kind == "call":
            return f"{node[1]}({', '.join(self.show(a) for a in node[2])})"
        if kind == "struct":
            parts = ", ".join(f"+{o}={self.show(f)}" for o, _s, f in node[4])
            return f"{node[1]}@{self.show(node[3])}{{{parts}}}"
        if kind == "setl":
            return f"setlow{node[1] * 8}({self.show(node[2])}, {self.show(node[3])})"
        if kind == "undef":
            return f"undefined({node[1]})"
        if kind == "ite":
            return f"ite({self.show_atom(node[1])}, {self.show(node[2])}, {self.show(node[3])})"
        return str(node)

    def show_atom(self, atom: int) -> str:
        node = self.nodes[atom]
        return f"{self.show(node[1])} {'<' if node[0] == 'lt' else '=='} {self.show(node[2])}"


# ------------------------------------------------------------- source syntax

_TOKEN = re.compile(r"\s+|#[^\n]*|([A-Za-z_][A-Za-z0-9_]*|0x[0-9A-Fa-f]+|\d+|=>|==|!=|<=|>=|<<|>>|&&|\|\||"
                    r"[(){},:;@=|+\-*&^<>!\[\]])")
_BINOPS = {"+": "add", "-": "sub", "*": "imul", "<<": "shl", ">>": "shr",
           "&": "and", "|": "or", "^": "xor"}
_CMPS = ("<", "<=", ">", ">=", "==", "!=")


@dataclass
class Ctor:
    name: str
    tag: Optional[int]
    is_null: bool
    index: int
    fields: list = field(default_factory=list)  # (name, type, off, size)


@dataclass
class TypeInfo:
    name: str
    ctors: list


@dataclass
class Fn:
    name: str
    params: list
    ret: object    # a type's name, or a tuple of them when it returns several values
    at: Optional[int]
    body: tuple


class SourceParser:
    """A plain recursive descent over the whole text."""

    def __init__(self, text: str):
        self.tokens = [str(int(m.group(1), 16)) if m.group(1).startswith("0x") else m.group(1)
                       for m in _TOKEN.finditer(text) if m.group(1)]
        self.at = 0
        self.types: dict[str, TypeInfo] = {}
        self.fns: dict[str, Fn] = {}
        self.states: list[tuple[str, str]] = []
        self.spans: dict[str, tuple[list, list]] = {}  # a record of many cells: their names, and fields in each
        self.records: dict[str, list] = {}  # the names of a record's fields, across all its cells
        self.enums: dict[str, tuple] = {}  # a constructor of an enum: the enum, its tag, and its fields where they go
        self.slots: set[str] = set()  # the slot types: a value of one is its number and its length, in 8 bytes

    def unspan(self) -> None:
        """Every function, in terms of records of one cell."""
        for fn in self.fns.values():
            fn.body = _Unspan(self.spans, self.records).expr(fn.body, {})

    def infer_cells(self) -> None:
        """`@` is optional, and a function that does not write it may still return a parameter's own cell.
        This works out which, by section 8.1's rule as src/righttoleft.py states it, sharing no code
        with it: if the two disagree, what the compiler wrote will not check against this."""
        scalars = {enum for enum, _tag, _fields in self.enums.values()}
        for fn in self.fns.values():
            if fn.at is None:
                fn.at = _returned_cell(fn, self.fns, scalars)

    def builtin_reserve(self) -> None:
        """LFPL-O's Free, if the program did not declare its own."""
        self.types.setdefault("Free", TypeInfo("Free", [
            Ctor("End", None, True, 0), Ctor("More", None, False, 1, [("next", "Free", 0, 8)])]))

    def peek(self) -> Optional[str]:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def next(self) -> str:
        token = self.tokens[self.at]
        self.at += 1
        return token

    def expect(self, token: str) -> None:
        got = self.next()
        if got != token:
            raise ValueError(f"expected {token}, got {got}")

    def parse(self) -> None:
        while self.peek() is not None:
            keyword = self.next()
            if keyword == "data":
                self._data()
            elif keyword == "record":
                self._record()
            elif keyword == "enum":
                self._enum()
            elif keyword in ("fn", "handler", "turn"):
                self._fn()
            elif keyword == "state":
                name = self.next()
                self.expect(":")
                self.states.append((name, self._type()))
                self.expect(";")
            elif keyword == "slot":
                self.slots.add(self.next())
                self.expect(";")
            else:
                raise ValueError(f"unexpected {keyword}")

    def _type(self) -> str:
        name = self.next()
        if name == "Array":
            self.expect("[")
            length = self.next()
            self.expect("]")
            return f"Array[{length}]:{self.next()}"
        if name == "Fin":
            bound = self.next()
            return f"Fin {SLOT_BYTES if bound in self.slots else bound}"
        return name

    def _fields(self, close: str) -> list:
        fields = []
        while True:
            name = self.next()
            self.expect(":")
            fields.append((name, self._type()))
            if self.next() == close:
                return fields

    def _data(self) -> None:
        name = self.next()
        if self.next() == ";":
            return  # a forward declaration says nothing about layout
        raw = []
        while True:
            ctor = self.next()
            fields = []
            if self.peek() == "(":
                self.next()
                fields = self._fields(")")
            raw.append((ctor, fields))
            if self.next() == ";":
                break
        counts = sorted(len(f) for _, f in raw)
        tagless = len(raw) == 2 and counts[0] == 0 and counts[1] > 0
        ctors = []
        for index, (ctor, fields) in enumerate(raw):
            tag = None if tagless else index
            ctors.append(Ctor(ctor, tag, tagless and not fields, index,
                              self._layout(fields, 0 if tagless else 1)))
        self.types[name] = TypeInfo(name, ctors)

    def _enum(self) -> None:
        """Section 4.11: one register, its tag in the low byte and each field after it at its natural alignment."""
        name = self.next()
        self.expect("=")
        tag = 0
        while True:
            ctor = self.next()
            fields = []
            if self.peek() == "(":
                self.next()
                fields = self._fields(")")
            self.enums[ctor] = (name, tag, self._layout(fields, 1))
            tag += 1
            if self.next() == ";":
                return

    def _record(self) -> None:
        name = self.next()
        self.expect("{")
        fields = self._fields("}")
        self.expect(";")
        self.records[name] = [field_name for field_name, _ty in fields]
        cells = _cells_of(fields)
        names = [name] + [f"{name}${index}" for index in range(1, len(cells))]
        for index, (part, fields_here) in enumerate(zip(names, cells)):
            if index + 1 < len(cells):
                fields_here = fields_here + [("$next", names[index + 1])]
            self.types[part] = TypeInfo(part, [Ctor(part, None, False, 0, self._layout(fields_here, 0))])
        if len(cells) > 1:
            self.spans[name] = (names, [len(fields_here) for fields_here in cells])

    @staticmethod
    def _layout(fields: list, start: int) -> list:
        placed = []
        off = start
        for name, ty in fields:
            size = FIELD_SIZES.get(ty, 8)
            off = (off + size - 1) // size * size
            placed.append((name, ty, off, size))
            off += size
        return placed

    def _fn(self) -> None:
        name = self.next()
        self.expect("(")
        params = []
        if self.peek() == ")":
            self.next()
        else:
            params = self._fields(")")
        self.expect(":")
        ret: object
        if self.peek() == "(":
            self.next()
            members = [self._type()]
            while self.next() == ",":
                members.append(self._type())
            ret = tuple(members)
        else:
            ret = self._type()
        at = None
        if self.peek() == "@":
            self.next()
            wanted = self.next()
            at = [p for p, _ in params].index(wanted)
        self.expect("=")
        body = self.expr()
        self.expect(";")
        self.fns[name] = Fn(name, params, ret, at, body)

    def _args(self) -> list:
        args = []
        if self.peek() == ")":
            self.next()
            return args
        while True:
            args.append(self.expr())
            if self.next() == ")":
                return args

    def expr(self) -> tuple:
        token = self.next()
        if token.isdigit():
            return ("int", int(token))
        if token in ("true", "false"):
            return ("int", int(token == "true"), "Bool")
        if token == "let":
            if self.peek() == "(":
                self.next()
                names = [self.next()]
                while self.next() == ",":
                    names.append(self.next())
                self.expect("=")
                bound = self.expr()
                self.expect("in")
                return ("lettuple", names, bound, self.expr())
            name = self.next()
            self.expect("=")
            bound = self.expr()
            self.expect("in")
            return ("let", name, bound, self.expr())
        if token == "(":
            members = [self.expr()]
            while self.next() == ",":
                members.append(self.expr())
            return ("tuple", members)
        if token in ("low8", "low16", "low32") and token not in self.fns and self.peek() == "(":
            self.next()
            inner = self.expr()
            self.expect(")")
            return ("low", int(token[3:]) // 8, inner)
        if token in ("idx", "get", "within") and token not in self.fns and self.peek() == "(":
            self.next()
            first = self.expr()
            self.expect(",")
            second = self.expr()
            self.expect(")")
            if token == "idx" and second[0] == "null" and second[1] in self.slots:
                second = ("int", SLOT_BYTES)  # a slot's size in bytes
            return (token, first, second)
        if token == "len" and token not in self.fns and self.peek() == "(":
            self.next()
            inner = self.expr()
            self.expect(")")
            return ("slen", inner)
        if token == "if":
            cond = self.expr()
            self.expect("then")
            then = self.expr()
            self.expect("else")
            return ("if", cond, then, self.expr())
        if token == "match":
            return self._match()
        if token == "{":
            base = self.expr()
            self.expect("with")
            name = self.next()
            self.expect("=")
            value = self.expr()
            self.expect("}")
            return ("with", base, name, value)
        if token == "pack":
            self.expect("<")
            widths = []
            while True:
                widths.append(int(self.next()))
                if self.next() == ">":
                    break
            self.expect("(")
            return ("pack", widths, self._args())
        if token in _BINOPS or token in _CMPS or token in ("&&", "||"):
            self.expect("(")
            left = self.expr()
            self.expect(",")
            right = self.expr()
            self.expect(")")
            return ("bin", token, left, right)
        if token == "!":
            self.expect("(")
            inner = self.expr()
            self.expect(")")
            return ("not", inner)
        if token[0].isupper():
            if self.peek() == "@":
                self.next()
                cell: object = self.next()
                if cell == "(":  # a record's cells, named one by one: `unspan` takes this form out
                    cell = [self.next()]
                    while self.next() == ",":
                        cell.append(self.next())
                if self.peek() == "{":
                    self.next()
                    return ("update", token, cell, self._named_fields())
                self.expect("(")
                return ("ctor", token, cell, self._args())
            if self.peek() == "(":
                self.next()
                return ("enum", token, self._args())
            if token in self.enums:
                return ("enum", token, [])
            return ("null", token)
        if self.peek() == "(":
            self.next()
            return ("call", token, self._args())
        if self.peek() == "[":
            self.next()
            index = self.expr()
            self.expect("]")
            return ("sget", token, index)
        return ("var", token)

    def _named_fields(self) -> list:
        named: list = []
        if self.peek() == "}":
            self.next()
            return named
        while True:
            name = self.next()
            self.expect("=")
            named.append((name, self.expr()))
            if self.next() == "}":
                return named

    def _match(self) -> tuple:
        scrutinee = self.expr()
        self.expect("with")
        arms = []
        while self.peek() != "end":
            if self.peek() == "|":
                self.next()
            ctor = self.next()
            binders = []
            if self.peek() == "(":
                self.next()
                while True:
                    binders.append(self.next())
                    if self.next() == ")":
                        break
            self.expect("=>")
            arms.append((ctor, binders, self.expr()))
        self.next()
        return ("match", scrutinee, arms)


# ------------------------------------------- a record that spans cells

def _cells_of(fields: list) -> list:
    """Which fields go in which cell: all that are left if they fit one, and otherwise what fits ahead of the link."""
    def fit(remaining: list, room: int) -> int:
        at = count = 0
        for _name, ty in remaining:
            size = FIELD_SIZES.get(ty, 8)
            at = (at + size - 1) // size * size
            if at + size > room:
                break
            at += size
            count += 1
        return count

    cells = []
    remaining = list(fields)
    while fit(remaining, CELL_BYTES) < len(remaining):
        count = fit(remaining, BYTES_AHEAD_OF_THE_LINK)
        cells.append(remaining[:count])
        remaining = remaining[count:]
    cells.append(remaining)
    return cells


@dataclass(frozen=True)
class _Stands:
    """What a name stands for: the cells after its own, if it has more. If it is the diamond of a pattern that an update
    uses it for, then also where that pattern left the fields, under names nothing can bind again."""
    others: list
    fields: list = field(default_factory=list)


def _has_update(e: object) -> bool:
    if isinstance(e, tuple) and e and e[0] == "update":
        return True
    return isinstance(e, (tuple, list)) and any(_has_update(part) for part in e)


class _Unspan:
    """A syntax tree, rewritten so that each record is one cell and each update a constructor. `groups` says what names
    stand for several cells: a pattern's diamond, then what `peel` leaves. The first cell keeps the name."""

    def __init__(self, spans: dict, records: dict):
        self.spans = spans
        self.records = records
        self.used = 0

    def fresh(self, stem: str) -> str:
        self.used += 1
        return f"{stem}${self.used}"

    def expr(self, e: tuple, groups: dict) -> tuple:
        kind = e[0]
        if kind in ("int", "var", "null"):
            return e
        if kind == "bin":
            return ("bin", e[1], self.expr(e[2], groups), self.expr(e[3], groups))
        if kind == "not":
            return ("not", self.expr(e[1], groups))
        if kind == "if":
            return ("if", *(self.expr(part, groups) for part in e[1:]))
        if kind == "low":
            return ("low", e[1], self.expr(e[2], groups))
        if kind in ("idx", "get", "within"):
            return (kind, self.expr(e[1], groups), self.expr(e[2], groups))
        if kind == "sget":
            return ("sget", e[1], self.expr(e[2], groups))
        if kind == "slen":
            return ("slen", self.expr(e[1], groups))
        if kind == "pack":
            return ("pack", e[1], [self.expr(a, groups) for a in e[2]])
        if kind == "enum":
            return ("enum", e[1], [self.expr(a, groups) for a in e[2]])
        if kind == "tuple":
            return ("tuple", [self.expr(m, groups) for m in e[1]])
        if kind == "call":
            return ("call", e[1], [self.expr(a, groups) for a in e[2]])
        if kind == "let":
            return ("let", e[1], self.expr(e[2], groups), self.expr(e[3], _without(groups, [e[1]])))
        if kind == "lettuple":
            return self._let_tuple(e, groups)
        if kind == "with":
            return ("with", self.expr(e[1], groups), e[2], self.expr(e[3], _without(groups, [e[2]])))
        if kind == "match":
            return ("match", self.expr(e[1], groups), [self._arm(arm, groups) for arm in e[2]])
        if kind == "ctor":
            return self._ctor(e, groups)
        if kind == "update":
            return self._update(e, groups)
        raise ValueError(f"unknown expression {kind}")

    def _let_tuple(self, e: tuple, groups: dict) -> tuple:
        _kind, names, bound, body = e
        if bound[0] == "call" and bound[1] == "peel":
            source = bound[2][0][1]
            cells = groups[source].others
            inner = _without(groups, names)
            if len(cells) > 1:
                inner[names[1]] = _Stands(cells[1:])
            return ("lettuple", names, ("tuple", [("var", source), ("var", cells[0])]), self.expr(body, inner))
        return ("lettuple", names, self.expr(bound, groups), self.expr(body, _without(groups, names)))

    def _arm(self, arm: tuple, groups: dict) -> tuple:
        ctor, binders, body = arm
        updated = ctor in self.records and _has_update(body)
        if ctor not in self.spans and not updated:
            return (ctor, binders, self.expr(body, _without(groups, binders)))
        fields = [self.fresh("$field") for _ in binders[1:]] if updated else binders[1:]
        names, counts = self.spans.get(ctor, ([ctor], [len(fields)]))
        cells = [self.fresh(binders[0]) for _ in names[1:]]
        inner = _without(groups, binders)
        inner[binders[0]] = _Stands(cells, fields)
        inside = self.expr(body, inner)
        if updated:
            for name, alias in reversed(list(zip(binders[1:], fields))):
                inside = ("let", name, ("var", alias), inside)
        at = sum(counts)
        result = None
        for index in range(len(names) - 1, -1, -1):
            at -= counts[index]
            here = fields[at:at + counts[index]]
            cell = binders[0] if index == 0 else cells[index - 1]
            if result is None:
                result = (names[index], [cell] + here, inside)
            else:
                link = self.fresh("$link")
                result = (names[index], [cell] + here + [link], ("match", ("var", link), [result]))
        return result

    def _ctor(self, e: tuple, groups: dict) -> tuple:
        _kind, name, cell, args = e
        args = [self.expr(a, groups) for a in args]
        if name in self.spans:
            names, counts = self.spans[name]
            cells = cell if isinstance(cell, list) else [cell] + groups[cell].others
            value = None
            at = sum(counts)
            for index in range(len(names) - 1, -1, -1):
                at -= counts[index]
                here = args[at:at + counts[index]]
                value = ("ctor", names[index], cells[index], here + ([] if value is None else [value]))
            return value
        if cell in groups and groups[cell].others:  # More@d(free): one per cell, every one of them given back
            others = groups[cell].others
            value = ("ctor", name, others[-1], args)
            for other in reversed(others[:-1]):
                value = ("ctor", name, other, [value])
            return ("ctor", name, cell, [value])
        return ("ctor", name, cell, args)

    def _update(self, e: tuple, groups: dict) -> tuple:
        """The constructor with every field written out: what is not named is what the pattern found."""
        _kind, name, cell, named = e
        given = dict(named)
        left = [("var", alias) for alias in groups[cell].fields]
        args = [given.get(field_name, kept) for field_name, kept in zip(self.records[name], left)]
        return self._ctor(("ctor", name, cell, args), groups)


def _without(groups: dict, names: list) -> dict:
    """A name bound again means something new."""
    return {name: cells for name, cells in groups.items() if name not in names}


# ------------------------------------------- which parameter's cell a function returns

_NOTHING_KNOWN, _NULL = "nothing known", "null"
_NOT_A_CELL = {"Int8", "Int16", "Int32", "Int64", "Bool", "Diamond"}


def _word_of(e: tuple, names: dict, fn: "Fn", fns: dict):
    """The word of a value: a parameter's position, null, or nothing known."""
    kind = e[0]
    if kind == "var":
        return names.get(e[1], _NOTHING_KNOWN)
    if kind == "null":
        return _NULL
    if kind == "ctor":
        return names.get(e[2], _NOTHING_KNOWN)
    if kind == "with":
        return _word_of(e[1], names, fn, fns)
    if kind == "let":
        return _word_of(e[3], {**names, e[1]: _word_of(e[2], names, fn, fns)}, fn, fns)
    if kind == "lettuple":
        return _word_of(e[3], {**names, **{name: _NOTHING_KNOWN for name in e[1]}}, fn, fns)
    if kind == "if":
        then, other = _word_of(e[2], names, fn, fns), _word_of(e[3], names, fn, fns)
        return then if then == other else _NOTHING_KNOWN
    if kind == "match":
        scrutinee = _word_of(e[1], names, fn, fns)
        words = {_word_of(body, _arm_names(binders, _cell_word(ctor, scrutinee), names), fn, fns)
                 for ctor, binders, body in e[2]}
        return words.pop() if len(words) == 1 else _NOTHING_KNOWN
    if kind == "call" and e[1] != fn.name and e[1] in fns:
        at = fns[e[1]].at
        if at is not None and at < len(e[2]):
            word = _word_of(e[2][at], names, fn, fns)
            return word if isinstance(word, int) else _NOTHING_KNOWN
    return _NOTHING_KNOWN


# The arms of a match on a Fin and on within: no cell is taken apart, so nothing they bind is a parameter's cell.
_COUNTING_ARMS = frozenset(("0", "down", "inside", "outside"))


def _cell_word(ctor: str, scrutinee):
    return _NOTHING_KNOWN if ctor in _COUNTING_ARMS else scrutinee


def _arm_names(binders: list, scrutinee, names: dict) -> dict:
    inner = dict(names)
    for position, name in enumerate(binders):
        inner[name] = scrutinee if position == 0 else _NOTHING_KNOWN
    return inner


def _allowed_by_exits(e: tuple, names: dict, nulls: frozenset, fn: "Fn", fns: dict) -> set:
    """The parameters whose cell every exit of this expression returns."""
    kind = e[0]
    if kind == "let":
        inner = {**names, e[1]: _word_of(e[2], names, fn, fns)}
        return _allowed_by_exits(e[3], inner, nulls, fn, fns)
    if kind == "lettuple":
        inner = {**names, **{name: _NOTHING_KNOWN for name in e[1]}}
        return _allowed_by_exits(e[3], inner, nulls, fn, fns)
    if kind == "if":
        return (_allowed_by_exits(e[2], names, nulls, fn, fns) & _allowed_by_exits(e[3], names, nulls, fn, fns))
    if kind == "match":
        scrutinee = _word_of(e[1], names, fn, fns)
        allowed = set(range(len(fn.params)))
        for ctor, binders, body in e[2]:
            word = _cell_word(ctor, scrutinee)
            known = nulls | {word} if not binders and isinstance(word, int) else nulls
            allowed &= _allowed_by_exits(body, _arm_names(binders, word, names), known, fn, fns)
        return allowed
    if kind == "call" and e[1] == fn.name:
        return {k for k, argument in enumerate(e[2]) if _word_of(argument, names, fn, fns) == k}
    word = _word_of(e, names, fn, fns)
    if isinstance(word, int):
        return {word}
    return set(nulls) if word == _NULL else set()


def _returned_cell(fn: "Fn", fns: dict, scalars: frozenset = frozenset()) -> Optional[int]:
    """`scalars` are the program's enums: a value of one is no cell, whatever its name."""
    if (not isinstance(fn.ret, str) or fn.ret in _NOT_A_CELL or fn.ret in scalars
            or fn.ret.startswith(("Array[", "Fin "))):
        return None
    names = {name: position for position, (name, _ty) in enumerate(fn.params)}
    allowed = _allowed_by_exits(fn.body, names, frozenset(), fn, fns)
    candidates = [k for k in sorted(allowed) if fn.params[k][1] == fn.ret]
    return candidates[0] if candidates else None


# ------------------------------------------------------- the source's meaning


def _norm(ty):
    """An integer type is its width in bytes; other types stay named. A field in network order holds an integer."""
    return {"Int8": 1, "Int16": 2, "Int32": 4, "Int64": 8, "Be16": 2, "Be32": 4, "Be64": 8}.get(ty, ty)


def _joined(a, b):
    """The type of a value two arms produced: a literal has none, so the other arm decides."""
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(_joined(x, y) for x, y in zip(a, b))
    return a


def _size_of_type(ty) -> int:
    if isinstance(ty, int):
        return ty
    return 1 if ty == "Bool" else 8


def _element(array_type: str) -> tuple[object, int]:
    """The element type of `Array[N]:t`, normalized, and its size."""
    element = array_type.split(":")[1]
    return _norm(element), FIELD_SIZES[element]


class SourceMeaning:
    """Evaluates a function body symbolically. The source has no heap to speak of:
    a built constructor is simply a struct leaf naming its cell and fields."""

    def __init__(self, terms: Terms, program: SourceParser):
        self.T = terms
        self.types = program.types
        self.fns = program.fns
        self.ctors = {c.name: (t, c) for t in program.types.values() for c in t.ctors}
        self.enums = program.enums
        self.orig: dict[int, Optional[int]] = {}  # struct leaf -> tag its cell held

    def function(self, fn: Fn) -> int:
        env = {}
        for index, (name, ty) in enumerate(fn.params):
            leaf = self.T.intern(("arg", index, FIELD_SIZES.get(ty, 8)))
            env[name] = (leaf, _norm(ty), None)
        return self.ev(fn.body, env)[0]

    def truth(self, term: int) -> int:
        T = self.T
        return T.neg(T.lift(lambda x: T.eq(x, T.ZERO), term, key=("is0",)))

    def ev(self, e: tuple, env: dict) -> tuple:
        """(term, type, the tag a diamond's cell is known to hold)."""
        T = self.T
        kind = e[0]
        if kind == "int":
            return T.const(e[1]), (e[2] if len(e) > 2 else None), None
        if kind == "var":
            return env[e[1]]
        if kind == "null":
            return T.ZERO, self.ctors[e[1]][0].name, None
        if kind == "bin":
            return self._binary(e, env)
        if kind == "not":
            return T.neg(self.truth(self.ev(e[1], env)[0])), "Bool", None
        if kind == "let":
            inner = dict(env)
            inner[e[1]] = self.ev(e[2], env)
            return self.ev(e[3], inner)
        if kind == "if":
            cond = self.truth(self.ev(e[1], env)[0])
            a, ta, _ = self.ev(e[2], env)
            b, tb, _ = self.ev(e[3], env)
            return T.select(cond, a, b), _joined(ta, tb), None
        if kind == "ctor":
            return self._ctor(e, env)
        if kind == "match":
            if e[1][0] == "within":
                return self._match_within(e, env)
            scrutinee_ty = self.ev(e[1], env)[1]
            if scrutinee_ty in {enum for enum, _tag, _fields in self.enums.values()}:
                return self._match_enum(e, env)
            if isinstance(scrutinee_ty, str) and scrutinee_ty.startswith("Fin "):
                return self._match_down(e, env)
            return self._match(e, env)
        if kind == "sget":
            return self._slot_byte(e, env)
        if kind == "slen":
            return T.lift(lambda s: T.shr(s, T.const(32)), self.ev(e[1], env)[0], key=("shr32",)), 2, None
        if kind == "with":
            return self._with(e, env)
        if kind == "call":
            fn = self.fns[e[1]]
            args = [self.ev(a, env)[0] for a in e[2]]
            size = FIELD_SIZES.get(fn.ret, 8)
            built = lambda *leaves: T.call(fn.name, leaves, size)  # noqa: E731
            result_ty = (tuple(_norm(m) for m in fn.ret) if isinstance(fn.ret, tuple)
                         else _norm(fn.ret))
            return T.lift(built, *args, key=("call", fn.name)), result_ty, None
        if kind == "pack":
            return self._pack(e, env)
        if kind == "enum":
            return self._enum(e, env)
        if kind == "tuple":
            evaluated = [self.ev(m, env) for m in e[1]]
            term = T.lift(lambda *leaves: T.intern(("tuple", leaves)), *(v[0] for v in evaluated))
            return term, tuple(v[1] for v in evaluated), None
        if kind == "lettuple":
            return self._let_tuple(e, env)
        if kind == "idx":
            value = self.ev(e[1], env)[0]
            bound = e[2][1]
            mask = T.const(bound - 1)
            return T.lift(lambda x: T.bitop("and", x, mask), value), f"Fin {bound}", None
        if kind == "get":
            return self._get(e, env)
        if kind == "low":
            size = e[1]
            return T.lift(lambda x: T.trunc(size, x), self.ev(e[2], env)[0], key=("trunc", size)), size, None
        raise ValueError(f"unknown expression {kind}")

    def _slot_byte(self, e: tuple, env: dict) -> tuple:
        """Byte i of slot s, where this rule says slot k is, or 0 at or past its length."""
        T = self.T
        slot, index = env[e[1]][0], self.ev(e[2], env)[0]

        def read(s: int, i: int) -> int:
            at = T.arith("add", 8, T.arith("shl", 8, T.trunc(4, s), T.const(SLOT_SHIFT)), i)
            return T.select(T.lt(i, T.shr(s, T.const(32))), T.intern(("sload", at)), T.ZERO)

        return T.lift(read, slot, index, key=("sget",)), 1, None

    def _match_down(self, e: tuple, env: dict) -> tuple:
        """`| 0 => e0 | down(j) => e1`: e0 if the Fin is 0, and e1 with j one less if not."""
        T = self.T
        _, scrutinee, arms = e
        term, ty, _ = self.ev(scrutinee, env)
        by_name = {name: (binders, body) for name, binders, body in arms}
        result_ty: list = [None]

        def on_leaf(leaf: int) -> int:
            zero, z_ty, _ = self.ev(by_name["0"][1], env)
            binders, body = by_name["down"]
            inner = dict(env)
            inner[binders[0]] = (T.arith("sub", 8, leaf, T.ONE), ty, None)
            down, d_ty, _ = self.ev(body, inner)
            result_ty[0] = _joined(_joined(result_ty[0], z_ty), d_ty)
            return T.select(T.eq(leaf, T.ZERO), zero, down)

        return T.lift(on_leaf, term), result_ty[0], None

    def _match_within(self, e: tuple, env: dict) -> tuple:
        """`| inside(i) => e1 | outside => e0`: e1 with i the integer if it is below the slot's length."""
        T = self.T
        _, (_within, slot_e, tested_e), arms = e
        slot, tested = self.ev(slot_e, env)[0], self.ev(tested_e, env)[0]
        by_name = {name: (binders, body) for name, binders, body in arms}
        binders, inside_e = by_name["inside"]
        inner = dict(env)
        inner[binders[0]] = (tested, f"Fin {SLOT_BYTES}", None)
        inside, i_ty, _ = self.ev(inside_e, inner)
        outside, o_ty, _ = self.ev(by_name["outside"][1], env)
        length = T.lift(lambda s: T.shr(s, T.const(32)), slot, key=("shr32",))
        below = T.lift(T.lt, tested, length, key=("lt",))
        return T.select(below, inside, outside), _joined(i_ty, o_ty), None

    def _get(self, e: tuple, env: dict) -> tuple:
        T = self.T
        array, array_ty, _ = self.ev(e[1], env)
        index = self.ev(e[2], env)[0]
        element_ty, size = _element(array_ty)

        def read(base: int, position: int) -> int:
            return T.intern(("tuple", (base, T.intern(("aload", base, position, size)))))

        return T.lift(read, array, index), (array_ty, element_ty), None

    def _let_tuple(self, e: tuple, env: dict) -> tuple:
        T = self.T
        _, names, bound, body = e
        term, ty, _ = self.ev(bound, env)
        member_types = ty if isinstance(ty, tuple) else tuple(self._result_types(bound))
        result_ty: list = [None]

        def on_leaf(leaf: int) -> int:
            node = T.nodes[leaf]
            inner = dict(env)
            for position, name in enumerate(names):
                member_ty = member_types[position]
                if node[0] == "tuple":
                    member = node[1][position]
                else:  # a call's results: each is a value of its own, named by its position
                    member = T.intern(("member", leaf, position, _size_of_type(member_ty)))
                inner[name] = (member, member_ty, None)
            result, arm_ty, _ = self.ev(body, inner)
            result_ty[0] = _joined(result_ty[0], arm_ty)
            return result

        return T.lift(on_leaf, term), result_ty[0], None

    def _result_types(self, e: tuple) -> list:
        """What a tuple-valued call returns: the only form whose type `ev` cannot carry alone."""
        if e[0] == "call":
            return [_norm(m) for m in self.fns[e[1]].ret]
        raise ValueError(f"cannot read a tuple type from {e[0]}")

    def _binary(self, e: tuple, env: dict) -> tuple:
        T = self.T
        _, op, left, right = e
        a, ta, _ = self.ev(left, env)
        b, tb, _ = self.ev(right, env)
        if op == "&&":
            return T.both(self.truth(a), self.truth(b)), "Bool", None
        if op == "||":
            return T.either(self.truth(a), self.truth(b)), "Bool", None
        if op in _CMPS:
            less = lambda x, y: T.lift(T.lt, x, y, key=("lt",))  # noqa: E731
            table = {"<": lambda: less(a, b), "<=": lambda: T.neg(less(b, a)),
                     ">": lambda: less(b, a), ">=": lambda: T.neg(less(a, b)),
                     "==": lambda: T.lift(T.eq, a, b, key=("eq",)),
                     "!=": lambda: T.neg(T.lift(T.eq, a, b, key=("eq",)))}
            return table[op](), "Bool", None
        width = ta if ta is not None else tb
        result_ty = width if width is not None else 8
        name = _BINOPS[op]
        if name in ("shl", "shr"):
            return (T.lift(lambda x, c: T.total_shift(name, result_ty, x, c), a, b,
                           key=("total", name, result_ty)), result_ty, None)
        if name in ("and", "or", "xor"):
            return T.lift(lambda x, y: T.bitop(name, x, y), a, b, key=("bit", name)), result_ty, None
        return (T.lift(lambda x, y: T.arith(name, result_ty, x, y), a, b,
                       key=("arith", name, result_ty)), result_ty, None)

    def _pack(self, e: tuple, env: dict) -> tuple:
        """Horner over a 64-bit word: every field keeps only its own low bits."""
        T = self.T
        _, widths, args = e
        acc = None
        for bits, arg in zip(widths, args):
            size = bits // 8
            term = T.lift(lambda x, w=size: T.trunc(w, x), self.ev(arg, env)[0], key=("trunc", size))
            if acc is None:
                acc = term
            else:
                count = T.const(bits)
                shifted = T.lift(lambda x: T.arith("shl", 8, x, count), acc, key=("shl8", bits))
                acc = T.lift(lambda x, y: T.bitop("or", x, y), shifted, term, key=("bit", "or"))
        assert acc is not None
        return acc, 8, None

    def _enum(self, e: tuple, env: dict) -> tuple:
        """The tag, with each field's own low bytes shifted to where the field is."""
        T = self.T
        _, name, args = e
        enum, tag, fields = self.enums[name]
        value = T.const(tag)
        for (_f, _t, off, size), arg in zip(fields, args):
            term = T.lift(lambda x, w=size: T.trunc(w, x), self.ev(arg, env)[0], key=("trunc", size))
            count = T.const(8 * off)
            shifted = T.lift(lambda x: T.arith("shl", 8, x, count), term, key=("shl8", 8 * off))
            value = T.lift(lambda x, y: T.bitop("or", x, y), value, shifted, key=("bit", "or"))
        return value, enum, None

    def _built(self, ctor: Ctor, cell: int, known: Optional[int], fields: tuple) -> int:
        """A cell whose fields and tag are what it already held was not built at all."""
        T = self.T
        if T.unchanged(cell, fields) and (ctor.tag is None or known == ctor.tag):
            return cell
        leaf = T.struct(ctor.name, ctor.tag, cell, fields)
        self.orig[leaf] = known
        return leaf

    def _stored(self, ty: str, size: int, term: int) -> int:
        """What is in a field of this type, for a value that goes there."""
        T = self.T
        if ty in WIRE_FIELDS:
            return T.lift(lambda x, w=size: T.swap(w, x), term, key=("swap", size))
        return T.lift(lambda x, w=size: T.trunc(w, x), term, key=("trunc", size))

    def _read_back(self, ty: str, size: int, term: int) -> int:
        """And what is in a value that comes from one."""
        if ty in WIRE_FIELDS:
            return self.T.lift(lambda x, w=size: self.T.swap(w, x), term, key=("swap", size))
        return term

    def _ctor(self, e: tuple, env: dict) -> tuple:
        T = self.T
        _, name, cell_name, args = e
        info, ctor = self.ctors[name]
        cell, _ty, known = env[cell_name]
        terms = []
        for (_fname, fty, _off, size), arg in zip(ctor.fields, args):
            terms.append(self._stored(fty, size, self.ev(arg, env)[0]))
        layout = [(off, size) for _f, _t, off, size in ctor.fields]

        def build(*leaves: int) -> int:
            fields = tuple((o, s, leaf) for (o, s), leaf in zip(layout, leaves))
            return self._built(ctor, cell, known, fields)

        return T.lift(build, *terms), info.name, None

    def _match(self, e: tuple, env: dict) -> tuple:
        T = self.T
        _, scrutinee, arms = e
        term, ty, _ = self.ev(scrutinee, env)
        info = self.types[ty]
        by_name = {arm[0]: arm for arm in arms}
        result_ty: list = [None]

        def run(arm: tuple, cell: int, known: Optional[int], fields: list) -> int:
            inner = dict(env)
            if arm[1]:
                inner[arm[1][0]] = (cell, "Diamond", known)
                for binder, value in zip(arm[1][1:], fields):
                    inner[binder] = value
            result, arm_ty, _ = self.ev(arm[2], inner)
            result_ty[0] = _joined(result_ty[0], arm_ty)
            return result

        def opaque_fields(ctor: Ctor, pointer: int) -> list:
            return [(self._read_back(fty, size, T.load(pointer, off, size)), _norm(fty), None)
                    for _f, fty, off, size in ctor.fields]

        def on_leaf(leaf: int) -> int:
            node = T.nodes[leaf]
            if node[0] == "struct":  # a known constructor: only its arm exists
                _i, ctor = self.ctors[node[1]]
                fields = [(self._read_back(meta[1], meta[3], f[2]), _norm(meta[1]), None)
                          for f, meta in zip(node[4], ctor.fields)]
                return run(by_name[ctor.name], node[3], self.orig.get(leaf), fields)
            ctors = info.ctors
            if leaf == T.ZERO:
                nulls = [c for c in ctors if c.is_null]
                return run(by_name[nulls[0].name], leaf, None, [])
            if len(ctors) == 1:
                only = ctors[0]
                return run(by_name[only.name], leaf, only.tag, opaque_fields(only, leaf))
            if ctors[0].tag is None:  # tagless: dispatch is a null test
                full = [c for c in ctors if not c.is_null][0]
                null = [c for c in ctors if c.is_null][0]
                return T.select(T.eq(leaf, T.ZERO), run(by_name[null.name], T.ZERO, None, []),
                                run(by_name[full.name], leaf, None, opaque_fields(full, leaf)))
            # Tagged: the arms test the tag in source order, and the last needs no test.
            ordered = [self.ctors[arm[0]][1] for arm in arms]
            last = ordered[-1]
            result = run(by_name[last.name], leaf, last.tag, opaque_fields(last, leaf))
            tag_read = T.load(leaf, 0, 1)
            for ctor in reversed(ordered[:-1]):
                body = run(by_name[ctor.name], leaf, ctor.tag, opaque_fields(ctor, leaf))
                result = T.select(T.eq(tag_read, T.const(ctor.tag)), body, result)
            return result

        return T.lift(on_leaf, term), result_ty[0], None

    def _match_enum(self, e: tuple, env: dict) -> tuple:
        """An enum's tag is its low byte, and each field is its own bytes of the value. The arms are tested in the
        order they are written, and the last is what is left."""
        T = self.T
        _, scrutinee, arms = e
        term = self.ev(scrutinee, env)[0]
        result_ty: list = [None]

        def on_leaf(leaf: int) -> int:
            result = None
            for name, binders, body in reversed(arms):
                _enum, tag, fields = self.enums[name]
                inner = dict(env)
                for binder, (_f, fty, off, size) in zip(binders, fields):
                    inner[binder] = (T.trunc(size, T.shr(leaf, T.const(8 * off))), _norm(fty), None)
                value, arm_ty, _ = self.ev(body, inner)
                result_ty[0] = _joined(result_ty[0], arm_ty)
                result = value if result is None else T.select(T.eq(T.trunc(1, leaf), T.const(tag)), value, result)
            assert result is not None
            return result

        return T.lift(on_leaf, term), result_ty[0], None

    def _with(self, e: tuple, env: dict) -> tuple:
        """Section 5.5: a rebuild in which every other field is its own old value."""
        T = self.T
        _, base, name, value = e
        term, ty, _ = self.ev(base, env)
        ctor = self.types[ty].ctors[0]
        position = [f[0] for f in ctor.fields].index(name)

        def on_leaf(leaf: int) -> int:
            node = T.nodes[leaf]
            if node[0] == "struct":
                cell, known = node[3], self.orig.get(leaf)
                old = [f[2] for f in node[4]]
            else:
                cell, known = leaf, None
                old = [T.load(leaf, off, size) for _f, _t, off, size in ctor.fields]
            inner = dict(env)
            field_type, size = ctor.fields[position][1], ctor.fields[position][3]
            inner[name] = (self._read_back(field_type, size, old[position]), _norm(field_type), None)
            new = self._stored(field_type, size, self.ev(value, inner)[0])

            def build(v: int) -> int:
                leaves = list(old)
                leaves[position] = v
                fields = tuple((f[2], f[3], x) for f, x in zip(ctor.fields, leaves))
                return self._built(ctor, cell, known, fields)

            return T.lift(build, new)

        return T.lift(on_leaf, term), ty, None


# ---------------------------------------------------------- the assembly's meaning

_REGISTERS: dict[str, tuple[str, int]] = {}
for _full, _low32, _low16, _low8 in (("rax", "eax", "ax", "al"), ("rbx", "ebx", "bx", "bl"),
                                     ("rcx", "ecx", "cx", "cl"), ("rdx", "edx", "dx", "dl"),
                                     ("rsi", "esi", "si", "sil"), ("rdi", "edi", "di", "dil"),
                                     ("rbp", "ebp", "bp", "bpl")):
    _REGISTERS.update({_full: (_full, 8), _low32: (_full, 4), _low16: (_full, 2), _low8: (_full, 1)})
for _n in range(8, 16):
    _REGISTERS.update({f"r{_n}": (f"r{_n}", 8), f"r{_n}d": (f"r{_n}", 4),
                       f"r{_n}w": (f"r{_n}", 2), f"r{_n}b": (f"r{_n}", 1)})
_PARAMS = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
_RESULTS = ("rax", "rdx", "rcx", "rsi", "rdi", "r8")
_SIZES = {"byte": 1, "word": 2, "dword": 4, "qword": 8}
_ARITH = {"add", "sub", "imul", "shl"}
_BITS = {"and", "or", "xor"}
_WRITERS = _ARITH | _BITS | {"shr", "mov", "movzx", "rol", "bswap"}


def _operand(text: str) -> tuple:
    text = text.strip()
    if text in _REGISTERS:
        base, width = _REGISTERS[text]
        return ("reg", base, width)
    match = re.fullmatch(r"(byte|word|dword|qword) \[(\w+)(?:\+(\d+))?\]", text)
    if match:
        return ("mem", match.group(2), int(match.group(3) or 0), _SIZES[match.group(1)])
    match = re.fullmatch(r"(byte|word|dword|qword) \[(\w+)\+(\w+)\*(\d)\]", text)
    if match:
        return ("element", match.group(2), match.group(3), int(match.group(4)), _SIZES[match.group(1)])
    match = re.fullmatch(r"byte \[cells\+(\w+)\]", text)
    if match:  # a byte of a slot, by its place in the cell store
        return ("slotbyte", match.group(1))
    if re.fullmatch(r"-?\d+", text):
        return ("imm", int(text) & MASK64)
    return ("name", text)


def parse_assembly(asm: str) -> tuple[dict, dict]:
    """Function bodies, in order, and the label aliases that `equ` declares."""
    bodies: dict[str, list] = {}
    aliases: dict[str, str] = {}
    current: Optional[list] = None
    for number, raw in enumerate(asm.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(("section", "global", "alignb", "cells:")):
            continue
        if " equ " in line:
            left, _, right = line.partition(" equ ")
            if left.startswith("L"):
                aliases[left] = right
            continue
        if line.startswith(";"):
            continue
        if line.endswith(":"):
            name = line[:-1]
            if name.startswith("fn_") and name.endswith(("_body", "$body")):
                current = bodies.setdefault(name[3:-5], [])
            elif name.startswith("fn_") or name == "_start" or name.startswith("."):
                current = None
            elif current is not None:
                current.append(("label", name, number))
            continue
        if current is None:
            continue
        head, _, rest = line.partition(" ")
        operands = [_operand(part) for part in rest.split(",")] if rest else []
        current.append(("ins", head, operands, number))
    return bodies, aliases


def _resolve_label(name: str, aliases: dict) -> str:
    while name in aliases:
        name = aliases[name]
    return name


@dataclass
class State:
    regs: dict
    mem: dict
    flags: Optional[tuple]
    cond: int
    tags: dict
    dead: dict     # a cell a callee owns -> the condition under which it does
    frame: int = 0  # how many `sub rsp` have not yet met their `add rsp`
    deep: dict = field(default_factory=dict)  # the same, for a value and whatever it still reaches

    def copy(self) -> "State":
        return State(dict(self.regs), dict(self.mem), self.flags, self.cond,
                     dict(self.tags), dict(self.dead), self.frame, dict(self.deep))


@dataclass(frozen=True)
class Problem:
    """The assembly and the source disagree, or the check could not decide."""
    function: str
    where: str
    detail: str
    inconclusive: bool = False

    def __str__(self) -> str:
        return f"fn {self.function}, {self.where}: {self.detail}"


class FunctionCheck:
    def __init__(self, terms: Terms, meaning: SourceMeaning, fn: Fn, body: list,
                 aliases: dict, clobbers: dict):
        self.T = terms
        self.meaning = meaning
        self.fn = fn
        self.body = body
        self.aliases = aliases
        self.clobbers = clobbers
        self.types = meaning.types
        self.fns = meaning.fns
        self.rsp = terms.intern(("rsp",))
        self.problems: list[Problem] = []
        self.steps = 0

    # --- memory: distinct pointer leaves are distinct cells, by linearity

    def _dead(self, S: State, pointer: int) -> int:
        """When does a callee own this? A value given to one takes along whatever it still
        reaches. A cell given as a capability does not, because linearity moved everything out of
        it first: a pointer read from it earlier is not the callee's. The answer is a condition,
        because one arm can give a cell to a callee while another keeps it."""
        T = self.T
        owned = T.either(S.dead.get(pointer, T.ZERO), S.deep.get(pointer, T.ZERO))
        node = T.nodes[pointer]
        if node[0] == "load":
            owned = T.either(owned, self._reached_from_deep(S, node[1]))
        return owned

    def _reached_from_deep(self, S: State, pointer: int) -> int:
        T = self.T
        owned = S.deep.get(pointer, T.ZERO)
        node = T.nodes[pointer]
        if node[0] == "load":
            owned = T.either(owned, self._reached_from_deep(S, node[1]))
        return owned

    def _read(self, S: State, pointer: int, off: int, size: int) -> int:
        T = self.T
        if pointer != self.rsp:
            owned = self._dead(S, pointer)
            if owned != T.ZERO:
                return T.select(owned, T.undef("read of a cell a callee now owns"),
                                self._read_memory(S, pointer, off, size))
        return self._read_memory(S, pointer, off, size)

    def _read_memory(self, S: State, pointer: int, off: int, size: int) -> int:
        T = self.T
        found = S.mem.get((pointer, off, size))
        if found is not None:
            return found
        for p, o, s in S.mem:
            if p == pointer and o < off + size and off < o + s:
                if o == off and s > size:  # the low bytes of a wider write
                    return T.lift(lambda x: T.trunc(size, x), S.mem[(p, o, s)])
                return T.undef("read overlaps a different write")
        if pointer == self.rsp:
            return T.undef("slot never written")
        if T.is_const(pointer):
            return T.undef("read through a constant")
        return T.load(pointer, off, size)

    def _store(self, S: State, pointer: int, off: int, size: int, value: int) -> None:
        for cond, leaf in self.T.paths(pointer):
            old = self._read(S, leaf, off, size)
            merged = self.T.select(cond, value, old)
            for key in [k for k in S.mem if k[0] == leaf and k[1] < off + size and off < k[1] + k[2]]:
                del S.mem[key]
            S.mem[(leaf, off, size)] = merged

    def _pointer(self, S: State, base: str) -> int:
        if base == "rsp":
            return self.rsp
        found = S.regs.get(base)
        return found if found is not None else self.T.undef(f"{base} was never set")

    def _val(self, S: State, o: tuple) -> int:
        T = self.T
        if o[0] == "imm":
            return T.const(o[1])
        if o[0] == "reg":
            found = S.regs.get(o[1])
            term = found if found is not None else T.undef(f"{o[1]} was clobbered or never set")
            return T.lift(lambda x, w=o[2]: T.trunc(w, x), term, key=("trunc", o[2]))
        if o[0] == "mem":
            return T.lift(lambda leaf: self._read(S, leaf, o[2], o[3]), self._pointer(S, o[1]))
        if o[0] == "slotbyte":
            return T.lift(lambda at: T.intern(("sload", at)), self._pointer(S, o[1]))
        if o[0] == "element":
            _kind, base, index, scale, size = o
            if scale != size:
                return T.undef(f"an element of {size} bytes read at a scale of {scale}")
            return T.lift(lambda b, i: T.intern(("aload", b, i, size)),
                          self._pointer(S, base), self._pointer(S, index))
        raise ValueError(f"cannot read {o}")

    def _binop(self, name: str, width: int, x: int, y: int) -> int:
        T = self.T
        if name in _ARITH:
            return T.lift(lambda a, b: T.arith(name, width, a, b), x, y, key=("arith", name, width))
        if name in _BITS:
            return T.lift(lambda a, b: T.bitop(name, a, b), x, y, key=("bit", name))
        return T.lift(T.shr, x, y, key=("shr",))

    def _hardware_count(self, S: State, source: tuple, width: int, where: str) -> int:
        """The count x86 actually shifts by. It keeps the low 6 bits for a 64-bit shift
        and the low 5 for a 32-bit one, of an immediate as much as of cl, which is how
        `shl rax, 70` comes to shift by 6. A register count equals what the source says
        only where the path has already established that it is in range."""
        T = self.T
        bound = 64 if width == 8 else 32
        if source[0] == "imm":
            return T.const(source[1] & bound - 1)
        if source[1] != "rcx":
            self.problems.append(Problem(self.fn.name, where, "a shift count must be in cl"))
        count = S.regs.get("rcx", T.undef("rcx was clobbered or never set"))
        for cond, leaf in T.paths(count):
            if not self._below(T.both(S.cond, cond), leaf, bound):
                self.problems.append(Problem(
                    self.fn.name, where,
                    f"shift by {T.show(leaf)} with nothing keeping it below {bound}, "
                    f"so the hardware may mask it"))
        return count

    def _below(self, cond: int, leaf: int, bound: int) -> bool:
        """On every path through cond, is leaf known to be below bound?"""
        T = self.T
        if cond == T.ZERO:  # this path never reaches the shift, whatever the count is
            return True
        if T.is_const(leaf):
            return T.value(leaf) < bound
        atom = T.top(cond)
        if atom == NO_ATOM:
            return False
        node = T.nodes[atom]
        settled = (node[0] == "lt" and node[1] == leaf and T.is_const(node[2])
                   and T.value(node[2]) <= bound)
        return ((settled or self._below(T.cof(cond, atom, True), leaf, bound))
                and self._below(T.cof(cond, atom, False), leaf, bound))

    def _cond(self, S: State, cc: str, where: str) -> int:
        T = self.T
        if S.flags is None:
            self.problems.append(Problem(self.fn.name, where, f"j{cc}: the flags hold no comparison"))
            return T.ZERO
        x, y = S.flags
        less = lambda a, b: T.lift(T.lt, a, b, key=("lt",))  # noqa: E731
        same = lambda: T.lift(T.eq, x, y, key=("eq",))  # noqa: E731
        return {"b": lambda: less(x, y), "ae": lambda: T.neg(less(x, y)),
                "a": lambda: less(y, x), "be": lambda: T.neg(less(y, x)),
                "e": same, "ne": lambda: T.neg(same())}[cc]()

    # --- what a pointer and the memory behind it mean, in the source's terms

    def _touched(self, S: State, pointer: int) -> bool:
        """Was this cell, or any cell reached through its fields, written? The root of
        a rebuilt structure can be untouched while a cell below it changed."""
        for key in S.mem:
            at = key[0]
            while True:
                if at == pointer:
                    return True
                node = self.T.nodes[at]
                if node[0] != "load":
                    break
                at = node[1]
        return False

    def _declared_type(self, pointer: int) -> Optional[str]:
        """What this pointer is according to the source, where that can be told from what it came from."""
        T = self.T
        node = T.nodes[pointer]
        if node[0] == "arg":
            return self.fn.params[node[1]][1]
        if node[0] == "call":
            ret = self.fns[node[1]].ret
            return ret if isinstance(ret, str) else None
        if node[0] == "member" and T.nodes[node[1]][0] == "call":
            ret = self.fns[T.nodes[node[1]][1]].ret
            return ret[node[2]] if isinstance(ret, tuple) else None
        if node[0] == "load":
            parent = self.types.get(self._declared_type(node[1]))
            if parent is not None:
                found = {ty for ctor in parent.ctors for _f, ty, off, size in ctor.fields
                         if off == node[2] and size == node[3]}
                return found.pop() if len(found) == 1 else None
        return None

    def _tag_of(self, fact: Optional[tuple], tag_leaf: int, info: TypeInfo, cell: int) -> Optional[int]:
        T = self.T
        if T.is_const(tag_leaf):
            return T.value(tag_leaf)
        if fact is None:
            return 0 if len(info.ctors) == 1 else None
        if fact[0] == "is":
            return fact[1]
        # What the tests ruled out is about the constructors of the type they tested, and the last arm of a match takes
        # no test. So the tag left over is that type's, whatever type the cell is being read as now.
        tested = self.types.get(self._declared_type(cell), info)
        left = [c.tag for c in tested.ctors if c.tag not in fact[1]]
        return left[0] if len(left) == 1 else None

    def _result_of(self, S: State) -> int:
        """What the function hands back, read from where the convention says it is."""
        T = self.T
        ret = self.fn.ret
        if not isinstance(ret, tuple):
            return self.reify(S, S.regs.get("rax", T.undef("rax never set")), _norm(ret))
        members = [self.reify(S, S.regs.get(_RESULTS[i], T.undef(f"{_RESULTS[i]} never set")), _norm(m))
                   for i, m in enumerate(ret)]
        return T.lift(lambda *leaves: T.intern(("tuple", leaves)), *members)

    def reify(self, S: State, term: int, ty) -> int:
        """Read a value the way the source would see it: a pointer to a rebuilt cell
        becomes the struct its memory now holds, so it compares with the source's."""
        if ty not in self.types:
            return term
        return self._reify_on(S, term, ty, S.cond, frozenset())

    def _on_paths(self, term: int, path: int, fn: Callable[[int, int], int]) -> int:
        """fn on each leaf this value can be, on a path that can happen. Following memory has
        to stop at impossible paths: a location merged from two arms names what it holds on
        each, and going through one arm's pointer into the other arm's contents can walk around
        a cycle that no execution has, since linear data are acyclic."""
        T = self.T
        node = T.nodes[term]
        if node[0] != "ite":
            return fn(term, path)
        test = T._node(node[1], T.ONE, T.ZERO)
        hi_path, lo_path = T.both(path, test), T.both(path, T.neg(test))
        hi = T.ZERO if hi_path == T.ZERO else self._on_paths(node[2], hi_path, fn)
        lo = T.ZERO if lo_path == T.ZERO else self._on_paths(node[3], lo_path, fn)
        return T.select(test, hi, lo)

    def _reify_on(self, S: State, term: int, ty, path: int, within: frozenset) -> int:
        if ty not in self.types:
            return term
        return self._on_paths(term, path, lambda leaf, p: self._reify_leaf(S, leaf, ty, p, within))

    def _unchanged_on(self, S: State, pointer: int, path: int) -> bool:
        """On this path, does this cell and everything reached through it hold what it held on
        entry? Another arm may have written to it, which says nothing about this path."""
        T = self.T
        for key, value in S.mem.items():
            at = key[0]
            while at != pointer:
                node = T.nodes[at]
                if node[0] != "load":
                    break
                at = node[1]
            if at != pointer:
                continue
            original = T.load(key[0], key[1], key[2])
            for cond, leaf in T.paths(value):
                if leaf != original and T.both(path, cond) != T.ZERO:
                    return False
        return True

    def _reify_leaf(self, S: State, leaf: int, ty: str, path: int, within: frozenset) -> int:
        T = self.T
        if leaf == T.ZERO or not self._touched(S, leaf) or self._unchanged_on(S, leaf, path):
            return leaf
        owned = self._dead(S, leaf)
        if owned != T.ZERO:
            alive = T.both(path, T.neg(owned))
            gone = T.undef("a callee owns this cell")
            if alive == T.ZERO:
                return gone
            return T.select(owned, gone, self._reify_alive(S, leaf, ty, alive, within))
        return self._reify_alive(S, leaf, ty, path, within)

    def _reify_alive(self, S: State, leaf: int, ty: str, path: int, within: frozenset) -> int:
        T = self.T
        if leaf in within:
            return T.undef("a cell that holds itself")
        within = within | {leaf}
        info = self.types[ty]
        if info.ctors[0].tag is None:
            ctor = [c for c in info.ctors if not c.is_null][0]
            return self._reify_fields(S, leaf, ctor, False, path, within)
        return self._reify_tagged(S, leaf, info, S.tags.get(leaf), path, within)

    def _reify_tagged(self, S: State, leaf: int, info: TypeInfo, fact: Optional[tuple], path: int,
                      within: frozenset) -> int:
        """What this tagged cell holds. A fact that came through a join is one fact per arm, and
        the arm's own condition chooses, which is the shape the source's match has too."""
        T = self.T
        if fact is not None and fact[0] == "cond":
            _kind, cond, first, second = fact
            hi_path, lo_path = T.both(path, cond), T.both(path, T.neg(cond))
            hi = T.ZERO if hi_path == T.ZERO else self._reify_tagged(S, leaf, info, first, hi_path, within)
            lo = T.ZERO if lo_path == T.ZERO else self._reify_tagged(S, leaf, info, second, lo_path, within)
            return T.select(cond, hi, lo)

        def on_tag(tag_leaf: int, p: int) -> int:
            if self._unchanged_on(S, leaf, p):  # the tag facts have split the path by now
                return leaf
            tag = self._tag_of(fact, tag_leaf, info, leaf)
            ctor = next((c for c in info.ctors if c.tag == tag), None)
            if ctor is None:
                return T.undef(f"tag of {T.show(leaf)} is {T.show(tag_leaf)}")
            return self._reify_fields(S, leaf, ctor, T.is_const(tag_leaf), p, within)

        return self._on_paths(self._read(S, leaf, 0, 1), path, on_tag)

    def _reify_fields(self, S: State, cell: int, ctor: Ctor, tag_written: bool, path: int,
                      within: frozenset) -> int:
        T = self.T
        terms = [self._reify_on(S, self._read(S, cell, off, size), _norm(fty), path, within)
                 for _f, fty, off, size in ctor.fields]
        layout = [(off, size) for _f, _t, off, size in ctor.fields]

        def build(*leaves: int) -> int:
            fields = tuple((o, s, leaf) for (o, s), leaf in zip(layout, leaves))
            if T.unchanged(cell, fields) and not tag_written:
                return cell
            return T.struct(ctor.name, ctor.tag, cell, fields)

        return T.lift(build, *terms)

    def _pointer_of(self, leaf: int) -> int:
        """The word a value occupies: a built cell's own, or, through an `@` callee,
        the word of the argument it was declared to return."""
        node = self.T.nodes[leaf]
        if node[0] == "struct":
            return node[3]
        if node[0] == "call":
            at = self.fns[node[1]].at
            if at is not None:
                return self._pointer_of(node[2][at])
        return leaf

    def _rewrite(self, S: State, pointer: int, cond: int, result: int) -> None:
        """After `f @k`, if cond holds, that word denotes the updated value, wherever it is held.
        This is why RESTORE compares equal: the skipped store's location already holds
        the word, and it holds the updated value only because the call really ran. What it denoted
        before can itself be the result of an earlier call on that word, which this one then updates."""
        T = self.T
        swap = lambda term: T.lift(  # noqa: E731
            lambda x: T.select(cond, result, x) if self._pointer_of(x) == pointer else x, term)
        node = T.nodes[pointer]
        implicit = None
        if node[0] == "load":
            held = self._read(S, node[1], node[2], node[3])
            if any(self._pointer_of(leaf) == pointer for _cond, leaf in T.paths(held)):
                implicit = (node[1], node[2], node[3]), swap(held)
        S.regs = {r: swap(t) for r, t in S.regs.items()}
        S.mem = {k: swap(t) for k, t in S.mem.items()}
        if implicit is not None:
            S.mem[implicit[0]] = implicit[1]

    def _invoke(self, S: State, callee: Fn) -> int:
        """The value of calling `callee` with the parameter registers as they stand."""
        T = self.T
        args = []
        for position, (_name, ty) in enumerate(callee.params):
            found = S.regs.get(_PARAMS[position])
            term = found if found is not None else T.undef(f"{_PARAMS[position]} was never set")
            args.append(self.reify(S, term, _norm(ty)))
        size = FIELD_SIZES.get(callee.ret, 8)
        result = T.lift(lambda *leaves: T.call(callee.name, leaves, size), *args,
                        key=("call", callee.name))
        for term, (_name, ty) in zip(args, callee.params):
            if _norm(ty) in self.types or ty == "Diamond":
                for cond, leaf in T.paths(term):
                    if ty == "Diamond":
                        S.dead[leaf] = T.either(S.dead.get(leaf, T.ZERO), cond)
                    else:
                        self._hand_over(S, leaf, cond)
        if callee.at is not None:
            for cond, leaf in T.paths(args[callee.at]):
                self._rewrite(S, self._pointer_of(leaf), cond, result)
        return result

    def _hand_over(self, S: State, leaf: int, cond: int) -> None:
        """The callee owns this cell and everything inside it from now on, if cond holds."""
        T = self.T
        node = T.nodes[leaf]
        if node[0] == "struct":
            # This cell was rebuilt, so what it used to reach has moved out already.
            S.dead[node[3]] = T.either(S.dead.get(node[3], T.ZERO), cond)
            for _o, _s, inner in node[4]:
                self._hand_over(S, inner, cond)
        elif not T.is_const(leaf):
            S.deep[leaf] = T.either(S.deep.get(leaf, T.ZERO), cond)

    # --- the obligations

    def _compare(self, cond: int, got: int, want: int, where: str) -> None:
        T = self.T

        def walk(c: int, g: int, w: int, sigma: dict) -> bool:
            if c == T.ZERO:
                return True
            self.steps += 1
            if self.steps > 2_000_000:
                raise TooLarge("too many paths")
            atom = min(T.top(c), T.top(g), T.top(w))
            if atom == NO_ATOM:
                memo: dict = {}
                a, b = T.subst(g, sigma, memo), T.subst(w, sigma, memo)
                if a != b:
                    self.problems.append(Problem(self.fn.name, where, self._explain(a, b, sigma)))
                    return False
                return True
            node = T.nodes[atom]
            hi_sigma: Optional[dict] = sigma
            if node[0] == "eq":  # on this path, x really is that constant
                x, y = node[1], node[2]
                if T.is_const(x) or T.is_const(y):
                    leaf, value = (y, x) if T.is_const(x) else (x, y)
                    if sigma.get(leaf, value) != value:  # it cannot be two constants at once
                        hi_sigma = None
                    else:
                        hi_sigma = dict(sigma)
                        hi_sigma[leaf] = value
            return ((hi_sigma is None or walk(T.cof(c, atom, True), T.cof(g, atom, True),
                                              T.cof(w, atom, True), hi_sigma))
                    and walk(T.cof(c, atom, False), T.cof(g, atom, False), T.cof(w, atom, False), sigma))

        walk(cond, got, want, {})

    def _explain(self, found: int, expected: int, sigma: dict) -> str:
        T = self.T
        a, b = T.nodes[found], T.nodes[expected]
        path = "" if not sigma else " when " + ", ".join(
            f"{T.show(k)} is {T.show(v)}" for k, v in sigma.items())
        if a[0] == "struct" and b[0] == "struct" and a[3] == b[3] and a[1] == b[1]:
            wrong = [f"field +{x[0]} holds {T.show(x[2])}, the source says {T.show(y[2])}"
                     for x, y in zip(a[4], b[4]) if x[2] != y[2]]
            return f"constructor {b[1]} in cell {T.show(b[3])}{path}: " + "; ".join(wrong)
        if b[0] == "struct":
            return (f"the source builds {T.show(expected)}{path}, "
                    f"but the machine produces {T.show(found)}")
        return f"expected {T.show(expected)}{path}, found {T.show(found)}"

    def _exit(self, S: State, got: int, want: int, where: str, reached: list) -> None:
        self._compare(S.cond, got, want, where)
        reached.append(S.cond)

    def _merge(self, states: list) -> State:
        """The merge rule of the module docstring, one incoming edge at a time."""
        T = self.T
        merged = states[0]
        for other in states[1:]:
            c = merged.cond
            regs = {}
            for r in set(merged.regs) | set(other.regs):
                a = merged.regs.get(r)
                b = other.regs.get(r)
                regs[r] = T.select(c, a if a is not None else T.undef(f"{r} unset on one path"),
                                   b if b is not None else T.undef(f"{r} unset on one path"))
            mem = {}
            for key in set(merged.mem) | set(other.mem):
                mem[key] = T.select(c, self._read(merged, *key), self._read(other, *key))
            tags = {}
            for pointer in set(merged.tags) | set(other.tags):
                x, y = merged.tags.get(pointer), other.tags.get(pointer)
                if x == y:
                    tags[pointer] = x
                elif x is not None and y is not None and x[0] == "not" and y[0] == "not":
                    tags[pointer] = ("not", x[1] & y[1])
                else:  # each arm knows its own, and the arm's condition says which applies
                    tags[pointer] = ("cond", c, x, y)
            dead = {p: T.select(c, merged.dead.get(p, T.ZERO), other.dead.get(p, T.ZERO))
                    for p in set(merged.dead) | set(other.dead)}
            deep = {p: T.select(c, merged.deep.get(p, T.ZERO), other.deep.get(p, T.ZERO))
                    for p in set(merged.deep) | set(other.deep)}
            if merged.frame != other.frame:
                self.problems.append(Problem(self.fn.name, "a join", "two edges with different frames"))
            merged = State(regs, mem, merged.flags if merged.flags == other.flags else None,
                           T.either(c, other.cond), tags, dead, merged.frame, deep)
        return merged

    # --- the walk over the function, in emitted order

    def run(self) -> list[Problem]:
        T = self.T
        fn = self.fn
        want = self.meaning.function(fn)
        regs = {}
        for position, (_name, ty) in enumerate(fn.params):
            regs[_PARAMS[position]] = T.intern(("arg", position, FIELD_SIZES.get(ty, 8)))
        current: Optional[State] = State(regs, {}, None, T.ONE, {}, {})
        pending: dict[str, list] = {}
        placed: set[str] = set()
        reached: list[int] = []
        for item in self.body:
            if item[0] == "label":
                name = item[1]
                placed.add(name)
                edges = pending.pop(name, [])
                if current is not None:
                    edges = edges + [current]
                current = self._merge(edges) if edges else None
                continue
            if current is None:
                continue
            current = self._step(current, item[1], item[2], item[3], want, pending, placed, reached)
        if current is not None:
            self.problems.append(Problem(fn.name, "end of body", "control runs off the function"))
        covered = T.ZERO
        for cond in reached:
            covered = T.either(covered, cond)
        if covered != T.ONE and not self.problems:
            self.problems.append(Problem(fn.name, "end of body", "some path never leaves the function"))
        return self.problems

    def _step(self, S: State, op: str, ops: list, line: int, want: int, pending: dict,
              placed: set, reached: list) -> Optional[State]:
        T = self.T
        where = f"line {line}"
        if op in ("sub", "add") and ops[0] == ("name", "rsp"):
            S.frame += 1 if op == "sub" else -1
            return S
        if op == "mov":
            dest, source = ops
            value = self._val(S, source)
            if dest[0] == "reg" and dest[2] == 8:
                S.regs[dest[1]] = value
            elif dest[0] == "reg" and dest[2] == 4:  # a 32-bit write clears the upper half
                S.regs[dest[1]] = T.lift(lambda x: T.trunc(4, x), value, key=("trunc", 4))
            elif dest[0] == "reg":
                old = S.regs.get(dest[1], T.undef("subregister of an unset register"))
                S.regs[dest[1]] = T.lift(lambda a, b, w=dest[2]: T.setl(w, a, T.trunc(w, b)), old, value)
            else:
                self._write(S, dest, value)
        elif op == "movzx":
            S.regs[ops[0][1]] = self._val(S, ops[1])
        elif op in _ARITH or op in _BITS or op == "shr":
            self._arith_step(S, op, ops, where)
        elif op == "rol":  # only ever by 8 on 16 bits, and the upper bits stay as they were
            if ops[0][0] != "reg" or ops[0][2] != 2 or ops[1] != ("imm", 8):
                self.problems.append(Problem(self.fn.name, where, "a rotate that is not a swap of 16 bits"))
            old = S.regs.get(ops[0][1], T.undef("rol of an unset register"))
            S.regs[ops[0][1]] = T.lift(lambda x: T.setl(2, x, T.swap(2, x)), old)
            S.flags = None
        elif op == "bswap":
            S.regs[ops[0][1]] = T.lift(lambda x, w=ops[0][2]: T.swap(w, x), self._val(S, ops[0]))
            S.flags = None
        elif op == "cmp":
            S.flags = (self._val(S, ops[0]), self._val(S, ops[1]))
        elif op == "test":
            if ops[0] != ops[1]:
                self.problems.append(Problem(self.fn.name, where, "test of two different operands"))
            S.flags = (self._val(S, ops[0]), T.ZERO)
        elif op == "ret":
            if S.frame:
                self.problems.append(Problem(self.fn.name, where, "ret with a frame still on the stack"))
            self._exit(S, self._result_of(S), want, f"ret at {where}", reached)
            return None
        elif op == "call":
            callee = self.fns[ops[0][1][3:]]
            result = self._invoke(S, callee)
            for register in self.clobbers.get(callee.name, ()):
                S.regs.pop(register, None)
            if isinstance(callee.ret, tuple):
                for position, member in enumerate(callee.ret):
                    size = FIELD_SIZES.get(member, 8)
                    S.regs[_RESULTS[position]] = T.lift(
                        lambda leaf, p=position, z=size: T.intern(("member", leaf, p, z)), result)
            else:
                S.regs["rax"] = result
            S.flags = None
        elif op == "jmp":
            return self._jump(S, ops[0][1], where, want, pending, placed, reached)
        elif op.startswith("cmov"):  # a 32-bit write: the upper half clears whichever way it goes
            dest, source = ops
            if dest[0] != "reg" or dest[2] != 4:
                self.problems.append(Problem(self.fn.name, where, "a cmov that is not into 32 bits"))
            old = T.lift(lambda x: T.trunc(4, x), self._val(S, dest), key=("trunc", 4))
            new = T.lift(lambda x: T.trunc(4, x), self._val(S, source), key=("trunc", 4))
            S.regs[dest[1]] = T.select(self._cond(S, op[4:], where), new, old)
        elif op.startswith("set"):
            old = S.regs.get(ops[0][1], T.undef("setcc into an unset register"))
            S.regs[ops[0][1]] = T.lift(lambda a, b: T.setl(1, a, b), old, self._cond(S, op[3:], where))
        elif op.startswith("j"):
            return self._branch(S, op[1:], ops[0][1], where, pending)
        else:
            self.problems.append(Problem(self.fn.name, where, f"unknown instruction {op}"))
        return S

    def _write(self, S: State, dest: tuple, value: int) -> None:
        T = self.T
        narrowed = T.lift(lambda x, w=dest[3]: T.trunc(w, x), value, key=("trunc", dest[3]))
        self._store(S, self._pointer(S, dest[1]), dest[2], dest[3], narrowed)

    def _arith_step(self, S: State, op: str, ops: list, where: str) -> None:
        T = self.T
        dest, source = ops
        if dest[0] == "reg" and dest[2] < 4:
            if op != "or":
                self.problems.append(Problem(self.fn.name, where, f"{op} on a subregister"))
            # Setting low bits in the Horner accumulator leaves the upper bits alone.
            S.regs[dest[1]] = self._binop("or", 8, S.regs.get(dest[1], T.undef("unset accumulator")),
                                          self._val(S, source))
        elif dest[0] == "reg" and op in ("shl", "shr"):
            count = self._hardware_count(S, source, dest[2], where)
            S.regs[dest[1]] = self._binop(op, dest[2], self._val(S, dest), count)
        elif dest[0] == "reg":
            S.regs[dest[1]] = self._binop(op, dest[2], self._val(S, dest), self._val(S, source))
        else:
            self._write(S, dest, self._binop(op, dest[3], self._val(S, dest), self._val(S, source)))
        S.flags = None

    def _branch(self, S: State, cc: str, target: str, where: str, pending: dict) -> Optional[State]:
        T = self.T
        taken_cond = self._cond(S, cc, where)
        taken = S.copy()
        taken.cond = T.both(S.cond, taken_cond)
        fall = S
        fall.cond = T.both(S.cond, T.neg(taken_cond))
        if cc in ("e", "ne") and S.flags is not None:
            equal, unequal = (taken, fall) if cc == "e" else (fall, taken)
            self._learn_tag(equal, unequal)
        if taken.cond != T.ZERO:
            pending.setdefault(_resolve_label(target, self.aliases), []).append(taken)
        return fall if fall.cond != T.ZERO else None

    def _learn_tag(self, equal: State, unequal: State) -> None:
        """A tag test tells each path which constructor the cell holds, so a rebuild
        with the same constructor can legitimately leave the tag alone."""
        T = self.T
        x, y = equal.flags if equal.flags is not None else (None, None)
        if x is None or not T.is_const(y):
            return
        node = T.nodes[x]
        if node[0] == "load" and node[2] == 0 and node[3] == 1:
            pointer, value = node[1], T.value(y)
            equal.tags[pointer] = ("is", value)
            fact = unequal.tags.get(pointer)
            ruled_out = fact[1] if fact is not None and fact[0] == "not" else frozenset()
            unequal.tags[pointer] = ("not", ruled_out | {value})

    def _jump(self, S: State, target: str, where: str, want: int, pending: dict, placed: set,
              reached: list) -> Optional[State]:
        T = self.T
        fn = self.fn
        if target.startswith("fn_"):  # a tail call to another function
            if S.frame:
                self.problems.append(Problem(fn.name, where, "a tail call with a frame on the stack"))
            result = self._invoke(S, self.fns[target[3:]])
            self._exit(S, result, want, f"tail call at {where}", reached)
            return None
        label = _resolve_label(target, self.aliases)
        if label not in placed:
            pending.setdefault(label, []).append(S)
            return None
        # A back-edge is a call of this function, taken to mean what it means.
        result = self._invoke(S, fn)
        if fn.at is not None:
            got = self.reify(S, S.regs.get("rax", T.undef("rax never set")), _norm(fn.ret))
        else:
            got = result
        self._exit(S, got, want, f"back-edge at {where}", reached)
        return None


# ------------------------------------------------------------------- the check


def _clobber_sets(bodies: dict) -> dict:
    """Which registers each function may change: read from its body, not trusted."""
    sets: dict[str, set] = {}
    for name, body in bodies.items():
        touched = {"rax"}
        for item in body:
            if item[0] != "ins":
                continue
            op, ops = item[1], item[2]
            if op in ("call", "jmp") and ops[0][0] == "name" and ops[0][1].startswith("fn_"):
                touched |= sets.get(ops[0][1][3:], set())
            elif op in _WRITERS or op.startswith(("set", "cmov")):
                if ops and ops[0][0] == "reg":
                    touched.add(ops[0][1])
        sets[name] = touched
    return sets


def check_translation(source: bytes, asm: str) -> list[Problem]:
    """Every problem found in `asm` as a translation of `source`; [] means agreement."""
    program = SourceParser(source.decode("latin-1"))
    program.parse()
    program.builtin_reserve()
    program.unspan()
    bodies, aliases = parse_assembly(asm)
    clobbers = _clobber_sets(bodies)
    problems: list[Problem] = []
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 100_000))
    try:
        program.infer_cells()
        for name, fn in program.fns.items():
            body = bodies.get(name)
            if body is None:
                problems.append(Problem(name, "whole function", "no code was emitted for it"))
                continue
            terms = Terms()
            check = FunctionCheck(terms, SourceMeaning(terms, program), fn, body, aliases, clobbers)
            try:
                problems.extend(check.run())
            except (TooLarge, RecursionError) as reason:
                problems.append(Problem(name, "whole function", f"could not decide: {reason}",
                                        inconclusive=True))
    finally:
        sys.setrecursionlimit(limit)
    return problems
