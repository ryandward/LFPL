"""The register-transfer checker: docs/rt-layer.md, one line at a time.

Lower writes the stream and this types it, sharing no code, which a test enforces by reading
the imports. Every rule moves tokens and creates none, so a value used twice, a store that was
skipped when it was needed, or a read of a cell that has been rebuilt is a local type error
here, found without running anything.

It keeps the first specification's memory bound for typing: the only state is the tokens of the
function being read, and one merged token state for each label still open. Control flow is
structured, so that is at most one state per enclosing construct. The bounds below add each finished
function's inequalities, because its callers copy them. Those are capped per function.

The same pass derives a handler's bounds, by the potential method of LFPL-O section 5. Every cell
of a value carries an unknown amount of each resource, which is released when the cell is opened
and paid again when a value is built in it. Each instruction costs a step, and each draw from a
Free costs a cell. A budget follows every path, and at each exit, join and call this writes down an
inequality that has to hold. Those are linear in the unknowns, so a handler's are one linear program.

This module does not solve it. Something else proposes the numbers, and `verify` evaluates every
inequality with them, exactly, so a wrong solver can only cost a refusal. The numbers filled into
the inequalities are the certificate of section 5.5.

A recursive function is where the bound comes from, and rule R3's decreasing set, which `end` states,
is what carries it: its parameters share one amount per cell. R3 says their total falls by at least
one cell at each recursive call, so each call releases that amount, and the inequality at the
recursive call says that pays for the iteration. Other linear parameters and results may carry
potential too, because a walk that moves cells onto an accumulator has to hand the second pass's
budget over with them. This is also a second proof of termination: a recursive call that released
nothing would have to pay for itself, and no numbers satisfy that.
"""
from __future__ import annotations

import re
import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Iterable, Mapping, Optional, Union

SCALARS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64}
WIRE = {"be16": 16, "be32": 32, "be64": 64}  # bits in network byte order: scalars, and no integer until bswap says so
SIZES = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "bool": 1}
OP_NAMES = {"add", "sub", "mul", "and", "or", "xor", "shl", "shr"}
READ_MODIFY_WRITE = {"add", "sub", "and", "or", "xor"}  # what one x86 instruction does in memory
CC_NAMES = {"b", "be", "a", "ae", "e", "ne"}
RESERVE = "Free"
NO_WORK = {"drop", "open", "move"}  # these move tokens and lower to nothing, so they cost no step
MAX_OBLIGATIONS = 1024  # one handler's, with its callees': the size of a linear program, section 8.2

Type = Union[str, tuple]  # a builtin name, ("named", id), ("enum", id), ("array", N, t) or ("fin", N)
ENUM_BYTES = 8  # an enum is one register: its tag in byte 0, and its fields after it
Word = tuple


class RTError(Exception):
    """The stream breaks a typing rule. Lower wrote something it should not have."""


class Unbounded(RTError):
    """A handler has no linear bound: a limit of the language, LFPL-O section 8.2."""


@dataclass(frozen=True)
class Budget:
    """What is available at a point in the code: unknowns times whole numbers, plus a constant for
    each resource. Used as an obligation, it says this much must not be negative."""
    terms: tuple = ()  # sorted (unknown, coefficient) pairs, none of them zero
    cells: int = 0
    steps: int = 0

    def plus(self, unknown: Optional[int], coefficient: int = 1) -> "Budget":
        if unknown is None:
            return self
        terms = dict(self.terms)
        terms[unknown] = terms.get(unknown, 0) + coefficient
        if not terms[unknown]:
            del terms[unknown]
        return Budget(tuple(sorted(terms.items())), self.cells, self.steps)

    def spend(self, *, cells: int = 0, steps: int = 0) -> "Budget":
        return Budget(self.terms, self.cells - cells, self.steps - steps)


@dataclass(frozen=True)
class Bound:
    """Section 6.1's admission table, for one handler: R(n) = a n + b and W(n) = c n + d."""
    a: int
    b: int
    c: int
    d: int


@dataclass(frozen=True)
class ResourceProgram:
    """One handler's linear program, as the checker derived it, for a solver to propose numbers.
    Every row must not be negative, and neither may any unknown."""
    rows: tuple
    item: Optional[int]
    entry: int
    n_max: int
    sizes: Optional[Mapping[int, int]] = None  # a turn's: the unknown per cell of each sized parameter, and its cells

    def objective(self) -> dict[int, int]:
        """Section 5.5: the bound at the largest item, or for a turn, with every sized parameter at its largest."""
        objective = {self.entry: 1}
        if self.sizes is not None:
            objective.update(self.sizes)
        elif self.item is not None:
            objective[self.item] = self.n_max
        return objective

    def constraints(self, resource: str) -> list[tuple[dict[int, int], int]]:
        return [(dict(row.terms), getattr(row, resource)) for row in self.rows]


def is_scalar(ty: Type) -> bool:
    return ty in SCALARS or ty in WIRE or ty in ("bool", "lit") or (isinstance(ty, tuple) and ty[0] in ("fin", "enum"))


def is_enum(ty: Type) -> bool:
    return isinstance(ty, tuple) and ty[0] == "enum"


def is_int(ty: Type) -> bool:
    return ty in SCALARS or ty == "lit"


def is_fin(ty: Type) -> bool:
    return isinstance(ty, tuple) and ty[0] == "fin"


@dataclass(frozen=True)
class Tok:
    ty: Type
    word: Word
    pot: Optional[int] = None  # the unknown per cell. None: it has none, or no cells to carry it


@dataclass(frozen=True)
class Loc:
    full: bool
    word: Word
    ty: Type
    pot: Optional[int] = None


@dataclass
class Cell:
    tag: Optional[tuple] = None  # ("is", k), ("not", {k...}) or None for nothing known
    locs: dict = field(default_factory=dict)  # (offset, size) -> Loc


@dataclass
class State:
    regs: dict
    cells: dict
    nulls: frozenset
    ruled: dict
    budget: Budget = Budget()
    unreachable: bool = False  # past the End arm of a reserve: it must type, and is charged nothing

    def copy(self) -> "State":
        return State(dict(self.regs),
                     {c: Cell(v.tag, dict(v.locs)) for c, v in self.cells.items()},
                     self.nulls, dict(self.ruled), self.budget, self.unreachable)


@dataclass
class Ctor:
    id: str
    type: str
    tag: str
    name: str
    fields: list = field(default_factory=list)  # (offset, size, type text, name)


@dataclass
class TypeDecl:
    kind: str
    name: str
    ctors: list = field(default_factory=list)


@dataclass
class Signature:
    ordinal: str
    name: str
    params: list
    ret: str
    at: Optional[int]
    handler: bool
    entry: int = 0                  # the unknown budget a call of this function must bring
    param_pots: list = field(default_factory=list)
    result_pots: list = field(default_factory=list)
    obligations: dict = field(default_factory=dict)  # inequalities, its callees' included: a set in order
    recursive: bool = False
    too_large: bool = False
    makes_a_free: bool = False      # on a path that can run: see _made
    turn: bool = False              # the runtime's turn: see turn_program
    # The inequalities past a reserve's End arm, which DRAW leaves out of a handler's bound. A turn counts its own
    # helpers': the Free it walks is what a handler gave back, and that one does run out.
    beyond_end: dict = field(default_factory=dict)


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


_REG = re.compile(r"r(\d+)$")
_LOC = re.compile(r"r(\d+)\[(\d+):(\d+)\]$")
_LIT = re.compile(r"#(\d+)$")
_INNER = re.compile(r"\[(\d+):(\d+)\]$")
_FUNC = re.compile(r"(\w+)\(\s*\[(\d+):(\d+)\]\s*,\s*(.+)\)$")
_HEADER = re.compile(r"(fn|handler|turn) (f\d+) (\w+)\((.*)\) : (\S+)(?: @(\d+))?$")
_LABEL = re.compile(r"(L\d+):\s*(.*)$")


class Checker:
    """Feed it lines. It raises RTError at the first one that breaks a rule."""

    def __init__(self) -> None:
        self.types: dict[str, TypeDecl] = {}
        self.aliases: dict[str, str] = {}
        self.composite: dict[str, Type] = {}
        self.tuples: dict[str, list[str]] = {}
        self.ctors: dict[str, Ctor] = {}
        self.functions: dict[str, Signature] = {}
        self.states: list[tuple[str, str]] = []
        self.fn: Optional[Signature] = None
        self.S: Optional[State] = None
        self.pending: dict[str, list[State]] = {}
        self.next_word = 0
        self.next_unknown = 0
        self.line_number = 0
        self.certified: dict[str, Bound] = {}

    # --- what the checker concluded, for the driver

    def _handler_named(self, name: str) -> Signature:
        for signature in self.functions.values():
            if signature.name == name:
                return signature
        raise KeyError(name)

    def _rows(self, handler: Signature) -> tuple:
        """Every inequality the handler must meet, which includes those of everything it calls.
        Section 3.1 gives a handler its reserve with nothing on it: those cells are paid for by
        being drawn."""
        if handler.too_large:
            raise Unbounded(f"section 8.2: {handler.name} and what it calls come to more than "
                            f"{MAX_OBLIGATIONS} inequalities, past what one linear program may hold")
        if handler.makes_a_free:
            raise Unbounded(f"section 5.3: {handler.name} or what it calls makes a Free of its own on "
                            f"a path that can run, so an End arm can be reached, and DRAW leaves "
                            f"those out of the bound")
        rows = dict(handler.obligations)
        reserve = handler.param_pots[-1]
        if reserve is not None:
            rows[Budget(((reserve, -1),))] = None
        return tuple(rows)

    def linear_program(self, name: str, n_max: int) -> ResourceProgram:
        handler = self._handler_named(name)
        return ResourceProgram(self._rows(handler), handler.param_pots[-2], handler.entry, n_max)

    def turn_program(self, name: str, sizes: Mapping[int, int]) -> ResourceProgram:
        """A turn's linear program. Each parameter in `sizes` is given an amount per cell, at the most cells it can
        hold, and every other parameter that holds cells is given nothing, since nothing bounds how many it holds:
        the pool. So the bound is for the whole turn, whatever its item, and it holds of every turn on its own."""
        turn = self._handler_named(name)
        if not turn.turn:
            raise RTError(f"{name} is not a turn")
        if turn.too_large:
            raise Unbounded(f"section 8.2: {name} and what it calls come to more than {MAX_OBLIGATIONS} inequalities")
        rows = {**turn.obligations, **turn.beyond_end}
        weights = {}
        for position, pot in enumerate(turn.param_pots):
            if pot is None:
                continue
            if position in sizes:
                weights[pot] = sizes[position]
            else:
                rows[Budget(((pot, -1),))] = None
        return ResourceProgram(tuple(rows), None, turn.entry, 0, weights)

    def verify_turn(self, name: str, sizes: Mapping[int, int], steps: Mapping[int, Fraction]) -> Bound:
        """The certificate for a turn's steps, checked in every inequality: W, with c = 0, since it is for any item."""
        program = self.turn_program(name, sizes)
        if any(value < 0 for value in steps.values()):
            raise RTError(f"the certificate for {name} gives an unknown a negative number of steps")
        for row in program.rows:
            total = sum((coefficient * Fraction(steps.get(unknown, 0)) for unknown, coefficient in row.terms),
                        Fraction(row.steps))
            if total < 0:
                raise RTError(f"the certificate for {name} falls {-total} steps short at {self._show_row(row)}")
        whole = sum((Fraction(weight) * Fraction(steps.get(unknown, 0))
                     for unknown, weight in program.objective().items()), Fraction(0))
        bound = Bound(0, 0, 0, math.ceil(whole))
        self.certified[name] = bound
        return bound

    def verify(self, name: str, cells: Mapping[int, Fraction], steps: Mapping[int, Fraction]) -> Bound:
        """The certificate: proposed numbers for each resource. They are evaluated in every
        inequality this module derived, so nothing is taken from whoever proposed them."""
        handler = self._handler_named(name)
        rows = self._rows(handler)
        for resource, values in (("cells", cells), ("steps", steps)):
            if any(value < 0 for value in values.values()):
                raise RTError(f"the certificate for {name} gives an unknown a negative {resource}")
            for row in rows:
                total = sum((coefficient * Fraction(values.get(unknown, 0))
                             for unknown, coefficient in row.terms), Fraction(getattr(row, resource)))
                if total < 0:
                    raise RTError(f"the certificate for {name} falls {-total} {resource} short at "
                                  f"{self._show_row(row)}")

        def whole(values: Mapping[int, Fraction], unknown: Optional[int]) -> int:
            return 0 if unknown is None else math.ceil(Fraction(values.get(unknown, 0)))

        item = handler.param_pots[-2]
        bound = Bound(whole(cells, item), whole(cells, handler.entry),
                      whole(steps, item), whole(steps, handler.entry))
        self.certified[name] = bound
        return bound

    @staticmethod
    def _show_row(row: Budget) -> str:
        terms = " + ".join(f"{c} u{u}" for u, c in row.terms)
        return f"{terms} + ({row.cells} cells, {row.steps} steps) >= 0"

    def bounds(self, name: str) -> Bound:
        """The admission table of a handler whose certificate has passed."""
        return self.certified[name]

    # --- the potential method

    def unknown(self) -> int:
        self.next_unknown += 1
        return self.next_unknown

    def _carries(self, ty: Type) -> bool:
        """Scalars have no cells, a capability's cell is empty, and an array is not made of cells."""
        return self._decl(ty) is not None

    def _pot_for(self, ty: Type) -> Optional[int]:
        return self.unknown() if self._carries(ty) else None

    def obligate(self, row: Budget, *, beyond_end: bool = False) -> None:
        fn = self.fn
        assert fn is not None
        if not row.terms and row.cells >= 0 and row.steps >= 0:
            return
        if fn.too_large:
            return
        held = fn.beyond_end if beyond_end or (self.S is not None and self.S.unreachable) else fn.obligations
        held[row] = None
        if len(fn.obligations) + len(fn.beyond_end) > MAX_OBLIGATIONS:
            fn.too_large = True
            fn.obligations = {}
            fn.beyond_end = {}

    def _made(self, ty: Type) -> None:
        """DRAW leaves the End arm out because a reserve of R(n) cells does not reach it, and that
        needs every Free to be the reserve. Linearity keeps there from being two of one, so all
        that is left to rule out is making one, which only a literal can do. Past an End arm that
        is fine, because the reserve has gone, and so has everything the bound says."""
        assert self.S is not None and self.fn is not None
        if self.is_reserve(ty) and not self.S.unreachable:
            self.fn.makes_a_free = True

    def at_least(self, have: Optional[int], need: Optional[int]) -> None:
        """A value carrying `have` per cell, used where `need` per cell is wanted. None on the
        left is a null, which has no cells to pay for."""
        if have is None or need is None or have == need:
            return
        self.obligate(Budget().plus(have).plus(need, -1))

    # --- helpers

    def fail(self, message: str) -> RTError:
        where = f"fn {self.fn.name}, " if self.fn is not None else ""
        return RTError(f"RT {where}line {self.line_number}: {message}")

    def _resolve(self, name: str) -> str:
        while name in self.aliases:
            name = self.aliases[name]
        return name

    def norm(self, text: str) -> Type:
        if text in SCALARS or text in WIRE or text in ("bool", "lit", "cap"):
            return text
        name = self._resolve(text)
        if name in self.composite:
            return self.composite[name]
        if name in self.types:
            return ("enum" if self.types[name].kind == "enum" else "named", name)
        raise self.fail(f"unknown type {text}")

    def _decl(self, ty: Type) -> Optional[TypeDecl]:
        return self.types[ty[1]] if isinstance(ty, tuple) and ty[0] == "named" else None

    def is_reserve(self, ty: Type) -> bool:
        decl = self._decl(ty)
        return decl is not None and decl.name == RESERVE

    def unify(self, a: Type, b: Type, where: str) -> Type:
        if a == b:
            return a
        if a == "lit" and b in SCALARS:
            return b
        if b == "lit" and a in SCALARS:
            return a
        raise self.fail(f"{where}: {a} against {b}")

    def fresh(self) -> Word:
        self.next_word += 1
        return ("fresh", self.next_word)

    def fits(self, token: Tok, ty: Type, where: str) -> None:
        """A literal has to be a value of the type it is used at. A register that holds one
        carries its value in its word, until a join with a different one forgets it."""
        if token.word[0] != "lit":
            return
        value = token.word[1]
        limit = 2 if ty == "bool" else 1 << {**SCALARS, **WIRE}[ty] if ty in SCALARS or ty in WIRE else None
        if limit is not None and value >= limit:
            raise self.fail(f"{where}: the literal {value} is not a {ty}")

    # --- reading tokens

    def reg(self, text: str) -> int:
        match = _REG.match(text)
        if match is None:
            raise self.fail(f"{text} is not a register")
        return int(match.group(1))

    def full(self, text: str) -> tuple[int, Tok]:
        r = self.reg(text)
        assert self.S is not None
        token = self.S.regs.get(r)
        if token is None:
            raise self.fail(f"r{r} holds nothing: it was consumed or never set")
        return r, token

    def set(self, text: str, token: Tok) -> None:
        r = self.reg(text)
        assert self.S is not None
        if r in self.S.regs:
            raise self.fail(f"r{r} is already full")
        self.S.regs[r] = token

    def cap(self, text: str) -> tuple[int, Word, Cell]:
        r, token = self.full(text)
        if token.ty != "cap":
            raise self.fail(f"r{r} is a {token.ty}, not a capability")
        assert self.S is not None
        cell = self.S.cells.get(token.word)
        if cell is None:
            raise self.fail(f"r{r} is a capability with no cell behind it")
        return r, token.word, cell

    def location(self, cell: Cell, key: tuple[int, int], base: str) -> Loc:
        found = cell.locs.get(key)
        if found is None:
            raise self.fail(f"{base}[{key[0]}:{key[1]}] is not a location of this cell")
        return found

    def operand(self, text: str, *, consume: bool = True) -> Tok:
        """A register, a literal or a location. Using a scalar consumes nothing. Using a
        linear value consumes it: a register goes free and a location becomes a hole."""
        literal = _LIT.match(text)
        if literal:
            return Tok("lit", ("lit", int(literal.group(1))))
        located = _LOC.match(text)
        if located:
            base, off, size = located.groups()
            _r, _word, cell = self.cap(f"r{base}")
            key = (int(off), int(size))
            loc = self.location(cell, key, f"r{base}")
            if not loc.full:
                raise self.fail(f"r{base}[{off}:{size}] is a hole: its value was taken")
            if consume and not is_scalar(loc.ty):
                cell.locs[key] = Loc(False, loc.word, loc.ty, loc.pot)
            return Tok(loc.ty, loc.word, loc.pot)
        r, token = self.full(text)
        if consume and not is_scalar(token.ty):
            assert self.S is not None
            del self.S.regs[r]
        return token

    def scalar(self, text: str, what: str) -> Tok:
        token = self.operand(text)
        if not is_scalar(token.ty):
            raise self.fail(f"{what} must be scalar, and {text} is {token.ty}")
        return token

    def integer(self, text: str, width: int, what: str, *, count: bool = False) -> None:
        """An operand at this width. A shift count is not a value of the type: section 4.4 gives
        any count a meaning, so it only has to be an integer."""
        token = self.scalar(text, what)
        if token.ty != "lit" and SCALARS.get(token.ty) != width:
            raise self.fail(f"{what} {text} is {token.ty}, and the operation is {width} bits")
        if not count:
            self.fits(token, f"i{width}", what)

    # --- the merge rule

    def merge(self, edges: list[State]) -> State:
        merged = edges[0]
        for other in edges[1:]:
            merged = self._merge_two(merged, other)
        return merged

    def _merge_word(self, a: Word, b: Word) -> Word:
        return a if a == b else self.fresh()

    def _merge_two(self, a: State, b: State) -> State:
        regs = {}
        for r in set(a.regs) | set(b.regs):
            x, y = a.regs.get(r), b.regs.get(r)
            if x is None or y is None:
                present = x if x is not None else y
                assert present is not None
                if not is_scalar(present.ty):
                    raise self.fail(f"r{r} holds {present.ty} on one edge and nothing on another")
                continue
            ty = self.unify(x.ty, y.ty, f"r{r} at a join") if x.ty != y.ty else x.ty
            regs[r] = Tok(ty, self._merge_word(x.word, y.word), self._merge_pot(x.pot, y.pot, a, b))
        cells = {}
        for word in set(a.cells) & set(b.cells):
            cells[word] = self._merge_cell(word, a.cells[word], b.cells[word], a, b)
        for r, token in regs.items():
            if token.ty == "cap" and token.word not in cells:
                # The capability of two different cells: which one is not known here, so all
                # that survives is an empty cell, and anything still in either would be lost.
                for edge in (a, b):
                    self._no_linear_in(edge.cells[edge.regs[r].word], f"r{r} at a join")
                cells[token.word] = Cell()
        held = {token.word for token in regs.values() if token.ty == "cap"}
        cells = {word: cell for word, cell in cells.items() if word in held}
        nulls = a.nulls & b.nulls
        ruled = {}
        for word in set(a.ruled) & set(b.ruled):
            ruled[word] = a.ruled[word] & b.ruled[word]
        return State(regs, cells, nulls, ruled, self._merge_budget(a, b),
                     a.unreachable and b.unreachable)

    def _merge_pot(self, x: Optional[int], y: Optional[int], a: State, b: State) -> Optional[int]:
        """After a join, a value carries what every edge that can run vouches for."""
        if b.unreachable or x == y or y is None:
            return x
        if a.unreachable or x is None:
            return y
        joined = self.unknown()
        self.at_least(x, joined)
        self.at_least(y, joined)
        return joined

    def _merge_budget(self, a: State, b: State) -> Budget:
        """After a join, only what every edge that can run has left can be spent. With the same
        unknowns that is the smaller constant, and needs no new unknown, which keeps an `if` from
        growing the linear program."""
        if b.unreachable:
            return a.budget
        if a.unreachable:
            return b.budget
        if a.budget.terms == b.budget.terms:
            return Budget(a.budget.terms, min(a.budget.cells, b.budget.cells),
                          min(a.budget.steps, b.budget.steps))
        joined = self.unknown()
        self.obligate(a.budget.plus(joined, -1))
        self.obligate(b.budget.plus(joined, -1))
        return Budget().plus(joined)

    def _merge_cell(self, word: Word, a: Cell, b: Cell, edge_a: State, edge_b: State) -> Cell:
        locs = {}
        for key in set(a.locs) | set(b.locs):
            x, y = a.locs.get(key), b.locs.get(key)
            if x is None or y is None:
                present = x if x is not None else y
                assert present is not None
                if not is_scalar(present.ty) and present.full:
                    raise self.fail(f"a linear location of {word} exists on only one edge")
                continue
            if x.ty != y.ty:
                # Two layouts at one offset. A value owed here would be lost, and a scalar or a
                # hole simply has no token after the join, because nothing says which it is.
                if (x.full and not is_scalar(x.ty)) or (y.full and not is_scalar(y.ty)):
                    raise self.fail(f"[{key[0]}:{key[1]}] of {word} holds {x.ty} on one edge and "
                                    f"{y.ty} on another")
                continue
            if x.full != y.full and not is_scalar(x.ty):
                raise self.fail(f"[{key[0]}:{key[1]}] of {word} is full on one edge and a hole "
                                f"on another")
            locs[key] = Loc(x.full and y.full, self._merge_word(x.word, y.word), x.ty,
                            self._merge_pot(x.pot, y.pot, edge_a, edge_b))
        tag = None
        if a.tag == b.tag:
            tag = a.tag
        elif a.tag is not None and b.tag is not None and a.tag[0] == b.tag[0] == "not":
            tag = ("not", a.tag[1] & b.tag[1])
        return Cell(tag, locs)

    def edge(self, label: str) -> None:
        """A jump. One back to a label already placed stays pending, so `end` refuses it: a loop
        is a tail call at this level, and nothing here remembers every label ever placed."""
        assert self.S is not None
        self.pending.setdefault(label, []).append(self.S.copy())

    # --- declarations

    def _declaration(self, head: str, parts: list[str]) -> None:
        if head == "type":
            self.types[parts[0]] = TypeDecl(parts[1], parts[2])
        elif head == "alias":
            self.aliases[parts[0]] = parts[1]
        elif head == "array":
            element = parts[2]
            if element not in SIZES:
                raise self.fail(f"array elements are scalars for now, not {element}")
            self.composite[parts[0]] = ("array", int(parts[1]), element)
        elif head == "fin":
            self.composite[parts[0]] = ("fin", int(parts[1]))
        elif head == "tuple":
            self.tuples[parts[0]] = parts[1:]
        elif head == "ctor":
            ctor = Ctor(parts[0], parts[1], parts[2], parts[3])
            self.ctors[parts[0]] = ctor
            self.types[parts[1]].ctors.append(ctor)
        elif head == "field":
            if parts[3] == "cap":
                raise self.fail(f"field {parts[4]} is a capability: one cannot be stored in a cell")
            ctor = self._ctor_of(parts[0])
            off, size = int(parts[1]), int(parts[2])
            if self.types[self._resolve(ctor.type)].kind == "enum":
                if parts[3] not in SCALARS or SIZES[parts[3]] != size or off % size or not (
                        1 <= off and off + size <= ENUM_BYTES):
                    raise self.fail(f"field {parts[4]} of {ctor.name} is {size} bytes at {off}, and an enum's fields "
                                    f"are integers that fit bytes 1 to {ENUM_BYTES - 1} of its register, after its tag")
            elif self._resolve(parts[3]) in self.types and self.types[self._resolve(parts[3])].kind == "enum":
                raise self.fail(f"field {parts[4]} is an enum: one cannot be stored in a cell yet")
            ctor.fields.append((off, size, parts[3], parts[4]))
        else:
            assert head == "state"
            self.states.append((parts[0], parts[1]))

    def _ctor_of(self, text: str) -> Ctor:
        ctor = self.ctors.get(text)
        if ctor is None:
            raise self.fail(f"unknown constructor {text}")
        return ctor

    def _open_cell(self, word: Word, ctor: Ctor, pot: Optional[int]) -> Cell:
        """A dispatched cell: every field of the constructor is full, each with a word of its
        own. That word cannot be just w.o: a cell that has been rebuilt and is opened again holds
        new values at the same offsets, and a value taken before the rebuild must not pass for
        what is there now, or RESTORE would skip a store that was needed. What a value carries
        per cell is carried by every cell reachable from it, so its linear fields carry the same."""
        locs = {}
        opening = self.fresh()
        for off, size, ty, _name in ctor.fields:
            norm = self.norm(ty)
            locs[(off, size)] = Loc(True, ("field", opening, off), norm,
                                    None if is_scalar(norm) else pot)
        tag = ("is", int(ctor.tag)) if ctor.tag.isdigit() else None
        return Cell(tag, locs)

    # --- the fold

    def feed(self, line: str) -> None:
        self.line_number += 1
        text = line.split(";")[0].strip()
        if not text:
            return
        label = _LABEL.match(text)
        if label:
            self._label(label.group(1))
            text = label.group(2)
            if not text:
                return
        head, _, rest = text.partition(" ")
        if head in ("type", "alias", "array", "fin", "tuple", "ctor", "field", "state"):
            if self.fn is not None:
                raise self.fail("a declaration inside a function")
            self._declaration(head, rest.split())
            return
        if head in ("fn", "handler", "turn"):
            self._header(text)
            return
        if self.fn is None:
            raise self.fail(f"{head} outside a function")
        if head == "end":
            self._end(rest)
            return
        if self.S is None:
            raise self.fail(f"{head} cannot be reached: control left before it")
        if head not in NO_WORK:
            self.S.budget = self.S.budget.spend(steps=1)
        handler = getattr(self, "i_" + head.replace(".", "_"), None)
        if handler is None:
            raise self.fail(f"unknown instruction {head}")
        handler(_args(rest))

    def finish(self) -> None:
        if self.fn is not None:
            raise self.fail("the stream ended inside a function")

    def _header(self, text: str) -> None:
        match = _HEADER.match(text)
        if match is None:
            raise self.fail(f"malformed function header: {text}")
        if self.fn is not None:
            raise self.fail("a function inside a function")
        kind, ordinal, name, params, ret, at = match.groups()
        parts = _args(params)
        signature = Signature(ordinal, name, [], ret, int(at) if at else None, kind == "handler", turn=kind == "turn")
        signature.entry = self.unknown()
        regs = {}
        for position, part in enumerate(parts):
            register, _, ty = part.partition(" : ")
            if self.reg(register) != position:
                raise self.fail(f"parameter {position} must be r{position}")
            signature.params.append(ty.strip())
            norm = self.norm(ty.strip())
            pot = self._pot_for(norm)
            signature.param_pots.append(pot)
            regs[position] = Tok(norm, ("arg", position), pot)
        members = self.tuples.get(self._resolve(ret), [ret])
        signature.result_pots = [self._pot_for(self.norm(member)) for member in members]
        self.fn = signature
        self.S = State(regs, {}, frozenset(), {}, Budget().plus(signature.entry))
        for position, token in regs.items():
            if token.ty == "cap":
                self.S.cells[token.word] = Cell()
        self.pending = {}
        if signature.handler:
            self._check_handler_shape(signature)
        if at is not None and not (0 <= int(at) < len(parts)):
            raise self.fail(f"@{at} names no parameter")
        if at is not None and self._resolve(ret) in self.tuples:
            raise self.fail(f"@{at} on a function that returns a tuple: which member would it be")

    def _check_handler_shape(self, signature: Signature) -> None:
        """Section 3.1: the state, the item and the reserve, in and back out."""
        results = self.tuples.get(self._resolve(signature.ret))
        count = len(self.states)
        verdict = results is not None and len(results) == count + 3 and is_enum(self.norm(results[count]))
        if results is None or len(results) != count + 2 + verdict or len(signature.params) != count + 2:
            raise self.fail("a handler takes its state, an item and a Free, and returns its "
                            "state, an output and a Free, or its state, a verdict, a record and a Free")
        for position, (_name, ty) in enumerate(self.states):
            if self.norm(signature.params[position]) != self.norm(ty):
                raise self.fail(f"parameter {position} of a handler must be state slot {position}")
            if self.norm(results[position]) != self.norm(ty):
                raise self.fail(f"result {position} of a handler must be state slot {position}")
        if not self.is_reserve(self.norm(signature.params[-1])) or not self.is_reserve(
                self.norm(results[-1])):
            raise self.fail("a handler's last parameter and last result must be Free")

    def _end(self, rest: str) -> None:
        if self.S is not None:
            raise self.fail("control runs off the end of the function")
        if self.pending:
            raise self.fail(f"labels never placed after a jump to them, or jumped back to: "
                            f"{sorted(self.pending)}")
        fn = self.fn
        assert fn is not None
        self._tie_decreasing_set(fn, rest)
        self.functions[fn.ordinal] = fn
        self.fn = None
        self.S = None

    def _tie_decreasing_set(self, fn: Signature, rest: str) -> None:
        """Rule R3's set carries the recursion: one amount per cell across all of it, since its
        total falls at each recursive call whichever member the cell left."""
        words = rest.split()
        if not words:
            if fn.recursive:
                raise self.fail(f"{fn.name} is recursive, and `end` names no decreasing set")
            return
        if words[0] != "decreasing" or len(words) != 2:
            raise self.fail(f"malformed end: {rest}")
        positions = [int(p) for p in words[1].split(",")]
        pots = [fn.param_pots[p] for p in positions]
        if any(pot is None for pot in pots):
            raise self.fail(f"a decreasing set of {fn.name} names a parameter with no cells")
        for first, second in zip(pots, pots[1:]):
            self.at_least(first, second)
            self.at_least(second, first)

    def _label(self, label: str) -> None:
        if self.fn is None:
            raise self.fail("a label outside a function")
        edges = self.pending.pop(label, [])
        if self.S is not None:
            edges.append(self.S)
        if not edges:
            raise self.fail(f"label {label} has no incoming edge")
        self.S = None  # the rows of a join belong to its edges, whatever path fell into it
        self.S = self.merge(edges)

    # --- values

    def i_const(self, a: list[str]) -> None:
        literal, _, ty = a[1].partition(" : ")
        match = _LIT.match(literal.strip())
        if match is None:
            raise self.fail(f"const needs a literal, not {literal}")
        value = int(match.group(1))
        norm = self.norm(ty.strip())
        word = self.fresh() if norm != "lit" else ("lit", value)
        if is_fin(norm):
            raise self.fail(f"const of {norm}: the only way to make a Fin is idx")
        if norm in SCALARS or norm == "bool":
            self.fits(Tok("lit", ("lit", value)), norm, "const")
        if is_enum(norm):
            raise self.fail(f"const of {a[1]}: an enum is made by enum, from its constructor")
        if not is_scalar(norm):
            decl = self._decl(norm)
            if decl is None or decl.kind != "tagless" or value != 0:
                raise self.fail(f"only a tagless type has a literal, and only 0: {a[1]}")
            word = ("null",)
            self._made(norm)
        self.set(a[0], Tok(norm, word))

    def i_move(self, a: list[str]) -> None:
        r, token = self.full(a[1])
        assert self.S is not None
        del self.S.regs[r]
        self.set(a[0], token)

    def i_copy(self, a: list[str]) -> None:
        _r, token = self.full(a[1])
        if not is_scalar(token.ty):
            raise self.fail(f"copy of {token.ty}: only a scalar can be copied")
        self.set(a[0], token)

    def i_drop(self, a: list[str]) -> None:
        r, token = self.full(a[0])
        if not is_scalar(token.ty):
            raise self.fail(f"drop of {token.ty}: only a scalar can be dropped")
        assert self.S is not None
        del self.S.regs[r]

    def i_op(self, a: list[str]) -> None:
        name, _, bits = a[1].partition("/")
        if name not in OP_NAMES or int(bits) not in SCALARS.values():
            raise self.fail(f"unknown operation {a[1]}")
        width = int(bits)
        self.integer(a[2], width, "an operand")
        self.integer(a[3], width, "an operand", count=name in ("shl", "shr"))
        self.set(a[0], Tok(f"i{width}", self.fresh()))

    def _comparable(self, left: str, right: str) -> None:
        x, y = self.scalar(left, "a comparison operand"), self.scalar(right, "a comparison operand")
        if is_enum(x.ty) or is_enum(y.ty):
            raise self.fail(f"a comparison of {x.ty} with {y.ty}: an enum is told apart by matching it, which is "
                            f"not built yet")
        if x.ty in WIRE or y.ty in WIRE:
            raise self.fail(f"a comparison of {x.ty} with {y.ty}: bits in network order have to be swapped first")
        if x.ty != y.ty and "lit" not in (x.ty, y.ty):
            raise self.fail(f"a comparison of {x.ty} with {y.ty}")
        self.fits(x, y.ty, "a comparison")
        self.fits(y, x.ty, "a comparison")

    def i_cmp(self, a: list[str]) -> None:
        if a[1] not in CC_NAMES:
            raise self.fail(f"unknown condition {a[1]}")
        self._comparable(a[2], a[3])
        self.set(a[0], Tok("bool", self.fresh()))

    def i_trunc(self, a: list[str]) -> None:
        width = int(a[1])
        token = self.scalar(a[2], "trunc's operand")
        if token.ty != "lit" and SCALARS.get(token.ty, 0) < width:
            raise self.fail(f"trunc of {token.ty} to {width} bits widens it")
        self.set(a[0], Tok(f"i{width}", self.fresh()))

    def i_pack(self, a: list[str]) -> None:
        widths = [int(w) for w in a[1].strip("<>").split()]
        if len(widths) != len(a) - 2:
            raise self.fail("pack needs one operand per width")
        for operand in a[2:]:
            ty = self.scalar(operand, "a pack field").ty
            if ty in WIRE:
                raise self.fail(f"a pack field {operand} is in network order, and has to be swapped first")
            if is_enum(ty):
                raise self.fail(f"a pack field {operand} is an enum, and a field is an integer")
        self.set(a[0], Tok("i64", self.fresh()))

    def _an_enum(self, text: str, what: str) -> tuple:
        token = self.scalar(text, what)
        if not is_enum(token.ty):
            raise self.fail(f"{what} {text} is {token.ty}, and only an enum holds its tag and fields in a register")
        return token.ty

    def i_etag(self, a: list[str]) -> None:
        """etag d, e: the tag of an enum, which is its low byte."""
        self._an_enum(a[1], "etag's operand")
        self.set(a[0], Tok("i8", self.fresh()))

    def i_efield(self, a: list[str]) -> None:
        """efield d, e, c, i: field i of constructor c, out of an enum. It reads bits, so it is defined whatever
        the tag says, and what it means when the tag is another constructor's is the symbolic check's business."""
        ty = self._an_enum(a[1], "efield's operand")
        ctor = self._ctor_of(a[2])
        if self._resolve(ctor.type) != ty[1]:
            raise self.fail(f"efield of {ctor.name}, which is not a constructor of {ty[1]}")
        index = int(a[3])
        if not 0 <= index < len(ctor.fields):
            raise self.fail(f"efield: {ctor.name} has no field {index}")
        self.set(a[0], Tok(ctor.fields[index][2], self.fresh()))

    def i_enum(self, a: list[str]) -> None:
        """enum d, c, a...: a value of an enum, in a register, from its constructor and a scalar per field."""
        ctor = self._ctor_of(a[1])
        enum = self._resolve(ctor.type)
        if self.types[enum].kind != "enum":
            raise self.fail(f"enum of {ctor.name}: {enum} is not an enum")
        if len(a) - 2 != len(ctor.fields):
            raise self.fail(f"{ctor.name} has {len(ctor.fields)} fields, and this gives {len(a) - 2} operands")
        for (_off, size, _ty, name), operand in zip(ctor.fields, a[2:]):
            self.integer(operand, 8 * size, f"field {name} of {ctor.name}")
        self.set(a[0], Tok(("enum", enum), self.fresh()))

    def i_bswap(self, a: list[str]) -> None:
        """bswap d, W, a. The one way between bits in network order and the integer they are, either way."""
        width = int(a[1])
        if width not in WIRE.values():
            raise self.fail(f"bswap of {width} bits: a byte has no order, and nothing else is that wide")
        if _LIT.match(a[2]):
            raise self.fail(f"bswap of {a[2]}: a literal is written already swapped, as a build's source")
        token = self.scalar(a[2], "bswap's operand")
        wire, integer = f"be{width}", f"i{width}"
        if token.ty == wire:
            result = integer
        elif token.ty in (integer, "lit"):
            self.fits(token, integer, "bswap's operand")
            result = wire
        else:
            raise self.fail(f"bswap of {width} bits on {a[2]}, which is {token.ty}")
        self.set(a[0], Tok(result, self.fresh()))

    # --- arrays

    def i_idx(self, a: list[str]) -> None:
        bound = int(a[1])
        if bound <= 0 or bound & bound - 1:
            raise self.fail(f"idx needs a power of two until division exists, not {bound}")
        token = self.scalar(a[2], "idx's operand")
        if not is_int(token.ty):
            raise self.fail(f"idx of {token.ty}")
        self.set(a[0], Tok(("fin", bound), self.fresh()))

    def i_aget(self, a: list[str]) -> None:
        _r, array = self.full(a[1])
        if not (isinstance(array.ty, tuple) and array.ty[0] == "array"):
            raise self.fail(f"aget on {array.ty}")
        _kind, bound, element = array.ty
        _i, index = self.full(a[2])
        if index.ty != ("fin", bound):
            raise self.fail(f"index of {index.ty} into an array of {bound}")
        if SIZES[element] != int(a[3]):
            raise self.fail(f"aget of {a[3]} bytes from an array of {element}")
        self.set(a[0], Tok(element, self.fresh()))

    # --- cells

    def i_read(self, a: list[str]) -> None:
        if not _LOC.match(a[1]):
            raise self.fail(f"read needs a location, not {a[1]}")
        token = self.operand(a[1], consume=False)
        if not is_scalar(token.ty):
            raise self.fail(f"read of a linear location: {a[1]} has to be taken")
        self.set(a[0], token)

    def i_take(self, a: list[str]) -> None:
        if not _LOC.match(a[1]):
            raise self.fail(f"take needs a location, not {a[1]}")
        token = self.operand(a[1], consume=False)
        if is_scalar(token.ty):
            raise self.fail(f"take of a scalar: {a[1]} is read")
        self.operand(a[1])
        self.set(a[0], token)

    def _build(self, base: str, ctor_id: str, tag: str, sources: list[str],
               star: Optional[Tok]) -> tuple[int, Ctor, int]:
        """The atomic constructor. `star` is the value `*` stands for, or None if it is banned."""
        r, word, cell = self.cap(base)
        ctor = self._ctor_of(ctor_id)
        decl = self.types[ctor.type]
        if len(sources) != len(ctor.fields):
            raise self.fail(f"{ctor.name} has {len(ctor.fields)} fields, and build gives "
                            f"{len(sources)} sources")
        named = set()
        stars = 0
        carried = self.unknown()  # what the new value carries per cell
        for source, (off, size, ty_text, name) in zip(sources, ctor.fields):
            ty = self.norm(ty_text)
            key = (off, size)
            where = f"{ctor.name}.{name}"
            held = cell.locs.get(key)
            if source == "=":
                self.at_least(held.pot if held is not None else None, None if is_scalar(ty) else carried)
                if held is None or not held.full:
                    raise self.fail(f"{where} says STAY, and [{off}:{size}] holds no full value")
                if held.ty != ty and not (is_int(held.ty) and is_int(ty)):
                    raise self.fail(f"{where} says STAY over a {held.ty}, and the field is {ty}")
                self._name_once(named, key, held, where)
            elif source == "*":
                stars += 1
                if star is None:
                    raise self.fail(f"{where}: `*` outside tailmod")
                if held is None or held.full or held.word != star.word:
                    raise self.fail(f"{where}: `*` needs a hole holding the word the call returns")
                self.unify(star.ty, ty, where)
                self.at_least(star.pot, carried)
            elif source.endswith("!"):
                r_source, token = self.full(source[:-1])
                if held is None or held.full or held.word != token.word:
                    raise self.fail(f"{where} says RESTORE, and [{off}:{size}] is not a hole "
                                    f"holding {token.word}")
                self.unify(token.ty, ty, where)
                self.at_least(token.pot, carried)
                assert self.S is not None
                del self.S.regs[r_source]
            elif _LIT.match(source):
                self._literal_fits(source, ty, where)
            elif _INNER.match(source):
                other = self._inner(source)
                found = self.location(cell, other, base)
                if other == key:
                    raise self.fail(f"{where} stores [{off}:{size}] over itself: that is `=`")
                if not found.full:
                    raise self.fail(f"{where}: {source} is a hole")
                self.unify(found.ty, ty, where)
                self._name_once(named, other, found, where)
                self.at_least(found.pot, None if is_scalar(ty) else carried)
            elif _FUNC.match(source):
                self._computed(source, cell, ty, where, key)
            else:
                token = self.operand(source)
                if held is not None and held.word == token.word:
                    raise self.fail(f"{where} stores a value back where it came from: that is "
                                    f"{'`=`' if held.full else 'RESTORE'}, Theorem 6.1")
                self.unify(token.ty, ty, where)
                self.fits(token, ty, where)
                self.at_least(token.pot, carried)
        if star is not None and stars != 1:
            raise self.fail("tailmod needs exactly one `*`")
        for key, loc in cell.locs.items():
            if loc.full and not is_scalar(loc.ty) and key not in named:
                raise self.fail(f"[{key[0]}:{key[1]}] still holds a linear {loc.ty} that "
                                f"{ctor.name} does not name, so it would be lost")
        self._tag_rule(tag, ctor, decl, cell)
        assert self.S is not None
        self.S.budget = self.S.budget.plus(carried, -1)  # the new cell's share, paid now
        return r, ctor, carried

    def _name_once(self, named: set, key: tuple[int, int], loc: Loc, where: str) -> None:
        """A linear value can stay or move once. Two sources naming it would be two owners."""
        if key in named and not is_scalar(loc.ty):
            raise self.fail(f"{where}: [{key[0]}:{key[1]}] is a linear {loc.ty}, and another "
                            f"field already takes it")
        named.add(key)

    @staticmethod
    def _inner(source: str) -> tuple[int, int]:
        match = _INNER.match(source)
        assert match is not None
        return int(match.group(1)), int(match.group(2))

    def _literal_fits(self, source: str, ty: Type, where: str) -> None:
        match = _LIT.match(source)
        assert match is not None
        if is_fin(ty):
            raise self.fail(f"{where}: a literal cannot be a {ty}: the only way to make one is idx")
        if is_scalar(ty):
            self.fits(Tok("lit", ("lit", int(match.group(1)))), ty, where)
            return
        decl = self._decl(ty)
        if decl is None or decl.kind != "tagless" or int(match.group(1)) != 0:
            raise self.fail(f"{where}: {source} cannot be a {ty}")
        self._made(ty)

    def _computed(self, source: str, cell: Cell, ty: Type, where: str, own: tuple[int, int]) -> None:
        match = _FUNC.match(source)
        assert match is not None
        name, off, size, operand = match.groups()
        if name not in READ_MODIFY_WRITE:
            raise self.fail(f"{where}: {name} is not an operation a field can do to itself")
        key = (int(off), int(size))
        if key != own:
            raise self.fail(f"{where}: a computed source works in place, and [{off}:{size}] is not "
                            f"this field: compute it into a register with op")
        loc = cell.locs.get(key)
        if loc is None or not loc.full or not is_int(loc.ty):
            raise self.fail(f"{where}: [{off}:{size}] holds no full integer")
        self.unify(loc.ty, ty, where)
        if _LOC.match(operand):
            raise self.fail(f"{where}: a computed source's operand is a register or a literal")
        token = self.scalar(operand, "a computed source's operand")
        if token.ty in WIRE:
            raise self.fail(f"{where}: {operand} is in network order, and has to be swapped first")
        self.fits(token, ty, where)

    def _tag_rule(self, tag: str, ctor: Ctor, decl: TypeDecl, cell: Cell) -> None:
        if decl.kind != "tagged":
            if tag != "none":
                raise self.fail(f"{ctor.name} has no tag, so build must say none, not {tag}")
            return
        known = cell.tag == ("is", int(ctor.tag))
        if tag == "keep" and not known:
            raise self.fail(f"keep: nothing shows the cell already holds tag {ctor.tag}")
        if tag not in ("keep", "write"):
            raise self.fail(f"{ctor.name} is tagged, so build must say keep or write, not {tag}")

    def i_build(self, a: list[str]) -> None:
        r, ctor, carried = self._build(a[0], a[1], a[2], a[3:], None)
        assert self.S is not None
        word = self.S.regs[r].word
        self.S.regs[r] = Tok(("named", self._resolve(ctor.type)), word, carried)
        del self.S.cells[word]
        self.S.ruled.pop(word, None)  # what a test ruled out was about what the cell used to hold

    # --- control

    def i_jump(self, a: list[str]) -> None:
        self.edge(a[0])
        self.S = None

    def i_br(self, a: list[str]) -> None:
        if a[0] not in CC_NAMES:
            raise self.fail(f"unknown condition {a[0]}")
        self._comparable(a[1].rstrip("!"), a[2].rstrip("!"))
        for operand in (a[1], a[2]):  # the comparison was its last use, so it goes free
            if operand.endswith("!"):
                assert self.S is not None
                del self.S.regs[self.reg(operand[:-1])]
        self.edge(a[3])
        self.edge(a[4])
        self.S = None

    def _dispatched(self, r: int, token: Tok, ctor: Ctor) -> None:
        """The cell opens, and what it carried becomes spendable."""
        assert self.S is not None
        self.S.regs[r] = Tok("cap", token.word)
        self.S.cells[token.word] = self._open_cell(token.word, ctor, token.pot)
        self.S.budget = self.S.budget.plus(token.pot)

    def i_br_null(self, a: list[str]) -> None:
        r, token = self.full(a[0])
        decl = self._decl(token.ty)
        if decl is None or decl.kind != "tagless":
            raise self.fail(f"br.null on {token.ty}: it is not a tagless type")
        assert self.S is not None
        base = self.S
        null_edge = base.copy()
        del null_edge.regs[r]
        null_edge.nulls = null_edge.nulls | {token.word}
        reserve = self.is_reserve(token.ty)
        # Section 5.3's DRAW: the arm with a cell costs a cell. The End arm has to type, and it is
        # not charged, because a reserve that holds R(n) cells does not reach it: see _made.
        null_edge.unreachable = base.unreachable or reserve
        self.S = null_edge
        self.edge(a[1])
        full_ctor = next(c for c in decl.ctors if c.tag != "null")
        self.S = base
        if reserve:
            self.S.budget = self.S.budget.spend(cells=1)
        self._dispatched(r, token, full_ctor)
        self.edge(a[2])
        self.S = None

    def i_br_tag(self, a: list[str]) -> None:
        r, token = self.full(a[0])
        decl = self._decl(token.ty)
        if decl is None or decl.kind != "tagged":
            raise self.fail(f"br.tag on {token.ty}: it is not a tagged type")
        tag = int(a[1])
        ctor = next((c for c in decl.ctors if c.tag == str(tag)), None)
        if ctor is None:
            raise self.fail(f"{decl.name} has no tag {tag}")
        assert self.S is not None
        ruled = self.S.ruled.get(token.word, frozenset())
        if tag in ruled:
            raise self.fail(f"tag {tag} of {decl.name} is already ruled out")
        base = self.S
        yes = base.copy()
        self.S = yes
        self._dispatched(r, token, ctor)
        self.edge(a[2])
        no = base.copy()
        no.ruled[token.word] = ruled | {tag}
        self.S = no
        self.edge(a[3])
        self.S = None

    def i_open(self, a: list[str]) -> None:
        r, token = self.full(a[0])
        decl = self._decl(token.ty)
        ctor = self._ctor_of(a[1])
        if decl is None or ctor not in decl.ctors:
            raise self.fail(f"open of {token.ty} as {ctor.name}")
        assert self.S is not None
        ruled = self.S.ruled.get(token.word, frozenset())
        left = [c for c in decl.ctors if not c.tag.isdigit() or int(c.tag) not in ruled]
        if left != [ctor]:
            raise self.fail(f"open as {ctor.name} with {[c.name for c in left]} still possible")
        self._dispatched(r, token, ctor)

    # --- calls and exits

    def _consume_arguments(self, signature: Signature, operands: list[str]) -> list[Tok]:
        if len(operands) != len(signature.params):
            raise self.fail(f"{signature.name} takes {len(signature.params)} arguments, "
                            f"not {len(operands)}")
        tokens = []
        for position, (operand, ty) in enumerate(zip(operands, signature.params)):
            want = self.norm(ty)
            token = self.operand(operand)
            self.fits(token, want, f"an argument of {signature.name}")
            self.at_least(token.pot, signature.param_pots[position])
            if want == "cap" or token.ty == "cap":
                if token.ty != want:
                    raise self.fail(f"argument {operand} is {token.ty}, and {signature.name} "
                                    f"wants {want}")
                self._hand_over_cell(token.word, operand)
            elif token.ty == "lit" and not self._literal_argument(token, want):
                raise self.fail(f"the literal {operand} cannot be a {want}")
            elif token.ty != "lit":
                self.unify(token.ty, want, f"an argument of {signature.name}")
            tokens.append(token)
        return tokens

    def _literal_argument(self, token: Tok, want: Type) -> bool:
        value = token.word[1] if token.word[0] == "lit" else None
        if want == "bool":
            return value is not None
        if is_scalar(want):
            return is_int(want)
        decl = self._decl(want)
        if decl is None or decl.kind != "tagless" or value != 0:
            return False
        self._made(want)
        return True

    def _hand_over_cell(self, word: Word, operand: str) -> None:
        assert self.S is not None
        cell = self.S.cells.pop(word, None)
        if cell is None:
            raise self.fail(f"{operand} is a capability with no cell behind it")
        self._no_linear_in(cell, f"{operand} leaves")

    def _no_linear_in(self, cell: Cell, where: str) -> None:
        for key, loc in cell.locs.items():
            if loc.full and not is_scalar(loc.ty):
                raise self.fail(f"{where} with a linear value still at [{key[0]}:{key[1]}]")

    def _callee(self, ordinal: str) -> Signature:
        """Whoever is called, the caller brings what it asks for."""
        fn = self.fn
        assert fn is not None and self.S is not None
        signature = self.functions.get(ordinal)
        if signature is None:
            if ordinal != fn.ordinal:
                raise self.fail(f"{ordinal} is not defined yet: a callee comes before its caller")
            fn.recursive = True
            signature = fn
        else:
            signature = self._at_its_own_numbers(signature)
        self.S.budget = self.S.budget.plus(signature.entry, -1)
        return signature

    def _at_its_own_numbers(self, callee: Signature) -> Signature:
        """A call of a finished function gets its own copy of the callee's unknowns and of every
        inequality over them, which is the callee's derivation again with other numbers in it. One
        set of numbers per function would refuse linear programs: reversing what a reversal
        returned would need the function's result to carry its parameter's amount, which pays for
        the walk and has its result's amount left, so that amount would have to exceed itself.
        Only a recursive call stays at the numbers of the call it is part of.

        A callee with no cells going in or out needs no copy, because nothing connects its calls:
        they all want one smallest entry, so sharing it costs nothing and keeps the rows few."""
        fn = self.fn
        assert fn is not None
        renamed: dict[int, int] = {}
        carries_nothing = all(pot is None for pot in (*callee.param_pots, *callee.result_pots))

        def copy(unknown: int) -> int:
            if carries_nothing:
                return unknown
            if unknown not in renamed:
                renamed[unknown] = self.unknown()
            return renamed[unknown]

        def copy_pot(unknown: Optional[int]) -> Optional[int]:
            return None if unknown is None else copy(unknown)

        fn.too_large = fn.too_large or callee.too_large
        if self.S is not None and not self.S.unreachable:
            fn.makes_a_free = fn.makes_a_free or callee.makes_a_free
        for row in callee.obligations:
            self.obligate(Budget(tuple(sorted((copy(u), c) for u, c in row.terms)), row.cells, row.steps))
        for row in callee.beyond_end:
            self.obligate(Budget(tuple(sorted((copy(u), c) for u, c in row.terms)), row.cells, row.steps),
                          beyond_end=True)
        return replace(callee, entry=copy(callee.entry),
                       param_pots=[copy_pot(pot) for pot in callee.param_pots],
                       result_pots=[copy_pot(pot) for pot in callee.result_pots])

    def i_call(self, a: list[str]) -> None:
        signature = self._callee(a[1])
        assert self.fn is not None
        if signature is self.fn:
            raise self.fail("a non-tail recursive call: DESCEND and ASCEND do not exist yet")
        tokens = self._consume_arguments(signature, a[2:])
        word = tokens[signature.at].word if signature.at is not None else self.fresh()
        results = self.tuples.get(self._resolve(signature.ret))
        if results is None:
            self._result(a[0], self.norm(signature.ret), word, signature.result_pots[0])
            return
        names = _args(a[0].strip("()"))
        if len(names) != len(results):
            raise self.fail(f"{signature.name} returns {len(results)} values")
        for name, ty, pot in zip(names, results, signature.result_pots):
            self._result(name, self.norm(ty), self.fresh(), pot)

    def _result(self, register: str, ty: Type, word: Word, pot: Optional[int]) -> None:
        self.set(register, Tok(ty, word, pot))
        if ty == "cap":  # a capability that comes back is an empty cell, whatever it held before
            assert self.S is not None
            self.S.cells[word] = Cell()

    def _no_linear_left(self, where: str) -> None:
        assert self.S is not None
        for r, token in self.S.regs.items():
            if not is_scalar(token.ty):
                raise self.fail(f"{where} with a linear value still in r{r}: {token.ty}")
        if self.S.cells:
            raise self.fail(f"{where} with a cell nothing holds: {sorted(self.S.cells)}")

    def _word_leaving(self, word: Word, where: str) -> None:
        fn = self.fn
        assert fn is not None and self.S is not None
        if fn.at is None:
            return
        wanted = ("arg", fn.at)
        if word != wanted and not (word == ("null",) and wanted in self.S.nulls):
            raise self.fail(f"{where}: {fn.name} is declared @{fn.at}, and this value has word "
                            f"{word}")

    def i_ret(self, a: list[str]) -> None:
        fn = self.fn
        assert fn is not None
        results = self.tuples.get(self._resolve(fn.ret))
        wanted = results if results is not None else [fn.ret]
        if len(a) != len(wanted):
            raise self.fail(f"ret of {len(a)} values from a function returning {len(wanted)}")
        for position, (operand, ty) in enumerate(zip(a, wanted)):
            if _LIT.match(operand):
                word = self._null_result(operand, self.norm(ty))
                if results is None:
                    self._word_leaving(word, "ret")
                continue
            r, token = self.full(operand)
            self.unify(token.ty, self.norm(ty), "a result")
            self.fits(token, self.norm(ty), "a result")
            self.at_least(token.pot, fn.result_pots[position])
            if not is_scalar(token.ty):
                assert self.S is not None
                del self.S.regs[r]
            if token.ty == "cap":
                self._hand_over_cell(token.word, operand)
            if results is None:
                self._word_leaving(token.word, "ret")
        self._no_linear_left("ret")
        assert self.S is not None
        self.obligate(self.S.budget)
        self.S = None

    def _null_result(self, operand: str, ty: Type) -> Word:
        """`ret #0`: the null of a tagless type, which needs no register."""
        decl = self._decl(ty)
        if decl is None or decl.kind != "tagless" or operand != "#0":
            raise self.fail(f"only a tagless type has a literal, and only 0: ret {operand} of {ty}")
        self._made(ty)
        return ("null",)

    def i_tail(self, a: list[str]) -> None:
        signature = self._callee(a[0])
        fn = self.fn
        assert fn is not None
        if signature.ret != fn.ret:
            raise self.fail(f"tail call of {signature.name}, which returns something else")
        tokens = self._consume_arguments(signature, a[1:])
        if fn.at is not None:
            if signature.at is None:
                raise self.fail(f"{fn.name} is declared @{fn.at}, and {signature.name} is not")
            self._word_leaving(tokens[signature.at].word, "tail")
        self._no_linear_left("tail")
        if signature is not fn:  # what it returns, this function returns
            for theirs, ours in zip(signature.result_pots, fn.result_pots):
                self.at_least(theirs, ours)
        assert self.S is not None
        self.obligate(self.S.budget)
        self.S = None

    def i_tailmod(self, a: list[str]) -> None:
        """tailmod p, C, tag, s_1 ... s_n, f, o_1 ... o_m: consume the operands, then build,
        then consume p, then jump. The count of sources comes from C."""
        ctor = self._ctor_of(a[1])
        count = len(ctor.fields)
        sources = a[3:3 + count]
        ordinal = a[3 + count]
        operands = a[4 + count:]
        signature = self._callee(ordinal)
        fn = self.fn
        assert fn is not None
        if signature is not fn:
            raise self.fail(f"tailmod of {signature.name} from {fn.name}: `*` links to what the "
                            f"next iteration of this function builds, so it can only be itself")
        if fn.at is None:
            raise self.fail(f"tailmod in {fn.name}, which is not declared @")
        tokens = self._consume_arguments(signature, operands)
        star = Tok(self.norm(signature.ret), tokens[fn.at].word, fn.result_pots[0])
        r, _built, carried = self._build(a[0], a[1], a[2], sources, star)
        self.at_least(carried, fn.result_pots[0])  # the root that comes back is one of these cells
        assert self.S is not None
        token = self.S.regs[r]
        self._word_leaving(token.word, "tailmod")
        del self.S.regs[r]
        del self.S.cells[token.word]
        self._no_linear_left("tailmod")
        self.obligate(self.S.budget)
        self.S = None


def check_stream(lines: Iterable[str]) -> Checker:
    checker = Checker()
    for line in lines:
        checker.feed(line)
    checker.finish()
    return checker
