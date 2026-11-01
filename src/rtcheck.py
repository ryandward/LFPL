"""The register-transfer checker: docs/rt-layer.md, one line at a time.

Lower writes the stream and this types it, sharing no code, which a test enforces by reading
the imports. Every rule moves tokens and creates none, so a value used twice, a store that was
skipped when it was needed, or a read of a cell that has been rebuilt is a local type error
here, found without running anything.

It keeps the first specification's memory bound: the only state is the tokens of the function
being read, and one merged token state for each label still open. Control flow is structured,
so that is at most one state per enclosing construct.

A pass over every instruction can also measure, so this is where a handler's bounds come from
until LFPL-O section 5 exists. A function with no recursion has no back-edge, so the number of
instructions in it, plus those of what it calls, bounds every path through it. A `br.null` on
a Free is a draw from the reserve. A recursive function has no bound, and a handler that
reaches one is refused.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Union

SCALARS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64}
SIZES = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "bool": 1}
OP_NAMES = {"add", "sub", "mul", "and", "or", "xor", "shl", "shr"}
CC_NAMES = {"b", "be", "a", "ae", "e", "ne"}
RESERVE = "Free"

Type = Union[str, tuple]  # a builtin name, ("named", id), ("array", N, t) or ("fin", N)
Word = tuple


class RTError(Exception):
    """The stream breaks a typing rule. Lower wrote something it should not have."""


class Unbounded(RTError):
    """A handler has no derivable bound: a limit of the language, LFPL-O section 8.2."""


def is_scalar(ty: Type) -> bool:
    return ty in SCALARS or ty in ("bool", "lit") or (isinstance(ty, tuple) and ty[0] == "fin")


def is_int(ty: Type) -> bool:
    return ty in SCALARS or ty == "lit"


@dataclass(frozen=True)
class Tok:
    ty: Type
    word: Word


@dataclass(frozen=True)
class Loc:
    full: bool
    word: Word
    ty: Type


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

    def copy(self) -> "State":
        return State(dict(self.regs),
                     {c: Cell(v.tag, dict(v.locs)) for c, v in self.cells.items()},
                     self.nulls, dict(self.ruled))


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
    steps: Optional[int] = None  # None: no bound, because recursion is reachable from here
    draws: Optional[int] = None


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
_HEADER = re.compile(r"(fn|handler) (f\d+) (\w+)\((.*)\) : (\S+)(?: @(\d+))?$")
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
        self.line_number = 0
        self.steps = 0
        self.draws = 0
        self.callee_steps: Optional[int] = 0
        self.callee_draws: Optional[int] = 0
        self.recursive = False

    # --- what the checker concluded, for the driver

    def bounds(self, name: str) -> tuple[int, int]:
        """(R, W) for a handler that has passed."""
        for signature in self.functions.values():
            if signature.name == name:
                assert signature.draws is not None and signature.steps is not None
                return signature.draws, signature.steps
        raise KeyError(name)

    # --- helpers

    def fail(self, message: str) -> RTError:
        where = f"fn {self.fn.name}, " if self.fn is not None else ""
        return RTError(f"RT {where}line {self.line_number}: {message}")

    def _resolve(self, name: str) -> str:
        while name in self.aliases:
            name = self.aliases[name]
        return name

    def norm(self, text: str) -> Type:
        if text in SCALARS or text in ("bool", "lit", "cap"):
            return text
        name = self._resolve(text)
        if name in self.composite:
            return self.composite[name]
        if name in self.types:
            return ("named", name)
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
                cell.locs[key] = Loc(False, loc.word, loc.ty)
            return Tok(loc.ty, loc.word)
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

    def integer(self, text: str, width: int, what: str) -> None:
        token = self.scalar(text, what)
        if token.ty != "lit" and SCALARS.get(token.ty) != width:
            raise self.fail(f"{what} {text} is {token.ty}, and the operation is {width} bits")

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
            if ty == "cap" and x.word != y.word:
                raise self.fail(f"r{r} is the capability of two different cells at a join")
            regs[r] = Tok(ty, self._merge_word(x.word, y.word))
        cells = {}
        for word in set(a.cells) & set(b.cells):
            cells[word] = self._merge_cell(word, a.cells[word], b.cells[word])
        nulls = a.nulls & b.nulls
        ruled = {}
        for word in set(a.ruled) & set(b.ruled):
            ruled[word] = a.ruled[word] & b.ruled[word]
        return State(regs, cells, nulls, ruled)

    def _merge_cell(self, word: Word, a: Cell, b: Cell) -> Cell:
        locs = {}
        for key in set(a.locs) | set(b.locs):
            x, y = a.locs.get(key), b.locs.get(key)
            if x is None or y is None:
                present = x if x is not None else y
                assert present is not None
                if not is_scalar(present.ty) and present.full:
                    raise self.fail(f"a linear location of {word} exists on only one edge")
                continue
            if x.full != y.full and not is_scalar(x.ty):
                raise self.fail(f"[{key[0]}:{key[1]}] of {word} is full on one edge and a hole "
                                f"on another")
            locs[key] = Loc(x.full and y.full, self._merge_word(x.word, y.word), x.ty)
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
            self.ctors[parts[0]].fields.append((int(parts[1]), int(parts[2]), parts[3], parts[4]))
        else:
            assert head == "state"
            self.states.append((parts[0], parts[1]))

    def _ctor_of(self, text: str) -> Ctor:
        ctor = self.ctors.get(text)
        if ctor is None:
            raise self.fail(f"unknown constructor {text}")
        return ctor

    def _open_cell(self, word: Word, ctor: Ctor) -> Cell:
        """A dispatched cell: every field of the constructor is full, each with a word of its
        own. That word cannot be just w.o: a cell that has been rebuilt and is opened again holds
        new values at the same offsets, and a value taken before the rebuild must not pass for
        what is there now, or RESTORE would skip a store that was needed."""
        locs = {}
        opening = self.fresh()
        for off, size, ty, _name in ctor.fields:
            locs[(off, size)] = Loc(True, ("field", opening, off), self.norm(ty))
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
        if head in ("fn", "handler"):
            self._header(text)
            return
        if self.fn is None:
            raise self.fail(f"{head} outside a function")
        if head == "end":
            self._end()
            return
        if self.S is None:
            raise self.fail(f"{head} cannot be reached: control left before it")
        self.steps += 1
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
        signature = Signature(ordinal, name, [], ret, int(at) if at else None, kind == "handler")
        regs = {}
        for position, part in enumerate(parts):
            register, _, ty = part.partition(" : ")
            if self.reg(register) != position:
                raise self.fail(f"parameter {position} must be r{position}")
            signature.params.append(ty.strip())
            regs[position] = Tok(self.norm(ty.strip()), ("arg", position))
        self.fn = signature
        self.S = State(regs, {}, frozenset(), {})
        for position, token in regs.items():
            if token.ty == "cap":
                self.S.cells[token.word] = Cell()
        self.pending = {}
        self.steps = self.draws = 0
        self.callee_steps = self.callee_draws = 0
        self.recursive = False
        if signature.handler:
            self._check_handler_shape(signature)
        if at is not None and not (0 <= int(at) < len(parts)):
            raise self.fail(f"@{at} names no parameter")

    def _check_handler_shape(self, signature: Signature) -> None:
        """Section 3.1: the state, the item and the reserve, in and back out."""
        results = self.tuples.get(self._resolve(signature.ret))
        count = len(self.states)
        if results is None or len(results) != count + 2 or len(signature.params) != count + 2:
            raise self.fail("a handler takes its state, an item and a Free, and returns its "
                            "state, an output and a Free")
        for position, (_name, ty) in enumerate(self.states):
            if self.norm(signature.params[position]) != self.norm(ty):
                raise self.fail(f"parameter {position} of a handler must be state slot {position}")
            if self.norm(results[position]) != self.norm(ty):
                raise self.fail(f"result {position} of a handler must be state slot {position}")
        if not self.is_reserve(self.norm(signature.params[-1])) or not self.is_reserve(
                self.norm(results[-1])):
            raise self.fail("a handler's last parameter and last result must be Free")

    def _end(self) -> None:
        if self.S is not None:
            raise self.fail("control runs off the end of the function")
        if self.pending:
            raise self.fail(f"labels never placed after a jump to them, or jumped back to: "
                            f"{sorted(self.pending)}")
        fn = self.fn
        assert fn is not None
        if self.recursive or self.callee_steps is None or self.callee_draws is None:
            fn.steps = fn.draws = None
        else:
            fn.steps = self.steps + self.callee_steps
            fn.draws = self.draws + self.callee_draws
        if fn.handler and fn.steps is None:
            raise Unbounded(f"section 8.2: handler {fn.name} reaches recursion, so it has no "
                            f"derivable bound until the resource analysis of section 5 exists")
        self.functions[fn.ordinal] = fn
        self.fn = None
        self.S = None

    def _label(self, label: str) -> None:
        if self.fn is None:
            raise self.fail("a label outside a function")
        edges = self.pending.pop(label, [])
        if self.S is not None:
            edges.append(self.S)
        if not edges:
            raise self.fail(f"label {label} has no incoming edge")
        self.S = self.merge(edges)

    # --- values

    def i_const(self, a: list[str]) -> None:
        literal, _, ty = a[1].partition(" : ")
        match = _LIT.match(literal.strip())
        if match is None:
            raise self.fail(f"const needs a literal, not {literal}")
        value = int(match.group(1))
        norm = self.norm(ty.strip())
        word = self.fresh()
        if not is_scalar(norm):
            decl = self._decl(norm)
            if decl is None or decl.kind != "tagless" or value != 0:
                raise self.fail(f"only a tagless type has a literal, and only 0: {a[1]}")
            word = ("null",)
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
        self.integer(a[3], width, "an operand")
        self.set(a[0], Tok(f"i{width}", self.fresh()))

    def _comparable(self, left: str, right: str) -> None:
        x, y = self.scalar(left, "a comparison operand"), self.scalar(right, "a comparison operand")
        if x.ty != y.ty and "lit" not in (x.ty, y.ty):
            raise self.fail(f"a comparison of {x.ty} with {y.ty}")

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
            self.scalar(operand, "a pack field")
        self.set(a[0], Tok("i64", self.fresh()))

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

    def _build(self, base: str, ctor_id: str, tag: str, sources: list[str], star: Optional[Word]
               ) -> tuple[int, Ctor]:
        """The atomic constructor. `star` is the word `*` must find, or None if it is banned."""
        r, word, cell = self.cap(base)
        ctor = self._ctor_of(ctor_id)
        decl = self.types[ctor.type]
        if len(sources) != len(ctor.fields):
            raise self.fail(f"{ctor.name} has {len(ctor.fields)} fields, and build gives "
                            f"{len(sources)} sources")
        named = set()
        stars = 0
        for source, (off, size, ty_text, name) in zip(sources, ctor.fields):
            ty = self.norm(ty_text)
            key = (off, size)
            where = f"{ctor.name}.{name}"
            held = cell.locs.get(key)
            if source == "=":
                if held is None or not held.full:
                    raise self.fail(f"{where} says STAY, and [{off}:{size}] holds no full value")
                if held.ty != ty and not (is_int(held.ty) and is_int(ty)):
                    raise self.fail(f"{where} says STAY over a {held.ty}, and the field is {ty}")
                named.add(key)
            elif source == "*":
                stars += 1
                if star is None:
                    raise self.fail(f"{where}: `*` outside tailmod")
                if held is None or held.full or held.word != star:
                    raise self.fail(f"{where}: `*` needs a hole holding the word the call returns")
            elif source.endswith("!"):
                r_source, token = self.full(source[:-1])
                if held is None or held.full or held.word != token.word:
                    raise self.fail(f"{where} says RESTORE, and [{off}:{size}] is not a hole "
                                    f"holding {token.word}")
                self.unify(token.ty, ty, where)
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
                named.add(other)
            elif _FUNC.match(source):
                self._computed(source, cell, ty, where)
            else:
                token = self.operand(source)
                if held is not None and held.word == token.word:
                    raise self.fail(f"{where} stores a value back where it came from: that is "
                                    f"{'`=`' if held.full else 'RESTORE'}, Theorem 6.1")
                self.unify(token.ty, ty, where)
        if star is not None and stars != 1:
            raise self.fail("tailmod needs exactly one `*`")
        for key, loc in cell.locs.items():
            if loc.full and not is_scalar(loc.ty) and key not in named:
                raise self.fail(f"[{key[0]}:{key[1]}] still holds a linear {loc.ty} that "
                                f"{ctor.name} does not name, so it would be lost")
        self._tag_rule(tag, ctor, decl, cell)
        return r, ctor

    @staticmethod
    def _inner(source: str) -> tuple[int, int]:
        match = _INNER.match(source)
        assert match is not None
        return int(match.group(1)), int(match.group(2))

    def _literal_fits(self, source: str, ty: Type, where: str) -> None:
        match = _LIT.match(source)
        assert match is not None
        if is_scalar(ty):
            return
        decl = self._decl(ty)
        if decl is None or decl.kind != "tagless" or int(match.group(1)) != 0:
            raise self.fail(f"{where}: {source} cannot be a {ty}")

    def _computed(self, source: str, cell: Cell, ty: Type, where: str) -> None:
        match = _FUNC.match(source)
        assert match is not None
        name, off, size, operand = match.groups()
        if name not in OP_NAMES:
            raise self.fail(f"{where}: unknown operation {name}")
        key = (int(off), int(size))
        loc = cell.locs.get(key)
        if loc is None or not loc.full or not is_scalar(loc.ty):
            raise self.fail(f"{where}: [{off}:{size}] holds no full scalar")
        self.unify(loc.ty, ty, where)
        self.scalar(operand, "a computed source's operand")

    def _tag_rule(self, tag: str, ctor: Ctor, decl: TypeDecl, cell: Cell) -> None:
        if decl.kind != "tagged":
            if tag != "none":
                raise self.fail(f"{ctor.name} has no tag, so build must say none, not {tag}")
            return
        known = cell.tag == ("is", int(ctor.tag))
        if tag == "keep" and not known:
            raise self.fail(f"keep: nothing shows the cell already holds tag {ctor.tag}")
        if tag == "write" and known:
            raise self.fail(f"write of tag {ctor.tag} over the same tag")
        if tag not in ("keep", "write"):
            raise self.fail(f"{ctor.name} is tagged, so build must say keep or write, not {tag}")

    def i_build(self, a: list[str]) -> None:
        r, ctor = self._build(a[0], a[1], a[2], a[3:], None)
        assert self.S is not None
        word = self.S.regs[r].word
        self.S.regs[r] = Tok(("named", self._resolve(ctor.type)), word)
        del self.S.cells[word]

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

    def _dispatched(self, r: int, word: Word, ctor: Ctor) -> None:
        assert self.S is not None
        self.S.regs[r] = Tok("cap", word)
        self.S.cells[word] = self._open_cell(word, ctor)

    def i_br_null(self, a: list[str]) -> None:
        r, token = self.full(a[0])
        decl = self._decl(token.ty)
        if decl is None or decl.kind != "tagless":
            raise self.fail(f"br.null on {token.ty}: it is not a tagless type")
        if self.is_reserve(token.ty):
            self.draws += 1
        assert self.S is not None
        base = self.S
        null_edge = base.copy()
        del null_edge.regs[r]
        null_edge.nulls = null_edge.nulls | {token.word}
        self.S = null_edge
        self.edge(a[1])
        full_ctor = next(c for c in decl.ctors if c.tag != "null")
        self.S = base
        self._dispatched(r, token.word, full_ctor)
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
        self._dispatched(r, token.word, ctor)
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
        self._dispatched(r, token.word, ctor)

    # --- calls and exits

    def _consume_arguments(self, signature: Signature, operands: list[str]) -> list[Tok]:
        if len(operands) != len(signature.params):
            raise self.fail(f"{signature.name} takes {len(signature.params)} arguments, "
                            f"not {len(operands)}")
        tokens = []
        for operand, ty in zip(operands, signature.params):
            want = self.norm(ty)
            token = self.operand(operand)
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
        value = token.word[1]
        if want == "bool":
            return value in (0, 1)
        if is_scalar(want):
            return is_int(want)
        decl = self._decl(want)
        return decl is not None and decl.kind == "tagless" and value == 0

    def _hand_over_cell(self, word: Word, operand: str) -> None:
        assert self.S is not None
        cell = self.S.cells.pop(word, None)
        if cell is None:
            raise self.fail(f"{operand} is a capability with no cell behind it")
        for key, loc in cell.locs.items():
            if loc.full and not is_scalar(loc.ty):
                raise self.fail(f"{operand} goes to a callee with a linear value still at "
                                f"[{key[0]}:{key[1]}]")

    def _callee(self, ordinal: str) -> Signature:
        signature = self.functions.get(ordinal)
        if signature is None:
            assert self.fn is not None
            if ordinal == self.fn.ordinal:
                self.recursive = True
                return self.fn
            raise self.fail(f"{ordinal} is not defined yet: a callee comes before its caller")
        if signature.steps is None:
            self.callee_steps = self.callee_draws = None
        elif self.callee_steps is not None and self.callee_draws is not None:
            assert signature.draws is not None
            self.callee_steps += signature.steps
            self.callee_draws += signature.draws
        return signature

    def i_call(self, a: list[str]) -> None:
        signature = self._callee(a[1])
        assert self.fn is not None
        if signature is self.fn:
            raise self.fail("a non-tail recursive call: DESCEND and ASCEND do not exist yet")
        tokens = self._consume_arguments(signature, a[2:])
        word = tokens[signature.at].word if signature.at is not None else self.fresh()
        results = self.tuples.get(self._resolve(signature.ret))
        if results is None:
            self.set(a[0], Tok(self.norm(signature.ret), word))
            return
        names = _args(a[0].strip("()"))
        if len(names) != len(results):
            raise self.fail(f"{signature.name} returns {len(results)} values")
        for name, ty in zip(names, results):
            self.set(name, Tok(self.norm(ty), self.fresh()))

    def _no_linear_left(self, where: str) -> None:
        assert self.S is not None
        for r, token in self.S.regs.items():
            if not is_scalar(token.ty):
                raise self.fail(f"{where} with a linear value still in r{r}: {token.ty}")

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
        for operand, ty in zip(a, wanted):
            r, token = self.full(operand)
            self.unify(token.ty, self.norm(ty), "a result")
            if not is_scalar(token.ty):
                assert self.S is not None
                del self.S.regs[r]
            if results is None:
                self._word_leaving(token.word, "ret")
        self._no_linear_left("ret")
        self.S = None

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
        if signature.at is None:
            raise self.fail(f"tailmod of {signature.name}, which is not declared @")
        if fn.at is None:
            raise self.fail(f"tailmod in {fn.name}, which is not declared @")
        tokens = self._consume_arguments(signature, operands)
        star = tokens[signature.at].word
        r, built = self._build(a[0], a[1], a[2], sources, star)
        assert self.S is not None
        token = self.S.regs[r]
        self._word_leaving(token.word, "tailmod")
        del self.S.regs[r]
        del self.S.cells[token.word]
        self._no_linear_left("tailmod")
        self.S = None


def check_stream(lines: Iterable[str]) -> Checker:
    checker = Checker()
    for line in lines:
        checker.feed(line)
    checker.finish()
    return checker
