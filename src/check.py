"""Check: first-order linear typing as a fold over the Dyck word.

Linear variables (diamond, data, records) are used exactly once on every path.
Scalars are erasable, and reading one does not consume it.

Rule R3 makes every function terminate. For some fixed set of linear argument positions,
not counting Free, the total number of cells must strictly fall at every recursive call.
That number is mu(C@e(args)) = 1 + the sum of mu over its linear args, and a match
replaces mu(x) with 1 + the sum of mu over the fields, or with 0 in a null arm. So each
value's count is a linear form, and comparing two is arithmetic. A count that cannot be
known, such as a call's result or the join of two arms, is opaque and never passes.

Exactly-once linearity keeps this small: an unknown can appear at most once in a form, so
a form is a constant plus a 64-bit mask, and it fits in the binder's own attribute cell.
Every subset of the positions starts as a candidate, and each call strikes out the ones
it does not shrink. With at most 6 parameters, those candidates fit in a 64-bit mask.

Usage state is a 64-bit mask, so a branch frame holds its snapshots in constant
space. That caps live linear binders per function at 64, a language limit that
depends only on the program text, like D_max. Names go no further than this
stage: Lower receives binder positions, offsets and widths in event payloads,
so it never compares a name and its frames stay small.

It checks LFPL-O's additions too, and writes the declarations out as lines of the RT layer's text
format, one event at a time, so they reach every reader of the stream:

  * handlers and state, with the shape section 3.1 fixes;
  * Array[N] t, Fin N and unboxed tuples, which are types with no entry in G, because they appear
    in the middle of a parameter list and G's entries for one function must stay consecutive;
  * idx, get and the narrowing low8, low16 and low32, which are primitives;
  * a built-in Free, and a type declared but never defined, which is the runtime's own to build.
    A program with a main has no runtime, so there it is refused.
"""
from __future__ import annotations

import struct
from typing import NamedTuple, Optional

from .dyck import (CLOSE, ELEMENT_CODES, OP_NAMES, K_AND, K_ARM, K_BINDER, K_BINOP, K_BOOL, K_CALL, K_CMP,
                   K_CTOR, K_CTORDECL, K_DATA, K_DIAMOND, K_EOF, K_FIELD, K_FN, K_GET, K_HANDLER,
                   K_IDX, K_IF, K_INT, K_LET, K_LIVE, K_LOW, K_MATCH, K_NOT, K_NULLCTOR, K_OR,
                   K_PACK, K_PARAM, K_RECORD, K_RESWORD, K_RTDECL, K_RTDECLMORE, K_STATE, K_TARRAY, K_TFIN,
                   K_TTUPLE, K_TUPLE, K_TYPE, K_VAR, K_VARLAST, K_WITH, LEAF, OPEN, SEP,
                   ev_class, ev_kind, ev_tag)
from .ring import FrameStack, Reject, Ring, Stage
from .tokens import PAYLOAD_BYTES

CELL_FIELDS_MAX = 16
LINEAR_BINDERS_MAX = 64
LINK_NAME = b"$next"  # the last field of a cell of a record that spans, as src/spans.py names it

# Type ids. Named types are G positions offset by FIRST_NAMED.
T_I8, T_I16, T_I32, T_I64, T_BOOL, T_DIAMOND, T_LIT, T_LIT8, T_LIT16, T_LIT32, T_BE16, T_BE32, T_BE64 = range(13)
FIRST_NAMED = 16
# A field in network byte order. It says how the field lies in its cell, and nothing else: what a pattern binds
# from one and what a constructor stores in one is the integer, swapped on the way.
_WIRE = {T_BE16: T_I16, T_BE32: T_I32, T_BE64: T_I64}
# A literal whose type nothing has fixed yet, by the bytes it needs.
_LITERALS = {T_LIT8: 1, T_LIT16: 2, T_LIT32: 4, T_LIT: 8}
_NARROWING = {b"low8": T_I8, b"low16": T_I16, b"low32": T_I32}
_PRIMITIVES = {b"idx", b"get", b"peel", *_NARROWING}
# Types with no entry in G. A range of ids per kind, so whether a type is linear can be read from
# its id alone, as it already can for every other type.
FIRST_ARRAY, FIRST_FIN, FIRST_TUPLE = 0x8000, 0xC000, 0xE000
TUPLE_MAX = 6  # as many results as x86 has registers for: section 4.1's pairs, at most six wide
_ELEMENT_TYPES = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}  # dyck's element codes, as the ids just below
_BUILTIN = {b"Int8": T_I8, b"Int16": T_I16, b"Int32": T_I32, b"Int64": T_I64,
            b"Bool": T_BOOL, b"Diamond": T_DIAMOND, b"Be16": T_BE16, b"Be32": T_BE32, b"Be64": T_BE64}
_SIZES = {T_I8: 1, T_I16: 2, T_I32: 4, T_I64: 8, T_BOOL: 1, T_LIT: 8, T_LIT8: 8, T_LIT16: 8, T_LIT32: 8,
          T_BE16: 2, T_BE32: 4, T_BE64: 8}

# G entries: name in the first cell, attributes in the second.
E_TYPE, E_CTOR, E_FIELD, E_FUNC, E_PARAM, E_STATE = range(6)
_ATTR = struct.Struct("<BBBBHHH")  # a, b, c, d, x, y, z
_FRAME = struct.Struct("<BBHHHHQQQ")  # a, b, t1, t2, c1, c2, m1, m2, m3
_BINDER = struct.Struct("<HBHQB")  # type, linear index or 255, then its cell count
_ARGUMENT = struct.Struct("<HQB")  # the cell count of one argument to a recursive call
_FORWARD, _RESOLVED = 254, 253  # the `a` attribute of a forward-declared type

# How a field is held, in the last byte of what Lower is told about one.
ATTR_LINEAR, ATTR_WIRE = 1, 2

# Payloads for Lower: positions, offsets and widths, and the id of whatever it must name.
P_VAR = struct.Struct("<HBB")     # src index, linear, size
P_FIELD = struct.Struct("<BBBB")  # index, off, size, attributes
P_BINDER = struct.Struct("<BBBB")  # 0 for diamond or 1 for a field, off, size, attributes
P_PARAM_RT = struct.Struct("<HBBH")      # src index, size, linear, type
P_DIAMOND_RT = struct.Struct("<HBBBBBBH")  # d src, tag, tagged, nfields, off, size, attributes, the constructor
P_ARM_RT = struct.Struct("<BBBBBH")      # tag, tagged, is the null constructor, arm index, nctors, the constructor
P_LET_RT = struct.Struct("<BBBB")        # separator 1, linear, size, how many names it binds
P_CALL_RT = struct.Struct("<HBBBB")      # function ordinal, is self, argument count, result count, @k + 1 or 0
P_WITH_RT = struct.Struct("<BBBHBB")     # off, size, attributes, the constructor, its position and count
P_TYPE_RT = struct.Struct("<H")          # a type
P_GET_RT = struct.Struct("<B")           # the element's size
P_IDX_RT = struct.Struct("<Q")           # the bound


class Cells(NamedTuple):
    """mu of a value, as a linear form: a constant plus some unknowns, each at least 0.
    Opaque means nothing is known about it. A form can be stored in one binder's cell."""
    constant: int = 0
    unknowns: int = 0
    opaque: bool = False

    def plus(self, other: "Cells") -> "Cells":
        return Cells(self.constant + other.constant, self.unknowns | other.unknowns,
                     self.opaque or other.opaque)


OPAQUE = Cells(opaque=True)
RESERVE_TYPE = b"Free"  # the reserve is never a measure: an item's work is paid by the item


def is_linear(ty: int) -> bool:
    return ty == T_DIAMOND or (ty >= FIRST_NAMED and not FIRST_FIN <= ty < FIRST_TUPLE)


def value_of(ty: int) -> int:
    """The type of what is in a field of this type, once it is out of the cell."""
    return _WIRE.get(ty, ty)


def attributes(ty: int) -> int:
    return (ATTR_LINEAR if is_linear(ty) else 0) | (ATTR_WIRE if ty in _WIRE else 0)


def is_composite(ty: int) -> bool:
    return ty >= FIRST_ARRAY


def size_of(ty: int) -> int:
    return _SIZES.get(ty, 8)


class CheckStage(Stage):
    """The typing judgment, one event at a time."""

    name = "check"

    def __init__(self, g_max: int = 2048, *, phase1: bool = False) -> None:
        super().__init__()
        self.g_max = g_max
        self.phase1 = phase1  # a program with a main, which _start calls and reads
        self.decl_from = self.decl_to = 0  # the G entries still to be written out as RT lines
        self.decl_text = b""               # the rest of the line being written out
        self.decl_queue: list[bytes] = []  # at most one declaration's few extra lines
        self.fn_handler = False
        self.fn_recursive = False
        self.fn_at = 0
        self.ty = T_LIT           # the type of the last completed expression
        self.used = 0             # U: linear binders consumed so far
        self.lin_held = 0         # linear indices in use
        self.fn_count = 0
        self.fn_self = -1         # G position of the function being checked
        self.ret_ty = 0
        self.decl_type = -1
        self.decl_field_count = 0
        self.decl_ctor = -1
        self.decl_forward = -1
        self.form = Cells()       # mu of the expression just finished
        self.closing = OPAQUE     # what a construct's close leaves in self.form
        self.entry: list[Cells] = []  # R3: mu of each parameter, as far as matches have shown
        self.viable = 0           # R3: subsets of positions that fell at every call so far
        self.pending_name = b""
        self.child_seen = False

    def attach(self, ring: Ring) -> None:
        super().attach(ring)
        self.G = self.stack()
        self.K = self.stack()
        self.B = self.stack()
        self.N = self.stack()  # the name of each let whose bound value is still open
        self.A = self.stack()  # R3: the cell count of each argument to a recursive call
        self.TA = self.stack()  # array types: length, then element
        self.TF = self.stack()  # Fin types: bound
        self.TT = self.stack()  # tuple types: their members
        self.TM = self.stack()  # the members of each tuple still open, type or expression
        self._seed_reserve()

    def _seed_reserve(self) -> None:
        """LFPL-O's Free is built in: End | More(next: Free), so every program and the runtime
        agree on what a reserve is. It costs G four entries before the program starts."""
        name = lambda text: text.ljust(PAYLOAD_BYTES, b"\0")  # noqa: E731
        self.decl_type = self._g_push(E_TYPE, name(RESERVE_TYPE), a=K_DATA)
        self.decl_ctor = self._g_push(E_CTOR, name(b"End"), x=self.decl_type)
        self.decl_field_count = 0
        self._finish_ctor()
        self.decl_ctor = self._g_push(E_CTOR, name(b"More"), x=self.decl_type)
        self._g_push(E_FIELD, name(b"next"), x=self.decl_ctor, z=self.decl_type + FIRST_NAMED)
        self.decl_field_count = 1
        self._finish_ctor()
        self._finish_data()
        self.decl_from, self.decl_to = 0, len(self.G)

    def has_work(self) -> bool:
        return bool(self.decl_text or self.decl_queue or self.decl_from < self.decl_to)

    # --- declarations, written out as RT lines: one chunk per firing, from what is in G

    def _rt_type(self, ty: int) -> str:
        names = {T_I8: "i8", T_I16: "i16", T_I32: "i32", T_I64: "i64", T_BOOL: "bool",
                 T_DIAMOND: "cap", T_LIT: "lit", T_LIT8: "lit", T_LIT16: "lit", T_LIT32: "lit",
                 T_BE16: "be16", T_BE32: "be32", T_BE64: "be64"}
        return names.get(ty, f"t{ty}")

    def _rt_line(self, index: int) -> bytes:
        kind, a, b, c, d, x, _y, z = self._g(index)
        name = self.G.at(index)[1][:PAYLOAD_BYTES].rstrip(b"\0").decode()
        if kind == E_TYPE:
            ctors = self._ctors(index)
            if a == K_RECORD:
                shape = "record"
            elif not ctors:
                shape = "abstract"
            else:
                shape = "tagged" if self._g(ctors[0])[4] else "tagless"
            return f"type t{index + FIRST_NAMED} {shape} {name}".encode()
        if kind == E_CTOR:
            tag = "null" if b else (str(a) if d else "-")
            return f"ctor c{index} t{x + FIRST_NAMED} {tag} {name}".encode()
        if kind == E_FIELD:
            return f"field c{x} {a} {b} {self._rt_type(z)} {name}".encode()
        return b""

    def _write_decl_chunk(self) -> None:
        while not self.decl_text:
            if self.decl_from < self.decl_to:
                self.decl_text = self._rt_line(self.decl_from)
                self.decl_from += 1
            else:
                self.decl_text = self.decl_queue.pop(0)
        chunk, self.decl_text = self.decl_text[:PAYLOAD_BYTES], self.decl_text[PAYLOAD_BYTES:]
        kind = K_RTDECLMORE if self.decl_text else K_RTDECL
        self.ring.emit_fresh(self, ev_tag(LEAF, kind), chunk)

    def depth(self) -> int:
        return len(self.K)

    # --- errors, G and binders

    def _reject(self, message: str) -> Reject:
        # "Near": the ring holds a few tokens between Lex and here, so the lexer's
        # line can be slightly ahead of the construct this stage is looking at.
        return Reject(f"near line {self.ring.line()}: {message}")

    def _g_push(self, kind: int, name: bytes, a=0, b=0, c=0, d=0, x=0, y=0, z=0) -> int:
        if len(self.G) >= self.g_max:
            raise self._reject(f"more than {self.g_max} global names: the language's "
                               f"G_max, which keeps F from running out (Theorem 7.2)")
        self.G.push(kind, name.ljust(PAYLOAD_BYTES, b"\0")
                    + _ATTR.pack(a, b, c, d, x, y, z))
        return len(self.G) - 1

    def _g(self, index: int) -> tuple:
        kind, data = self.G.at(index)
        return (kind, *_ATTR.unpack_from(data, PAYLOAD_BYTES))

    def _g_set(self, index: int, a, b, c, d, x, y, z) -> None:
        kind, data = self.G.at(index)
        self.G.rewrite(index, kind, data[:PAYLOAD_BYTES] + _ATTR.pack(a, b, c, d, x, y, z))

    def _g_find(self, kind: int, name: bytes) -> Optional[int]:
        for index in range(len(self.G) - 1, -1, -1):
            entry_kind, data = self.G.at(index)
            if entry_kind == kind and data[:PAYLOAD_BYTES] == name:
                return index
        return None

    # --- array, Fin and tuple types: interned, so one type always has one id

    def _intern(self, table: FrameStack, first: int, data: bytes) -> tuple[int, bool]:
        for index in range(len(table)):
            if table.at(index)[1].rstrip(b"\0") == data.rstrip(b"\0"):
                return first + index, False
        if len(table) >= self.g_max:
            raise self._reject(f"more than {self.g_max} distinct types")
        table.push(0, data)
        return first + len(table) - 1, True

    def _array_type(self, length: int, element: int) -> int:
        if length == 0 or length & length - 1:
            raise self._reject(f"section 4.5: an array of {length}: until division exists an "
                               f"array's length must be a power of two, so idx is a mask")
        ty, new = self._intern(self.TA, FIRST_ARRAY, struct.pack("<QB", length, element + 1))
        if new:
            self.decl_queue.append(f"array t{ty} {length} {self._rt_type(element)}".encode())
        return ty

    def _array_of(self, ty: int) -> tuple[int, int]:
        length, element = struct.unpack_from("<QB", self.TA.at(ty - FIRST_ARRAY)[1])
        return length, element - 1

    def _fin_type(self, bound: int, declared: bool) -> int:
        ty, _new = self._intern(self.TF, FIRST_FIN, struct.pack("<Q", bound))
        if declared:  # in a body the type needs no line: no reader is told about it by id
            self.decl_queue.append(f"fin t{ty} {bound}".encode())
        return ty

    def _fin_of(self, ty: int) -> int:
        return struct.unpack_from("<Q", self.TF.at(ty - FIRST_FIN)[1])[0]

    def _tuple_type(self, members: list[int], declared: bool = False) -> int:
        if len(members) > TUPLE_MAX:
            raise self._reject(f"a tuple of {len(members)}: at most {TUPLE_MAX}")
        if any(m >= FIRST_TUPLE for m in members):
            raise self._reject("a tuple cannot hold a tuple: tuples are flat, because they are "
                               "only ever registers")
        ty, _new = self._intern(self.TT, FIRST_TUPLE,
                                struct.pack(f"<{len(members)}H", *(m + 1 for m in members)))
        if declared:
            names = " ".join(self._rt_type(m) for m in members)
            self.decl_queue.append(f"tuple t{ty} {names}".encode())
        return ty

    def _tuple_of(self, ty: int) -> list[int]:
        data = self.TT.at(ty - FIRST_TUPLE)[1].rstrip(b"\0")
        count = (len(data) + 1) // 2
        return [m - 1 for m in struct.unpack_from(f"<{count}H", data.ljust(2 * count, b"\0"))]

    def _is_tuple(self, ty: int) -> bool:
        return ty >= FIRST_TUPLE

    def _members_open(self) -> list[int]:
        """Pop the member types the tuple being closed has collected."""
        frame = self._top()
        count = len(self.TM) - frame[4]
        members = [struct.unpack_from("<H", self.TM.at(frame[4] + i)[1])[0] for i in range(count)]
        for _ in range(count):
            self.TM.pop()
        return members

    def _type_token(self, kind: int, payload: bytes) -> int:
        if kind == K_TARRAY:
            length = struct.unpack_from("<Q", payload)[0]
            element = _ELEMENT_TYPES[payload[8]]
            return self._array_type(length, element)
        if kind == K_TFIN:
            bound = struct.unpack_from("<Q", payload)[0]
            if bound == 0 or bound & bound - 1:
                raise self._reject(f"section 4.5: Fin {bound}: a power of two until division exists")
            return self._fin_type(bound, True)
        return self._type(payload)

    def _type(self, name: bytes) -> int:
        key = name.rstrip(b"\0")
        if key in _BUILTIN:
            return _BUILTIN[key]
        found = self._g_find(E_TYPE, name)
        if found is None:
            raise self._reject(f"unknown type {key.decode()}")
        return found + FIRST_NAMED

    def _bind(self, name: bytes, ty: int, form: Optional[Cells] = None) -> int:
        """A new binder. By default its cell count is an unknown of its own, except that
        a diamond is exactly one cell and a scalar owns none."""
        linear = 255
        if is_linear(ty):
            for bit in range(LINEAR_BINDERS_MAX):
                if not self.lin_held >> bit & 1:
                    linear = bit
                    self.lin_held |= 1 << bit
                    break
            else:
                raise self._reject(
                    f"more than {LINEAR_BINDERS_MAX} live linear binders in a function")
        if form is None:
            if ty == T_DIAMOND:
                form = Cells(1)
            elif linear != 255:
                form = Cells(unknowns=1 << linear)
            else:
                form = Cells()
        self.B.push(0, name.ljust(PAYLOAD_BYTES, b"\0")
                    + _BINDER.pack(ty, linear, form.constant, form.unknowns, form.opaque))
        return len(self.B) - 1

    def _unbind_to(self, mark: int) -> None:
        while len(self.B) > mark:
            _, data = self.B.peek()
            _ty, linear, *_form = _BINDER.unpack_from(data, PAYLOAD_BYTES)
            if linear != 255:
                if not self.used >> linear & 1:
                    name = data[:PAYLOAD_BYTES].rstrip(b"\0").decode()
                    if "$" in name:  # src/spans.py's name for a cell of a record that spans
                        raise self._reject(f"a cell of {name.split('$')[0] or 'a record'} is never used: it spans "
                                           f"cells, and all of them have to go somewhere")
                    raise self._reject(f"linear variable {name} is never used")
                self.used &= ~(1 << linear)
                self.lin_held &= ~(1 << linear)
            self.B.pop()

    def _lookup(self, name: bytes) -> tuple[int, int, int]:
        for index in range(len(self.B) - 1, -1, -1):
            _, data = self.B.at(index)
            if data[:PAYLOAD_BYTES] == name:
                ty, linear, *_form = _BINDER.unpack_from(data, PAYLOAD_BYTES)
                return index, ty, linear
        raise self._reject(f"unbound variable {name.rstrip(bytes(1)).decode()}")

    def _form_of(self, index: int) -> Cells:
        _ty, _linear, constant, unknowns, opaque = _BINDER.unpack_from(
            self.B.at(index)[1], PAYLOAD_BYTES)
        return Cells(constant, unknowns, bool(opaque))

    def _use(self, name: bytes) -> tuple[int, int, int]:
        index, ty, linear = self._lookup(name)
        if linear != 255:
            if self.used >> linear & 1:
                raise self._reject(
                    f"linear variable {name.rstrip(bytes(1)).decode()} used twice")
            self.used |= 1 << linear
        return index, ty, linear

    def _unify(self, a: int, b: int, where: str) -> int:
        if a == b:
            return a
        if a >= FIRST_TUPLE and b >= FIRST_TUPLE:
            x, y = self._tuple_of(a), self._tuple_of(b)
            if len(x) == len(y):
                return self._tuple_type([self._unify(m, n, where) for m, n in zip(x, y)])
        if a in _LITERALS and b in _LITERALS:
            return a if _LITERALS[a] > _LITERALS[b] else b
        if a in _LITERALS and b in (T_I8, T_I16, T_I32, T_I64):
            return self._literal_meets(a, b, where)
        if b in _LITERALS and a in (T_I8, T_I16, T_I32, T_I64):
            return self._literal_meets(b, a, where)
        raise self._reject(f"type mismatch in {where}: {self._show(a)} and {self._show(b)}")

    def _literal_meets(self, literal: int, ty: int, where: str) -> int:
        """A literal has to be a value of the type it is used at. If it could be more, a register
        would hold an Int8 of 300, and the same value would be 44 to + and not 44 to ==."""
        if _LITERALS[literal] > _SIZES[ty]:
            raise self._reject(f"a literal in {where} does not fit {self._show(ty)}")
        return ty

    def _show(self, ty: int) -> str:
        for name, value in _BUILTIN.items():
            if value == ty:
                return name.decode()
        if ty in _LITERALS:
            return "integer literal"
        if ty >= FIRST_TUPLE:
            return "(" + ", ".join(self._show(m) for m in self._tuple_of(ty)) + ")"
        if ty >= FIRST_FIN:
            return f"Fin {self._fin_of(ty)}"
        if ty >= FIRST_ARRAY:
            length, element = self._array_of(ty)
            return f"Array[{length}] {self._show(element)}"
        _, data = self.G.at(ty - FIRST_NAMED)
        return data[:PAYLOAD_BYTES].rstrip(b"\0").decode()

    def _need_int(self, ty: int, where: str) -> None:
        if ty not in (T_I8, T_I16, T_I32, T_I64) and ty not in _LITERALS:
            raise self._reject(f"{where} needs an integer, got {self._show(ty)}")

    # --- frames

    def _frame(self, kind: int, a=0, b=0, t1=0, t2=0, c1=0, c2=0, m1=0, m2=0, m3=0) -> None:
        self.K.push(kind, _FRAME.pack(a, b, t1, t2, c1, c2, m1, m2, m3))

    def _top(self) -> list:
        return list(_FRAME.unpack_from(self.K.peek()[1]))

    def _store(self, frame: list) -> None:
        self.K.rewrite(len(self.K) - 1, self.K.peek()[0], _FRAME.pack(*frame))

    # --- the fold

    def fire(self, cell: Optional[int]) -> None:
        if cell is None:
            self._write_decl_chunk()
            return
        store = self.ring.store
        tag, payload = store.tag(cell), store.payload(cell)
        cls, kind = ev_class(tag), ev_kind(tag)
        if kind in (K_DATA, K_CTORDECL, K_RECORD, K_STATE) or (
                len(self.K) and self.K.peek()[0] in (K_DATA, K_CTORDECL, K_RECORD, K_STATE)):
            self._declaration(cls, kind, payload)
            self.ring.drop(self, cell)
            return
        if (cls in (SEP, CLOSE) and kind == K_CALL and len(self.K)
                and self.K.peek()[0] in (K_IDX, K_GET, K_LOW)):
            kind = self.K.peek()[0]
        result = (self._on_open, self._on_sep, self._on_leaf, self._on_close)[cls](
            kind, payload)
        if cls == CLOSE:
            self.form, self.closing = self.closing, OPAQUE
        if result is None:
            self.ring.drop(self, cell)
        else:
            new_kind, new_payload = result
            self.ring.pass_(self, cell, ev_tag(cls, new_kind), new_payload)

    # --- declarations: consumed here, never forwarded

    def _declaration(self, cls: int, kind: int, payload: bytes) -> None:
        if cls == OPEN and kind == K_STATE:
            if len(self.G) and self._g_find(E_STATE, payload) is not None:
                raise self._reject("duplicate state name")
            self._frame(K_STATE)
            self._g_push(E_STATE, payload)
        elif cls == LEAF and kind in (K_TYPE, K_TARRAY, K_TFIN) and self.K.peek()[0] == K_STATE:
            ty = self._type_token(kind, payload)
            if not FIRST_ARRAY <= ty < FIRST_FIN:
                raise self._reject("section 4.5: a state slot holding cells waits for swap, so "
                                   "state is arrays for now")
            state = len(self.G) - 1
            _, a, b, c, d, x, y, _z = self._g(state)
            self._g_set(state, a, b, c, d, x, y, ty)
            name = self.G.at(state)[1][:PAYLOAD_BYTES].rstrip(b"\0").decode()
            self.decl_queue.append(f"state {name} {self._rt_type(ty)}".encode())
        elif cls == CLOSE and kind == K_STATE:
            self.K.pop()
        elif cls == OPEN and kind in (K_DATA, K_RECORD):
            found = self._g_find(E_TYPE, payload)
            self.decl_forward = -1
            if found is not None:
                # G's first entry is the built-in Free, which a Phase 1 program declares for
                # itself. The runtime still checks the shape of what a handler is given.
                if self._g(found)[1] != _FORWARD and found != 0:
                    raise self._reject("duplicate type name")
                self.decl_forward = found
            self.decl_type = self._g_push(E_TYPE, payload, a=kind)
            self._frame(kind, c1=len(self.G))
            self.decl_field_count = 0
            if kind == K_RECORD:
                self.decl_ctor = self._g_push(E_CTOR, payload, x=self.decl_type)
        elif cls == OPEN and kind == K_CTORDECL:
            self.decl_ctor = self._g_push(E_CTOR, payload, x=self.decl_type)
            self._frame(kind)
            self.decl_field_count = 0
        elif cls == LEAF and kind == K_FIELD:
            self._g_push(E_FIELD, payload, x=self.decl_ctor)
        elif cls == LEAF and kind in (K_TYPE, K_TFIN, K_TARRAY):
            field = len(self.G) - 1
            if kind == K_TARRAY:
                raise self._reject("section 4.5: an array lives in its own region of the image, "
                                   "so it cannot be stored in a cell")
            field_ty = self._type_token(kind, payload)
            if field_ty == T_DIAMOND:
                raise self._reject("a diamond cannot be stored in a field")
            _, a, b, c, d, x, y, _z = self._g(field)
            self._g_set(field, a, b, c, d, x, y, field_ty)
            self.decl_field_count += 1
        elif cls == CLOSE and kind == K_CTORDECL:
            self.K.pop()
            self._finish_ctor()
        elif cls == CLOSE and kind == K_RECORD:
            self.K.pop()
            self._finish_ctor()
            self._layout(self.decl_type, tagged=False, tagless=False)
            self._write_out_declaration()
            self._resolve_forward()
        elif cls == CLOSE and kind == K_DATA:
            self.K.pop()
            if self._ctors(self.decl_type):
                self._finish_data()
                self._write_out_declaration()
                self._resolve_forward()
            elif self.decl_forward >= 0:
                raise self._reject("a type is declared forward twice")
            else:
                self._write_out_declaration()
                _, _a, b, c, d, x, y, z = self._g(self.decl_type)
                self._g_set(self.decl_type, _FORWARD, b, c, d, x, y, z)
        else:
            raise self._reject("malformed declaration")

    def _write_out_declaration(self) -> None:
        self.decl_from, self.decl_to = self.decl_type, len(self.G)

    def _resolve_forward(self) -> None:
        """Fields written against the forward declaration now name the definition.
        G stays linear in declarations: this scan runs once per forward type."""
        if self.decl_forward < 0:
            return
        old = self.decl_forward + FIRST_NAMED
        new = self.decl_type + FIRST_NAMED
        self.decl_queue.append(f"alias t{old} t{new}".encode())
        for index in range(len(self.G)):
            entry = self._g(index)
            if entry[0] == E_FIELD and entry[7] == old:
                self._g_set(index, *entry[1:7], new)
        _, _a, b, c, d, x, y, z = self._g(self.decl_forward)
        self._g_set(self.decl_forward, _RESOLVED, b, c, d, x, y, z)

    def _check_forwards_resolved(self) -> None:
        if not self.phase1:
            return  # a type never defined is one the runtime builds and the program only moves
        for index in range(len(self.G)):
            entry = self._g(index)
            if entry[0] == E_TYPE and entry[1] == _FORWARD:
                name = self.G.at(index)[1][:PAYLOAD_BYTES].rstrip(b"\0").decode()
                raise self._reject(f"type {name} is declared but never defined")

    def _finish_ctor(self) -> None:
        _, a, b, _c, d, x, y, z = self._g(self.decl_ctor)
        self._g_set(self.decl_ctor, a, b, self.decl_field_count, d, x, y, z)

    def _ctors(self, type_index: int) -> list[int]:
        ctors = []
        for index in range(type_index + 1, len(self.G)):
            entry = self._g(index)
            if entry[0] == E_CTOR and entry[5] == type_index:
                ctors.append(index)
            elif entry[0] != E_FIELD:
                break
        return ctors

    def _state_types(self) -> list[tuple[bytes, int]]:
        return [(self.G.at(i)[1][:PAYLOAD_BYTES], self._g(i)[7]) for i in range(len(self.G))
                if self._g(i)[0] == E_STATE]

    def _check_handler(self, first: int, nparams: int) -> None:
        """Section 3.1: state, an item and a Free go in; state, an output and a Free come out,
        each state slot in the position and under the name its declaration gave it."""
        states = self._state_types()
        reserve = self._g_find(E_TYPE, RESERVE_TYPE.ljust(PAYLOAD_BYTES, b"\0"))
        assert reserve is not None
        reserve += FIRST_NAMED
        if nparams != len(states) + 2:
            raise self._reject("section 3.1: a handler takes each state slot, an item and a Free")
        for position, (name, ty) in enumerate(states):
            if self.G.at(first + position)[1][:PAYLOAD_BYTES] != name or self._g(first + position)[5] != ty:
                raise self._reject(f"section 3.1: parameter {position} must be state slot "
                                   f"{name.rstrip(bytes(1)).decode()}")
        if self._g(first + nparams - 1)[5] != reserve:
            raise self._reject("section 3.1: a handler's last parameter must be a Free")
        if not self._is_tuple(self.ret_ty):
            raise self._reject("section 3.1: a handler returns a tuple")
        results = self._tuple_of(self.ret_ty)
        if len(results) != len(states) + 2 or results[-1] != reserve or any(
                results[i] != states[i][1] for i in range(len(states))):
            raise self._reject("section 3.1: a handler returns each state slot, an output and a Free")

    def _finish_data(self) -> None:
        ctors = self._ctors(self.decl_type)
        fields = [self._g(c)[3] for c in ctors]
        tagless = (len(ctors) == 2 and sorted(fields)[0] == 0 and sorted(fields)[1] > 0)
        self._layout(self.decl_type, tagged=not tagless, tagless=tagless)

    def _layout(self, type_index: int, tagged: bool, tagless: bool) -> None:
        """Place fields in the 16-byte cell, or reject. Section 3.1's ceiling."""
        ctors = self._ctors(type_index)
        if len(ctors) > LINEAR_BINDERS_MAX:
            raise self._reject(f"more than {LINEAR_BINDERS_MAX} constructors")
        for tag, ctor in enumerate(ctors):
            count = self._g(ctor)[3]
            offset = 1 if tagged else 0
            if count > CELL_FIELDS_MAX:
                raise self._reject("constructor does not fit in a diamond")
            for position in range(ctor + 1, ctor + 1 + count):
                kind, a, b, c, d, x, y, field_ty = self._g(position)
                size = size_of(field_ty)
                offset = (offset + size - 1) // size * size
                if offset + size > 16:
                    name = self.G.at(ctor)[1][:PAYLOAD_BYTES].rstrip(b"\0").decode()
                    raise self._reject(
                        f"constructor {name} does not fit in a diamond (16 bytes)")
                self._g_set(position, offset, size, c, d, x, y, field_ty)
                offset += size
            is_null = tagless and count == 0
            _, a, b, c, d, x, y, z = self._g(ctor)
            self._g_set(ctor, tag, is_null, count, tagged, x, y, z)

    # --- open

    def _on_open(self, kind: int, payload: bytes):
        if len(self.K) and self.K.peek()[0] == K_IDX and self._top()[1]:
            raise self._reject("the bound of idx must be a literal")
        if kind == K_TTUPLE:
            self._frame(K_TTUPLE, c1=len(self.TM))
            return None
        if kind in (K_FN, K_HANDLER):
            if self._g_find(E_FUNC, payload) is not None:
                raise self._reject("duplicate function name")
            if payload.rstrip(bytes(1)) in _PRIMITIVES:
                raise self._reject(f"{payload.rstrip(bytes(1)).decode()} is a primitive")
            self.fn_handler = kind == K_HANDLER
            self.fn_recursive = False
            self.fn_self = self._g_push(E_FUNC, payload, b=self.fn_handler, x=self.fn_count)
            self.used = self.lin_held = 0
            self.fn_at = 0
            self.entry = []
            self._frame(K_FN, c1=len(self.G))
            return kind, payload
        if kind in (K_LET, K_WITH):
            self._frame(kind, c1=len(self.B), c2=len(self.N))
        elif kind == K_BINOP:
            self._frame(kind, a=payload[0], t1=T_LIT)
        elif kind in (K_IF, K_MATCH, K_CMP, K_AND, K_OR, K_NOT):
            self._frame(kind, t1=T_LIT)
        elif kind == K_TUPLE:
            self._frame(K_TUPLE, c1=len(self.TM))
        elif kind == K_CALL and payload.rstrip(bytes(1)) in (b"idx", b"get"):
            primitive = K_IDX if payload.rstrip(bytes(1)) == b"idx" else K_GET
            self._frame(primitive)
            return primitive, b""
        elif kind == K_CALL and payload.rstrip(bytes(1)) in _NARROWING:
            self._frame(K_LOW, t1=_NARROWING[payload.rstrip(bytes(1))])
            self.child_seen = False
            return K_LOW, b""
        elif kind == K_CALL:
            found = self._g_find(E_FUNC, payload)
            if found is None:
                raise self._reject(
                    f"unknown function {payload.rstrip(bytes(1)).decode()}: a callee "
                    f"is defined before its caller, so the call graph stays acyclic")
            _, nparams, is_handler, at, _d, ordinal, ret, first = self._g(found)
            if is_handler:
                raise self._reject("a handler is an entry point, and nothing can call one")
            self._frame(K_CALL, a=nparams, b=found == self.fn_self, t1=ret, c1=first)
            self.child_seen = False
            results = len(self._tuple_of(ret)) if self._is_tuple(ret) else 1
            return kind, P_CALL_RT.pack(ordinal, found == self.fn_self, nparams, results, at)
        elif kind == K_CTOR:
            found = self._g_find(E_CTOR, payload)
            if found is None:
                raise self._reject(f"unknown constructor {payload.rstrip(bytes(1)).decode()}")
            _, _tag, is_null, count, _tagged, type_index, _y, _z = self._g(found)
            if is_null:
                raise self._reject("a null constructor takes no cell")
            self._frame(K_CTOR, a=count, t1=type_index + FIRST_NAMED, c1=found)
            self.child_seen = False
            return kind, bytes((count,))
        elif kind == K_PACK:
            self._frame(K_PACK, a=len(payload.rstrip(bytes(1))))
            self.child_seen = False
        elif kind == K_ARM:
            return self._open_arm(payload)
        return kind, payload

    def _open_arm(self, payload: bytes):
        frame = self._top()
        found = self._g_find(E_CTOR, payload)
        type_index = frame[2] - FIRST_NAMED
        if found is None or self._g(found)[5] != type_index:
            raise self._reject(f"{payload.rstrip(bytes(1)).decode()} is not a constructor "
                               f"of {self._show(frame[2])}")
        _, tag, is_null, count, tagged, _x, _y, _z = self._g(found)
        seen = frame[7]
        if seen >> tag & 1:
            raise self._reject("duplicate match arm")
        arm_index = frame[0]
        frame[7] = seen | 1 << tag
        frame[0] = arm_index + 1
        self._store(frame)
        self.used = frame[6]  # every arm starts from the state at the match
        nctors = len(self._ctors(type_index))
        # R3: on this path, the unknown matched here is 0, or a cell plus its fields. The
        # fields are added as they are bound, and the arm's close undoes all of it.
        position = saved = None
        if frame[4]:
            bit = 1 << frame[4] - 1
            position = next((k for k, f in enumerate(self.entry) if f.unknowns & bit), None)
            if position is not None:
                saved = self.entry[position]
                self.entry[position] = Cells(saved.constant + (not is_null), saved.unknowns & ~bit)
        self._frame(K_ARM, a=0 if is_null else count + 1,
                    t1=0 if position is None else position + 1,
                    t2=0 if saved is None else saved.constant, c1=len(self.B), c2=found,
                    m1=0 if saved is None else saved.unknowns)
        return K_ARM, P_ARM_RT.pack(tag, tagged, is_null, arm_index, nctors, found)

    # --- sep

    def _on_sep(self, kind: int, payload: bytes):
        frame = self._top()
        index = payload[0]
        if kind in (K_FN, K_HANDLER):
            if index == 0:
                frame[0] = 1  # parameters finished; the next type is the result
                self._store(frame)
                return None
            func = self.fn_self
            first = frame[4]
            nparams = len(self.G) - first
            if self.fn_handler:
                self._check_handler(first, nparams)
            elif self.phase1 and self.G.at(func)[1][:PAYLOAD_BYTES].rstrip(b"\0") == b"main":
                self._check_main(first, nparams)
            _, _a, b, _c, d, ordinal, _y, _z = self._g(func)
            self._g_set(func, nparams, b, self.fn_at, d, ordinal, self.ret_ty, first)
            self.viable = self._candidate_sets(first, nparams)
            return kind, payload
        if kind == K_LET:
            ty = self.ty
            count = len(self.N) - frame[5]
            if count > 1 or self._is_tuple(ty):
                return self._bind_tuple(kind, ty, frame[5], count)
            name = self.N.peek()[1][:PAYLOAD_BYTES]
            self.N.pop()
            # let is a renaming, so the value's count carries over to the name
            self._bind(name, ty, self.form if is_linear(ty) else Cells())
            return kind, P_LET_RT.pack(1, is_linear(ty), size_of(ty), 1)
        if kind == K_TUPLE:
            if len(self.TM) - frame[4] >= TUPLE_MAX:
                raise self._reject(f"a tuple of more than {TUPLE_MAX}")
            self.TM.push(0, struct.pack("<H", self.ty))
            return kind, payload
        if kind == K_LOW:
            raise self._reject("low8, low16 and low32 take one integer")
        if kind == K_IDX:
            self._need_int(self.ty, "idx's operand")
            frame[1] = 1  # the bound comes next, and it must be a literal
            self._store(frame)
            return None
        if kind == K_GET:
            if not FIRST_ARRAY <= self.ty < FIRST_FIN:
                raise self._reject(f"get needs an array, got {self._show(self.ty)}")
            frame[2] = self.ty
            self._store(frame)
            return kind, payload
        if kind == K_IF:
            if index == 1:
                if self.ty != T_BOOL:
                    raise self._reject("an if condition must be Bool")
                frame[6] = self.used
            else:
                frame[2] = self.ty
                frame[7] = self.used
                self.used = frame[6]
            self._store(frame)
        elif kind == K_MATCH:
            if self.ty < FIRST_NAMED or is_composite(self.ty):
                raise self._reject(f"cannot match on {self._show(self.ty)}")
            frame[2] = self.ty
            # R3: only a value whose count is exactly one unknown can have it expanded
            form = self.form
            single = (form.constant == 0 and not form.opaque and form.unknowns
                      and not form.unknowns & form.unknowns - 1)
            frame[4] = form.unknowns.bit_length() if single else 0
            frame[6] = self.used
            self._store(frame)
        elif kind == K_ARM:
            if frame[1] != frame[0]:
                raise self._reject(f"pattern needs {frame[0]} binders, got {frame[1]}")
        elif kind == K_WITH:
            if index == 1:
                record = self.ty
                if (record < FIRST_NAMED or is_composite(record)
                        or self._g(record - FIRST_NAMED)[1] != K_RECORD):
                    raise self._reject("'with' needs a record")
                frame[2] = record
                self._store(frame)
        elif kind in (K_BINOP, K_CMP):
            self._need_int(self.ty, "an arithmetic operand")
            frame[2] = self.ty
            self._store(frame)
        elif kind in (K_AND, K_OR):
            if self.ty != T_BOOL:
                raise self._reject("a logical operand must be Bool")
        elif kind == K_CALL:
            self._arg(frame)
        elif kind == K_CTOR:
            self._arg(frame)
            field = self._g(frame[4] + 1 + frame[5])
            return kind, P_FIELD.pack(frame[5], field[1], field[2], attributes(field[7]))
        elif kind == K_PACK:
            self._need_int(self.ty, "a pack field")
            frame[5] += 1
            self._store(frame)
        return kind, payload

    def _bind_tuple(self, kind: int, ty: int, first_name: int, count: int):
        """let (a, b) = e in ...: each name gets its member. A count is only known through a
        single value, so a linear member's count is opaque, which R3 never trusts."""
        if not self._is_tuple(ty):
            raise self._reject(f"{count} names bound to {self._show(ty)}, which is not a tuple")
        members = self._tuple_of(ty)
        if len(members) != count:
            raise self._reject(f"{count} names bound to a tuple of {len(members)}")
        names = [self.N.at(first_name + i)[1][:PAYLOAD_BYTES] for i in range(count)]
        for _ in range(count):
            self.N.pop()
        for name, member in zip(names, members):
            self._bind(name, member, OPAQUE if is_linear(member) else Cells())
        return kind, P_LET_RT.pack(1, 1, 8, count)

    def _check_main(self, first: int, nparams: int) -> None:
        """_start hands main the whole of Lambda_cell as one free list, and reads
        its result. Exactly-once linearity shapes both ends of that boundary."""
        if nparams == 0:
            if is_linear(self.ret_ty):
                raise self._reject("main without parameters owns no cell to return one in")
            return
        if nparams > 1:
            raise self._reject("main takes nothing, or one free list")
        param_ty = self._g(first)[5]
        if param_ty < FIRST_NAMED or len(self._ctors(param_ty - FIRST_NAMED)) != 2:
            raise self._reject("main's parameter must be a free list: End | More(next: Free)")
        for ctor in self._ctors(param_ty - FIRST_NAMED):
            _, _tag, is_null, count, _tagged, _x, _y, _z = self._g(ctor)
            if not is_null and (count != 1 or self._g(ctor + 1)[7] != param_ty):
                raise self._reject("main's free list must be End | More(next: Free)")
        ret = self.ret_ty
        if ret < FIRST_NAMED or len(self._ctors(ret - FIRST_NAMED)) != 2:
            raise self._reject("main must return Fail | Done(value: Int64, ...)")
        for ctor in self._ctors(ret - FIRST_NAMED):
            _, _tag, is_null, count, _tagged, _x, _y, _z = self._g(ctor)
            if not is_null and (count == 0 or self._g(ctor + 1)[7] != T_I64
                                or self._g(ctor + 1)[1] != 0):
                raise self._reject("main must return Fail | Done(value: Int64, ...)")

    def _arg(self, frame: list) -> None:
        """Check the argument just finished against parameter or field frame[5]."""
        position = frame[5]
        if position >= frame[0]:
            raise self._reject(f"too many arguments: expected {frame[0]}")
        if self.K.peek()[0] == K_CTOR:
            wanted = value_of(self._g(frame[4] + 1 + position)[7])
        else:
            wanted = self._g(frame[4] + position)[5]
        self._unify(wanted, self.ty, "an argument")
        if self.K.peek()[0] == K_CTOR:
            if is_linear(wanted):
                frame[3] += self.form.constant
                frame[6] |= self.form.unknowns
                frame[1] = frame[1] or self.form.opaque
        elif frame[1]:  # an argument to a recursive call: kept until the call closes
            self.A.push(0, _ARGUMENT.pack(self.form.constant, self.form.unknowns, self.form.opaque))
        frame[5] = position + 1
        self._store(frame)

    def _candidate_sets(self, first: int, nparams: int) -> int:
        """Every nonempty set of linear positions, as a bit per subset. A set of positions
        is a number below 64, because a function has at most 6 parameters."""
        positions = 0
        for position in range(nparams):
            ty = self._g(first + position)[5]
            if is_composite(ty):
                continue  # an array has a fixed size: it can never be what shrinks
            if is_linear(ty) and not (
                    ty >= FIRST_NAMED and self.G.at(ty - FIRST_NAMED)[1][:PAYLOAD_BYTES].rstrip(b"\0")
                    == RESERVE_TYPE):
                positions |= 1 << position
        sets = 0
        for subset in range(1, 1 << nparams):
            if not subset & ~positions:
                sets |= 1 << subset
        return sets

    def _rule_r3(self, count: int) -> None:
        """Strike out every set of positions this recursive call fails to shrink. What is
        left at the end is a natural number that falls at every call, so f terminates."""
        args = []
        for position in range(count):
            data = self.A.at(len(self.A) - count + position)[1]
            constant, unknowns, opaque = _ARGUMENT.unpack_from(data)
            args.append(Cells(constant, unknowns, bool(opaque)))
        for _ in range(count):
            self.A.pop()
        surviving = 0
        for subset in range(1, 1 << count):
            if self.viable >> subset & 1 and self._falls(subset, args):
                surviving |= 1 << subset
        self.viable = surviving
        self.fn_recursive = True
        if not surviving:
            name = self.G.at(self.fn_self)[1][:PAYLOAD_BYTES].rstrip(b"\0").decode()
            raise self._reject(
                f"rule R3: no fixed set of linear arguments of {name} loses cells at every "
                f"recursive call so far, not counting Free, so nothing shows the recursion ends")

    def _decreasing_set(self) -> int:
        """The set of positions rule R3 settled on, which the RT layer's potential sits on: the
        smallest, since a smaller set asks less of its callers. 0 if nothing recursed."""
        if not self.fn_recursive:
            return 0
        sets = [s for s in range(1, 64) if self.viable >> s & 1]
        return min(sets, key=lambda s: (bin(s).count("1"), s))

    def _falls(self, subset: int, args: list[Cells]) -> bool:
        """before - after must be at least 1 whatever the unknowns are, and they can all be
        0. So after may use only unknowns before has, and the constants must leave 1."""
        before = after = Cells()
        for position, arg in enumerate(args):
            if subset >> position & 1:
                before = before.plus(self.entry[position])
                after = after.plus(arg)
        return (not after.opaque and not after.unknowns & ~before.unknowns
                and before.constant - after.constant >= 1)

    # --- leaf

    def _on_leaf(self, kind: int, payload: bytes):
        if kind == K_LIVE:  # the right-to-left pass, on its way to Lower: not a value
            return kind, payload
        if kind == K_EOF:
            self._check_forwards_resolved()
            return kind, payload
        if kind == K_PARAM:
            self.pending_name = payload
            return None
        if kind in (K_TYPE, K_TARRAY, K_TFIN):
            frame = self._top()
            ty = self._type_token(kind, payload)
            if ty in _WIRE:
                raise self._reject(f"{self._show(ty)} says how a field lies in a cell, and a value "
                                   f"is an integer: {self._show(_WIRE[ty])} here")
            if self.K.peek()[0] == K_TTUPLE:
                if len(self.TM) - frame[4] >= TUPLE_MAX:
                    raise self._reject(f"a tuple of more than {TUPLE_MAX}")
                self.TM.push(0, struct.pack("<H", ty))
                return None
            if frame[0] == 1:
                self.ret_ty = ty
                return K_TYPE, P_TYPE_RT.pack(ty)
            if len(self.G) - frame[4] == 6:
                raise self._reject("more than 6 parameters (the register budget)")
            src = self._bind(self.pending_name, ty)
            self.entry.append(self._form_of(src))
            self._g_push(E_PARAM, self.pending_name, x=ty)
            return K_PARAM, P_PARAM_RT.pack(src, size_of(ty), is_linear(ty), ty)
        if kind == K_RESWORD:
            src, ty, _ = self._lookup(payload)
            if ty != self.ret_ty:
                raise self._reject("'@' must name a parameter of the result type")
            self.fn_at = src + 1
            return kind, struct.pack("<H", src)
        if kind == K_BINDER:
            return self._binder(payload)
        if kind == K_FIELD:
            return self._with_field(payload)
        if kind == K_DIAMOND:
            src, ty, _ = self._use(payload)
            if ty != T_DIAMOND:
                raise self._reject("a constructor's cell must have type Diamond")
            frame = self._top()
            _, tag, _null, count, tagged, _x, _y, _z = self._g(frame[4])
            off = size = linear = 0
            if count:
                field = self._g(frame[4] + 1)
                off, size, linear = field[1], field[2], attributes(field[7])
            self.child_seen = False
            return kind, P_DIAMOND_RT.pack(src, tag, tagged, count, off, size, linear, frame[4])
        if len(self.K) and self.K.peek()[0] == K_IDX and self._top()[1]:
            if kind != K_INT:
                raise self._reject("the bound of idx must be a literal")
            bound = struct.unpack_from("<Q", payload)[0]
            if bound == 0 or bound & bound - 1:
                raise self._reject(f"section 4.4: idx by {bound}: a power of two until division "
                                   f"exists, because idx is then a mask")
            frame = self._top()
            frame[1] = 2
            frame[7] = bound
            self._store(frame)
            return None
        self.child_seen = True
        self.form = Cells()  # literals, Bool and null constructors own no cells
        if kind in (K_VAR, K_VARLAST):
            src, ty, linear = self._use(payload)
            self.ty = ty
            self.form = self._form_of(src)
            return kind, P_VAR.pack(src, linear != 255, size_of(ty))
        if kind == K_INT:
            value = struct.unpack_from("<Q", payload)[0]
            self.ty = (T_LIT8 if value < 1 << 8 else T_LIT16 if value < 1 << 16
                       else T_LIT32 if value < 1 << 32 else T_LIT)
        elif kind == K_BOOL:
            self.ty = T_BOOL
        elif kind == K_NULLCTOR:
            found = self._g_find(E_CTOR, payload)
            if found is None or not self._g(found)[2]:
                raise self._reject(
                    f"{payload.rstrip(bytes(1)).decode()} is not a null constructor")
            self.ty = self._g(found)[5] + FIRST_NAMED
            return kind, P_TYPE_RT.pack(self.ty)
        return kind, payload

    def _binder(self, payload: bytes):
        frame = self._top()
        if self.K.peek()[0] == K_LET:
            if len(self.N) - frame[5] >= TUPLE_MAX:
                raise self._reject(f"more than {TUPLE_MAX} names in one let")
            self.N.push(0, payload)
            return K_BINDER, b""
        position = frame[1]
        if position >= frame[0]:
            raise self._reject(f"pattern has too many binders: expected {frame[0]}")
        frame[1] = position + 1
        self._store(frame)
        if position == 0:
            self._bind(payload, T_DIAMOND)
            return K_BINDER, P_BINDER.pack(0, 0, 8, 1)
        field = self._g(frame[5] + position)
        index = self._bind(payload, value_of(field[7]))
        if frame[2]:  # the scrutinee was an unknown of this parameter: these fields replace it
            self.entry[frame[2] - 1] = self.entry[frame[2] - 1].plus(self._form_of(index))
        return K_BINDER, P_BINDER.pack(1, field[1], field[2], attributes(field[7]))

    def _with_field(self, payload: bytes):
        frame = self._top()
        ctor = frame[2] - FIRST_NAMED + 1
        count = self._g(ctor)[3]
        for position in range(ctor + 1, ctor + 1 + count):
            if self.G.at(position)[1][:PAYLOAD_BYTES] == payload:
                field = self._g(position)
                frame[3] = value_of(field[7])
                self._store(frame)
                self._bind(payload, frame[3])
                return K_FIELD, P_WITH_RT.pack(field[1], field[2], attributes(field[7]), ctor,
                                               position - ctor - 1, count)
        if count and self.G.at(ctor + count)[1][:PAYLOAD_BYTES].rstrip(b"\0") == LINK_NAME:
            raise self._reject(f"{{ with }} reaches the first cell of {self._show(frame[2])}, and "
                               f"{payload.rstrip(bytes(1)).decode()} is not there: rebuild it with a match")
        raise self._reject(f"no field {payload.rstrip(bytes(1)).decode()} in {self._show(frame[2])}")

    # --- close

    def _on_close(self, kind: int, payload: bytes):
        frame = self._top()
        if kind == K_TTUPLE:
            members = self._members_open()
            self.K.pop()
            self.ret_ty = self._tuple_type(members, True)
            return K_TYPE, P_TYPE_RT.pack(self.ret_ty)
        if kind == K_TUPLE:
            self.TM.push(0, struct.pack("<H", self.ty))
            members = self._members_open()
            self.ty = self._tuple_type(members)
            self.K.pop()
            self.child_seen = True
            return kind, payload
        if kind == K_LOW:
            if not self.child_seen:
                raise self._reject("low8, low16 and low32 take one integer")
            self._need_int(self.ty, "a narrowing")
            self.ty = frame[2]
            self.K.pop()
            self.child_seen = True
            return kind, bytes((size_of(self.ty),))
        if kind == K_IDX:
            if frame[1] != 2:
                raise self._reject("idx takes a value and a literal bound")
            self.ty = self._fin_type(frame[7], False)
            self.K.pop()
            self.child_seen = True
            return kind, P_IDX_RT.pack(frame[7])
        if kind == K_GET:
            array = frame[2]
            if not FIRST_ARRAY <= array < FIRST_FIN:
                raise self._reject("get takes an array and an index")
            length, element = self._array_of(array)
            if not FIRST_FIN <= self.ty < FIRST_TUPLE or self._fin_of(self.ty) != length:
                raise self._reject(f"get's index must be Fin {length}, got {self._show(self.ty)}")
            self.ty = self._tuple_type([array, element])
            self.K.pop()
            self.child_seen = True
            return kind, P_GET_RT.pack(size_of(element))
        if kind in (K_FN, K_HANDLER):
            self._unify(self.ret_ty, self.ty, "a function result")
            self._unbind_to(0)
            self.fn_count += 1
            self.K.pop()
            return kind, bytes((self._decreasing_set(),))
        if kind == K_LET:
            self.closing = self.form  # the value of a let is the value of its body
            self._unbind_to(frame[4])
        elif kind == K_IF:
            self.ty = self._unify(frame[2], self.ty, "the arms of an if")
            if self.used != frame[7]:
                raise self._reject("the arms of an if consume different linear variables")
        elif kind == K_MATCH:
            type_index = frame[2] - FIRST_NAMED
            if frame[0] != len(self._ctors(type_index)):
                raise self._reject(f"non-exhaustive match on {self._show(frame[2])}")
            self.ty = frame[3]
            self.used = frame[8]
        elif kind == K_ARM:
            self._unbind_to(frame[4])
            if frame[2]:
                self.entry[frame[2] - 1] = Cells(frame[3], frame[6])
            self.K.pop()
            parent = self._top()
            if parent[1]:
                self.ty = self._unify(parent[3], self.ty, "the arms of a match")
                if self.used != parent[8]:
                    raise self._reject("the arms of a match consume different linear variables")
            else:
                parent[1] = 1
                parent[8] = self.used
            parent[3] = self.ty
            self._store(parent)
            return kind, payload
        elif kind == K_WITH:
            self._unify(frame[3], self.ty, "a field update")
            self._unbind_to(frame[4])
            self.ty = frame[2]
        elif kind == K_BINOP:
            self._need_int(self.ty, "an arithmetic operand")
            if OP_NAMES[frame[0]] in ("shl", "shr") and self.ty in _LITERALS:
                self.ty = frame[2]  # section 4.4: a count is not a value of the type, and any has a meaning
            else:
                self.ty = self._unify(frame[2], self.ty, "an arithmetic operation")
            if self.ty in _LITERALS:
                self.ty = T_I64  # arithmetic on two literals has no context to adopt
            self.K.pop()
            self.child_seen = True
            return kind, bytes((size_of(self.ty),))  # Lower never guesses a width
        elif kind == K_CMP:
            self._need_int(self.ty, "a comparison operand")
            self._unify(frame[2], self.ty, "a comparison")
            self.ty = T_BOOL
        elif kind in (K_AND, K_OR, K_NOT):
            if self.ty != T_BOOL:
                raise self._reject("a logical operand must be Bool")
        elif kind in (K_CALL, K_CTOR):
            if self.child_seen:
                self._arg(frame)
                frame = self._top()
            if frame[5] != frame[0]:
                raise self._reject(f"wrong argument count: expected {frame[0]}")
            if kind == K_CTOR:  # mu(C@e(args)) = 1 + the sum of mu over the linear args
                self.closing = Cells(1 + frame[3], frame[6], bool(frame[1]))
            elif frame[1]:
                self._rule_r3(frame[0])
            self.ty = frame[2]
            if kind == K_CALL:
                members = self._tuple_of(self.ty) if self._is_tuple(self.ty) else [self.ty]
                mask = sum(1 << i for i, m in enumerate(members) if is_linear(m))
                self.K.pop()
                self.child_seen = True
                return kind, bytes((mask,))
        elif kind == K_PACK:
            if not self.child_seen:
                raise self._reject("pack needs one expression per width")
            self._need_int(self.ty, "a pack field")
            if frame[5] + 1 != frame[0]:
                raise self._reject("pack needs one expression per width")
            self.ty = T_I64
        self.K.pop()
        self.child_seen = True
        return kind, payload
