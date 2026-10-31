"""Check: first-order linear typing as a fold over the Dyck word.

Linear variables (diamond, data, records) are used exactly once on every path.
Scalars are erasable, and reading one does not consume it. Lower enforces the
one further rule a forward fold requires: overwriting a location erases the
scalar homed there.

Usage state is a 64-bit mask, so a branch frame holds its snapshots in constant
space. That caps live linear binders per function at 64, a language limit that
depends only on the program text, like D_max. Names go no further than this
stage: Lower receives binder positions, offsets and widths in event payloads,
so it never compares a name and its frames stay small.
"""
from __future__ import annotations

import struct
from typing import Optional

from .dyck import (CLOSE, K_AND, K_ARM, K_BINDER, K_BINOP, K_BOOL, K_CALL, K_CMP,
                   K_CTOR, K_CTORDECL, K_DATA, K_DIAMOND, K_EOF, K_FIELD, K_FN, K_IF,
                   K_INT, K_LET, K_MATCH, K_NOT, K_NULLCTOR, K_OR, K_PACK, K_PARAM,
                   K_RECORD, K_RESWORD, K_TYPE, K_VAR, K_WITH, LEAF, OPEN,
                   ev_class, ev_kind, ev_tag)
from .ring import Reject, Ring, Stage
from .tokens import PAYLOAD_BYTES

CELL_FIELDS_MAX = 16
LINEAR_BINDERS_MAX = 64

# Type ids. Named types are G positions offset by FIRST_NAMED.
T_I8, T_I16, T_I32, T_I64, T_BOOL, T_DIAMOND, T_LIT = range(7)
FIRST_NAMED = 16
_BUILTIN = {b"Int8": T_I8, b"Int16": T_I16, b"Int32": T_I32, b"Int64": T_I64,
            b"Bool": T_BOOL, b"Diamond": T_DIAMOND}
_SIZES = {T_I8: 1, T_I16: 2, T_I32: 4, T_I64: 8, T_BOOL: 1, T_LIT: 8}

# G entries: name in the first cell, attributes in the second.
E_TYPE, E_CTOR, E_FIELD, E_FUNC, E_PARAM = range(5)
_ATTR = struct.Struct("<BBBBHHH")  # a, b, c, d, x, y, z
_FRAME = struct.Struct("<BBHHHHQQQ")  # a, b, t1, t2, c1, c2, m1, m2, m3
_BINDER = struct.Struct("<HB")  # type, linear index or 255
_FORWARD, _RESOLVED = 254, 253  # the `a` attribute of a forward-declared type

# Payloads for Lower.
P_VAR = struct.Struct("<HBB")     # src index, linear, size
P_CALL = struct.Struct("<HBB")    # function ordinal, is self, argument count
P_DIAMOND = struct.Struct("<HBBBBBB")  # d src, tag, tagged, nfields, off, size, linear
P_FIELD = struct.Struct("<BBBB")  # index, off, size, linear
P_ARM = struct.Struct("<BBBBB")   # tag, tagged, is the null constructor, arm index, nctors
P_BINDER = struct.Struct("<BBBB")  # 0 for diamond or 1 for a field, off, size, linear
P_LET = struct.Struct("<BBB")     # separator 1, linear, size
P_PARAM = struct.Struct("<HBB")   # src index, size, linear
P_WITH = struct.Struct("<BBB")    # off, size, linear


def is_linear(ty: int) -> bool:
    return ty == T_DIAMOND or ty >= FIRST_NAMED


def size_of(ty: int) -> int:
    return _SIZES.get(ty, 8)


class CheckStage(Stage):
    """The typing judgment, one event at a time."""

    name = "check"

    def __init__(self, g_max: int = 2048) -> None:
        super().__init__()
        self.g_max = g_max
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
        self.pending_name = b""
        self.child_seen = False

    def attach(self, ring: Ring) -> None:
        super().attach(ring)
        self.G = self.stack()
        self.K = self.stack()
        self.B = self.stack()
        self.N = self.stack()  # the name of each let whose bound value is still open

    def depth(self) -> int:
        return len(self.K)

    # --- errors, G and binders

    def _reject(self, message: str) -> Reject:
        # "Near": the ring holds a few tokens between Lex and here, so the lexer's
        # line can be slightly ahead of the construct this stage is looking at.
        return Reject(f"near line {self.ring.stages[0].line}: {message}")

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

    def _type(self, name: bytes) -> int:
        key = name.rstrip(b"\0")
        if key in _BUILTIN:
            return _BUILTIN[key]
        found = self._g_find(E_TYPE, name)
        if found is None:
            raise self._reject(f"unknown type {key.decode()}")
        return found + FIRST_NAMED

    def _bind(self, name: bytes, ty: int) -> int:
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
        self.B.push(0, name.ljust(PAYLOAD_BYTES, b"\0") + _BINDER.pack(ty, linear))
        return len(self.B) - 1

    def _unbind_to(self, mark: int) -> None:
        while len(self.B) > mark:
            _, data = self.B.peek()
            ty, linear = _BINDER.unpack_from(data, PAYLOAD_BYTES)
            if linear != 255:
                if not self.used >> linear & 1:
                    name = data[:PAYLOAD_BYTES].rstrip(b"\0").decode()
                    raise self._reject(f"linear variable {name} is never used")
                self.used &= ~(1 << linear)
                self.lin_held &= ~(1 << linear)
            self.B.pop()

    def _lookup(self, name: bytes) -> tuple[int, int, int]:
        for index in range(len(self.B) - 1, -1, -1):
            _, data = self.B.at(index)
            if data[:PAYLOAD_BYTES] == name:
                ty, linear = _BINDER.unpack_from(data, PAYLOAD_BYTES)
                return index, ty, linear
        raise self._reject(f"unbound variable {name.rstrip(bytes(1)).decode()}")

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
        if a == T_LIT and b in (T_I8, T_I16, T_I32, T_I64):
            return b
        if b == T_LIT and a in (T_I8, T_I16, T_I32, T_I64):
            return a
        raise self._reject(f"type mismatch in {where}: {self._show(a)} and {self._show(b)}")

    def _show(self, ty: int) -> str:
        for name, value in _BUILTIN.items():
            if value == ty:
                return name.decode()
        if ty == T_LIT:
            return "integer literal"
        _, data = self.G.at(ty - FIRST_NAMED)
        return data[:PAYLOAD_BYTES].rstrip(b"\0").decode()

    def _need_int(self, ty: int, where: str) -> None:
        if ty not in (T_I8, T_I16, T_I32, T_I64, T_LIT):
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
        assert cell is not None
        store = self.ring.store
        tag, payload = store.tag(cell), store.payload(cell)
        cls, kind = ev_class(tag), ev_kind(tag)
        if kind in (K_DATA, K_CTORDECL, K_RECORD) or (
                len(self.K) and self.K.peek()[0] in (K_DATA, K_CTORDECL, K_RECORD)):
            self._declaration(cls, kind, payload)
            self.ring.drop(self, cell)
            return
        result = (self._on_open, self._on_sep, self._on_leaf, self._on_close)[cls](
            kind, payload)
        if result is None:
            self.ring.drop(self, cell)
        else:
            new_kind, new_payload = result
            self.ring.pass_(self, cell, ev_tag(cls, new_kind), new_payload)

    # --- declarations: consumed here, never forwarded

    def _declaration(self, cls: int, kind: int, payload: bytes) -> None:
        if cls == OPEN and kind in (K_DATA, K_RECORD):
            found = self._g_find(E_TYPE, payload)
            self.decl_forward = -1
            if found is not None:
                if self._g(found)[1] != _FORWARD:
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
        elif cls == LEAF and kind == K_TYPE:
            field = len(self.G) - 1
            field_ty = self._type(payload)
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
            self._resolve_forward()
        elif cls == CLOSE and kind == K_DATA:
            self.K.pop()
            if self._ctors(self.decl_type):
                self._finish_data()
                self._resolve_forward()
            elif self.decl_forward >= 0:
                raise self._reject("a type is declared forward twice")
            else:
                _, _a, b, c, d, x, y, z = self._g(self.decl_type)
                self._g_set(self.decl_type, _FORWARD, b, c, d, x, y, z)
        else:
            raise self._reject("malformed declaration")

    def _resolve_forward(self) -> None:
        """Fields written against the forward declaration now name the definition.
        G stays linear in declarations: this scan runs once per forward type."""
        if self.decl_forward < 0:
            return
        old = self.decl_forward + FIRST_NAMED
        new = self.decl_type + FIRST_NAMED
        for index in range(len(self.G)):
            entry = self._g(index)
            if entry[0] == E_FIELD and entry[7] == old:
                self._g_set(index, *entry[1:7], new)
        _, _a, b, c, d, x, y, z = self._g(self.decl_forward)
        self._g_set(self.decl_forward, _RESOLVED, b, c, d, x, y, z)

    def _check_forwards_resolved(self) -> None:
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
        if kind == K_FN:
            if self._g_find(E_FUNC, payload) is not None:
                raise self._reject("duplicate function name")
            self.fn_self = self._g_push(E_FUNC, payload, x=self.fn_count)
            self.used = self.lin_held = 0
            self._frame(K_FN, c1=len(self.G))
            return kind, payload
        if kind in (K_LET, K_WITH):
            self._frame(kind, c1=len(self.B))
        elif kind in (K_IF, K_MATCH, K_BINOP, K_CMP, K_AND, K_OR, K_NOT):
            self._frame(kind, t1=T_LIT)
        elif kind == K_CALL:
            found = self._g_find(E_FUNC, payload)
            if found is None:
                raise self._reject(
                    f"unknown function {payload.rstrip(bytes(1)).decode()}: a callee "
                    f"is defined before its caller, so the call graph stays acyclic")
            _, nparams, _b, _c, _d, ordinal, ret, first = self._g(found)
            self._frame(K_CALL, a=nparams, t1=ret, c1=first)
            self.child_seen = False
            return kind, P_CALL.pack(ordinal, found == self.fn_self, nparams)
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
        self._frame(K_ARM, a=0 if is_null else count + 1, c1=len(self.B), c2=found)
        return K_ARM, P_ARM.pack(tag, tagged, is_null, arm_index, nctors)

    # --- sep

    def _on_sep(self, kind: int, payload: bytes):
        frame = self._top()
        index = payload[0]
        if kind == K_FN:
            if index == 0:
                frame[0] = 1  # parameters finished; the next type is the result
                self._store(frame)
                return None
            func = self.fn_self
            first = frame[4]
            nparams = len(self.G) - first
            if self.G.at(func)[1][:PAYLOAD_BYTES].rstrip(b"\0") == b"main":
                self._check_main(first, nparams)
            _, _a, b, c, d, ordinal, _y, _z = self._g(func)
            self._g_set(func, nparams, b, c, d, ordinal, self.ret_ty, first)
            return kind, payload
        if kind == K_LET:
            ty = self.ty
            name = self.N.peek()[1][:PAYLOAD_BYTES]
            self.N.pop()
            self._bind(name, ty)
            return kind, P_LET.pack(1, is_linear(ty), size_of(ty))
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
            if self.ty < FIRST_NAMED:
                raise self._reject(f"cannot match on {self._show(self.ty)}")
            frame[2] = self.ty
            frame[6] = self.used
            self._store(frame)
        elif kind == K_ARM:
            if frame[1] != frame[0]:
                raise self._reject(f"pattern needs {frame[0]} binders, got {frame[1]}")
        elif kind == K_WITH:
            if index == 1:
                record = self.ty
                if record < FIRST_NAMED or self._g(record - FIRST_NAMED)[1] != K_RECORD:
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
            return kind, P_FIELD.pack(frame[5], field[1], field[2], is_linear(field[7]))
        elif kind == K_PACK:
            self._need_int(self.ty, "a pack field")
            frame[5] += 1
            self._store(frame)
        return kind, payload

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
            wanted = self._g(frame[4] + 1 + position)[7]
        else:
            wanted = self._g(frame[4] + position)[5]
        self._unify(wanted, self.ty, "an argument")
        frame[5] = position + 1
        self._store(frame)

    # --- leaf

    def _on_leaf(self, kind: int, payload: bytes):
        if kind == K_EOF:
            self._check_forwards_resolved()
            return kind, payload
        if kind == K_PARAM:
            self.pending_name = payload
            return None
        if kind == K_TYPE:
            frame = self._top()
            ty = self._type(payload)
            if frame[0] == 1:
                self.ret_ty = ty
                return None
            if len(self.G) - frame[4] == 6:
                raise self._reject("more than 6 parameters (the register budget)")
            src = self._bind(self.pending_name, ty)
            self._g_push(E_PARAM, self.pending_name, x=ty)
            return K_PARAM, P_PARAM.pack(src, size_of(ty), is_linear(ty))
        if kind == K_RESWORD:
            src, ty, _ = self._lookup(payload)
            if ty != self.ret_ty:
                raise self._reject("'@' must name a parameter of the result type")
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
                off, size, linear = field[1], field[2], is_linear(field[7])
            self.child_seen = False
            return kind, P_DIAMOND.pack(src, tag, tagged, count, off, size, linear)
        self.child_seen = True
        if kind == K_VAR:
            src, ty, linear = self._use(payload)
            self.ty = ty
            return kind, P_VAR.pack(src, linear != 255, size_of(ty))
        if kind == K_INT:
            self.ty = T_LIT
        elif kind == K_BOOL:
            self.ty = T_BOOL
        elif kind == K_NULLCTOR:
            found = self._g_find(E_CTOR, payload)
            if found is None or not self._g(found)[2]:
                raise self._reject(
                    f"{payload.rstrip(bytes(1)).decode()} is not a null constructor")
            self.ty = self._g(found)[5] + FIRST_NAMED
        return kind, payload

    def _binder(self, payload: bytes):
        frame = self._top()
        if self.K.peek()[0] == K_LET:
            self.N.push(0, payload)
            return None
        position = frame[1]
        if position >= frame[0]:
            raise self._reject(f"pattern has too many binders: expected {frame[0]}")
        frame[1] = position + 1
        self._store(frame)
        if position == 0:
            self._bind(payload, T_DIAMOND)
            return K_BINDER, P_BINDER.pack(0, 0, 8, 1)
        field = self._g(frame[5] + position)
        self._bind(payload, field[7])
        return K_BINDER, P_BINDER.pack(1, field[1], field[2], is_linear(field[7]))

    def _with_field(self, payload: bytes):
        frame = self._top()
        ctor = frame[2] - FIRST_NAMED + 1
        count = self._g(ctor)[3]
        for position in range(ctor + 1, ctor + 1 + count):
            if self.G.at(position)[1][:PAYLOAD_BYTES] == payload:
                field = self._g(position)
                frame[3] = field[7]
                self._store(frame)
                self._bind(payload, field[7])
                return K_FIELD, P_WITH.pack(field[1], field[2], is_linear(field[7]))
        raise self._reject(f"no field {payload.rstrip(bytes(1)).decode()} in {self._show(frame[2])}")

    # --- close

    def _on_close(self, kind: int, payload: bytes):
        frame = self._top()
        if kind == K_FN:
            self._unify(self.ret_ty, self.ty, "a function result")
            self._unbind_to(0)
            self.fn_count += 1
            self.K.pop()
            return kind, payload
        if kind == K_LET:
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
            self.ty = self._unify(frame[2], self.ty, "an arithmetic operation")
            if self.ty == T_LIT:
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
            self.ty = frame[2]
        elif kind == K_PACK:
            self._need_int(self.ty, "a pack field")
            if frame[5] + 1 != frame[0]:
                raise self._reject("pack needs one expression per width")
            self.ty = T_I64
        self.K.pop()
        self.child_seen = True
        return kind, payload
