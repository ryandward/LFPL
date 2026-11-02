"""Lower: the checked event word to the register-transfer stream, as a fold.

Each event is handled against the frame of the construct that encloses it, so a child's
destination is fixed when it opens, exactly as the translation calculus asks. This is what the
old lower.py did, without the part that made it 2,600 lines: no physical register, frame slot,
flag or spill appears here. A value is a register with no limit on how many there are, or a
literal, or a location of an open cell that has not needed to move. docs/rt-layer.md is the
language it writes, and rtcheck types every line of it before anything is done with it.

Frames, values and binders live in real cells, on the ring, so this stage keeps the first
specification's O(D) bound the same way every other stage does. The text it writes leaves on c5.

Reading only forward leaves one thing it cannot know, which is whether a scalar is wanted after its cell is
rebuilt. src/righttoleft.py works that out ahead of it, and says so as this stage reads: `live_now` is the set of
names with a use still to come on this path, so that a scalar moves out exactly when something reads it later.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, replace
from typing import BinaryIO, Callable, Generic, Iterable, Iterator, NamedTuple, Optional, Protocol, TypeVar

from .check import (P_ARM_RT, P_BINDER, P_CALL_RT, P_DIAMOND_RT, P_FIELD, P_GET_RT, P_IDX_RT,
                    P_LET_RT, P_PARAM_RT, P_TYPE_RT, P_VAR, P_WITH_RT, CheckStage)
from .dyck import (CC_NAMES, CLOSE, K_AND, K_ARM, K_BINDER, K_BINOP, K_BOOL, K_CALL, K_CMP, K_CTOR,
                   K_DIAMOND, K_EOF, K_FIELD, K_FN, K_GET, K_HANDLER, K_IDX, K_IF, K_INT, K_LET, K_LIVE,
                   K_LOW, K_MATCH, K_NOT, K_NULLCTOR, K_OR, K_PACK, K_PARAM, K_RESWORD, K_RTDECL,
                   K_RTDECLMORE, K_TUPLE, K_TYPE, K_VAR, K_VARLAST, K_WITH, LEAF, OP_NAMES, OPEN, SEP, Lexer,
                   Parser, ev_class, ev_kind)
from .righttoleft import CELLS_PER_DEPTH, MASK_BITS, RightToLeft
from .ring import FrameStack, Reject, Ring, Stage
from .tokens import PAYLOAD_BYTES

LINE_MORE, LINE_END = 0, 1
_RT_OPS = {"imul": "mul"}
_RMW = frozenset(("add", "sub", "and", "or", "xor"))
UNKNOWN_TAG = 255
MAX_BINDERS = 128  # LFPL-O section 8.2: names in scope in one function
MAX_REGISTERS = 65535  # registers one function may write, which a value's two bytes can name
NO_REFUGE = MAX_REGISTERS  # a scalar's refuge is chosen the first time it has to move
E_MAX = 4096       # the most cells of text one event may write: a settle of a whole value stack


class Limit(Exception):
    """A limit of the language that a program can run into and fix: LFPL-O section 8.2. Any
    other failure in this module is a bug in the compiler."""


# --------------------------------------------------------------------- values

V_LIT, V_NULL, V_REG, V_LOC, V_CAP, V_TUPLE, V_VIRT, V_CALL, V_FUNC = range(9)
OWNED, LINEAR, CONSUMED, BOOL = 1, 2, 4, 8
_VALUE = struct.Struct("<BBHBBII")
_LITERAL = struct.Struct("<BBQ")


@dataclass(frozen=True)
class Value:
    """One of the things an expression can leave behind. `reg` is the register, or for a location
    the capability of its cell. For the rest it holds what the docstring of each kind says."""
    kind: int
    reg: int = 0    # V_NULL: its type. V_TUPLE: how many. V_CALL: the callee's ordinal.
    off: int = 0    # V_CAP: the tag known to be in the cell. V_VIRT, V_CALL: how many fields, args.
    size: int = 0
    word: int = 0   # V_VIRT: the constructor. V_FUNC: the operation's index.
    cell: int = 0   # which dispatch opened the cell. V_CALL: the word the call returns.
    flags: int = 0
    n: int = 0      # V_LIT

    def pack(self) -> bytes:
        if self.kind == V_LIT:
            return _LITERAL.pack(self.kind, self.flags, self.n)
        return _VALUE.pack(self.kind, self.flags, self.reg, self.off, self.size, self.word, self.cell)

    @staticmethod
    def unpack(data: bytes) -> "Value":
        if data[0] == V_LIT:
            kind, flags, n = _LITERAL.unpack_from(data)
            return Value(kind, flags=flags, n=n)
        kind, flags, reg, off, size, word, cell = _VALUE.unpack_from(data)
        return Value(kind, reg, off, size, word, cell, flags)


D_ANY, D_RET, D_REG, D_COND = range(4)
RETURN_IT = 2  # Frame.push: the value is the function's result, so `ret` it at close


class Dest(NamedTuple):
    kind: int
    d: int = 0
    count: int = 1
    t: int = 0
    f: int = 0


_FRAME = struct.Struct("<BBBBBBHHHHHHHIIIIIIIH")


@dataclass
class Frame:
    kind: int
    state: int = 0
    dest: int = D_ANY
    count: int = 1
    push: int = 0     # the dest was D_ANY: leave the joined value on the value stack at close
    noted: int = 0    # an arm has delivered, so this join's Delivered records hold what it did
    d: int = 0
    reg: int = 0      # the scrutinee, or the capability being built
    bmark: int = 0
    vmark: int = 0
    gmark: int = 0
    umark: int = 0
    aux: int = 0
    t: int = 0
    f: int = 0
    l1: int = 0
    l2: int = 0
    l3: int = 0
    word: int = 0     # the scrutinee's word, or the capability's
    cell: int = 0
    jmark: int = 0    # where this join's Delivered records start

    def pack(self) -> bytes:
        return _FRAME.pack(self.kind, self.state, self.dest, self.count, self.push, self.noted,
                           self.d, self.reg, self.bmark, self.vmark, self.gmark, self.umark, self.aux,
                           self.t, self.f, self.l1, self.l2, self.l3, self.word, self.cell, self.jmark)

    @staticmethod
    def unpack(data: bytes) -> "Frame":
        return Frame(*_FRAME.unpack_from(data))

    def as_dest(self) -> Dest:
        return Dest(self.dest, self.d, self.count, self.t, self.f)


class Packed(Protocol):
    def pack(self) -> bytes: ...

    @staticmethod
    def unpack(data: bytes) -> "Packed": ...


R = TypeVar("R", bound=Packed)


class Records(Generic[R]):
    """A stack of fixed-format records, in cells."""

    def __init__(self, frames: FrameStack, cls: type[R]):
        self.frames = frames
        self.cls = cls

    def __len__(self) -> int:
        return len(self.frames)

    def push(self, record: R) -> None:
        self.frames.push(0, record.pack())

    def at(self, index: int) -> R:
        return self.cls.unpack(self.frames.at(index)[1])

    def top(self) -> R:
        return self.at(len(self) - 1)

    def set(self, index: int, record: R) -> None:
        self.frames.rewrite(index, 0, record.pack())

    def set_top(self, record: R) -> None:
        self.set(len(self) - 1, record)

    def pop(self) -> R:
        record = self.top()
        self.frames.pop()
        return record

    def truncate(self, length: int) -> None:
        while len(self) > length:
            self.frames.pop()


@dataclass(frozen=True)
class Binder:
    """A value with a name. `refuge` is the register a field moves to if its cell is rebuilt while it
    is still there. A linear field's is chosen when the name is bound, and a scalar's when it first
    has to move, which most never do. Going back to the start of a branch does not forget it, so that
    every arm that has to move it moves it to the same register, and the arms still agree at the join."""
    value: Value
    refuge: int = 0

    def pack(self) -> bytes:
        return self.value.pack().ljust(15, b"\0") + struct.pack("<H", self.refuge)

    @staticmethod
    def unpack(data: bytes) -> "Binder":
        return Binder(Value.unpack(data), struct.unpack_from("<H", data, 15)[0])


@dataclass(frozen=True)
class Undo:
    """What a binder held before an arm changed it."""
    index: int
    binder: Binder

    def pack(self) -> bytes:
        return struct.pack("<H", self.index) + self.binder.pack()

    @staticmethod
    def unpack(data: bytes) -> "Undo":
        return Undo(struct.unpack_from("<H", data)[0], Binder.unpack(data[2:]))


@dataclass(frozen=True)
class Layout:
    """Where one field of the constructor being built goes."""
    off: int
    size: int
    linear: int

    def pack(self) -> bytes:
        return bytes((self.off, self.size, self.linear))

    @staticmethod
    def unpack(data: bytes) -> "Layout":
        return Layout(data[0], data[1], data[2])


@dataclass(frozen=True)
class Delivered:
    """What an arm left in one member of a destination, for the join to compare. A word survives
    a join only if every arm delivered that word. A capability keeps its cell the same way."""
    word: int = 0
    linear: int = 0
    cell: int = 0
    tag: int = UNKNOWN_TAG

    def pack(self) -> bytes:
        return struct.pack("<IBIB", self.word, self.linear, self.cell, self.tag)

    @staticmethod
    def unpack(data: bytes) -> "Delivered":
        return Delivered(*struct.unpack_from("<IBIB", data))


@dataclass(frozen=True)
class Cell:
    number: int

    def pack(self) -> bytes:
        return struct.pack("<I", self.number)

    @staticmethod
    def unpack(data: bytes) -> "Cell":
        return Cell(struct.unpack_from("<I", data)[0])


@dataclass(frozen=True)
class Ctor:
    """The constructor being built by the frame below it."""
    ctor: int
    tag: int
    tagged: int

    def pack(self) -> bytes:
        return struct.pack("<HBB", self.ctor, self.tag, self.tagged)

    @staticmethod
    def unpack(data: bytes) -> "Ctor":
        return Ctor(*struct.unpack_from("<HBB", data))


# ----------------------------------------------------------------- the stage


class RTLower(Stage):
    name = "lower"

    def __init__(self) -> None:
        super().__init__()
        self.out: list[tuple[int, bytes]] = []
        self.dead = 0              # how deep inside an arm a known constructor rules out
        self.live = False          # whether control can reach the next instruction
        self.ordinal = -1
        self.next_reg = 0
        self.next_label = 0
        self.next_word = 2
        self.next_cell = 1
        self.fn_name = b""
        self.fn_handler = False
        self.ret_ty = 0
        self.at: Optional[int] = None
        self.live_now = 0          # the names with a use still to come on this path: a set, as an integer
        self.param_types: list[int] = []  # at most 6
        self.last_into: list[Delivered] = [Delivered()]  # what was last delivered, per member

    def attach(self, ring: Ring) -> None:
        super().attach(ring)
        self.F = Records(self.stack(), Frame)
        self.V = Records(self.stack(), Value)
        self.B = Records(self.stack(), Binder)
        self.G = Records(self.stack(), Cell)  # the cells rebuilt or handed over on this path
        self.U = Records(self.stack(), Undo)
        self.L = Records(self.stack(), Layout)
        self.C = Records(self.stack(), Ctor)
        self.J = Records(self.stack(), Delivered)

    def depth(self) -> int:
        return len(self.F)

    # --- the text

    def emit(self, line: str) -> None:
        data = line.encode()
        while len(data) > PAYLOAD_BYTES:
            self.out.append((LINE_MORE, data[:PAYLOAD_BYTES]))
            data = data[PAYLOAD_BYTES:]
        self.out.append((LINE_END, data))

    def i(self, line: str) -> None:
        assert self.live, f"an instruction control cannot reach: {line}"
        self.emit("    " + line)

    def label(self, label: int) -> None:
        self.emit(f"L{label}:")
        self.live = True

    def jump(self, label: int) -> None:
        if self.live:
            self.i(f"jump L{label}")
            self.live = False

    def fresh(self, count: int = 1) -> int:
        first = self.next_reg
        self.next_reg += count
        if self.next_reg > MAX_REGISTERS:
            raise Limit(f"section 8.2: fn {self.fn_name.decode()} writes more than {MAX_REGISTERS} "
                        f"registers")
        return first

    def new_label(self) -> int:
        self.next_label += 1
        return self.next_label

    def new_word(self) -> int:
        self.next_word += 1
        return self.next_word

    def new_cell(self) -> int:
        self.next_cell += 1
        return self.next_cell

    # --- the fold

    def fire(self, cell: Optional[int]) -> None:
        assert cell is not None
        store = self.ring.store
        tag, payload = store.tag(cell), store.payload(cell)
        self.out = []
        self._event(ev_class(tag), ev_kind(tag), payload)
        if len(self.out) > E_MAX:
            raise Limit(f"section 8.2: one construct wrote {len(self.out)} cells of RT, past "
                        f"E = {E_MAX}")
        self.ring.expand(self, cell, self.out)

    def _event(self, cls: int, kind: int, payload: bytes) -> None:
        if self.dead:
            if cls == OPEN:
                self.dead += 1
            elif cls == CLOSE:
                self.dead -= 1
            return
        if kind == K_RTDECLMORE:
            self.out.append((LINE_MORE, payload))
        elif kind == K_RTDECL:
            self.out.append((LINE_END, payload))
        elif cls == OPEN:
            self._open(kind, payload)
        elif cls == SEP:
            self._sep(kind, payload)
        elif cls == LEAF:
            self._leaf(kind, payload)
        else:
            self._close(kind, payload)

    # --- destinations

    def _dest(self) -> Dest:
        """Where the expression opening now delivers, as its enclosing construct decides."""
        top = self.F.top()
        kind = top.kind
        if kind in (K_FN, K_HANDLER):
            return top.as_dest()
        if kind == K_LET:
            return Dest(D_ANY, count=top.aux) if top.state == 0 else top.as_dest()
        if kind == K_IF:
            return Dest(D_COND, t=top.l1, f=top.l2) if top.state == 0 else top.as_dest()
        if kind == K_ARM:
            return top.as_dest()
        if kind == K_AND:
            return Dest(D_COND, t=top.l3, f=top.f) if top.state == 0 else Dest(D_COND, t=top.t, f=top.f)
        if kind == K_OR:
            return Dest(D_COND, t=top.t, f=top.l3) if top.state == 0 else Dest(D_COND, t=top.t, f=top.f)
        if kind == K_NOT:
            return Dest(D_COND, t=top.f, f=top.t)
        return Dest(D_ANY)

    def _resolve(self, dest: Dest) -> tuple[Dest, int]:
        """A construct with arms must deliver every arm to one place, so `any` becomes a register
        now, and the value goes on the stack when the construct closes."""
        if dest.kind == D_ANY:
            return Dest(D_REG, self.fresh(dest.count), dest.count), 1
        return dest, 0

    def _deliver(self, value: Value, dest: Dest) -> None:
        if dest.kind == D_ANY:
            self.V.push(value)
        elif dest.kind == D_RET:
            self._return(value)
        elif dest.kind == D_REG:
            self._into(value, dest.d)
        else:
            self._branch_on(value, dest.t, dest.f)

    def _return(self, value: Value) -> None:
        if value.kind == V_TUPLE:
            members = [self.V.pop() for _ in range(value.reg)]
            members.reverse()
            registers = [self._as_reg(m) for m in members]
            self.i("ret " + ", ".join(f"r{r}" for r in registers))
        else:
            self.i(f"ret r{self._as_reg(value)}")
        self.live = False

    def _as_reg(self, value: Value) -> int:
        if value.kind in (V_REG, V_CAP):
            return value.reg
        register = self.fresh()
        self._into(value, register)
        return register

    def _into(self, value: Value, d: int) -> None:
        """Deliver to a named register. This records what arrived, for the join to compare."""
        if value.kind == V_TUPLE:
            members = [self.V.pop() for _ in range(value.reg)]
            members.reverse()
            arrived = list(members)
            # A capability moves last: a member read through it needs it where it still is.
            for position in sorted(range(len(members)), key=lambda i: members[i].kind == V_CAP):
                self._into(members[position], d + position)
                arrived[position] = self.last_into[0]
            self.last_into = arrived
            return
        word = value.word
        linear = bool(value.flags & LINEAR)
        cell, tag = 0, UNKNOWN_TAG
        if value.kind == V_LIT:
            ty = "bool" if value.flags & BOOL else "lit"
            self.i(f"const r{d}, #{value.n} : {ty}")
            word = self.new_word()
        elif value.kind == V_NULL:
            self.i(f"const r{d}, #0 : t{value.reg}")
            word, linear = 1, True
        elif value.kind == V_LOC:
            verb = "take" if linear else "read"
            self.i(f"{verb} r{d}, r{value.reg}[{value.off}:{value.size}]")
        elif value.kind == V_CAP:
            # Nothing names the cell's locations through its new register, so leaving here is
            # like going to a callee: what it still holds moves out, and a scalar is left behind.
            self._close_cell(value.cell, frozenset(), len(self.V))
            self.i(f"move r{d}, r{value.reg}")
            linear, cell, tag = True, value.cell, value.off
        else:
            assert value.kind == V_REG, value
            if value.reg != d:
                verb = "move" if value.flags & OWNED else "copy"
                self.i(f"{verb} r{d}, r{value.reg}")
        self.last_into = [Delivered(word, int(linear), cell, tag)]

    def _branch_on(self, value: Value, t: int, f: int) -> None:
        if value.kind == V_TUPLE:  # delivered here before Check reaches the end of the condition
            raise Reject("a condition must be Bool, and this is a tuple")
        if value.kind == V_LIT:
            register = self.fresh()
            self._into(value, register)
            text = f"r{register}!"
        else:
            text = self._operand(value, last_use=True)
        self.i(f"br ne, {text}, #0, L{t}, L{f}")
        self.live = False

    def _operand(self, value: Value, last_use: bool = False) -> str:
        if value.kind == V_LIT:
            return f"#{value.n}"
        if value.kind == V_NULL:
            return "#0"
        if value.kind == V_LOC:
            return f"r{value.reg}[{value.off}:{value.size}]"
        assert value.kind in (V_REG, V_CAP), value
        if last_use and value.flags & OWNED and not value.flags & LINEAR:
            return f"r{value.reg}!"
        return f"r{value.reg}"

    def _release(self, value: Value) -> None:
        """A temporary scalar has been read for the last time."""
        if value.kind == V_REG and value.flags & OWNED and not value.flags & LINEAR and self.live:
            self.i(f"drop r{value.reg}")

    # --- open

    def _open(self, kind: int, payload: bytes) -> None:
        if kind in (K_FN, K_HANDLER):
            self.fn_name = payload.rstrip(b"\0")
            self.fn_handler = kind == K_HANDLER
            self.ordinal += 1
            self.next_reg = self.next_label = 0
            self.next_word = 2
            self.next_cell = 1
            self.at = None
            self.param_types = []
            self.F.push(Frame(kind, dest=D_RET))
            return
        if kind == K_ARM:
            self._open_arm(payload)
            return
        dest = self._dest()
        if kind == K_LET:
            self.F.push(Frame(K_LET, dest=dest.kind, d=dest.d, count=dest.count, t=dest.t, f=dest.f,
                              bmark=len(self.B)))
        elif kind == K_IF:
            self._settle_pending()
            dest, push = self._resolve(dest)
            self.F.push(Frame(K_IF, dest=dest.kind, d=dest.d, count=dest.count, t=dest.t, f=dest.f,
                              push=push, l1=self.new_label(), l2=self.new_label(), l3=self.new_label(),
                              bmark=len(self.B), gmark=len(self.G), umark=len(self.U),
                              jmark=self._open_join(dest)))
        elif kind == K_MATCH:
            dest, push = self._resolve(dest)
            self.F.push(Frame(K_MATCH, dest=dest.kind, d=dest.d, count=dest.count, t=dest.t, f=dest.f,
                              push=push, l3=self.new_label(), vmark=len(self.V), bmark=len(self.B),
                              gmark=len(self.G), umark=len(self.U), jmark=self._open_join(dest)))
        elif kind in (K_AND, K_OR, K_NOT):
            self._open_logical(kind, dest)
        elif kind == K_WITH:
            self.F.push(Frame(K_WITH, dest=dest.kind, d=dest.d, count=dest.count, t=dest.t, f=dest.f,
                              bmark=len(self.B), vmark=len(self.V), gmark=len(self.G)))
        elif kind == K_CALL:
            ordinal, is_self, argc, results, at = P_CALL_RT.unpack_from(payload)
            self.F.push(Frame(K_CALL, dest=dest.kind, d=dest.d, count=dest.count, t=dest.t, f=dest.f,
                              state=is_self, aux=ordinal, l1=argc, l2=results, l3=at,
                              vmark=len(self.V)))
        elif kind == K_CTOR:
            self.F.push(Frame(K_CTOR, dest=dest.kind, d=dest.d, count=dest.count, t=dest.t, f=dest.f,
                              aux=payload[0], vmark=len(self.V), l1=len(self.L)))
        elif kind == K_PACK:
            widths = payload.rstrip(b"\0")
            self.F.push(Frame(K_PACK, dest=dest.kind, d=dest.d, count=dest.count, t=dest.t, f=dest.f,
                              aux=len(widths), l1=int.from_bytes(widths[:4], "little"),
                              l2=int.from_bytes(widths[4:8], "little"), vmark=len(self.V)))
        else:
            aux = payload[0] if kind in (K_BINOP, K_CMP) else 0
            self.F.push(Frame(kind, dest=dest.kind, d=dest.d, count=dest.count, t=dest.t, f=dest.f,
                              aux=aux, vmark=len(self.V)))

    def _open_join(self, dest: Dest) -> int:
        mark = len(self.J)
        if dest.kind == D_REG:
            for _ in range(dest.count):
                self.J.push(Delivered())
        return mark

    def _open_logical(self, kind: int, dest: Dest) -> None:
        """Jumping code. Given labels to jump to, it writes no value at all. Otherwise it needs
        its own, and materializes 1 or 0 at them when it closes."""
        if dest.kind == D_COND:
            self.F.push(Frame(kind, dest=D_COND, t=dest.t, f=dest.f, l3=self.new_label()))
            return
        if dest.kind == D_RET:
            resolved, push = Dest(D_REG, self.fresh()), RETURN_IT
        else:
            resolved, push = self._resolve(dest)
        self.F.push(Frame(kind, dest=resolved.kind, d=resolved.d, count=resolved.count,
                          t=self.new_label(), f=self.new_label(), l3=self.new_label(),
                          l1=self.new_label(), push=push))

    def _settle_pending(self) -> None:
        """A branching construct is about to start, and a location waiting on the value stack may
        be overwritten in one arm and not another. Reading it now means every arm finds it in the
        same register."""
        for index in range(len(self.V)):
            value = self.V.at(index)
            if value.kind == V_LOC:
                register = self.fresh()
                self._into(value, register)
                self.V.set(index, Value(V_REG, register, word=value.word,
                                        flags=OWNED | (value.flags & LINEAR)))

    # --- leaves

    def _leaf(self, kind: int, payload: bytes) -> None:
        if kind == K_PARAM:
            src, _size, linear, ty = P_PARAM_RT.unpack_from(payload)
            self.B.push(Binder(Value(V_REG, src, word=self.new_word(),
                                     flags=OWNED | (LINEAR if linear else 0))))
            self.param_types.append(ty)
        elif kind == K_TYPE:
            self.ret_ty = P_TYPE_RT.unpack_from(payload)[0]
        elif kind == K_RESWORD:
            self.at = struct.unpack_from("<H", payload)[0]
        elif kind == K_BINDER:
            self._leaf_binder(payload)
        elif kind == K_DIAMOND:
            self._ctor_cell(payload)
        elif kind == K_FIELD:
            self._with_field(payload)
        elif kind == K_LIVE:
            self.live_now = int.from_bytes(payload, "little")
        elif kind == K_EOF:
            pass
        else:
            self._deliver(self._leaf_value(kind, payload), self._dest())

    def _leaf_value(self, kind: int, payload: bytes) -> Value:
        if kind == K_INT:
            return Value(V_LIT, n=struct.unpack_from("<Q", payload)[0])
        if kind == K_BOOL:
            return Value(V_LIT, flags=BOOL, n=payload[0])
        if kind == K_NULLCTOR:
            return Value(V_NULL, P_TYPE_RT.unpack_from(payload)[0], flags=LINEAR)
        assert kind in (K_VAR, K_VARLAST), kind
        src, linear, _size = P_VAR.unpack_from(payload)
        value = self._use(src, bool(linear))
        if kind == K_VARLAST:
            self.live_now &= ~(1 << src)
        return value

    def _set_binder(self, index: int, binder: Binder) -> None:
        """A change a later arm must not see goes in the undo log. Only a name older than the
        innermost open branch can be seen by one, and such a name is consumed at most once and
        moved at most once along any path, so the log is bounded by the names on this path, not
        by the length of the function."""
        if index < self._branch_floor():
            self.U.push(Undo(index, self.B.at(index)))
        self.B.set(index, binder)

    def _branch_floor(self) -> int:
        for position in range(len(self.F) - 1, -1, -1):
            frame = self.F.at(position)
            if frame.kind in (K_IF, K_MATCH) and frame.state != 0:
                return frame.bmark
        return 0

    def _use(self, index: int, linear: bool) -> Value:
        binder = self.B.at(index)
        value = binder.value
        if linear:
            self._set_binder(index, replace(binder, value=replace(value, flags=value.flags | CONSUMED)))
        if value.kind == V_LOC:
            assert not self._gone(value.cell), (
                f"fn {self.fn_name.decode()} reads name {index} after its cell closed, and the "
                f"right-to-left pass said nothing would")
            return value
        if value.kind == V_REG:
            return replace(value, flags=(value.flags & (LINEAR | OWNED)) if linear
                           else value.flags & LINEAR)
        return value

    def _gone(self, cell: int) -> bool:
        return any(self.G.at(i).number == cell for i in range(len(self.G)))

    # --- let

    def _leaf_binder(self, payload: bytes) -> None:
        top = self.F.top()
        if top.kind == K_LET:
            top.aux += 1
            self.F.set_top(top)
            return
        kind, off, size, linear = P_BINDER.unpack_from(payload)
        position = top.aux
        top.aux += 1
        self.F.set_top(top)
        if top.state == 1:  # a known constructor: the names are its field values
            if kind == 0:
                self._push_binder(Binder(Value(V_CAP, top.reg, off=top.l1, word=top.word,
                                               cell=top.cell, flags=LINEAR)))
                return
            match = self.F.at(len(self.F) - 2)
            value = self.V.at(match.vmark + position - 1)
            self._push_binder(Binder(value, self.fresh() if linear else NO_REFUGE))
            return
        if kind == 0:
            self._push_binder(Binder(Value(V_CAP, top.reg, off=top.l1, word=top.word, cell=top.cell,
                                           flags=LINEAR)))
            return
        self._push_binder(Binder(Value(V_LOC, top.reg, off, size, self.new_word(), top.cell,
                                       LINEAR if linear else 0), self.fresh() if linear else NO_REFUGE))

    def _sep_let(self, top: Frame, payload: bytes) -> None:
        _one, _linear, _size, count = P_LET_RT.unpack_from(payload)
        if count > 1:
            marker = self.V.pop()
            assert marker.kind == V_TUPLE, marker
            members = [self.V.pop() for _ in range(marker.reg)]
            members.reverse()
            for member in members:
                self._bind_value(member)
        else:
            self._bind_value(self.V.pop())
        top.state = 1
        self.F.set_top(top)

    def _bind_value(self, value: Value) -> None:
        self._push_binder(Binder(value, self.fresh() if value.kind == V_LOC and value.flags & LINEAR
                                 else NO_REFUGE))

    def _push_binder(self, binder: Binder) -> None:
        if len(self.B) >= MAX_BINDERS:
            raise Limit(f"section 8.2: fn {self.fn_name.decode()} has more than {MAX_BINDERS} names "
                        f"in scope at once")
        self.B.push(binder)

    def _pop_binders(self, mark: int, values_from: int = 0) -> None:
        """Names go out of scope. A scalar's register goes with its owner, unless the value the
        construct leaves behind is that very register: then the value owns it now."""
        while len(self.B) > mark:
            value = self.B.pop().value
            if value.kind != V_REG or value.flags & LINEAR or not value.flags & OWNED:
                continue
            # The last value naming it is the one that owns it, since a tuple moves its members
            # in order and the earlier ones are copies by then.
            kept = False
            for index in range(len(self.V) - 1, values_from - 1, -1):
                held = self.V.at(index)
                if held.kind == V_REG and held.reg == value.reg:
                    self.V.set(index, replace(held, flags=held.flags | OWNED))
                    kept = True
                    break
            if not kept and self.live:
                self.i(f"drop r{value.reg}")

    # --- if and match

    def _note_arm(self, frame: Frame) -> None:
        """Rule of the merge: a join's word survives only if every arm delivered the same one."""
        if frame.dest != D_REG:
            return
        for position in range(frame.count):
            came = self.last_into[position]
            at = frame.jmark + position
            if not frame.noted:
                self.J.set(at, came)
                continue
            held = self.J.at(at)
            if held.word != came.word:
                held = replace(held, word=self.new_word())
            if held.cell != came.cell:
                held = replace(held, cell=0, tag=UNKNOWN_TAG)
            elif held.tag != came.tag:
                held = replace(held, tag=UNKNOWN_TAG)
            self.J.set(at, held)
        frame.noted = 1

    def _rollback(self, frame: Frame) -> None:
        """The next arm starts from the state at the branch."""
        while len(self.U) > frame.umark:
            undo = self.U.pop()
            if undo.index < len(self.B):  # a name bound inside the arm is out of scope already
                self.B.set(undo.index, replace(undo.binder, refuge=self.B.at(undo.index).refuge))
        self.G.truncate(frame.gmark)

    def _wanted_later(self, index: int) -> bool:
        """A name too far along for the annotation is taken to be wanted, which costs a read and is right."""
        return index >= MASK_BITS or bool(self.live_now >> index & 1)

    def _finish_moves_of_arm(self, frame: Frame) -> None:
        """A scalar this arm moved out of a cell, for a read that has now happened. Another arm may not
        have moved it at all, so it cannot be left in a register that the name still claims after the
        join. If something after the join reads it, every arm had to move it, and it stays."""
        for position in range(len(self.U) - 1, frame.umark - 1, -1):
            undo = self.U.at(position)
            if undo.index >= len(self.B):
                continue
            before, now = undo.binder.value, self.B.at(undo.index).value
            if (before.kind == V_LOC and not before.flags & LINEAR and now.kind == V_REG
                    and now.flags & OWNED and not self._wanted_later(undo.index)):
                if self.live:
                    self.i(f"drop r{now.reg}")
                self.B.set(undo.index, replace(undo.binder, refuge=now.reg))

    def _close_join(self, frame: Frame) -> None:
        if frame.dest != D_REG:
            return
        members = [self.J.at(frame.jmark + position) for position in range(frame.count)]
        self.J.truncate(frame.jmark)
        self.last_into = members
        if not frame.push:
            return
        for position, member in enumerate(members):
            if member.cell:
                self.V.push(Value(V_CAP, frame.d + position, off=member.tag, word=member.word,
                                  cell=member.cell, flags=LINEAR))
            else:
                self.V.push(Value(V_REG, frame.d + position, word=member.word,
                                  flags=OWNED | (LINEAR if member.linear else 0)))
        if frame.count > 1:
            self.V.push(Value(V_TUPLE, frame.count))

    def _sep_match(self, top: Frame) -> None:
        value = self.V.pop()
        if value.kind == V_VIRT:
            top.state = 1
            top.reg = value.reg
            top.word = value.word
            top.cell = value.cell
            top.aux = value.off    # how many field values wait on the stack
            top.l1 = self.C.pop().ctor
            top.l2 = value.size    # the tag the cell still holds: nothing was built
            self.F.set_top(top)
            return
        self._settle_pending()
        if value.kind == V_REG and value.flags & OWNED:
            register = value.reg
        else:
            register = self.fresh()
            self._into(value, register)
        top.reg = register
        top.word = value.word if value.kind != V_NULL else 1
        top.state = 2
        top.gmark, top.umark = len(self.G), len(self.U)  # consuming the scrutinee stays done
        self.F.set_top(top)

    def _open_arm(self, payload: bytes) -> None:
        tag, tagged, is_null, arm_index, nctors, ctor = P_ARM_RT.unpack_from(payload)
        match = self.F.top()
        if match.state == 1:
            if ctor != match.l1:
                self.dead = 1
                return
            self.F.push(Frame(K_ARM, state=1, dest=match.dest, d=match.d, count=match.count, t=match.t,
                              f=match.f, reg=match.reg, cell=match.cell, word=match.word,
                              bmark=len(self.B), l1=match.l2))
            return
        self._rollback(match)
        cell = self.new_cell()
        known = tag if tagged else UNKNOWN_TAG
        if nctors == 1:
            self.i(f"open r{match.reg}, c{ctor}")
        elif not tagged:
            if arm_index == 0:
                match.l1, match.l2 = self.new_label(), self.new_label()
                self.i(f"br.null r{match.reg}, L{match.l1}, L{match.l2}")
                self.live = False
            self.label(match.l1 if is_null else match.l2)
        elif arm_index < nctors - 1:
            if arm_index > 0:
                self.label(match.l1)
            yes, no = self.new_label(), self.new_label()
            self.i(f"br.tag r{match.reg}, {tag}, L{yes}, L{no}")
            self.live = False
            match.l1 = no
            self.label(yes)
        else:
            self.label(match.l1)
            self.i(f"open r{match.reg}, c{ctor}")
        self.F.set_top(match)
        self.F.push(Frame(K_ARM, dest=match.dest, d=match.d, count=match.count, t=match.t, f=match.f,
                          reg=match.reg, word=match.word, cell=cell, bmark=len(self.B), l1=known))

    def _forget_cells_from(self, first: int) -> None:
        """The cells opened since this number are out of scope, and nothing can name them. What
        is left is one cell per enclosing arm, so this list is bounded by depth too."""
        kept = [self.G.at(i) for i in range(len(self.G)) if self.G.at(i).number < first]
        self.G.truncate(0)
        for cell in kept:
            self.G.push(cell)

    def _close_arm(self) -> None:
        top = self.F.pop()
        self._pop_binders(top.bmark)
        if top.state != 1:
            self._forget_cells_from(top.cell)
        match = self.F.top()
        self._finish_moves_of_arm(match)
        self._note_arm(match)
        self.F.set_top(match)
        self.jump(match.l3)

    def _close_match(self) -> None:
        top = self.F.pop()
        if top.dest == D_REG:
            self.label(top.l3)
        self._close_join(top)

    def _sep_if(self, top: Frame, payload: bytes) -> None:
        if payload[0] == 1:
            top.gmark, top.umark = len(self.G), len(self.U)  # what the condition did, stays done
            self.label(top.l1)
            top.state = 1
        else:
            self._finish_moves_of_arm(top)
            self._note_arm(top)
            self.jump(top.l3)
            self._rollback(top)
            self.label(top.l2)
            top.state = 2
        self.F.set_top(top)

    def _close_if(self) -> None:
        top = self.F.pop()
        self._finish_moves_of_arm(top)
        self._note_arm(top)
        if top.dest == D_REG:
            self.jump(top.l3)
            self.label(top.l3)
        self._close_join(top)

    # --- logical connectives

    def _sep_logical(self, top: Frame) -> None:
        self.label(top.l3)
        top.state = 1
        self.F.set_top(top)

    def _close_logical(self) -> None:
        top = self.F.pop()
        if top.dest == D_COND:
            return
        self.label(top.t)
        self.i(f"const r{top.d}, #1 : bool")
        self.jump(top.l1)
        self.label(top.f)
        self.i(f"const r{top.d}, #0 : bool")
        self.jump(top.l1)
        self.label(top.l1)
        word = self.new_word()
        self.last_into = [Delivered(word)]
        if top.push == RETURN_IT:
            self.i(f"ret r{top.d}")
            self.live = False
        elif top.push:
            self.V.push(Value(V_REG, top.d, word=word, flags=OWNED))

    # --- operators

    def _result(self, top: Frame) -> int:
        return top.d if top.dest == D_REG else self.fresh()

    def _finish(self, top: Frame, register: int) -> None:
        """A scalar has been computed into the register `_result` chose for it."""
        word = self.new_word()
        value = Value(V_REG, register, word=word, flags=OWNED)
        if top.dest == D_REG:
            self.last_into = [Delivered(word)]
        else:
            self._deliver(value, top.as_dest())

    def _close_binop(self, payload: bytes) -> None:
        top = self.F.pop()
        right, left = self.V.pop(), self.V.pop()
        width = payload[0]
        name = OP_NAMES[top.aux]
        parent = self.F.top()
        phantom = len(self.F) > 1 and self.F.at(len(self.F) - 2).kind == K_MATCH and (
            self.F.at(len(self.F) - 2).state == 0)
        if (parent.kind == K_CTOR and not phantom and top.dest == D_ANY and left.kind == V_LOC
                and name in _RMW and left.cell == parent.cell and right.kind in (V_LIT, V_REG)):
            field = self.L.top()
            if field.off == left.off and field.size == left.size == width:
                self.V.push(right)
                self.V.push(Value(V_FUNC, left.reg, left.off, left.size, top.aux, left.cell))
                return
        register = self._result(top)
        self.i(f"op r{register}, {_RT_OPS.get(name, name)}/{8 * width}, "
               f"{self._operand(left)}, {self._operand(right)}")
        self._release(left)
        self._release(right)
        self._finish(top, register)

    def _close_cmp(self) -> None:
        top = self.F.pop()
        right, left = self.V.pop(), self.V.pop()
        cc = CC_NAMES[top.aux]
        if top.dest == D_COND:
            self.i(f"br {cc}, {self._operand(left, True)}, {self._operand(right, True)}, "
                   f"L{top.t}, L{top.f}")
            self.live = False
            return
        register = self._result(top)
        self.i(f"cmp r{register}, {cc}, {self._operand(left)}, {self._operand(right)}")
        self._release(left)
        self._release(right)
        self._finish(top, register)

    def _close_pack(self) -> None:
        top = self.F.pop()
        widths = (top.l1.to_bytes(4, "little") + top.l2.to_bytes(4, "little"))[:top.aux]
        values = [self.V.pop() for _ in range(top.aux)]
        values.reverse()
        register = self._result(top)
        operands = ", ".join(self._operand(v) for v in values)
        self.i(f"pack r{register}, <{' '.join(str(w) for w in widths)}>, {operands}")
        for value in values:
            self._release(value)
        self._finish(top, register)

    def _close_idx(self, payload: bytes) -> None:
        top = self.F.pop()
        value = self.V.pop()
        register = self._result(top)
        self.i(f"idx r{register}, {P_IDX_RT.unpack_from(payload)[0]}, {self._operand(value)}")
        self._release(value)
        self._finish(top, register)

    def _close_low(self, payload: bytes) -> None:
        top = self.F.pop()
        value = self.V.pop()
        register = self._result(top)
        self.i(f"trunc r{register}, {8 * payload[0]}, {self._operand(value)}")
        self._release(value)
        self._finish(top, register)

    def _close_get(self, payload: bytes) -> None:
        top = self.F.pop()
        index, array = self.V.pop(), self.V.pop()
        size = P_GET_RT.unpack_from(payload)[0]
        base = self._as_reg(array)
        position = self._as_reg(index)
        element = self.fresh()
        self.i(f"aget r{element}, r{base}, r{position}, {size}")
        if index.kind != V_REG or index.flags & OWNED:  # a temporary, read for the last time
            self.i(f"drop r{position}")
        self.V.push(Value(V_REG, base, word=array.word, flags=OWNED | LINEAR))
        self.V.push(Value(V_REG, element, word=self.new_word(), flags=OWNED))
        self._deliver(Value(V_TUPLE, 2), top.as_dest())

    def _close_tuple(self) -> None:
        top = self.F.pop()
        count = len(self.V) - top.vmark
        self._deliver(Value(V_TUPLE, count), top.as_dest())

    # --- calls

    def _close_call(self, payload: bytes) -> None:
        top = self.F.pop()
        is_self, ordinal, argc, results, at = bool(top.state), top.aux, top.l1, top.l2, top.l3
        linear_mask = payload[0]
        parent = self.F.top()
        arguments_from = len(self.V) - argc
        if is_self and parent.kind == K_CTOR and parent.dest == D_RET and self.at is not None:
            field = self.L.top()
            word = self.V.at(arguments_from + self.at).word
            if self._is_hole_of(word, parent.cell, field.off, field.size):
                self.V.push(Value(V_CALL, ordinal, argc, cell=word))
                return
        if is_self and top.dest != D_RET:
            raise Limit(f"section 8.2: fn {self.fn_name.decode()} calls itself outside tail "
                        f"position, and non-tail recursion is refused until DESCEND and ASCEND exist")
        for index in range(arguments_from, len(self.V)):
            value = self.V.at(index)
            if value.kind == V_CAP:  # a location of this cell among the arguments is read first
                self._close_cell(value.cell, frozenset(), len(self.V))
        values = [self.V.pop() for _ in range(argc)]
        values.reverse()
        operands = ", ".join(self._operand(v) for v in values)
        separator = ", " if operands else ""
        if top.dest == D_RET:
            self.i(f"tail f{ordinal}{separator}{operands}")
            self.live = False
            return
        word = values[at - 1].word if at else 0
        if results > 1:
            first = self._result(top) if top.dest == D_REG else self.fresh(results)
            names = ", ".join(f"r{first + i}" for i in range(results))
            self.i(f"call ({names}), f{ordinal}{separator}{operands}")
            self._release_all(values)
            members = [Value(V_REG, first + i, word=self.new_word(),
                             flags=OWNED | (LINEAR if linear_mask >> i & 1 else 0))
                       for i in range(results)]
            if top.dest == D_REG:
                self.last_into = [Delivered(m.word, int(bool(m.flags & LINEAR))) for m in members]
                return
            for member in members:
                self.V.push(member)
            self._deliver(Value(V_TUPLE, results), top.as_dest())
            return
        register = self._result(top)
        self.i(f"call r{register}, f{ordinal}{separator}{operands}")
        self._release_all(values)
        value = Value(V_REG, register, word=word or self.new_word(),
                      flags=OWNED | (LINEAR if linear_mask & 1 else 0))
        if top.dest == D_REG:
            self.last_into = [Delivered(value.word, linear_mask & 1)]
        else:
            self._deliver(value, top.as_dest())

    def _release_all(self, values: list[Value]) -> None:
        for value in values:
            self._release(value)

    # --- constructors

    def _ctor_cell(self, payload: bytes) -> None:
        src, tag, tagged, count, off, size, linear, ctor = P_DIAMOND_RT.unpack_from(payload)
        top = self.F.top()
        cap = self._use(src, True)
        assert cap.kind in (V_CAP, V_REG), cap
        top.reg = cap.reg
        top.word = cap.word
        top.cell = cap.cell
        top.l2 = cap.off if cap.kind == V_CAP else UNKNOWN_TAG
        self.F.set_top(top)
        self.C.push(Ctor(ctor, tag, tagged))
        if count:
            self.L.push(Layout(off, size, linear))

    def _sep_ctor(self, payload: bytes) -> None:
        _index, off, size, linear = P_FIELD.unpack_from(payload)
        self.L.push(Layout(off, size, linear))

    def _close_ctor(self) -> None:
        top = self.F.pop()
        info = self.C.pop()
        count = top.aux
        parent = self.F.top()
        if parent.kind == K_MATCH and parent.state == 0:
            self.L.truncate(top.l1)
            self.C.push(info)
            self.V.push(Value(V_VIRT, top.reg, count, top.l2, top.word, top.cell))
            return
        fields = [self.L.at(top.l1 + i) for i in range(count)]
        self.L.truncate(top.l1)
        # Values come off in reverse, and a pending call or computed source owns those below it.
        entries: list[tuple] = []
        call: Optional[tuple] = None
        for position in range(count - 1, -1, -1):
            value = self.V.pop()
            if value.kind == V_CALL:
                arguments = [self.V.pop() for _ in range(value.off)]
                arguments.reverse()
                call = (position, value, arguments)
                entries.append((value, None))
            elif value.kind == V_FUNC:
                entries.append((value, self.V.pop()))
            else:
                entries.append((value, None))
        entries.reverse()
        sources = self._field_sources(top, fields, entries)
        named = frozenset((f.off, f.size) for f, s in zip(fields, sources) if s == "=")
        named |= frozenset(self._inner_places(sources))
        self._close_cell(top.cell, named, len(self.V), arguments=call[2] if call else [])
        tag = "none"
        if info.tagged:
            tag = "keep" if top.l2 == info.tag else "write"
        joined = ", ".join(sources)
        separator = ", " if joined else ""
        if call is not None:
            operands = ", ".join(self._operand(v) for v in call[2])
            self.i(f"tailmod r{top.reg}, c{info.ctor}, {tag}{separator}{joined}, f{call[1].reg}"
                   f"{', ' if operands else ''}{operands}")
            self.live = False
            return
        self.i(f"build r{top.reg}, c{info.ctor}, {tag}{separator}{joined}")
        for value, _operand in entries:
            self._release(value)
        for value, extra in entries:
            if extra is not None:
                self._release(extra)
        value = Value(V_REG, top.reg, word=top.word, flags=OWNED | LINEAR)
        if top.dest == D_REG:
            self._into(value, top.d)
        else:
            self._deliver(value, top.as_dest())

    @staticmethod
    def _inner_places(sources: list[str]) -> Iterator[tuple[int, int]]:
        for source in sources:
            if source.startswith("["):
                off, size = source[1:-1].split(":")
                yield int(off), int(size)

    def _mark_gone(self, cell: int) -> None:
        self.G.push(Cell(cell))

    def _is_hole_of(self, word: int, cell: int, off: int, size: int) -> bool:
        """Does this location of this cell belong to a name whose value this word is? If so the
        value can go back with no store: that is RESTORE."""
        if not cell:
            return False
        for index in range(len(self.B) - 1, -1, -1):
            value = self.B.at(index).value
            if (value.cell == cell and value.word == word and value.off == off
                    and value.size == size and value.kind in (V_LOC, V_REG)):
                return True
        return False

    def _field_sources(self, top: Frame, fields: list[Layout], entries: list[tuple]) -> list[str]:
        sources = []
        for field, (value, extra) in zip(fields, entries):
            if value.kind == V_LIT:
                sources.append(f"#{value.n}")
            elif value.kind == V_NULL:
                sources.append("#0")
            elif value.kind == V_CALL:
                sources.append("*")
            elif value.kind == V_FUNC:
                sources.append(f"{OP_NAMES[value.word]}([{value.off}:{value.size}], "
                               f"{self._operand(extra)})")
            elif value.kind == V_LOC and value.cell == top.cell:
                same = value.off == field.off and value.size == field.size
                sources.append("=" if same else f"[{value.off}:{value.size}]")
            elif value.kind == V_LOC:
                register = self.fresh()
                self._into(value, register)
                sources.append(f"r{register}")
            else:
                assert value.kind == V_REG, value
                if self._is_hole_of(value.word, top.cell, field.off, field.size):
                    sources.append(f"r{value.reg}!" if value.flags & LINEAR else "=")
                else:
                    sources.append(f"r{value.reg}")
        return sources

    def _close_cell(self, cell: int, named: frozenset, values_to: int, arguments: Iterable = ()) -> None:
        """A cell is about to be rebuilt or handed over, so nothing can be left in it. A linear
        field not named by the new constructor moves to its name's refuge, and so does a scalar that
        something still reads. A location waiting on the value stack is read while it can be."""
        for index in range(values_to):
            value = self.V.at(index)
            if value.kind == V_LOC and value.cell == cell:
                register = self.fresh()
                self._into(value, register)
                self.V.set(index, Value(V_REG, register, word=value.word,
                                        flags=OWNED | (value.flags & LINEAR)))
        for index in range(len(self.B)):
            binder = self.B.at(index)
            value = binder.value
            if value.kind != V_LOC or value.cell != cell:
                continue
            if value.flags & LINEAR:
                if not value.flags & CONSUMED and (value.off, value.size) not in named:
                    self.i(f"take r{binder.refuge}, r{value.reg}[{value.off}:{value.size}]")
                    self._set_binder(index, Binder(Value(V_REG, binder.refuge, off=value.off,
                                                         size=value.size, word=value.word, cell=cell,
                                                         flags=OWNED | LINEAR), binder.refuge))
            elif self._wanted_later(index):
                refuge = self.fresh() if binder.refuge == NO_REFUGE else binder.refuge
                self.i(f"read r{refuge}, r{value.reg}[{value.off}:{value.size}]")
                self._set_binder(index, Binder(Value(V_REG, refuge, off=value.off, size=value.size,
                                                     word=value.word, cell=cell, flags=OWNED), refuge))
        if cell:
            self._mark_gone(cell)

    # --- with

    def _sep_with(self, top: Frame, payload: bytes) -> None:
        if payload[0] == 1:
            value = self.V.pop()
            register = self._as_owned(value)
            top.reg = register
            top.word = value.word
            top.cell = self.new_cell()
            top.state = 1
            self.F.set_top(top)
        else:
            top.state = 2
            self.F.set_top(top)

    def _as_owned(self, value: Value) -> int:
        if value.kind == V_REG and value.flags & OWNED:
            return value.reg
        register = self.fresh()
        self._into(value, register)
        return register

    def _with_field(self, payload: bytes) -> None:
        off, size, linear, ctor, position, count = P_WITH_RT.unpack_from(payload)
        top = self.F.top()
        self.i(f"open r{top.reg}, c{ctor}")
        top.aux = position
        top.l1 = ctor
        top.l2 = count
        top.l3 = off | size << 8 | linear << 16
        self.F.set_top(top)
        self.B.push(Binder(Value(V_LOC, top.reg, off, size, self.new_word(), top.cell,
                                 LINEAR if linear else 0), self.fresh() if linear else NO_REFUGE))

    def _close_with(self) -> None:
        top = self.F.pop()
        value = self.V.pop()
        off, size = top.l3 & 255, top.l3 >> 8 & 255
        field = Layout(off, size, top.l3 >> 16)
        source = self._field_sources(top, [field], [(value, None)])[0]
        sources = ["="] * top.l2
        sources[top.aux] = source
        self._pop_binders(top.bmark)
        self._close_cell(top.cell, frozenset(), len(self.V))
        self.i(f"build r{top.reg}, c{top.l1}, none, {', '.join(sources)}")
        self._release(value)
        result = Value(V_REG, top.reg, word=top.word, flags=OWNED | LINEAR)
        if top.dest == D_REG:
            self._into(result, top.d)
        else:
            self._deliver(result, top.as_dest())

    # --- sep and close, dispatched

    def _sep(self, kind: int, payload: bytes) -> None:
        top = self.F.top()
        if kind in (K_FN, K_HANDLER):
            self._header()
        elif kind == K_LET:
            self._sep_let(top, payload)
        elif kind == K_IF:
            self._sep_if(top, payload)
        elif kind == K_MATCH:
            self._sep_match(top)
        elif kind == K_ARM:
            if top.state == 1:  # every field of a known constructor now has its name
                self.V.truncate(self.F.at(len(self.F) - 2).vmark)
        elif kind == K_WITH:
            self._sep_with(top, payload)
        elif kind in (K_AND, K_OR):
            self._sep_logical(top)
        elif kind == K_CTOR:
            self._sep_ctor(payload)
        elif kind in (K_BINOP, K_CMP, K_CALL, K_TUPLE, K_GET, K_PACK):
            pass
        else:
            raise AssertionError(f"sep of kind {kind}")

    def _header(self) -> None:
        params = [f"r{index} : {_type_text(ty)}" for index, ty in enumerate(self.param_types)]
        word = "handler" if self.fn_handler else "fn"
        at = f" @{self.at}" if self.at is not None else ""
        self.emit(f"{word} f{self.ordinal} {self.fn_name.decode()}({', '.join(params)}) : "
                  f"{_type_text(self.ret_ty)}{at}")
        self.live = True
        self.next_reg = len(self.param_types)
        top = self.F.top()
        top.count = 1
        self.F.set_top(top)

    def _close(self, kind: int, payload: bytes) -> None:
        if kind == K_TYPE:  # a tuple type: Check has interned its members into one id
            self.ret_ty = P_TYPE_RT.unpack_from(payload)[0]
        elif kind in (K_FN, K_HANDLER):
            self.F.pop()
            self.B.truncate(0)
            self.U.truncate(0)
            self.G.truncate(0)
            decreasing = [str(p) for p in range(8) if payload[0] >> p & 1]
            self.emit("end decreasing " + ",".join(decreasing) if decreasing else "end")
        elif kind == K_LET:
            top = self.F.pop()
            self._pop_binders(top.bmark)
        elif kind == K_IF:
            self._close_if()
        elif kind == K_MATCH:
            self._close_match()
        elif kind == K_ARM:
            self._close_arm()
        elif kind in (K_AND, K_OR, K_NOT):
            self._close_logical()
        elif kind == K_WITH:
            self._close_with()
        elif kind == K_BINOP:
            self._close_binop(payload)
        elif kind == K_CMP:
            self._close_cmp()
        elif kind == K_CALL:
            self._close_call(payload)
        elif kind == K_CTOR:
            self._close_ctor()
        elif kind == K_PACK:
            self._close_pack()
        elif kind == K_IDX:
            self._close_idx(payload)
        elif kind == K_LOW:
            self._close_low(payload)
        elif kind == K_GET:
            self._close_get(payload)
        elif kind == K_TUPLE:
            self._close_tuple()
        else:
            raise AssertionError(f"close of kind {kind}")


def _type_text(ty: int) -> str:
    names = {0: "i8", 1: "i16", 2: "i32", 3: "i64", 4: "bool", 5: "cap", 6: "lit"}
    return names.get(ty, f"t{ty}")


# ------------------------------------------------------------------ occupancy and the driver

_CELL_FIELDS = 16
_ARGUMENTS = 6
_FRAME_CELLS = (_FRAME.size + PAYLOAD_BYTES - 1) // PAYLOAD_BYTES
# Cells per open construct: its frame; the values waiting on it, a field or an argument each, with
# a computed source's operand under it; the names an arm binds, and the undo records one path can
# hold for them, consumed once and moved once; the layouts, the constructor and the one cell.
S_LOWER = (_FRAME_CELLS + (2 * _CELL_FIELDS + _ARGUMENTS) + 2 * (_CELL_FIELDS + 1)
           + 2 * 2 * (_CELL_FIELDS + 1) + _CELL_FIELDS + 2 + _ARGUMENTS)
S_PARSE = 1
S_CHECK = 3 + 1 + 2 * (_CELL_FIELDS + 1) + 6 + 2 * _ARGUMENTS
G_CELLS_PER_ENTRY = 2
DEFAULT_G_MAX = 2048


def occupancy_bound(depth: int, g_cells: int, l_max: int = PAYLOAD_BYTES, w: int = 1) -> int:
    """Theorem 7.2, for this ring: W + 5 + (E - 1) + L_max + (per-stage cells) * D + s * G. Five channels and
    held cells now, with the right-to-left pass among the stages."""
    return w + 5 + (E_MAX - 1) + l_max + (S_PARSE + CELLS_PER_DEPTH + S_CHECK + S_LOWER) * depth + g_cells


def required_cells(d_max: int, g_max: int = DEFAULT_G_MAX) -> int:
    # G, the type tables, and one cell per function for what it returns.
    return occupancy_bound(d_max, 2 * G_CELLS_PER_ENTRY * g_max + g_max) + 1


def file_bytes(handle: BinaryIO, chunk: int = 4096) -> Iterator[int]:
    while True:
        block = handle.read(chunk)
        if not block:
            return
        yield from block


def rt_lines(handle: BinaryIO, on_step: Optional[Callable[[Ring], None]] = None,
             *, phase1: bool = False) -> Iterator[str]:
    """The stream, one line at a time, as the ring writes it. Nothing is held in between: the
    ring runs only as far as it takes to produce the next line."""
    pending: list[str] = []
    text = bytearray()

    def sink(tag: int, payload: bytes) -> None:
        text.extend(payload.rstrip(b"\0"))
        if tag == LINE_END:
            pending.append(text.decode())
            text.clear()

    stages = [Lexer(), Parser(), RightToLeft(), CheckStage(lfplo=True, phase1=phase1), RTLower()]
    ring = Ring(required_cells(64), file_bytes(handle), stages, sink, on_step)
    for _ in _steps(ring):
        while pending:
            yield pending.pop(0)
    while pending:
        yield pending.pop(0)


def _steps(ring: Ring) -> Iterator[None]:
    """Ring.run, one sweep at a time, so a consumer can stop at the first line it refuses."""
    last = len(ring.stages) - 1
    while True:
        fired = False
        if ring.c5_queue:
            ring.egress()
            ring._observe()
            fired = True
        for index in range(last, -1, -1):
            stage = ring.stages[index]
            blocked = (bool(ring.c5_queue) if index == last else ring.channels[index + 1] is not None)
            if blocked:
                continue
            incoming = None if stage.has_work() else ring.channels[index]
            if incoming is None and not stage.has_work():
                continue
            if incoming is not None:
                ring.channels[index] = None
                stage.sigma.add(incoming)
            ring.current = stage
            stage.fire(incoming)
            ring.current = None
            ring._observe()
            fired = True
        if not ring.ingest_finished and ring.channels[0] is None:
            ring.ingest()
            ring._observe()
            fired = True
        if not fired:
            return
        yield None
