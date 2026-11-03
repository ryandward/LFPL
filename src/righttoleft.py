"""The right-to-left pass, LFPL-O section 8.1: the facts a forward fold cannot see.

Lower reads a program forward, once, and two things it needs are decided by what comes later:

    the last use of each binder    A scalar field lives in a location of its cell, and rebuilding the cell or
                                   handing it to a callee ends that. So the scalar has to move out first, if
                                   and only if something reads it later. Which is a fact about the future.

    the cell every exit returns    `@k` says a function returns parameter k's own cell. Lower has to know
                                   that at the function's first instruction, and it follows from the exits.

This stage sits between Parse and Check and works one function at a time, in source order, because what a
function returns can depend on what its callees return, and a callee always comes first. Every event goes through
src/spans.py on its way in, so a record that spans cells has become the chain of records it is before any of this
looks at it. It has three steps.

Forward, as Parse writes the function's events to a file on disk, it resolves each name to its position in
scope, just as Check will, and follows each value's word: which parameter's cell the value is, if any. That
much has to be done going forward, because a let's name comes before the expression it is bound to, so going
backward a use cannot yet be told from a use of an outer name. This is also where the exits are met.

Backward, it reads the file from the end, holding the set of binders that have a use still to come, plus two
sets per open branch. A use of a binder not in the set is its last on this path. At the start of an arm or a
scope it writes the whole set into that event's record: that is the annotation.

Forward again, it sends the events on to Check as if Parse had just written them, with these additions:

    leaf(RESWORD, name)   before the body, if `@` was not written and exactly the rule below finds a cell
    leaf(VARLAST)         in place of leaf(VAR), at a binder's last use on this path
    leaf(LIVE, mask)      after each separator that starts an arm or a scope: binders still to be used

A field a record update leaves where it is comes from src/spans.py as leaf(VARAT, position), which is a use like any
other, except that it says which binding it means instead of a name. Its last use is marked in its own payload.

A declared `@k` is left alone. Either way rtcheck holds every exit to it, so an inference that was wrong
could only ever refuse a program, and never make a wrong one.

The rule. W(e) is the word of a value, which is parameter p, null, or nothing known. A parameter has its own
word, and a let's name has its expression's, except that the names of a tuple are nothing known. C@d(...) has d's
word, a null constructor is null, and { e with f = v } has e's. In an arm of `match s`, the diamond has W(s),
and the other fields are nothing known. An if or a match has its arms' word if they all agree, and otherwise
nothing known. A call of g @j, other than a recursive one, has W of its argument j, unless that is null. Anything
else is nothing known. The exits are found by going down the last expression of a let, and each arm of an if or
match. An exit allows p if W is p, or if W is null and p is known to be null here, which is inside the arm of a
match on p with no binders. A recursive call allows each k whose argument k has word k. The function's cell
is the lowest position, among parameters of the result type, that every exit allows. src/symcheck.py works this
out again on its own, sharing no code with this, and a disagreement refuses the build.

State is O(D): a frame per open construct, the names in scope, which are bounded per construct, and one cell
per function for what it returns, which is the O(G) term of Theorem 7.2. Neither the file nor any of this holds
need(e), frame sizes or distances to the next use: those are the x86 backend's own business.
"""
from __future__ import annotations

import struct
import tempfile
from dataclasses import dataclass
from typing import BinaryIO, Optional

from .check import CELL_FIELDS_MAX, TUPLE_MAX
from .dyck import (CLOSE, DEFAULT_D_MAX, ARM_WORDS, K_ARM, K_BINDER, K_BOOL, K_CALL, K_CTOR, K_DIAMOND, K_ENUM, K_EOF, K_FIELD,
                   K_FN, K_HANDLER, K_IF, K_INT, K_LET, K_LIVE, K_MATCH, K_NULLCTOR, K_PARAM, K_RECORD, K_RESWORD,
                   K_TARRAY, K_TFIN, K_TFINNAME, K_TTUPLE, K_TUPLE, K_TYPE, K_VAR, K_VARAT, K_VARLAST, K_WITH, LEAF, OPEN,
                   SEP, VARAT, ev_class, ev_kind, ev_tag, FUNCTIONS)
from .ring import Reject, Stage
from .spans import CELLS_PER_DEPTH as SPAN_CELLS_PER_DEPTH, Spans
from .tokens import PAYLOAD_BYTES

# A value's word. A parameter's position is 0 or more.
NOTHING_KNOWN, NULL, NOT_YET = -1, -2, -3
MAX_PARAMETERS = 6     # as Check refuses more, and so a set of them fits in a byte
MASK_BITS = 8 * PAYLOAD_BYTES  # names beyond this are taken to be wanted: Lower refuses at 128 anyway
NO_POSITION = 0xFFFF   # a name Check is going to refuse, or an event that binds nothing
# Theorem 7.2. One open construct holds: its frame, or a branch's two sets going backward; the names an arm or a
# let binds, and a let's while it waits; and the types of a function's parameters, counted at every depth.
CELLS_PER_DEPTH = 3 + (CELL_FIELDS_MAX + 2) + TUPLE_MAX + MAX_PARAMETERS + SPAN_CELLS_PER_DEPTH
_NOT_A_CELL = frozenset((b"Int8", b"Int16", b"Int32", b"Int64", b"Bool", b"Diamond"))

# One event on disk: what Parse wrote, the line it was on, the name's position or the first position a let
# binds, and how many, then what the backward step decides.
_RECORD = struct.Struct("<B15sIHBB15s")
F_LIVE, F_LAST = 1, 2

SPILLING, BACKWARD, REPLAYING = range(3)
S_PARAMETERS, S_RESULT, S_BODY = range(3)  # where a function's frame has got to


@dataclass
class _Open:
    """One open construct, in one cell. What `word`, `join` and `args` hold depends on the construct."""
    state: int = 0
    exit: int = 0              # the value of this construct is what the function returns
    mark: int = 0              # how many names were in scope when it opened
    word: int = NOTHING_KNOWN  # a match's scrutinee, a constructor's diamond, a with's record
    join: int = NOT_YET        # what the arms so far agree on, or a call's callee's cell
    nulls: int = 0             # an arm: the parameters known to be null outside it
    count: int = 0             # names so far, or arguments so far
    args: tuple = (NOTHING_KNOWN,) * MAX_PARAMETERS

    _PACKED = struct.Struct("<BBHbbBB6b")

    def pack(self) -> bytes:
        return self._PACKED.pack(self.state, self.exit, self.mark, self.word, self.join, self.nulls, self.count,
                                 *self.args)

    @classmethod
    def unpack(cls, data: bytes) -> "_Open":
        state, exit_, mark, word, join, nulls, count, *args = cls._PACKED.unpack_from(data)
        return cls(state, exit_, mark, word, join, nulls, count, tuple(args))


class RightToLeft(Stage):
    name = "right-to-left"

    def __init__(self, d_max: int = DEFAULT_D_MAX) -> None:
        super().__init__()
        self.d_max = d_max
        self.spans = Spans(self)
        self.file: Optional[BinaryIO] = None
        self.records = 0
        self.phase = SPILLING
        self.in_function = False
        self.cursor = 0
        self.fn_name = b""
        self.pending_name = b""
        self.result_type = b""
        self.word = NOTHING_KNOWN  # the word of the expression that just finished
        self.child = False         # whether an argument has finished since the last separator
        self.nulls = 0             # the parameters known to be null on this path
        self.allowed = 0           # the parameters every exit so far allows
        self.inferring = False     # no `@` was written, and the result could be a parameter's cell
        self.declared_at = -1
        self.write_at = b""        # the name to send as `@`, once the backward step is done
        self.live = 0
        self.after: list[tuple[int, bytes]] = []  # at most two: the rest of one record's events

    def attach(self, ring) -> None:
        super().attach(ring)
        self.K = self.stack()  # going forward: a frame per open construct
        self.B = self.stack()  # going forward: the names in scope, each one's tag holding its word
        self.N = self.stack()  # going forward: the names of each let still working out its value
        self.P = self.stack()  # going forward: the type of each parameter
        self.R = self.stack()  # going backward: for each open branch, the set after it, and so far before it
        self.T = self.stack()  # for all time: the name of each function, its tag holding its cell + 1
        self.E = self.stack()  # for all time: the name of each enum, which is a scalar, whatever it is called
        self.spans.attach()

    def depth(self) -> int:
        return max(len(self.K), len(self.R))

    def table_cells(self) -> int:
        """What this stage keeps for all time: the O(G) term of Theorem 7.2."""
        return len(self.T.cells) + len(self.E.cells) + self.spans.table_cells()

    def has_work(self) -> bool:
        return self.phase != SPILLING or self.spans.emitting is not None

    def _reject(self, message: str) -> Reject:
        return Reject(f"near line {self.ring.stages[0].line}: {message}")

    # --- the ring

    def fire(self, cell: Optional[int]) -> None:
        if cell is None:
            if self.spans.emitting is not None:
                self._declare()
            elif self.phase == BACKWARD:
                self._step_back()
            else:
                self._replay()
            return
        store = self.ring.store
        tag, payload = store.tag(cell), store.payload(cell)
        cls, kind = ev_class(tag), ev_kind(tag)
        if kind == K_RECORD or self.spans.holding:
            self.spans.hold(cls, kind, payload)
            self.ring.drop(self, cell)
            return
        if not self.in_function and not (cls == OPEN and kind in FUNCTIONS):
            self.ring.replay_line = None
            if cls == OPEN and kind == K_ENUM:
                self.E.push(0, payload)
            if kind == K_EOF and self.file is not None:
                self.file.close()
                self.file = None
            self.ring.pass_(self, cell, tag, payload)
            return
        for new_tag, new_payload in self.spans.events(cls, kind, payload):
            new_payload = new_payload.ljust(PAYLOAD_BYTES, b"\0")
            stamp, count = self._forward(ev_class(new_tag), ev_kind(new_tag), new_payload)
            self._append(new_tag, new_payload, stamp, count)
        self.ring.drop(self, cell)
        if cls == CLOSE and kind in FUNCTIONS:
            self.in_function = False
            self.cursor = self.records
            self.live = 0
            self.phase = BACKWARD

    def _declare(self) -> None:
        """One event of a record on its way to Check, now that it is known what the record is. The last
        call has nothing to send, and that is a firing with nothing to show for it, which is all right."""
        self.ring.replay_line = None
        event = self.spans.next_declaration_event()
        if event is not None:
            self.ring.emit_fresh(self, event[0], event[1])

    # --- the file

    def _append(self, tag: int, payload: bytes, stamp: int, count: int) -> None:
        if self.file is None:
            self.file = tempfile.TemporaryFile()
        self.file.seek(self.records * _RECORD.size)
        self.file.write(_RECORD.pack(tag, payload, self.ring.stages[0].line, stamp, count, 0, bytes(PAYLOAD_BYTES)))
        self.records += 1

    def _read(self, position: int) -> tuple:
        assert self.file is not None
        self.file.seek(position * _RECORD.size)
        return _RECORD.unpack(self.file.read(_RECORD.size))

    def _annotate(self, position: int, record: tuple, flags: int, mask: int) -> None:
        tag, payload, line, stamp, count, _flags, _mask = record
        assert self.file is not None
        self.file.seek(position * _RECORD.size)
        self.file.write(_RECORD.pack(tag, payload, line, stamp, count, flags, mask.to_bytes(PAYLOAD_BYTES, "little")))

    # --- going forward: names and words

    def _top(self) -> tuple[int, _Open]:
        kind, data = self.K.peek()
        return kind, _Open.unpack(data)

    def _set_top(self, frame: _Open) -> None:
        kind, _data = self.K.peek()
        self.K.rewrite(len(self.K) - 1, kind, frame.pack())

    def _push(self, kind: int, frame: _Open) -> None:
        """Parse has kept the source's nesting to the limit already. What a record that spans adds is counted here."""
        if len(self.K) >= self.d_max:
            raise self._reject(f"nesting too deep: more than {self.d_max} open constructs, counting two for each "
                               f"cell of a record that spans after its first (the Depth rule)")
        self.K.push(kind, frame.pack())

    def _lookup(self, name: bytes) -> tuple[int, int]:
        """A name's position, and its word. Check will refuse a name that is not there."""
        for position in range(len(self.B) - 1, -1, -1):
            tag, data = self.B.at(position)
            if data == name:
                return position, tag - 2
        return NO_POSITION, NOTHING_KNOWN

    def _bind(self, name: bytes, word: int) -> int:
        self.B.push(word + 2, name)
        return len(self.B) - 1

    def _unbind_to(self, mark: int) -> None:
        while len(self.B) > mark:
            self.B.pop()
        self.spans.unbind_to(mark)

    def _in_exit_position(self) -> bool:
        """Is the expression starting now the last thing this path of the function evaluates?"""
        kind, frame = self._top()
        if kind in FUNCTIONS:
            return frame.state == S_BODY
        if kind in (K_LET, K_ARM):
            return frame.state == 1 and bool(frame.exit)
        if kind == K_IF:
            return frame.state != 0 and bool(frame.exit)
        return False

    def _exit(self, word: int) -> None:
        if word >= 0:
            self.allowed &= 1 << word
        elif word == NULL:
            self.allowed &= self.nulls
        else:
            self.allowed = 0

    def _finished(self, word: int, exit_: bool) -> None:
        self.word = word
        self.child = True
        if exit_:
            self._exit(word)

    def _forward(self, cls: int, kind: int, payload: bytes) -> tuple[int, int]:
        if cls == OPEN:
            self._open(kind, payload)
        elif cls == SEP:
            return self._sep(kind, payload)
        elif cls == LEAF:
            return self._leaf(kind, payload), 0
        else:
            self._close(kind)
        return NO_POSITION, 0

    def _open(self, kind: int, payload: bytes) -> None:
        if kind in FUNCTIONS:
            self.in_function = True
            self.fn_name = payload
            self.inferring = kind == K_FN
            self.declared_at = -1
            self.write_at = b""
            self.allowed = 0
            self.nulls = 0
            self.records = 0
            self._push(kind, _Open(state=S_PARAMETERS))
            return
        if kind == K_TTUPLE:
            self.inferring = False  # `@k` says which one cell comes back
            self._push(kind, _Open())
            return
        if kind == K_ARM:
            _match_kind, match = self._top()
            # An arm of a match on a Fin or on `within`, `0`, `down(j)`, `inside(i)` or `outside`, has no cell, and
            # what it binds is no parameter's.
            counting = payload.rstrip(b"\0") in (b"", *ARM_WORDS)
            self._push(K_ARM, _Open(exit=match.exit, mark=len(self.B), word=NOTHING_KNOWN if counting else match.word,
                                    nulls=self.nulls))
            return
        frame = _Open(exit=int(self._in_exit_position()), mark=len(self.B))
        if kind == K_CALL:
            frame.state = int(payload == self.fn_name)
            frame.join = self._cell_of(payload)
            self.child = False
        self._push(kind, frame)

    def _cell_of(self, name: bytes) -> int:
        for position in range(len(self.T)):
            tag, data = self.T.at(position)
            if data == name:
                return tag - 1
        return NOTHING_KNOWN

    def _sep(self, kind: int, payload: bytes) -> tuple[int, int]:
        top_kind, frame = self._top()
        index = payload[0]
        if kind in FUNCTIONS:
            if index == 0:
                frame.state = S_RESULT
            else:
                frame.state = S_BODY
                self._start_body()
            self._set_top(frame)
        elif kind == K_LET:
            count = frame.count
            first = len(self.B)
            names = [self.N.at(len(self.N) - count + position)[1] for position in range(count)]
            for _ in range(count):
                self.N.pop()
            for name in names:
                self._bind(name, self.word if count == 1 else NOTHING_KNOWN)
            frame.state = 1
            self._set_top(frame)
            return first, count
        elif kind == K_IF:
            if index == 2:
                frame.join = self.word
            frame.state = index
            self._set_top(frame)
        elif kind == K_MATCH:
            frame.word = self.word
            frame.state = 1
            self._set_top(frame)
        elif kind == K_ARM:
            if frame.count == 0 and frame.word >= 0:  # no binders: the null constructor of a tagless type
                self.nulls |= 1 << frame.word
            frame.state = 1
            self._set_top(frame)
        elif kind == K_WITH:
            if index == 1:
                frame.word = self.word
            frame.state = index
            self._set_top(frame)
        elif top_kind == K_CALL:
            self._argument(frame)
            self._set_top(frame)
        elif top_kind == K_TUPLE:
            # Check would say this at the same member, and Parse may find something else first.
            frame.count += 1
            if frame.count == TUPLE_MAX:
                raise self._reject(f"section 8.2: a tuple of more than {TUPLE_MAX}, which is as many registers as a result comes back in")
            self._set_top(frame)
        return NO_POSITION, 0

    def _start_body(self) -> None:
        self.allowed = 0
        if not self.inferring:
            return
        if self.result_type.rstrip(b"\0") in _NOT_A_CELL or not self.result_type or self._is_enum(self.result_type):
            self.inferring = False
            return
        for position in range(len(self.P)):
            if self.P.at(position)[1] == self.result_type:
                self.allowed |= 1 << position

    def _is_enum(self, name: bytes) -> bool:
        return any(self.E.at(position)[1] == name for position in range(len(self.E)))

    def _argument(self, frame: _Open) -> None:
        if self.child:
            if frame.count < MAX_PARAMETERS:
                args = list(frame.args)
                args[frame.count] = self.word
                frame.args = tuple(args)
            frame.count = min(frame.count + 1, 255)
            self.child = False

    def _leaf(self, kind: int, payload: bytes) -> int:
        if kind == K_PARAM:
            self.pending_name = payload
            return NO_POSITION
        if kind in (K_TYPE, K_TARRAY, K_TFIN, K_TFINNAME):
            return self._type(kind, payload)
        if kind == K_RESWORD:
            self.inferring = False
            position = self._lookup(payload)[0]  # Check will refuse a name that is not a parameter
            self.declared_at = position if position != NO_POSITION else -1
            return NO_POSITION
        if kind == K_BINDER:
            return self._binder(payload)
        if kind == K_FIELD:  # the name a { e with f = v } binds, for use in v
            return self._bind(payload, NOTHING_KNOWN)
        if kind == K_DIAMOND:
            position, word = self._lookup(payload)
            _kind, frame = self._top()
            frame.word = word
            self._set_top(frame)
            return position
        exit_ = self._in_exit_position()
        if kind == K_VAR:
            position, word = self._lookup(payload)
            self._finished(word, exit_)
            return position
        if kind == K_VARAT:
            position = VARAT.unpack_from(payload)[0]
            self._finished(self.B.at(position)[0] - 2, exit_)
            return position
        if kind == K_NULLCTOR:
            self._finished(NULL, exit_)
        elif kind in (K_INT, K_BOOL):
            self._finished(NOTHING_KNOWN, exit_)
        return NO_POSITION

    def _type(self, kind: int, payload: bytes) -> int:
        top_kind, frame = self._top()
        if top_kind == K_TTUPLE:
            return NO_POSITION
        name = payload if kind == K_TYPE else b""
        if frame.state == S_RESULT:
            self.result_type = name
            return NO_POSITION
        if len(self.P) == MAX_PARAMETERS:
            raise self._reject(f"more than {MAX_PARAMETERS} parameters (the register budget)")
        self.P.push(0, name)
        return self._bind(self.pending_name, len(self.P) - 1)

    def _binder(self, payload: bytes) -> int:
        top_kind, frame = self._top()
        if top_kind == K_LET:
            if frame.count == TUPLE_MAX:
                raise self._reject(f"section 8.2: more than {TUPLE_MAX} names in one let")
            self.N.push(0, payload)
            frame.count += 1
            self._set_top(frame)
            return NO_POSITION
        if frame.count > CELL_FIELDS_MAX:
            raise self._reject(f"pattern has too many binders: a cell has at most {CELL_FIELDS_MAX} fields")
        position = self._bind(payload, frame.word if frame.count == 0 else NOTHING_KNOWN)
        frame.count += 1
        self._set_top(frame)
        return position

    def _close(self, kind: int) -> None:
        _kind, frame = self._top()
        self.K.pop()
        if kind in FUNCTIONS:
            self._end_function()
        elif kind == K_TTUPLE:
            pass
        elif kind == K_LET:
            self._unbind_to(frame.mark)
            self.child = True
        elif kind == K_IF:
            self.word = self.word if self.word == frame.join else NOTHING_KNOWN
            self.child = True
        elif kind == K_ARM:
            self._unbind_to(frame.mark)
            self.nulls = frame.nulls
            _match_kind, match = self._top()
            match.join = self.word if match.join in (NOT_YET, self.word) else NOTHING_KNOWN
            self._set_top(match)
        elif kind == K_MATCH:
            self.word = frame.join if frame.join != NOT_YET else NOTHING_KNOWN
            self.child = True
        elif kind == K_WITH:
            self._unbind_to(frame.mark)
            self._finished(frame.word, bool(frame.exit))
        elif kind == K_CTOR:
            self._finished(frame.word, bool(frame.exit))
        elif kind == K_CALL:
            self._close_call(frame)
        else:
            self._finished(NOTHING_KNOWN, bool(frame.exit))

    def _close_call(self, frame: _Open) -> None:
        self._argument(frame)
        if frame.state:  # recursive: `@k` holds of it if it passes parameter k's own word along as argument k
            self.word = NOTHING_KNOWN
            self.child = True
            if frame.exit:
                self.allowed &= sum(1 << k for k in range(min(frame.count, MAX_PARAMETERS)) if frame.args[k] == k)
            return
        word = NOTHING_KNOWN
        if 0 <= frame.join < min(frame.count, MAX_PARAMETERS) and frame.args[frame.join] >= 0:
            word = frame.args[frame.join]
        self._finished(word, bool(frame.exit))

    def _end_function(self) -> None:
        cell = self.declared_at
        if self.inferring and self.allowed:
            cell = (self.allowed & -self.allowed).bit_length() - 1
            self.write_at = self.B.at(cell)[1]
        self.T.push(cell + 1, self.fn_name)
        self._unbind_to(0)
        while len(self.P):
            self.P.pop()

    # --- going backward: who is still to be used

    def _step_back(self) -> None:
        self.cursor -= 1
        record = self._read(self.cursor)
        tag, payload, _line, stamp, count, flags, _mask = record
        cls, kind = ev_class(tag), ev_kind(tag)
        if cls == LEAF:
            flags = self._back_leaf(kind, stamp, flags)
            if flags:
                self._annotate(self.cursor, record, flags, 0)
        elif cls == SEP:
            self._back_sep(kind, payload[0], record, stamp, count)
        elif cls == CLOSE:
            if kind in (K_MATCH, K_IF):
                self.R.push(0, self.live.to_bytes(PAYLOAD_BYTES, "little") + bytes(PAYLOAD_BYTES))
            elif kind == K_ARM:
                self.live = self._branch()[0]
        elif kind == K_ARM:
            after, before = self._branch()
            self._set_branch(after, before | self.live)
        if self.cursor == 0:
            self.phase = REPLAYING

    def _branch(self) -> tuple[int, int]:
        data = self.R.peek()[1]
        return int.from_bytes(data[:PAYLOAD_BYTES], "little"), int.from_bytes(data[PAYLOAD_BYTES:], "little")

    def _set_branch(self, after: int, before: int) -> None:
        self.R.rewrite(len(self.R) - 1, 0, after.to_bytes(PAYLOAD_BYTES, "little") + before.to_bytes(PAYLOAD_BYTES, "little"))

    def _back_leaf(self, kind: int, stamp: int, flags: int) -> int:
        if stamp >= MASK_BITS:
            return flags
        bit = 1 << stamp
        if kind in (K_VAR, K_VARAT, K_DIAMOND):
            if kind != K_DIAMOND and not self.live & bit:
                flags |= F_LAST
            self.live |= bit
        elif kind in (K_BINDER, K_FIELD):
            self.live &= ~bit
        return flags

    def _back_sep(self, kind: int, index: int, record: tuple, stamp: int, count: int) -> None:
        if kind == K_MATCH:
            _after, self.live = self._branch()
            self.R.pop()
        elif kind == K_IF:
            self._annotate(self.cursor, record, F_LIVE, self.live)
            after, before = self._branch()
            if index == 2:
                self._set_branch(after, self.live)
                self.live = after
            else:
                self.live |= before
                self.R.pop()
        elif kind == K_ARM or (kind == K_WITH and index == 2) or (kind in FUNCTIONS and index == 1):
            self._annotate(self.cursor, record, F_LIVE, self.live)
        elif kind == K_LET:
            self._annotate(self.cursor, record, F_LIVE, self.live)
            for position in range(stamp, min(stamp + count, MASK_BITS)):
                self.live &= ~(1 << position)

    # --- forward again: on to Check

    def _replay(self) -> None:
        if not self.after:
            self._next_record()
        tag, payload = self.after.pop(0)
        self.ring.emit_fresh(self, tag, payload)
        if not self.after and self.cursor == self.records:
            self.phase = SPILLING
            self.records = 0

    def _next_record(self) -> None:
        tag, payload, line, _stamp, _count, flags, mask = self._read(self.cursor)
        self.cursor += 1
        self.ring.replay_line = line
        cls, kind = ev_class(tag), ev_kind(tag)
        if cls == SEP and kind in FUNCTIONS and payload[0] == 1 and self.write_at:
            self.after.append((ev_tag(LEAF, K_RESWORD), self.write_at))
        if flags & F_LAST and kind == K_VARAT:
            payload = VARAT.pack(VARAT.unpack_from(payload)[0], 1)
        elif flags & F_LAST:
            tag = ev_tag(LEAF, K_VARLAST)
        self.after.append((tag, payload))
        if flags & F_LIVE:
            self.after.append((ev_tag(LEAF, K_LIVE), mask))
