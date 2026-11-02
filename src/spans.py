"""A record that spans cells: LFPL-O section 4.8.

The source declares one record, and a cell has 16 bytes. So this lays the fields out across as many cells as they
take, in declaration order and at their natural alignment, and every cell but the last gives its second 8 bytes to
a link to the next. All of it happens to the event word, ahead of the right-to-left pass, which sees what a person
would have written by hand:

    record Frame { a: Int64, b: Int64, c: Int64 };     data $1.1;
                                                       record Frame { a: Int64, $next: $1.1 };
                                                       record $1.1 { b: Int64, c: Int64 };

    match f with Frame(d, a, b, c) => BODY end         match f with Frame(d, a, $2) =>
                                                         match $2 with $1.1(d$0.0, b, c) => BODY end end

    Frame@d(x, y, z)                                   Frame@d(x, $1.1@d$0.0(y, z))

Nothing after this stage knows that spans exist: Check types those matches, Lower writes `=` for what stays in each
cell, rtcheck holds every word, and n and R(n) count each of those cells because each of them is one. That is why this
adds nothing to what has to be trusted. `src/symcheck.py` does the same rewrite on its own, sharing no code with this.
A `$` cannot be in a source name, because the lexer has no such character.

What a pattern's `d` means is the only new idea. It stands for every cell of the record, and exactly-once linearity
holds of each of them, because each is a binder downstream: the name itself is the first cell, and the others are
`d$p.0`, `d$p.1` and so on, for the name at position p in scope, which no other name in scope is. So a group is only a
note, against that position, of how many cells it stands for. A group can be used in three ways, and only these, since a
cell cannot silently be left behind:

    R@d(...)                    builds a record of exactly that many cells in them
    More@d(free)                gives every one of them back
    let (c, rest) = peel(d)     c is the first, and rest is the others: `(d, d$p.0)` as a tuple, downstream

A record with no group to build in names its cells, each of which came from somewhere, usually a draw on the reserve,
which is where R(n) counts it:

    R@(a, b, c)(...)            builds it in those, in that order

A record that fits one cell is left exactly as it was written.
"""
from __future__ import annotations

import struct
from dataclasses import astuple, dataclass
from typing import Iterator, Optional

from .dyck import (CLOSE, K_ARM, K_BINDER, K_CALL, K_CTOR, K_DATA, K_DIAMOND, K_FIELD, K_LET, K_MATCH, K_RECORD,
                   K_TUPLE, K_TYPE, K_VAR, LEAF, OPEN, SEP, ev_kind, ev_tag)
from .tokens import PAYLOAD_BYTES

CELL = 16
LINK = 8               # where a part's link to the next part is, and how many bytes come ahead of it
LINK_NAME = b"$next"
FIELDS_MAX = 64        # section 8.2: the fields of one record
# Of a hidden name: then `$`, a position in scope of up to five digits, `.`, and which cell, at up to two.
PREFIX_ROOM = PAYLOAD_BYTES - 9
# Theorem 7.2. One record's fields, and its name and its close, held until the close says whether it spans.
HELD_CELLS = 2 * FIELDS_MAX + 2
# Theorem 7.2, per open construct: what is being rewritten there, the note of a group bound there, and
# the names of the cells a constructor listed, of which there is at most one per field.
CELLS_PER_DEPTH = 2 + FIELDS_MAX
SIZES = {b"Int8": 1, b"Int16": 2, b"Int32": 4, b"Int64": 8, b"Bool": 1, b"Be16": 2, b"Be32": 4, b"Be64": 8}
NO_RECORD = 0xFFFF     # what is open is a `More@d(free)`, which is cells and no record
NO_GROUP = 0xFFFF      # its cells are named one by one
PEEL = b"peel"
EVENT = tuple[int, bytes]
_RECORD_HEAD = struct.Struct("<HBB")  # its number, its parts and its fields, after the name
_GROUP = struct.Struct("<HHBB")       # the name's position in scope, where its cells were numbered, the first, how many


def parts_of(sizes: list[int]) -> list[int]:
    """How many fields go in each cell. A cell takes all that are left if they fit it. If not, it takes what fits
    ahead of the link, and any field fits at offset 0, so each cell holds at least one."""
    parts = []
    start = 0
    while True:
        if _fitting(sizes[start:], CELL) == len(sizes) - start:
            parts.append(len(sizes) - start)
            return parts
        count = _fitting(sizes[start:], LINK)
        parts.append(count)
        start += count


def _fitting(sizes: list[int], room: int) -> int:
    offset = count = 0
    for size in sizes:
        offset = (offset + size - 1) // size * size
        if offset + size > room:
            break
        offset += size
        count += 1
    return count


@dataclass
class _Rewriting:
    """A spanning match arm or constructor, or a `More@d`, that is open. `depth` is how many frames the pass will have
    open when its deepest part is on top, which is how an event is known to belong to it."""
    kind: int
    depth: int
    record: int = NO_RECORD
    part: int = 0
    in_part: int = 0     # fields so far in this part. A `More@d`: how many more closes it needs
    seen: int = 0        # binders or arguments finished so far, as the source counts them
    group: int = NO_GROUP  # the position in scope of the group that names its cells
    listed: int = 0      # or how many cells it named
    mark: int = 0        # and where those names start

    _PACKED = struct.Struct("<BHHBBBHBH")

    def pack(self) -> bytes:
        return self._PACKED.pack(*astuple(self))

    @classmethod
    def unpack(cls, data: bytes) -> "_Rewriting":
        return cls(*cls._PACKED.unpack_from(data))


@dataclass(frozen=True)
class _Group:
    position: int
    origin: int  # the position of the pattern's diamond, which stays in scope as long as any of these names do
    first: int
    cells: int
    prefix: bytes

    def cell(self, index: int) -> bytes:
        """The name, downstream, of one cell of the group after its first."""
        return self.prefix + b"$" + str(self.origin).encode() + b"." + str(self.first + index - 1).encode()


@dataclass(frozen=True)
class _Span:
    name: bytes
    number: int
    counts: bytes

    @property
    def total(self) -> int:
        return sum(self.counts)

    def part_name(self, part: int) -> bytes:
        return self.name if part == 0 else f"${self.number}.{part}".encode()


class Spans:
    """The rewrite, for the right-to-left stage to run each event through. Everything it holds is in cells."""

    def __init__(self, stage) -> None:
        self.stage = stage
        self.ctor = b""            # the constructor just opened, whose diamond comes next
        self.peeling = 0         # after `peel(`: 1 wants the name, 2 wants the `)`, and 3 waits for the `in`
        self.rest: Optional[_Group] = None  # what the second name of that let will stand for
        self.holding = False
        self.emitting: Optional[Iterator[EVENT]] = None
        self.after_diamond = False  # the last event named a constructor's cell, and another may follow

    def attach(self) -> None:
        self.S = self.stage.stack()  # for all time: each record that spans
        self.A = self.stage.stack()  # the groups in scope, in order of position
        self.O = self.stage.stack()  # what is open and being rewritten
        self.D = self.stage.stack()  # the record being held, one event per frame, with its tag
        self.Q = self.stage.stack()  # the cells listed by the constructors open now, after each one's first

    def table_cells(self) -> int:
        return len(self.S.cells)

    def unbind_to(self, mark: int) -> None:
        while len(self.A) and self._group_at(len(self.A) - 1).position >= mark:
            self.A.pop()

    # --- declarations

    def hold(self, cls: int, kind: int, payload: bytes) -> None:
        """A record's events wait here for its close, which is when its size is known."""
        if len(self.D) >= HELD_CELLS:
            name = self.D.at(0)[1].rstrip(b"\0").decode()
            raise self.stage._reject(f"section 8.2: record {name} has more than {FIELDS_MAX} fields")
        self.D.push(ev_tag(cls, kind), payload)
        self.holding = cls != CLOSE
        if cls == CLOSE:
            self.emitting = self._declaration()

    def next_declaration_event(self) -> Optional[EVENT]:
        assert self.emitting is not None
        event = next(self.emitting, None)
        if event is None:
            self.emitting = None
            while len(self.D):
                self.D.pop()
        return event

    def _declaration(self) -> Iterator[EVENT]:
        name = self.D.at(0)[1]
        fields = (len(self.D) - 2) // 2
        counts = parts_of([self._size(1 + 2 * index) for index in range(fields)])
        if len(counts) == 1:
            for index in range(len(self.D)):
                yield self.D.at(index)
            return
        span = _Span(name, len(self.S) + 1, bytes(counts))
        self.S.push(0, name + _RECORD_HEAD.pack(span.number, len(counts), fields) + span.counts)
        for part in range(1, len(counts)):
            yield ev_tag(OPEN, K_DATA), span.part_name(part)
            yield ev_tag(CLOSE, K_DATA), b""
        event = 1
        for part, count in enumerate(counts):
            yield ev_tag(OPEN, K_RECORD), span.part_name(part)
            for _ in range(2 * count):
                yield self.D.at(event)
                event += 1
            if part + 1 < len(counts):
                yield ev_tag(LEAF, K_FIELD), LINK_NAME
                yield ev_tag(LEAF, K_TYPE), span.part_name(part + 1)
            yield ev_tag(CLOSE, K_RECORD), b""

    def _size(self, index: int) -> int:
        tag, data = self.D.at(index + 1)
        return SIZES.get(data.rstrip(b"\0"), 8) if ev_kind(tag) == K_TYPE else 8

    # --- the tables

    def _span_at(self, index: int) -> _Span:
        data = self.S.at(index)[1]
        number, parts, _fields = _RECORD_HEAD.unpack_from(data, PAYLOAD_BYTES)
        at = PAYLOAD_BYTES + _RECORD_HEAD.size
        return _Span(data[:PAYLOAD_BYTES], number, data[at:at + parts])

    def _span_named(self, name: bytes) -> Optional[int]:
        for index in range(len(self.S)):
            if self.S.at(index)[1][:PAYLOAD_BYTES] == name:
                return index
        return None

    def _group_at(self, index: int) -> _Group:
        data = self.A.at(index)[1]
        position, origin, first, cells = _GROUP.unpack_from(data)
        return _Group(position, origin, first, cells, data[_GROUP.size:].rstrip(b"\0"))

    def _group_of(self, name: bytes) -> Optional[_Group]:
        position = self.stage._lookup(name)[0]
        for index in range(len(self.A) - 1, -1, -1):
            group = self._group_at(index)
            if group.position == position:
                return group
        return None

    def _note_group(self, group: _Group) -> None:
        self.A.push(0, _GROUP.pack(group.position, group.origin, group.first, group.cells) + group.prefix)

    def _open(self) -> Optional[_Rewriting]:
        return _Rewriting.unpack(self.O.peek()[1]) if len(self.O) else None

    def _save(self, what: _Rewriting) -> None:
        self.O.rewrite(len(self.O) - 1, 0, what.pack())

    def _here(self, kind: int) -> Optional[_Rewriting]:
        """What is being rewritten, if this event is in the deepest part of it."""
        what = self._open()
        return what if what is not None and what.kind == kind and what.depth == len(self.stage.K) else None

    # --- events

    def events(self, cls: int, kind: int, payload: bytes) -> list[EVENT]:
        """What this event is, for everything downstream. At most one record's worth."""
        if self.peeling:
            return self._in_peel(cls, kind, payload)
        if self.after_diamond and (cls != LEAF or kind != K_DIAMOND):
            self._every_cell_named()
        same = [(ev_tag(cls, kind), payload)]
        if cls == OPEN:
            if kind == K_CALL and payload.rstrip(b"\0") == PEEL:
                return self._open_peel()
            if kind == K_CTOR:
                self.ctor = payload
            if kind in (K_ARM, K_CTOR):
                record = self._span_named(payload)
                if record is not None:
                    self.O.push(0, _Rewriting(kind, len(self.stage.K) + 1, record).pack())
            return same
        if cls == LEAF:
            if kind == K_BINDER and self._here(K_ARM) is not None:
                return self._binder(payload)
            if kind == K_DIAMOND:
                return self._diamond(payload)
            if kind == K_VAR:
                self._refuse_a_group(payload)
            return same
        if cls == SEP:
            if kind == K_CTOR:
                what = self._here(K_CTOR)
                if what is not None and what.record != NO_RECORD:
                    return self._next_argument(what)
            elif kind == K_ARM:
                what = self._here(K_ARM)
                if what is not None:
                    self._all_bound(what)
            return same
        if kind == K_CTOR:
            what = self._here(K_CTOR)
            if what is not None:
                return self._close_constructor(what)
        elif kind == K_ARM:
            what = self._here(K_ARM)
            if what is not None:
                self.O.pop()
                nested = [(ev_tag(CLOSE, K_ARM), b""), (ev_tag(CLOSE, K_MATCH), b"")] * what.part
                return nested + same
        return same

    # --- a pattern

    def _binder(self, name: bytes) -> list[EVENT]:
        what = self._here(K_ARM)
        assert what is not None
        span = self._span_at(what.record)
        out = [(ev_tag(LEAF, K_BINDER), name)]
        if what.seen == 0:  # its diamond, which stands for every cell. It is bound next, at this position
            what.group = len(self.stage.B)
            self._note_group(_Group(what.group, what.group, 0, len(span.counts), name.rstrip(b"\0")[:PREFIX_ROOM]))
        elif what.seen > span.total:
            raise self.stage._reject(f"pattern has too many binders: expected {span.total + 1}")
        else:
            what.in_part += 1
            if what.in_part == span.counts[what.part] and what.part + 1 < len(span.counts):
                link = b"$" + str(len(self.stage.B) + 1).encode()  # bound after the field this event binds
                what.part += 1
                what.in_part = 0
                what.depth += 2
                out += [(ev_tag(LEAF, K_BINDER), link),
                        (ev_tag(SEP, K_ARM), b"\1"),
                        (ev_tag(OPEN, K_MATCH), b""),
                        (ev_tag(LEAF, K_VAR), link),
                        (ev_tag(SEP, K_MATCH), b"\1"),
                        (ev_tag(OPEN, K_ARM), span.part_name(what.part)),
                        (ev_tag(LEAF, K_BINDER), self._group_named_at(what.group).cell(what.part))]
        what.seen += 1
        self._save(what)
        return out

    def _group_named_at(self, position: int) -> _Group:
        for index in range(len(self.A) - 1, -1, -1):
            group = self._group_at(index)
            if group.position == position:
                return group
        raise AssertionError(f"no group at position {position}")

    def _all_bound(self, what: _Rewriting) -> None:
        span = self._span_at(what.record)
        if what.seen != span.total + 1:
            raise self.stage._reject(f"pattern needs {span.total + 1} binders, got {what.seen}")

    # --- a constructor

    def _diamond(self, name: bytes) -> list[EVENT]:
        group = self._group_of(name)
        cells = group.cells if group is not None else 1
        same = [(ev_tag(LEAF, K_DIAMOND), name)]
        what = self._here(K_CTOR)
        shown = name.rstrip(b"\0").decode()
        ctor = self.ctor.rstrip(b"\0")
        again = self.after_diamond
        self.after_diamond = True
        if what is not None and what.record != NO_RECORD:
            span = self._span_at(what.record)
            if group is not None:
                if again or cells != len(span.counts):
                    raise self.stage._reject(f"{ctor.decode()} takes {len(span.counts)} cells, and {shown} is {cells}")
                what.group = group.position
                self._save(what)
                return same
            if not again:  # the first of a list goes out now, and the others when the constructor gets to them
                what.mark, what.listed = len(self.Q), 1
                self._save(what)
                return same
            self.Q.push(0, name)
            what.listed += 1
            self._save(what)
            return []
        if again:
            raise self.stage._reject(f"{ctor.decode()} takes one cell, and this names more")
        if cells == 1:
            return same
        assert group is not None
        if ctor != b"More":
            raise self.stage._reject(f"{ctor.decode()} takes one cell, and {shown} is {cells}: "
                                     f"let (c, rest) = peel({shown}) gives you one")
        for index in range(1, cells):
            same += [(ev_tag(OPEN, K_CTOR), self.ctor), (ev_tag(LEAF, K_DIAMOND), group.cell(index))]
        self.O.push(0, _Rewriting(K_CTOR, len(self.stage.K) + cells - 1, in_part=cells - 1).pack())
        return same

    def _every_cell_named(self) -> None:
        self.after_diamond = False
        what = self._here(K_CTOR)
        if what is None or what.record == NO_RECORD or what.group != NO_GROUP:
            return
        span = self._span_at(what.record)
        if what.listed != len(span.counts):
            raise self.stage._reject(f"{span.name.rstrip(bytes(1)).decode()} takes {len(span.counts)} cells, and "
                                     f"this names {what.listed}")

    def _cell_of(self, what: _Rewriting, part: int) -> bytes:
        if what.group != NO_GROUP:
            return self._group_named_at(what.group).cell(part)
        return self.Q.at(what.mark + part - 1)[1]

    def _next_argument(self, what: _Rewriting) -> list[EVENT]:
        span = self._span_at(what.record)
        what.seen += 1
        if what.seen >= span.total:
            raise self.stage._reject(f"too many arguments: expected {span.total}")
        what.in_part += 1
        out = [(ev_tag(SEP, K_CTOR), bytes((what.in_part,)))]
        if what.in_part == span.counts[what.part]:
            what.part += 1
            what.in_part = 0
            what.depth += 1
            out += [(ev_tag(OPEN, K_CTOR), span.part_name(what.part)),
                    (ev_tag(LEAF, K_DIAMOND), self._cell_of(what, what.part))]
        self._save(what)
        return out

    def _close_constructor(self, what: _Rewriting) -> list[EVENT]:
        self.O.pop()
        if what.record == NO_RECORD:
            return [(ev_tag(CLOSE, K_CTOR), b"")] * (what.in_part + 1)
        span = self._span_at(what.record)
        if what.seen + 1 != span.total:
            raise self.stage._reject(f"wrong argument count: expected {span.total}")
        for _ in range(what.listed - 1):
            self.Q.pop()
        return [(ev_tag(CLOSE, K_CTOR), b"")] * (what.part + 1)

    # --- peel, and what else a group cannot be

    def _refuse_a_group(self, name: bytes) -> None:
        group = self._group_of(name)
        if group is not None:
            raise self.stage._reject(f"{name.rstrip(bytes(1)).decode()} is {group.cells} cells, and only a constructor "
                                     f"or peel can take them")

    def _open_peel(self) -> list[EVENT]:
        kind, frame = self.stage._top()
        if kind != K_LET or frame.state != 0 or frame.count != 2:
            raise self.stage._reject("peel is written let (c, rest) = peel(d)")
        self.peeling = 1
        return [(ev_tag(OPEN, K_TUPLE), b"")]

    def _in_peel(self, cls: int, kind: int, payload: bytes) -> list[EVENT]:
        if self.peeling == 1:
            if cls != LEAF or kind != K_VAR:
                raise self.stage._reject("peel takes the name of a record's diamond")
            group = self._group_of(payload)
            if group is None:
                raise self.stage._reject(f"{payload.rstrip(bytes(1)).decode()} is one cell, and peel takes "
                                         f"the diamond of a record that spans")
            self.peeling = 2
            # The second name stands for the rest: it is the first of them, and the others keep their names.
            self.rest = _Group(len(self.stage.B) + 1, group.origin, group.first + 1, group.cells - 1, group.prefix)
            return [(ev_tag(LEAF, K_VAR), payload), (ev_tag(SEP, K_TUPLE), b"\1"),
                    (ev_tag(LEAF, K_VAR), group.cell(1))]
        if self.peeling == 2:
            if cls != CLOSE or kind != K_CALL:
                raise self.stage._reject("peel takes one diamond")
            self.peeling = 3
            return [(ev_tag(CLOSE, K_TUPLE), b"")]
        assert cls == SEP and kind == K_LET, "a parser that lets anything but `in` follow a bound expression"
        self.peeling = 0
        if self.rest is not None and self.rest.cells > 1:
            self._note_group(self.rest)
        self.rest = None
        return [(ev_tag(cls, kind), payload)]
