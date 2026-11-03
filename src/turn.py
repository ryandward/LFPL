"""The runtime's turn in LFPL, for one handler: src/turn.lfpl, filled in from the handler's own checked stream.

What the turn does is fixed, and written once in src/turn.lfpl. What depends on the handler is filled in here: the
record its item is, which field of it is the slot, the verdict it returns, and how many places the output queue has.
The result is LFPL, appended to the handler's program, and it goes through every stage the handler does. So the turn
is typed and its bound certified like any other code, and src/rtverilog.py lowers it with the handler.

It serves the shape examples/ipv4.lfpl has, and refuses any other with a reason: a handler with no state slots, that
draws nothing from its reserve, takes a record whose payload is one slot, and returns a verdict beside that record.
"""
from __future__ import annotations

import io
import os
import string
from dataclasses import dataclass

from .certify import certify
from .check import SLOT_BYTES
from .rtcheck import Checker, TypeDecl
from .rtlower import rt_lines
from .runtime import Walk, _walk
from .spans import FIELDS_MAX, SIZES

# A turn is told a frame's length in 15 bits, beside the bit that says egress has sent, and ingress counts a frame to
# one byte past what it holds. Section 8.2 keeps that far below 2^15 for every record: at most 64 fields on the wire,
# none of more than 8 bytes, and one slot. So no handler needs refusing for it.
assert FIELDS_MAX * max(SIZES.values()) + SLOT_BYTES + 1 < 1 << 15

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "turn.lfpl")
TURN = "serve"
LINK = "$next"
_TYPES = {"Pool", "Spares", "Out", "Queue", "Turned"}
_CTORS = {"Dry", "Cell", "NoSpare", "Spare", "None", "Pending", "Idle", "Queued", "Dropped", "Runt", "TooLarge",
          "Short", "Full"}
_FUNCTIONS = {"give", "give_free", "give_frame", "give_out", "is_full", "release", "enqueue", TURN}
# What the turn returns in its last result, by tag, which is the order src/turn.lfpl declares them in. Queued and
# Dropped carry the tag of the handler's verdict in the byte above their own.
OUTCOMES = ("idle", "queued", "dropped", "runt", "too_large", "short", "full")
DECIDED = ("queued", "dropped")


class TurnRefused(Exception):
    """The handler has a shape the turn does not serve yet."""


@dataclass(frozen=True)
class Shape:
    """What the hardware around the turn needs to know: where a frame's bytes go, and where the queue's head is."""
    handler: str
    item: str                 # the record's name
    item_walk: Walk           # its cells, scalars, links, slot and length, as src/runtime.py walks them
    parts: int                # the cells of the record itself
    wire: int                 # bytes of it on the wire
    slot_bytes: int           # what its slot holds
    k_out: int                # outputs the queue holds: its head and the places after it
    verdicts: tuple           # (name, tag, the port's offset in the register, or None)

    @property
    def capacity(self) -> int:
        """The most bytes a frame in ingress can have: the record's on the wire, and a full slot."""
        return self.wire + self.slot_bytes

    @property
    def n_max(self) -> int:
        """The largest n of a frame: its record's cells, and a full slot's bytes."""
        return self.parts + self.slot_bytes


def turn_source(program: bytes, *, k_out: int = 4) -> tuple[str, Shape]:
    """The handler's program with the turn appended, and what the machine needs to know about it."""
    if k_out < 2:
        raise TurnRefused(f"section 8.2: K_out of {k_out}: the queue has its head and at least one place after it")
    checker = Checker()
    for line in rt_lines(io.BytesIO(program)):
        checker.feed(line)
    checker.finish()
    handlers = [f for f in checker.functions.values() if f.handler]
    if len(handlers) != 1:
        raise TurnRefused(f"section 3.1: a turn serves one handler, and the program declares {len(handlers)}")
    handler = handlers[0]
    if checker.states:
        raise TurnRefused("the turn passes no state slots yet, and the handler takes some")
    results = checker.tuples.get(checker._resolve(handler.ret), [])
    verdict_type = checker.norm(results[0]) if len(results) == 3 else None
    if not (isinstance(verdict_type, tuple) and verdict_type[0] == "enum"):
        raise TurnRefused("the turn reads a verdict, and the handler returns an output instead")
    item_type = checker.norm(handler.params[0])
    decl = checker._decl(item_type)
    if decl is None or decl.kind != "record" or checker.norm(results[1]) != item_type:
        raise TurnRefused("the turn queues the record the handler was given, and this handler returns another")
    walk = _walk(checker, decl.name, decl.ctors[0].fields, from_the_wire=True)
    fields, parts = _source_fields(checker, decl)
    if walk.cells != parts:
        raise TurnRefused(f"{decl.name} holds a record in a field, and the turn frees only its own cells and its slot")
    linear = [index for index, (_name, ty) in enumerate(fields) if checker._decl(checker.norm(ty)) is not None]
    if walk.slot is None or len(linear) != 1:
        raise TurnRefused(f"the turn takes a record whose payload is one slot, and {decl.name} has "
                          f"{'a chain' if walk.chain is not None else f'{len(linear)} linear fields'}")
    slot_decl = checker._decl(checker.norm(fields[linear[0]][1]))
    assert slot_decl is not None
    bound = certify(checker, handler.name, parts + slot_decl.bytes)
    if bound.a or bound.b:
        raise TurnRefused(f"the turn hands the handler an empty reserve, and {handler.name} draws "
                          f"R(n) = {bound.a} n + {bound.b}")
    _no_clash(checker)
    verdicts = _verdicts(checker, checker.types[verdict_type[1]])
    shape = Shape(handler.name, decl.name, walk, parts, walk.wire_bytes, slot_decl.bytes, k_out, verdicts)
    text = _fill(handler.name, decl.name, fields, linear[0], slot_decl.name, parts, walk.wire_bytes, shape.capacity,
                 k_out, checker.types[verdict_type[1]])
    return program.decode() + "\n" + text, shape


def _source_fields(checker: Checker, decl: TypeDecl) -> tuple[list, int]:
    """The record's fields as its source declared them, across its cells, and how many cells those are."""
    fields, parts, ctor = [], 1, decl.ctors[0]
    while True:
        following = None
        for _off, _size, ty, name in ctor.fields:
            if name == LINK:
                following = checker._decl(checker.norm(ty))
            else:
                fields.append((name, ty))
        if following is None:
            return fields, parts
        parts += 1
        ctor = following.ctors[0]


def _no_clash(checker: Checker) -> None:
    taken = ({decl.name for decl in checker.types.values()}, {ctor.name for ctor in checker.ctors.values()},
             {fn.name for fn in checker.functions.values()})
    for names, ours in zip(taken, (_TYPES, _CTORS, _FUNCTIONS)):
        clash = sorted(names & ours)
        if clash:
            raise TurnRefused(f"the turn declares {clash[0]}, and so does the program")


def _verdicts(checker: Checker, decl: TypeDecl) -> tuple:
    verdicts = []
    for ctor in decl.ctors:
        port = None
        if ctor.fields:
            (off, _size, ty, name), *rest = ctor.fields
            if rest or name != "port" or checker.norm(ty) != "i8":
                raise TurnRefused(f"{ctor.name} says which port the record goes to, as port: Int8, or says nothing")
            port = off
        verdicts.append((ctor.name, int(ctor.tag), port))
    return tuple(verdicts)


def _fill(handler: str, item: str, fields: list, slot: int, slot_type: str, parts: int, wire: int, capacity: int,
          k_out: int, verdict: TypeDecl) -> str:
    binders = [f"x{index}" for index in range(len(fields))]
    cells = [f"c{index}" for index in range(parts)]
    places = [f"q{index}" for index in range(1, k_out)]  # after the head

    def nest(names: list, inner: str) -> str:
        for name in reversed(names):
            inner = f"Cell@{name}({inner})"
        return inner

    peel, peeled = "", "d"
    for index in range(parts - 1):
        following = cells[index + 1] if index == parts - 2 else f"r{index}"
        peel += f"let ({cells[index]}, {following}) = peel({peeled}) in "
        peeled = following
    taken = cells + ["e"]
    take, pools = [], ["pool0"] + [f"p{index}" for index in range(len(taken))]
    for index, name in enumerate(taken):
        take.append(f"  match {pools[index]} with Dry => ({nest(taken[:index], 'Dry')}, spares0, ingress, head0, queue0, "
                    f"Short) | Cell({name}, {pools[index + 1]}) =>")
    built = cells[0] if parts == 1 else "(" + ", ".join(cells) + ")"
    out = [f"y{index}" for index in range(len(fields))]  # the record as the handler gave it back
    copy = f"{item}@{built}({', '.join(out)})"  # what the queue holds: ingress's record keeps its cells
    enqueue, head = [], "Pending@h(port, frame)"
    for index, place in enumerate(places):
        kept = [f"{places[j]} = Pending@e{j}(p{j}, f{j})" for j in range(index)]
        enqueue.append(f"  match {place} with None => ({head}, Queue@d {{ {', '.join(kept + [f'{place} = out'])} }}, "
                       f"None) | Pending(e{index}, p{index}, f{index}) =>")
    full = [f"{places[j]} = Pending@e{j}(p{j}, f{j})" for j in range(len(places))]
    enqueue.append(f"    ({head}, Queue@d {{ {', '.join(full)} }}, out)" + " end" * (len(places) + 2))
    shifted = [f"{places[j]} = {places[j + 1]}" for j in range(len(places) - 1)] + [f"{places[-1]} = None"]
    arms = []
    for ctor in verdict.ctors:
        if ctor.fields:
            arms.append(f"      | {ctor.name}(port) => let (head1, queue2, extra) = enqueue(head0, queue1, "
                        f"Pending@e(port, {copy})) in let (pool3, spares2) = give_out(extra, Cell@s(pool2), spares1) in "
                        f"(pool3, spares2, {item}@o {{ {fields[slot][0]} = fresh }}, head1, queue2, Queued({ctor.tag}))")
        else:
            arms.append(f"      | {ctor.name} => ({nest(taken, 'pool2')}, Spare@s(fresh, spares1), {item}@o {{}}, head0, "
                        f"queue1, Dropped({ctor.tag}))")
    with open(TEMPLATE) as handle:
        template = string.Template(handle.read())
    return template.substitute(
        slot_type=slot_type, item=item, k_out=k_out,
        queue_fields=", ".join(f"{place}: Out" for place in places),
        queue_pattern=f"Queue(d, {', '.join(places)})", first_place=places[0], last_place=places[-1],
        shifted=", ".join(shifted),
        enqueue_body="\n".join(enqueue), pattern=f"{item}(d, {', '.join(binders)})", slot_binder=binders[slot],
        peel_cells=peel, cells_to_pool=nest(cells, "pool") if parts > 1 else "Cell@d(pool)",
        take="\n".join(take), take_end=" end" * len(taken), after_take=pools[-1], taken_back=nest(taken, pools[-1]),
        wire=wire, capacity=capacity, handler=handler, out_pattern=f"{item}(o, {', '.join(out)})",
        done_arms="\n".join(arms))
