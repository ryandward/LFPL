"""The runtime's turn in LFPL, for one handler: src/turn.lfpl, filled in from the handler's own checked stream.

What the turn does is fixed, and written once in src/turn.lfpl. What depends on the handler is filled in here: the
record its item is, which field of it is the chain, the verdict it returns, and how many places the output queue has.
The result is LFPL, appended to the handler's program, and it goes through every stage the handler does. So the turn
is typed and its bound certified like any other code, and src/rtverilog.py lowers it with the handler.

It serves the shape examples/ipv4.lfpl has, and refuses any other with a reason: a handler with no state slots, that
draws nothing from its reserve, takes a record with one chain, and returns a verdict beside that same record.
"""
from __future__ import annotations

import io
import os
import string
from dataclasses import dataclass

from .certify import certify
from .rtcheck import Checker, TypeDecl
from .rtlower import rt_lines
from .runtime import BODY_PAYLOAD, Walk, _is_chain, _walk

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "turn.lfpl")
TURN = "serve"
LINK = "$next"
_TYPES = {"Pool", "Out", "Queue", "Turned"}
_CTORS = {"Dry", "Cell", "None", "Pending", "Idle", "Queued", "Dropped", "Runt", "TooLarge", "Short", "Full"}
_FUNCTIONS = {"give", "give_chain", "give_free", "give_frame", "give_out", "split", "unsplit", "refill", "is_full",
              "release", "enqueue", TURN}
# What the turn returns in its last result, by tag, which is the order src/turn.lfpl declares them in.
OUTCOMES = ("idle", "queued", "dropped", "runt", "too_large", "short", "full")


class TurnRefused(Exception):
    """The handler has a shape the turn does not serve yet."""


@dataclass(frozen=True)
class Shape:
    """What the hardware around the turn needs to know: where a frame's bytes go, and where the queue's head is."""
    handler: str
    item: str                 # the record's name
    item_walk: Walk           # its cells, scalars, links, chain and length, as src/runtime.py walks them
    parts: int                # the cells of the record itself
    wire: int                 # bytes of it on the wire
    capacity: int             # the most bytes a frame in ingress can have: what N_max cells hold
    n_max: int                # the most cells a frame takes in
    n_out: int                # and the most an output can hold going out
    k_out: int                # outputs the queue holds: its head and the places after it
    verdicts: tuple           # (name, tag, the port's offset in the register, or None)


def turn_source(program: bytes, *, n_max: int, k_out: int = 4) -> tuple[str, Shape]:
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
    bound = certify(checker, handler.name, n_max)
    if bound.a or bound.b:
        raise TurnRefused(f"the turn hands the handler an empty reserve, and {handler.name} draws "
                          f"R(n) = {bound.a} n + {bound.b}")
    walk = _walk(checker, decl.name, decl.ctors[0].fields, from_the_wire=True)
    fields, parts = _source_fields(checker, decl)
    if walk.cells != parts:
        raise TurnRefused(f"{decl.name} holds a record in a field, and the turn frees only its own cells and its chain")
    chain = [index for index, (_name, ty) in enumerate(fields) if checker._decl(checker.norm(ty)) is not None]
    if walk.chain is None or len(chain) != 1:
        raise TurnRefused(f"the turn takes a record with one chain, and {decl.name} has {len(chain)} linear fields")
    chain_decl = checker._decl(checker.norm(fields[chain[0]][1]))
    assert chain_decl is not None
    last, chunk, definition = _chain(checker, chain_decl, fields[chain[0]][1])
    _no_clash(checker, {last, chunk} if definition else set())
    verdicts = _verdicts(checker, checker.types[verdict_type[1]])
    capacity = walk.wire_bytes + BODY_PAYLOAD * (n_max - parts)
    if capacity + 1 >= 1 << 16:  # ingress counts a frame up to one byte past what it holds
        raise TurnRefused(f"section 8.2: N_max of {n_max} holds {capacity} bytes, and one more is past what a 16-bit "
                          f"length counts")
    text = _fill(handler.name, decl.name, fields, chain[0], chain_decl.name, last, chunk, definition, parts,
                 walk.wire_bytes, capacity, k_out, checker.types[verdict_type[1]])
    # The handler draws nothing, so what it sends is made of the cells it was given: no more than N_max.
    shape = Shape(handler.name, decl.name, walk, parts, walk.wire_bytes, capacity, n_max, n_max, k_out, verdicts)
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


def _chain(checker: Checker, decl: TypeDecl, ty: str) -> tuple[str, str, str]:
    """The chain's two constructors, and its definition if the program left it abstract."""
    if decl.kind == "abstract":
        bytes_ = ", ".join(f"b{index}: Int8" for index in range(BODY_PAYLOAD))
        return "Last", "Chunk", f"data {decl.name} = Last | Chunk(used: Int8, {bytes_}, next: {decl.name});\n"
    if not _is_chain(checker, ty):
        raise TurnRefused(f"{decl.name} is the chain, and it is not laid out as ingress fills one")
    null = next(c for c in decl.ctors if c.tag == "null")
    full = next(c for c in decl.ctors if c.tag != "null")
    return null.name, full.name, ""


def _no_clash(checker: Checker, more: set) -> None:
    taken = ({decl.name for decl in checker.types.values()}, {ctor.name for ctor in checker.ctors.values()},
             {fn.name for fn in checker.functions.values()})
    for names, ours in zip(taken, (_TYPES, _CTORS | more, _FUNCTIONS)):
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


def _fill(handler: str, item: str, fields: list, chain: int, chain_type: str, last: str, chunk: str, definition: str,
          parts: int, wire: int, capacity: int, k_out: int, verdict: TypeDecl) -> str:
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
        take.append(f"  match {pools[index]} with Dry => ({nest(taken[:index], 'Dry')}, slot, head0, queue0, Short) "
                    f"| Cell({name}, {pools[index + 1]}) =>")
    zeros = []
    for index, (_name, ty) in enumerate(fields):
        zeros.append("refill(fresh, rest)" if index == chain else "false" if ty == "bool" else "0")
    built = cells[0] if parts == 1 else "(" + ", ".join(cells) + ")"
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
                        f"Pending@e(port, out)) in (give_out(extra, pool2), slot1, head1, queue2, Queued)")
        else:
            arms.append(f"      | {ctor.name} => (give_frame(out, Cell@e(pool2)), slot1, head0, queue1, Dropped)")
    with open(TEMPLATE) as handle:
        template = string.Template(handle.read())
    return template.substitute(
        chain_definition=definition, item=item, k_out=k_out, chain_type=chain_type, last=last, chunk=chunk,
        queue_fields=", ".join(f"{place}: Out" for place in places),
        queue_pattern=f"Queue(d, {', '.join(places)})", first_place=places[0], last_place=places[-1],
        shifted=", ".join(shifted),
        enqueue_body="\n".join(enqueue), pattern=f"{item}(d, {', '.join(binders)})", chain_binder=binders[chain],
        chain_field=fields[chain][0], peel_cells=peel, cells_to_pool=nest(cells, "pool") if parts > 1 else "Cell@d(pool)",
        take="\n".join(take), take_end=" end" * len(taken), after_take=pools[-1], taken_back=nest(taken, "pool1"),
        wire=wire, capacity=capacity, handler=handler, fresh_record=f"{item}@{built}({', '.join(zeros)})",
        done_arms="\n".join(arms))
