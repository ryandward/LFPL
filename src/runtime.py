"""The fixed image and the serving loop: LFPL-O sections 3, 6 and 7, as x86-64 assembly.

This is the part LFPL-O section 11.3 says must be trusted, so it is meant to be read: no
libc, no allocator, and a handful of system calls. Everything the program will ever use is
laid out at link time, locked and touched at startup.

An item arrives without being parsed. `recvmsg` scatters the datagram straight into cells,
through an iovec that names each scalar field of the item's record and then the payload of
each body cell, so the wire format follows from the item's declared fields and no byte of it is
copied twice. Egress gathers the same way with `sendmsg`. Refusing an item costs nothing: its
cells never left ingress, so the next datagram simply overwrites them.

A field whose type is a record is that record's cell, which is how a record that spans cells gets
here, since src/spans.py has made it a chain of records by now. So an item or an output is a walk over
those cells, and its n counts all of them.

What the source does not say, this file decides:

    Wire format     an item is its record's scalar fields in declaration order, packed, then the
                    bytes of its one chain field. An IntN is little endian, and a BeN is as it
                    came, in network order, so the kernel writes the one as it does the other.
    Length          a field named `length`, of type Int16, is on the wire neither way. Admission
                    fills it in with how long the datagram was, because a program that does not
                    walk its body has no other way to find out.
    Chain type      a chain of cells: used count at 0, 7 payload bytes, next at 8, and null for
                    an empty body. A program can leave the type abstract, or declare exactly
                    that layout to walk it. Bytes past the used count read as zero, so nothing
                    from an earlier item can be seen.
    Slot            in place of a chain, a slot: the bytes after the scalars go into one, and the
                    record's field holds its number and how many they are. Slot k is 2048 k bytes
                    into the cell store, after the cells, as on the hardware target. Ingress holds
                    one, and admission gives it a free one in place of the one the item takes. A byte
                    a handler reads past the length is 0, whatever an earlier item left in it, because
                    that is what reading a slot means, and not because anything here clears it. An
                    item's n is its record's cells and its slot's bytes, which is what the bound counts.
    Output          a constructor with a field named `port` is sent to `out_base + port`,
                    as its other scalar fields followed by its chain. Any other constructor
                    sends nothing, and its cells go back to the pool.

The control port is an ingress source like any other, bound to the loopback address as the rest
are. `S` answers with the counters, walking the free list to count the pool instead of trusting
a counter. `P` pauses egress and `R` resumes it, for maintenance, and because loopback sends never
block, so without it nothing here could show that a full queue refuses. Each answers with the
counters too, so whoever sent it knows it has happened. `Q` drains and shuts down.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .rtcheck import SIZES, Bound, Checker, Ctor

PAGE = 4096
CELL = 16
BODY_PAYLOAD = 7     # bytes of one chain cell
LENGTH_FIELD = "length"
K_IN = 1             # recvmsg takes one datagram per turn: the socket's own buffer is the ring
O_MAX = 1            # a handler returns one output
CONTROL_EVERY = 64   # turns between looks at the control port, so a flood cannot starve it
RUNTIME_DEPTH = 48   # the deepest the loop's own calls go: control, drain, free_chain, push_cell
IDLE_MICROSECONDS = 1000
MAX_IMAGE_BYTES = 1 << 30  # the code reaches the image with 32-bit displacements, which stop at 2 GiB
WORD_LIMIT = 1 << 63       # the admission table and the arithmetic on it are 64-bit, signed in imul
IOV_MAX = 1024             # what one sendmsg takes: a longer chain counts as a send error
SOL_SOCKET = 1
SO_TIMESTAMPNS = 35        # and the type of its control message: a timespec of CLOCK_REALTIME
SO_RXQ_OVFL = 40           # and of its: an unsigned 32-bit count of what the socket has dropped
CONTROL_BYTES = 64         # a timestamp takes 32, and a count 24
# admitted_cells adds up n, as admission worked it out from each datagram's length. budget_steps
# adds up W(n) for those. steps is what ran, in a build that counts them, and is otherwise 0.
# kernel_dropped is what the ingress socket has said it dropped, which arrives on the next datagram
# it delivers, so it lags until there is one. max_age_ns is the longest an admitted item waited, from
# the kernel's timestamp to admission, in a build that asks for it, since that takes a system call
# for each item, and is otherwise 0. Nothing refuses on it yet.
COUNTERS = ("capacity", "pool", "served", "forwarded", "dropped", "too_large", "short", "full",
            "malformed", "send_errors", "recv_errors", "admitted_cells", "budget_steps", "steps",
            "kernel_dropped", "max_age_ns", "slot_capacity", "slots")
# slot_capacity and slots are the free slots when nothing is held, and now, counted as the pool is. After them, one
# counter for each value of a handler's verdict, in the order its enum declares them.
PARAMS = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
RESULTS = ("rax", "rdx", "rcx", "rsi", "rdi", "r8")
_WIDTH = {1: "byte", 2: "word", 4: "dword", 8: "qword"}


class BindingError(Exception):
    """The handler's declared types do not fit what this runtime can receive or send. Rule text
    starts with the section that declares the limit."""


@dataclass(frozen=True)
class StateSlot:
    name: str
    length: int
    element_size: int
    initial: Optional[str]  # a file of exactly length * element_size bytes, or None for zeros


@dataclass(frozen=True)
class Walk:
    """The cells of a record or an output as this runtime goes through them, numbered from the one it
    starts at. `scalars` are in wire order, which is declaration order, going into a record where it is met."""
    cells: int
    scalars: tuple[tuple[int, int, int], ...]  # cell, offset, size
    links: tuple[tuple[int, int, int], ...]    # cell, offset, the cell the field points to
    chain: Optional[tuple[int, int]]           # cell, offset
    length: Optional[tuple[int, int]]          # cell, offset
    slot: Optional[tuple[int, int]] = None     # cell, offset: in place of a chain

    @property
    def wire_bytes(self) -> int:
        return sum(size for _cell, _off, size in self.scalars)


@dataclass(frozen=True)
class OutCtor:
    name: str
    tag: Optional[int]       # None: the null constructor of a tagless type
    port: Optional[tuple[int, int]]  # offset, size
    walk: Walk


@dataclass(frozen=True)
class Options:
    in_port: int
    out_base: int
    ctl_port: int
    n_max: int = 216     # the largest item in cells: a record of 1 and 215 of 7 bytes, for a 1500 MTU
    k_out: int = 64      # the output queue, a power of two
    cells: Optional[int] = None  # None: section 3.4's C. Fewer can make admission refuse with Short
    measure_age: bool = False    # ask for each datagram's arrival time, and read the clock at admission

    def __post_init__(self) -> None:
        for name, port in (("ingress", self.in_port), ("the control port", self.ctl_port)):
            if not 1 <= port <= 65535:
                raise BindingError(f"section 7.3: {name} at {port} is not a port")
        if not 1 <= self.out_base <= 65535 - 255:
            raise BindingError(f"section 7.3: egress at {self.out_base} plus a port number can "
                               f"pass 65535")
        if self.n_max < 1:
            raise BindingError(f"section 8.2: N_max of {self.n_max}: an item is at least one cell")
        if self.k_out < 1 or self.k_out & self.k_out - 1:
            raise BindingError(f"section 8.2: K_out of {self.k_out} must be a power of two")


@dataclass(frozen=True)
class Image:
    handler: str
    states: tuple[StateSlot, ...]
    item: Walk
    out_tagged: bool
    out: tuple[OutCtor, ...]
    bound: Bound
    depth: int
    options: Options
    verdict: bool = False  # what goes out is a record, and which way it goes is an enum in a register beside it
    slot_bytes: int = 0    # a slot's size, if the item holds one

    @property
    def largest_output(self) -> int:
        """N_out: an output can hold every cell of its item and of its reserve."""
        return self.in_cells + self.reserve_at_most

    @property
    def slotted(self) -> bool:
        return self.item.slot is not None

    @property
    def in_cells(self) -> int:
        """The cells ingress holds: N_max for a chain, and the record's own for a slot, which holds the rest."""
        return self.item.cells if self.slotted else self.options.n_max

    @property
    def n_largest(self) -> int:
        """The largest n: N_max cells, or the record's cells and a full slot's bytes."""
        return self.item.cells + self.slot_bytes if self.slotted else self.options.n_max

    @property
    def slot_count(self) -> int:
        """Ingress's, one for each output the queue holds, and the one a new item takes before ingress gets another."""
        return self.options.k_out + 1 if self.slotted else 0

    @property
    def first_slot(self) -> int:
        """Slots come after the cells, and slot k starts 2048 k bytes into the store."""
        return (CELL * self.cell_count + self.slot_bytes - 1) // self.slot_bytes if self.slotted else 0

    @property
    def cell_count(self) -> int:
        """Section 3.4: C = K_in N_max + (R(N_max) + N_max) + K_out O_max N_out + S_cap. At this
        size Short cannot happen: a queue with room always leaves a maximal transaction's cells."""
        n_max = self.in_cells
        if self.options.cells is not None:
            return self.options.cells
        return (K_IN * n_max + self.reserve_at_most + n_max
                + self.options.k_out * O_MAX * self.largest_output)

    @property
    def reserve_at_most(self) -> int:
        """R(N_max)."""
        return self.bound.a * self.n_largest + self.bound.b

    @property
    def capacity(self) -> int:
        return self.cell_count - K_IN * self.in_cells


    @property
    def stack_bytes(self) -> int:
        needed = max(self.depth + 8, RUNTIME_DEPTH)
        return (needed + PAGE - 1) // PAGE * PAGE

    @property
    def body_cells(self) -> int:
        return self.options.n_max - self.item.cells if self.item.chain is not None else 0

    @property
    def out_cells(self) -> int:
        return max(ctor.walk.cells for ctor in self.out)

    @property
    def out_scalars(self) -> int:
        return max(len(ctor.walk.scalars) for ctor in self.out)

    @property
    def bytes(self) -> int:
        return (self.store_bytes + self.stack_bytes + sum(s.length * s.element_size for s in self.states))

    @property
    def store_bytes(self) -> int:
        """The cells, and the slots after them."""
        if self.slotted:
            return (self.first_slot + self.slot_count) * self.slot_bytes
        return self.cell_count * CELL


def handler_slots(checker: Checker, initial: dict[str, str], *, verdicts: bool = False) -> tuple:
    """The handler, the type of its item and of its output, and its state slots, which are the same
    whichever device the items come from. With a verdict, the output is the record it is about."""
    fn = next(f for f in checker.functions.values() if f.handler)
    item_type = checker.norm(fn.params[-2])
    results = checker.tuples[checker._resolve(fn.ret)]
    if len(results) == len(checker.states) + 3 and not verdicts:
        raise BindingError("section 3.1: this runtime sends an output, and the handler returns a verdict")
    out_type = checker.norm(results[-2])
    states = []
    for name, ty in checker.states:
        norm = checker.norm(ty)
        assert isinstance(norm, tuple) and norm[0] == "array"
        states.append(StateSlot(name, norm[1], SIZES[norm[2]], initial.pop(name, None)))
    if initial:
        raise BindingError(f"no state slot named {sorted(initial)[0]}")
    if len(fn.params) != len(states) + 2:
        raise BindingError(f"section 3.1: {fn.name} takes {len(fn.params) - 2} state slots, and the "
                           f"program declares {len(states)}: a slot declared after the handler is "
                           f"one it never receives")
    return fn, item_type, out_type, tuple(states)


def bind(checker: Checker, handler_name: str, depth: int, options: Options,
         initial: dict[str, str]) -> Image:
    """Read the handler's types from the checked stream, and refuse what cannot be served."""
    fn, item_type, out_type, states = handler_slots(checker, initial, verdicts=True)
    _check_reserve(checker, fn.params[-1])
    decl = checker._decl(item_type)
    if decl is None or decl.kind != "record":
        raise BindingError(f"section 3.1: this runtime receives an item as a record, and "
                           f"{item_type} is not one")
    item = _walk(checker, decl.name, decl.ctors[0].fields, from_the_wire=True)
    if options.n_max < item.cells:
        raise BindingError(f"section 8.2: N_max of {options.n_max}, and {decl.name} alone is {item.cells} cells")
    slot_bytes = next((t.bytes for t in checker.types.values() if t.kind == "slot"), 0) if item.slot else 0
    out_decl = checker._decl(out_type)
    verdict = _verdict_of(checker, fn, states)
    if verdict is not None:
        if out_decl is None or out_decl.kind != "record":
            raise BindingError(f"section 3.1: a verdict goes with a record, and {out_type} is not one")
        walk = _walk(checker, out_decl.name, out_decl.ctors[0].fields)
        out = tuple(_verdict_ctor(checker, ctor, walk) for ctor in verdict.ctors)
    elif out_decl is None or out_decl.kind not in ("tagged", "tagless", "record"):
        raise BindingError(f"section 3.1: the output {out_type} is not a data type")
    else:
        out = tuple(_out_ctor(checker, ctor) for ctor in out_decl.ctors)
    for ctor in out:
        if len(ctor.walk.scalars) > IOV_MAX:
            raise BindingError(f"section 8.2: {ctor.name} has {len(ctor.walk.scalars)} fields to send, "
                               f"and one message gathers {IOV_MAX}")
    image = Image(handler_name, states, item, verdict is not None or out_decl.kind == "tagged", out,
                  checker.bounds(handler_name), depth, options, verdict is not None, slot_bytes)
    if item.slot is not None and any(ctor.walk.slot is None for ctor in out if ctor.port is not None):
        raise BindingError("section 3.1: an output sent from a slot item sends its slot")
    return image


def largest_n(checker: Checker, handler_name: str, options: Options) -> int:
    """The n a handler's bound is certified at: N_max cells, or for an item that holds a slot, its record's cells
    and a full slot's bytes, since n counts both."""
    if checker.held_slots(handler_name):
        cells = checker.record_cells(handler_name)
        slot_bytes = next(t.bytes for t in checker.types.values() if t.kind == "slot")
        if cells is not None:
            return cells + slot_bytes
    return options.n_max


def _verdict_of(checker: Checker, fn, states: tuple) -> Optional[TypeDecl]:
    """The enum a handler returns ahead of its record, if it returns one."""
    results = checker.tuples[checker._resolve(fn.ret)]
    if len(results) != len(states) + 3:
        return None
    ty = checker.norm(results[len(states)])
    assert isinstance(ty, tuple) and ty[0] == "enum", ty  # rtcheck types a handler's shape
    return checker.types[ty[1]]


def _verdict_ctor(checker: Checker, ctor: Ctor, walk: Walk) -> OutCtor:
    """Each value of the verdict sends the record to the port it names, or gives it back."""
    if not ctor.fields:
        return OutCtor(ctor.name, int(ctor.tag), None, walk)
    (off, size, ty, name), *rest = ctor.fields
    if rest or name != "port" or checker.norm(ty) != "i8":
        raise BindingError(f"section 3.1: {ctor.name} says which port the record goes to, as port: Int8, or says "
                           f"nothing and the record is given back")
    return OutCtor(ctor.name, int(ctor.tag), (off, size), walk)


def _check_reserve(checker: Checker, reserve: str) -> None:
    """The pool and every reserve are threaded through offset 0, as the built-in Free is. A
    program may still declare its own Free, and then it has to have that shape."""
    ty = checker.norm(reserve)
    decl = checker._decl(ty)
    assert decl is not None
    full = [c for c in decl.ctors if c.tag != "null"]
    fields = full[0].fields if len(full) == 1 else []
    if (decl.kind != "tagless" or len(fields) != 1 or fields[0][:2] != (0, 8)
            or checker.norm(fields[0][2]) != ty):
        raise BindingError("section 3.1: Free must be End | More(next: Free)")


def _is_chain(checker: Checker, ty: str) -> bool:
    """A type the runtime can build from a datagram: abstract, or declared with the very layout it
    uses, which is what lets a program walk it."""
    norm = checker.norm(ty)
    decl = checker._decl(norm)
    if decl is None:
        return False
    if decl.kind == "abstract":
        return True
    full = [c for c in decl.ctors if c.tag != "null"]
    if decl.kind != "tagless" or len(full) != 1 or len(full[0].fields) != BODY_PAYLOAD + 2:
        return False
    fields = full[0].fields
    bytes_in_order = all(field[:3] == (position, 1, "i8") for position, field in enumerate(fields[:-1]))
    return bytes_in_order and fields[-1][:2] == (8, 8) and checker.norm(fields[-1][2]) == norm


def _walk(checker: Checker, owner: str, fields: list, *, from_the_wire: bool = False) -> Walk:
    """Where everything is, going through these fields and the record of any field that is one."""
    scalars: list = []
    links: list = []
    found: dict[str, Optional[tuple[int, int]]] = {"chain": None, "length": None, "slot": None}
    cells = 1

    def visit(cell: int, owner: str, fields: list, within: frozenset) -> None:
        nonlocal cells
        for off, size, ty, name in fields:
            norm = checker.norm(ty)
            decl = checker._decl(norm)
            if name == LENGTH_FIELD and norm == "i16":
                if found["length"] is not None:
                    raise BindingError(f"section 3.1: {owner}.{name} is a second field saying how long it is")
                found["length"] = (cell, off)
            elif isinstance(norm, tuple) and norm[0] == "fin" and from_the_wire:
                # A Fin is what makes an array read need no bounds check, and only idx may make one.
                # Filled from the wire, it would let a sender choose an address.
                raise BindingError(f"section 4.5: {owner}.{name} is a Fin, and an item's fields "
                                   f"come from the wire: take an integer and use idx")
            elif decl is None:
                scalars.append((cell, off, size))
            elif decl.kind == "record":
                if decl.name in within:
                    raise BindingError(f"section 3.1: {owner}.{name} holds a {decl.name} inside a {decl.name}, "
                                       f"which has no end this runtime could find on the wire")
                inner = cells
                cells += 1
                links.append((cell, off, inner))
                visit(inner, decl.name, decl.ctors[0].fields, within | {decl.name})
            elif _is_chain(checker, ty) and found["chain"] is None and found["slot"] is None:
                found["chain"] = (cell, off)
            elif decl.kind == "slot" and found["chain"] is None and found["slot"] is None:
                found["slot"] = (cell, off)
            else:
                raise BindingError(f"section 3.1: {owner}.{name} is {decl.name}, and this runtime "
                                   f"carries scalars, records, and one chain or one slot")

    visit(0, owner, fields, frozenset((owner,)))
    return Walk(cells, tuple(scalars), tuple(links), found["chain"], found["length"], found["slot"])


def _out_ctor(checker: Checker, ctor: Ctor) -> OutCtor:
    port = None
    rest = []
    for entry in ctor.fields:
        if entry[3] == "port":
            if checker.norm(entry[2]) != "i8":
                raise BindingError(f"section 3.1: {ctor.name}.port must be Int8")
            port = (entry[0], entry[1])
        else:
            rest.append(entry)
    tag = int(ctor.tag) if ctor.tag.isdigit() else (None if ctor.tag == "null" else 0)
    return OutCtor(ctor.name, tag, port, _walk(checker, ctor.name, rest))


def _be16(port: int) -> str:
    return f"{port >> 8}, {port & 255}"


def _is_null(ctor: OutCtor, tagged: bool) -> bool:
    return ctor.tag is None and not tagged


def _out_cell(add: Callable[[str], None], register: str, cell: int) -> None:
    """A cell of the output in r12, into a register. The first is r12's, and _read_out_cells has the others."""
    add(f"    mov {register}, {'r12' if cell == 0 else f'[out_cells+{8 * cell}]'}")


def _read_out_cells(add: Callable[[str], None], walk: Walk) -> None:
    for cell, off, inner in walk.links:
        _out_cell(add, "rax", cell)
        add(f"    mov rax, [rax+{off}]")
        add(f"    mov [out_cells+{8 * inner}], rax")


def _free_all_but_the_first(add: Callable[[str], None], walk: Walk) -> None:
    """The slot or the chain first, because a cell's next field is the first thing giving it back overwrites."""
    if walk.slot is not None:
        _out_cell(add, "rax", walk.slot[0])
        add(f"    mov eax, [rax+{walk.slot[1]}]")
        add("    call push_slot")
    if walk.chain is not None:
        _out_cell(add, "rdi", walk.chain[0])
        add(f"    mov rdi, [rdi+{walk.chain[1]}]")
        add("    call free_chain")
    for _cell, _off, inner in walk.links:
        add(f"    mov rdi, [out_cells+{8 * inner}]")
        add("    call push_cell")


def assembly(image: Image) -> str:
    """The runtime and the image, as NASM, to follow the compiled functions."""
    o = image.options
    item = image.item
    item_count = len(item.scalars)
    iov_in = item_count + (1 if image.slotted else image.body_cells)
    if iov_in > IOV_MAX:
        raise BindingError("section 8.2: N_max is past what one recvmsg can scatter")
    if image.bytes > MAX_IMAGE_BYTES:
        raise BindingError(f"section 8.2: {image.handler} needs an image of {image.bytes} bytes, and "
                           f"this runtime takes at most {MAX_IMAGE_BYTES}")
    if image.bound.c * image.n_largest + image.bound.d >= WORD_LIMIT:
        raise BindingError(f"section 8.2: W(N_max) of {image.handler} is "
                           f"{image.bound.c * image.n_largest + image.bound.d} steps, which admission's "
                           f"64-bit arithmetic cannot hold")
    smallest = K_IN * image.in_cells + image.reserve_at_most + image.in_cells
    if image.cell_count < smallest:
        raise BindingError(f"section 3.4: {image.cell_count} cells cannot hold ingress and one "
                           f"maximal transaction, which take {smallest}")
    lines: list[str] = []
    add = lines.append
    out_iov = min(IOV_MAX, image.out_scalars + max(1, image.largest_output))
    k = len(image.states)

    add("section .text")
    add("global _start")
    add(f"handler_entry equ fn_{image.handler}")
    fixed_image_startup(add)
    add("    ; 5: ingress takes the first N_max cells, or its record's for a slot, and the rest are the pool")
    add("    mov edi, cells")
    for cell in range(item.cells):
        add(f"    lea rax, [rdi+{CELL * cell}]")
        add(f"    mov [slot_cells+{8 * cell}], rax")
        add(f"    call point_cell_{cell}")
    for position, (_cell, _off, size) in enumerate(item.scalars):
        add(f"    mov qword [in_iov+{16 * position + 8}], {size}")
    add(f"    lea rax, [rdi+{CELL * item.cells}]")
    add(f"    mov ecx, {image.body_cells}")
    add(f"    mov edx, in_iov+{16 * item_count}")
    add(".body_iovec:")
    add("    test rcx, rcx")
    add("    jz .body_done")
    add("    lea rbx, [rax+1]")
    add("    mov [rdx], rbx")
    add(f"    mov qword [rdx+8], {BODY_PAYLOAD}")
    add("    add rax, 16")
    add("    add rdx, 16")
    add("    dec rcx")
    add("    jmp .body_iovec")
    add(".body_done:")
    if image.slotted:
        add("    ; and the first slot, and a stack of the free ones")
        add(f"    mov qword [in_slot], {image.first_slot}")
        add(f"    mov qword [in_iov+{16 * item_count}], cells+{image.slot_bytes * image.first_slot}")
        add(f"    mov qword [in_iov+{16 * item_count + 8}], {image.slot_bytes}")
        add(f"    mov eax, {image.first_slot + 1}")
        add("    xor edx, edx")
        add(".free_slots:")
        add("    mov [slot_free+rdx*8], rax")
        add("    inc rax")
        add("    inc rdx")
        add(f"    cmp rdx, {image.slot_count - 1}")
        add("    jb .free_slots")
        add("    mov [slot_free_count], rdx")
        add("    mov [cnt_slot_capacity], rdx")
    add(f"    mov eax, cells+{CELL * image.in_cells}")
    add("    xor esi, esi")
    add(f"    mov ecx, {image.capacity}")
    add(".thread:")
    add("    mov [rax], rsi")
    add("    mov rsi, rax")
    add("    add rax, 16")
    add("    dec rcx")
    add("    jnz .thread")
    add("    mov [pool], rsi")
    add(f"    mov qword [pool_count], {image.capacity}")
    add(f"    mov qword [cnt_capacity], {image.capacity}")
    for position, state in enumerate(image.states):
        add(f"    mov qword [state_{position}], array_{state.name}")
    add("    ; 6: open the endpoints")
    add("    mov eax, 41")
    add("    mov edi, 2")
    add("    mov esi, 2")
    add("    xor edx, edx")
    add("    syscall")
    add("    test rax, rax")
    add("    js .refuse_endpoints")
    add("    mov [in_fd], rax")
    for option in (SO_RXQ_OVFL, SO_TIMESTAMPNS) if o.measure_age else (SO_RXQ_OVFL,):
        # before bind, so that no datagram goes without
        add("    mov eax, 54")
        add("    mov rdi, [in_fd]")
        add(f"    mov esi, {SOL_SOCKET}")
        add(f"    mov edx, {option}")
        add("    mov r10d, option_on")
        add("    mov r8d, 4")
        add("    syscall")
        add("    test rax, rax")
        add("    jnz .refuse_endpoints")
    add("    mov rdi, [in_fd]")
    add("    mov eax, 49")
    add("    mov esi, in_addr")
    add("    mov edx, 16")
    add("    syscall")
    add("    test rax, rax")
    add("    jnz .refuse_endpoints")
    add("    mov eax, 54")
    add("    mov rdi, [in_fd]")
    add("    mov esi, 1")
    add("    mov edx, 20")
    add("    mov r10d, idle_time")
    add("    mov r8d, 16")
    add("    syscall")
    add("    test rax, rax")
    add("    jnz .refuse_endpoints")
    add("    mov eax, 41")
    add("    mov edi, 2")
    add("    mov esi, 2")
    add("    xor edx, edx")
    add("    syscall")
    add("    test rax, rax")
    add("    js .refuse_endpoints")
    add("    mov [out_fd], rax")
    add("    mov eax, 41")
    add("    mov edi, 2")
    add("    mov esi, 2")
    add("    xor edx, edx")
    add("    syscall")
    add("    test rax, rax")
    add("    js .refuse_endpoints")
    add("    mov [ctl_fd], rax")
    add("    mov rdi, rax")
    add("    mov eax, 49")
    add("    mov esi, ctl_addr")
    add("    mov edx, 16")
    add("    syscall")
    add("    test rax, rax")
    add("    jnz .refuse_endpoints")
    add("    jmp serve")
    for label, status in ((".refuse_signals", 1), (".refuse_lock", 2), (".refuse_guard", 4),
                          (".refuse_endpoints", 6)):
        add(f"{label}:")
        add("    mov eax, 231")
        add(f"    mov edi, {status}")
        add("    syscall")
    add("")
    add("; ---------------------------------------------------- the serving loop, section 3.3")
    add("serve:")
    add("    call drain")
    add("    inc qword [turn]")
    add(f"    test qword [turn], {CONTROL_EVERY - 1}")
    add("    jnz .arrive")
    add("    call control")
    add(".arrive:")
    add("    ; ARRIVE: the device writes one item into the slot, or the wait ends with none")
    add(f"    mov qword [in_msg+40], {CONTROL_BYTES}")
    if o.measure_age:
        add("    mov qword [arrival_sec], 0")
    add("    mov eax, 47")
    add("    mov rdi, [in_fd]")
    add("    mov esi, in_msg")
    add("    xor edx, edx")
    add("    syscall")
    add("    test rax, rax")
    add("    js .no_item")
    add("    call read_control")
    add("    ; Section 6.1, in order. A refusal counts itself and leaves the cells in ingress.")
    add("    test dword [in_msg+48], 32")
    add("    jnz .too_large")
    add(f"    cmp rax, {item.wire_bytes}")
    add("    jb .malformed")
    add(f"    sub rax, {item.wire_bytes}")
    add("    mov r12, rax")
    if image.slotted:
        add("    ; n is the record's cells and the slot's bytes: the record's cells are new ones, and so is a slot")
        add("    xor r13d, r13d")
    else:
        if item.chain is None:
            add("    test rax, rax")
            add("    jnz .too_large")
        add(f"    add rax, {BODY_PAYLOAD - 1}")
        add("    xor edx, edx")
        add(f"    mov ecx, {BODY_PAYLOAD}")
        add("    div rcx")
        add("    mov r13, rax")
    add(f"    lea r14, [rax+{item.cells}]")
    add("    mov rax, [admission]")
    add("    imul rax, r14")
    add("    add rax, [admission+8]")
    add("    mov r15, rax")
    add(f"    add rax, {item.cells}" if image.slotted else "    add rax, r14")  # R(n), and the cells the item takes
    add("    cmp [pool_count], rax")
    add("    jb .short")
    add("    mov rax, [q_count]")
    add(f"    add rax, {O_MAX}")
    add(f"    cmp rax, {image.options.k_out}")
    add("    ja .full")
    if image.slotted:
        add("    ; a free slot for ingress to take, last, as the turn takes a spare: a full queue holds every other one")
        add("    cmp qword [slot_free_count], 0")
        add("    je .short")
    if o.measure_age:
        add("    call note_age")
    add("    ; n comes from the datagram's length, and R(n) and W(n) from the table")
    add("    add [cnt_admitted_cells], r14")
    add("    mov rax, [admission+16]")
    add("    imul rax, r14")
    add("    add rax, [admission+24]")
    add("    add [cnt_budget_steps], rax")
    add("    ; ADMIT: the item's cells leave ingress, and fresh ones from the pool take their place")
    add("    test r13, r13")
    add("    jz .linking")
    add("    mov rax, r13")
    add("    dec rax")
    add(f"    imul rcx, rax, {BODY_PAYLOAD}")
    add("    mov rdx, r12")
    add("    sub rdx, rcx")
    add("    shl rax, 4")
    add(f"    mov r10, [in_iov+{16 * item_count}+rax]")
    add("    dec r10")
    add("    ; the last cell keeps its used count and that many bytes: the rest of it is an old item's")
    add("    lea rcx, [rdx+1]")
    add("    shl rcx, 3")
    add("    cmp rcx, 64")
    add("    jae .linking")
    add("    mov eax, 1")
    add("    shl rax, cl")
    add("    dec rax")
    add("    and [r10], rax")
    add(".linking:")
    add("    xor r8d, r8d")
    add("    mov rcx, r13")
    add(".link:")
    add("    test rcx, rcx")
    add("    jz .linked")
    add("    dec rcx")
    add("    mov rax, rcx")
    add("    shl rax, 4")
    add(f"    lea rbx, [in_iov+{16 * item_count}+rax]")
    add("    mov rdi, [rbx]")
    add("    dec rdi")
    add(f"    mov byte [rdi], {BODY_PAYLOAD}")
    add("    mov [rdi+8], r8")
    add("    mov r8, rdi")
    add("    call pop_cell")
    add("    inc rax")
    add("    mov [rbx], rax")
    add("    jmp .link")
    add(".linked:")
    add("    test r13, r13")
    add("    jz .body_counted")
    add("    mov [r10], dl")
    add(".body_counted:")
    for cell, off, inner in item.links:
        add(f"    mov rax, [slot_cells+{8 * inner}]")
        add(f"    mov rcx, [slot_cells+{8 * cell}]")
        add(f"    mov [rcx+{off}], rax")
    if item.chain is not None:
        add(f"    mov rcx, [slot_cells+{8 * item.chain[0]}]")
        add(f"    mov [rcx+{item.chain[1]}], r8")
    if item.slot is not None:
        add("    ; the slot ingress filled is the item's: its number, and how many bytes it holds")
        add("    mov rax, r12")
        add("    shl rax, 32")
        add("    or rax, [in_slot]")
        add(f"    mov rcx, [slot_cells+{8 * item.slot[0]}]")
        add(f"    mov [rcx+{item.slot[1]}], rax")
        add("    ; and ingress takes a free one")
        add("    dec qword [slot_free_count]")
        add("    mov rax, [slot_free_count]")
        add("    mov rax, [slot_free+rax*8]")
        add("    mov [in_slot], rax")
        add(f"    shl rax, {image.slot_bytes.bit_length() - 1}")
        add("    add rax, cells")
        add(f"    mov [in_iov+{16 * item_count}], rax")
    if item.length is not None:
        add(f"    lea rax, [r12+{item.wire_bytes}]")
        add(f"    mov rcx, [slot_cells+{8 * item.length[0]}]")
        add(f"    mov word [rcx+{item.length[1]}], ax")
    add("    mov r14, [slot_cells]")
    for cell in range(item.cells):
        add("    call pop_cell")
        add(f"    mov [slot_cells+{8 * cell}], rax")
        add(f"    call point_cell_{cell}")
    add("    ; the item's reserve: R(n) cells, as a Free")
    add("    xor edx, edx")
    add(".reserve:")
    add("    test r15, r15")
    add("    jz .reserved")
    add("    dec r15")
    add("    call pop_cell")
    add("    mov [rax], rdx")
    add("    mov rdx, rax")
    add("    jmp .reserve")
    add(".reserved:")
    add(f"    mov {PARAMS[k + 1]}, rdx")
    add(f"    mov {PARAMS[k]}, r14")
    for position in range(k):
        add(f"    mov {PARAMS[position]}, [state_{position}]")
    add(f"    call handler_entry")
    if image.verdict:
        add("    ; DONE: the state, the verdict, the record and what is left of the reserve come back")
    else:
        add("    ; DONE: the state, the output and what is left of the reserve come back")
    for position in range(k):
        add(f"    mov [state_{position}], {RESULTS[position]}")
    if image.verdict:
        add(f"    mov r15, {RESULTS[k]}")
    add(f"    mov r12, {RESULTS[k + image.verdict]}")
    add(f"    mov r13, {RESULTS[k + image.verdict + 1]}")
    add(".give_back:")
    add("    test r13, r13")
    add("    jz .given_back")
    add("    mov rdi, r13")
    add("    mov r13, [r13]")
    add("    call push_cell")
    add("    jmp .give_back")
    add(".given_back:")
    add("    inc qword [cnt_served]")
    if image.verdict:  # its tag is the register's low byte
        for position, ctor in enumerate(image.out[:-1]):
            add(f"    cmp r15b, {ctor.tag}")
            add(f"    je .out_{position}")
        add(f"    jmp .out_{len(image.out) - 1}")
    else:
        _dispatch(add, image, "r12", ".out")
    for position, ctor in enumerate(image.out):
        add(f".out_{position}:")
        if image.verdict:
            add(f"    inc qword [cnt_verdict_{position}]")
        if ctor.port is not None:
            add("    inc qword [cnt_forwarded]")
            add("    mov rax, [q_head]")
            add("    add rax, [q_count]")
            add(f"    and rax, {image.options.k_out - 1}")
            add("    mov [queue+rax*8], r12")
            if image.verdict:  # the port is in the register, and the register does not wait in the queue
                add("    mov rcx, r15")
                add(f"    shr rcx, {8 * ctor.port[0]}")
                add("    mov [queue_ports+rax], cl")
            add("    inc qword [q_count]")
            add("    jmp serve")
        else:
            add("    inc qword [cnt_dropped]")
            if not _is_null(ctor, image.out_tagged):
                _read_out_cells(add, ctor.walk)
                _free_all_but_the_first(add, ctor.walk)
            if not _is_null(ctor, image.out_tagged):
                add("    mov rdi, r12")
                add("    call push_cell")
            add("    jmp serve")
    for label, counter in ((".too_large", "too_large"), (".malformed", "malformed"),
                           (".short", "short"), (".full", "full")):
        add(f"{label}:")
        add(f"    inc qword [cnt_{counter}]")
        add("    jmp serve")
    add(".no_item:")
    add("    cmp rax, -11")
    add("    je .idle")
    add("    inc qword [cnt_recv_errors]")
    add(".idle:")
    add("    call control")
    add("    jmp serve")
    add("")
    add("; The control messages of the recvmsg just done. Each is a length, a level and a type, then")
    add("; its data, and the next starts at the next multiple of 8. Keeps rax.")
    add("read_control:")
    add("    mov edi, in_control")
    add("    mov rsi, [in_msg+40]")
    add("    add rsi, rdi")
    add(".next:")
    add("    lea rdx, [rdi+16]")
    add("    cmp rdx, rsi")
    add("    ja .done")
    add("    mov rcx, [rdi]")
    add("    cmp rcx, 16")
    add("    jb .done")
    add(f"    cmp dword [rdi+8], {SOL_SOCKET}")
    add("    jne .advance")
    add(f"    cmp dword [rdi+12], {SO_RXQ_OVFL}")
    if o.measure_age:
        add("    je .drops")
        add(f"    cmp dword [rdi+12], {SO_TIMESTAMPNS}")
        add("    jne .advance")
        add("    mov rdx, [rdi+16]")
        add("    mov [arrival_sec], rdx")
        add("    mov rdx, [rdi+24]")
        add("    mov [arrival_nsec], rdx")
        add("    jmp .advance")
        add(".drops:")
    else:
        add("    jne .advance")
    add("    ; The socket's count since it was made, which wraps at 32 bits: so the 32-bit difference.")
    add("    mov edx, [rdi+16]")
    add("    mov r8d, edx")
    add("    sub edx, [last_drops]")
    add("    add [cnt_kernel_dropped], rdx")
    add("    mov [last_drops], r8d")
    add(".advance:")
    add("    add rcx, 7")
    add("    and rcx, -8")
    add("    add rdi, rcx")
    add("    jmp .next")
    add(".done:")
    add("    ret")
    add("; end read_control")
    add("")
    if o.measure_age:
        add("; An admitted item's arrival, kept while it is the item being served, and how long it waited.")
        add("; Nothing refuses on this yet. Keeps r12 to r15, which admission is using.")
        add("note_age:")
        add("    mov rdx, [arrival_sec]")
        add("    test rdx, rdx")
        add("    jz .done")
        add("    mov [item_arrival_sec], rdx")
        add("    mov rdx, [arrival_nsec]")
        add("    mov [item_arrival_nsec], rdx")
        add("    mov eax, 228")
        add("    xor edi, edi")
        add("    mov esi, clock_now")
        add("    syscall")
        add("    mov rax, [clock_now]")
        add("    sub rax, [item_arrival_sec]")
        add("    imul rax, rax, 1000000000")
        add("    add rax, [clock_now+8]")
        add("    sub rax, [item_arrival_nsec]")
        add("    jns .known")
        add("    xor eax, eax")
        add("    ; the clock was stepped back between the two: a negative age is 0")
        add(".known:")
        add("    cmp rax, [cnt_max_age_ns]")
        add("    jbe .done")
        add("    mov [cnt_max_age_ns], rax")
        add(".done:")
        add("    ret")
        add("; end note_age")
        add("")
    add("; DRAIN: send what is queued, until it is empty or the socket would block")
    add("drain:")
    add("    cmp qword [egress_paused], 0")
    add("    jne .done")
    add("    cmp qword [q_count], 0")
    add("    je .done")
    add("    mov rax, [q_head]")
    add("    mov r12, [queue+rax*8]")
    if image.verdict:  # one record, whichever value sent it, and its port beside it in the queue
        add("    movzx eax, byte [queue_ports+rax]")
        _send(add, image, 0, image.out[0].walk, out_iov)
    else:
        _dispatch(add, image, "r12", ".send")
        for position, ctor in enumerate(image.out):
            add(f".send_{position}:")
            if ctor.port is None:
                add("    jmp .sent")  # never queued: only constructors with a port are
                continue
            port_off, port_size = ctor.port
            add(f"    movzx eax, {_WIDTH[port_size]} [r12+{port_off}]")
            _send(add, image, position, ctor.walk, out_iov)
    add(".sent:")
    add("    mov rdi, r12")
    add("    call push_cell")
    add("    mov rax, [q_head]")
    add("    inc rax")
    add(f"    and rax, {image.options.k_out - 1}")
    add("    mov [q_head], rax")
    add("    dec qword [q_count]")
    add("    jmp drain")
    add(".done:")
    add("    ret")
    add("")
    _control_and_the_rest(add, image, item, iov_in, out_iov)
    return "\n".join(lines) + "\n"


def _send(add: Callable[[str], None], image: Image, position: int, walk: Walk, out_iov: int) -> None:
    """One output in r12 to the port in eax, and then to .sent. Its cells but the first go back."""
    add(f"    add eax, {image.options.out_base}")
    add("    xchg al, ah")
    add("    mov word [out_addr+2], ax")
    _read_out_cells(add, walk)
    for index, (cell, off, size) in enumerate(walk.scalars):
        _out_cell(add, "rax", cell)
        add(f"    lea rax, [rax+{off}]")
        add(f"    mov [out_iov+{16 * index}], rax")
        add(f"    mov qword [out_iov+{16 * index + 8}], {size}")
    add(f"    mov ecx, {len(walk.scalars)}")
    add(f"    mov ebx, out_iov+{16 * len(walk.scalars)}")
    if walk.slot is not None:  # the slot's bytes, as many as it says it holds
        _out_cell(add, "rax", walk.slot[0])
        add(f"    mov rax, [rax+{walk.slot[1]}]")
        add("    mov rdx, rax")
        add("    shr rdx, 32")
        add("    mov [rbx+8], rdx")
        add("    mov eax, eax")
        add(f"    shl rax, {image.slot_bytes.bit_length() - 1}")
        add("    add rax, cells")
        add("    mov [rbx], rax")
        add("    add rbx, 16")
        add("    inc rcx")
    if walk.chain is not None:
        _out_cell(add, "rdi", walk.chain[0])
        add(f"    mov rdi, [rdi+{walk.chain[1]}]")
        add(".chain_%d:" % position)
        add("    test rdi, rdi")
        add(f"    jz .send_ready_{position}")
        add(f"    cmp rcx, {out_iov}")
        add(f"    jae .too_long_{position}")
        add("    lea rax, [rdi+1]")
        add("    mov [rbx], rax")
        add("    ; a program can write this count, so it can name no byte outside the cell")
        add("    movzx eax, byte [rdi]")
        add(f"    cmp eax, {BODY_PAYLOAD}")
        add(f"    jbe .counted_bytes_{position}")
        add(f"    mov eax, {BODY_PAYLOAD}")
        add(f".counted_bytes_{position}:")
        add("    mov [rbx+8], rax")
        add("    add rbx, 16")
        add("    inc rcx")
        add("    mov rdi, [rdi+8]")
        add(f"    jmp .chain_{position}")
    add(f".send_ready_{position}:")
    add("    mov [out_msg+24], rcx")
    add("    mov eax, 46")
    add("    mov rdi, [out_fd]")
    add("    mov esi, out_msg")
    add("    mov edx, 64")
    add("    syscall")
    add("    cmp rax, -11")
    add("    je .done")
    add("    test rax, rax")
    add(f"    jns .counted_{position}")
    if walk.chain is not None:  # a chain a program grew past what one message can gather
        add(f".too_long_{position}:")
    add("    inc qword [cnt_send_errors]")
    add(f".counted_{position}:")
    _free_all_but_the_first(add, walk)
    add("    jmp .sent")


def _control_and_the_rest(add: Callable[[str], None], image: Image, item: Walk, iov_in: int, out_iov: int) -> None:
    """The control port, the item's scatter, the pool, and the data and bss of the image."""
    o = image.options
    k = len(image.states)
    add("; The control port. Never blocks, and bounded by the pool's size.")
    add("control:")
    add("    mov dword [peer_len], 16")
    add("    mov eax, 45")
    add("    mov rdi, [ctl_fd]")
    add("    mov esi, control_buffer")
    add("    mov edx, 16")
    add("    mov r10d, 64")
    add("    mov r8d, peer")
    add("    mov r9d, peer_len")
    add("    syscall")
    add("    cmp rax, 1")
    add("    jl .done")
    add("    mov al, [control_buffer]")
    add("    cmp al, 81")
    add("    je .quit")
    add("    cmp al, 80")
    add("    jne .not_pause")
    add("    mov qword [egress_paused], 1")
    add(".not_pause:")
    add("    cmp al, 82")
    add("    jne .answer")
    add("    mov qword [egress_paused], 0")
    add(".answer:")
    add("    xor ecx, ecx")
    add("    mov rdi, [pool]")
    add(".walk:")
    add("    test rdi, rdi")
    add("    jz .walked")
    add("    inc rcx")
    add(f"    cmp rcx, {image.cell_count + 1}")
    add("    ja .walked")
    add("    mov rdi, [rdi]")
    add("    jmp .walk")
    add(".walked:")
    add("    mov [cnt_pool], rcx")
    if image.slotted:
        add("    mov rax, [slot_free_count]")
        add("    mov [cnt_slots], rax")
    add("    mov eax, 44")
    add("    mov rdi, [ctl_fd]")
    add("    mov esi, counters")
    add(f"    mov edx, {8 * (len(COUNTERS) + image.verdict * len(image.out))}")
    add("    mov r10d, 64")
    add("    mov r8d, peer")
    add("    mov r9d, 16")
    add("    syscall")
    add(".done:")
    add("    ret")
    add(".quit:")
    add("    mov qword [egress_paused], 0")
    add("    call drain")
    add("    mov eax, 231")
    add("    xor edi, edi")
    add("    syscall")
    add("")
    add("; rax: a cell of the item's record. The iovec names its scalar fields, so the kernel fills them.")
    for cell in range(item.cells):
        add(f"point_cell_{cell}:")
        for position, (where, off, _size) in enumerate(item.scalars):
            if where == cell:
                add(f"    lea rsi, [rax+{off}]")
                add(f"    mov [in_iov+{16 * position}], rsi")
        add("    ret")
    add("")
    pool_routines(add)
    add("")
    add("section .data align=4096")
    add("data_start:")
    add("all_signals: dq -1")
    add("ignore_action: dq 1, 0, 0, 0")
    add(f"idle_time: dq 0, {IDLE_MICROSECONDS}")
    add(f"in_addr: dw 2\n    db {_be16(o.in_port)}\n    db 127, 0, 0, 1\n    dq 0")
    add(f"ctl_addr: dw 2\n    db {_be16(o.ctl_port)}\n    db 127, 0, 0, 1\n    dq 0")
    add("out_addr: dw 2\n    db 0, 0\n    db 127, 0, 0, 1\n    dq 0")
    add(f"in_msg: dq 0, 0, in_iov, {iov_in}, in_control, {CONTROL_BYTES}, 0")
    add("option_on: dd 1")
    add("out_msg: dq out_addr, 16, out_iov, 0, 0, 0, 0")
    add("; the admission table of section 6.1: R(n) = a n + b, W(n) = c n + d")
    add(f"admission: dq {image.bound.a}, {image.bound.b}, {image.bound.c}, {image.bound.d}")
    for state in image.states:
        if state.initial is not None:
            add("align 16")
            add(f"array_{state.name}: incbin \"{state.initial}\"")
    add("data_end:")
    add("")
    add("section .bss nobits alloc noexec write align=4096")
    add("bss_start:")
    add(f"stack_guard: resb {PAGE}")
    add(f"resb {image.stack_bytes}")
    add("stack_top:")
    add("alignb 16")
    add("alignb 4096")
    add(f"cells: resb {image.store_bytes}")
    for state in image.states:
        if state.initial is None:
            add("alignb 16")
            add(f"array_{state.name}: resb {state.length * state.element_size}")
    add(f"queue: resq {image.options.k_out}")
    if image.verdict:
        add(f"queue_ports: resb {image.options.k_out}")
        add("alignb 8")
    add(f"in_iov: resq {2 * iov_in}")
    add(f"in_control: resq {CONTROL_BYTES // 8}")
    if o.measure_age:
        add("arrival_sec: resq 1")
        add("arrival_nsec: resq 1")
        add("item_arrival_sec: resq 1")
        add("item_arrival_nsec: resq 1")
        add("clock_now: resq 2")
    add("last_drops: resq 1")
    add(f"out_iov: resq {2 * out_iov}")
    add(f"out_cells: resq {image.out_cells}")
    add(f"slot_cells: resq {item.cells}")
    for position in range(k):
        add(f"state_{position}: resq 1")
    for name in ("pool", "pool_count", "q_head", "q_count", "turn", "in_fd", "out_fd",
                 "ctl_fd", "egress_paused", "in_slot", "slot_free_count"):
        add(f"{name}: resq 1")
    add(f"slot_free: resq {max(1, image.slot_count)}")
    add("counters:")
    for name in COUNTERS:
        add(f"cnt_{name}: resq 1")
    for position in range(len(image.out) if image.verdict else 0):
        add(f"cnt_verdict_{position}: resq 1")
    add("peer: resb 16")
    add("peer_len: resd 1")
    add("alignb 8")
    add("control_buffer: resb 16")
    add(f"bss_end_page: resb {PAGE}")
    add("bss_end:")
    add("section .note.GNU-stack noalloc noexec nowrite progbits")


def fixed_image_startup(add: Callable[[str], None]) -> None:
    """Section 7.2's first four steps, which do not depend on the device. Exits with 1, 2 or 4."""
    add("_start:")
    add("    ; Section 7.2. Each step that can fail exits with its own number, before any input.")
    add("    ; 1: block every asynchronous signal, and ignore SIGPIPE")
    add("    mov eax, 14")
    add("    xor edi, edi")
    add("    mov esi, all_signals")
    add("    xor edx, edx")
    add("    mov r10d, 8")
    add("    syscall")
    add("    test rax, rax")
    add("    jnz .refuse_signals")
    add("    mov eax, 13")
    add("    mov edi, 13")
    add("    mov esi, ignore_action")
    add("    xor edx, edx")
    add("    mov r10d, 8")
    add("    syscall")
    add("    test rax, rax")
    add("    jnz .refuse_signals")
    add("    ; 2: lock the image into memory")
    add("    mov eax, 151")
    add("    mov edi, 3")
    add("    syscall")
    add("    test rax, rax")
    add("    jnz .refuse_lock")
    add("    ; 3: touch every page, so no page fault can happen during service")
    add("    mov edi, data_start")
    add(".touch_data:")
    add("    or byte [rdi], 0")
    add(f"    add rdi, {PAGE}")
    add("    cmp rdi, data_end")
    add("    jb .touch_data")
    add("    mov edi, bss_start")
    add(".touch_bss:")
    add("    or byte [rdi], 0")
    add(f"    add rdi, {PAGE}")
    add("    cmp rdi, bss_end")
    add("    jb .touch_bss")
    add("    ; 4: the guard page, and the stack inside the image")
    add("    mov eax, 10")
    add("    mov edi, stack_guard")
    add(f"    mov esi, {PAGE}")
    add("    xor edx, edx")
    add("    syscall")
    add("    test rax, rax")
    add("    jnz .refuse_guard")
    add("    mov esp, stack_top")


def pool_routines(add: Callable[[str], None]) -> None:
    add("; The free list. Admission has already shown a cell is there. pop_cell returns one in")
    add("; rax and uses r9. push_cell takes one in rdi and uses rax. free_chain takes a chain in")
    add("; rdi and also uses rbx.")
    add("pop_cell:")
    add("    mov rax, [pool]")
    add("    mov r9, [rax]")
    add("    mov [pool], r9")
    add("    dec qword [pool_count]")
    add("    ret")
    add("push_cell:")
    add("    mov rax, [pool]")
    add("    mov [rdi], rax")
    add("    mov [pool], rdi")
    add("    inc qword [pool_count]")
    add("    ret")
    add("; A slot, by its number in eax, back on the stack of free ones. Uses rcx.")
    add("push_slot:")
    add("    mov rcx, [slot_free_count]")
    add("    mov [slot_free+rcx*8], rax")
    add("    inc qword [slot_free_count]")
    add("    ret")
    add("free_chain:")
    add("    test rdi, rdi")
    add("    jz .done")
    add("    mov rbx, [rdi+8]")
    add("    call push_cell")
    add("    mov rdi, rbx")
    add("    jmp free_chain")
    add(".done:")
    add("    ret")


def dispatch_on(add: Callable[[str], None], tags: Sequence[Optional[int]], tagged: bool, register: str,
                stem: str) -> None:
    """Jump to the arm for the constructor this register holds, given each one's tag, or None for the
    null constructor of a tagless type. The last needs no test."""
    for position, tag in enumerate(tags[:-1]):
        if tagged:
            add(f"    cmp byte [{register}], {tag}")
            add(f"    je {stem}_{position}")
        else:
            add(f"    test {register}, {register}")
            add(f"    {'jz' if tag is None else 'jnz'} {stem}_{position}")
    add(f"    jmp {stem}_{len(tags) - 1}")


def _dispatch(add: Callable[[str], None], image: Image, register: str, stem: str) -> None:
    dispatch_on(add, [ctor.tag for ctor in image.out], image.out_tagged, register, stem)
