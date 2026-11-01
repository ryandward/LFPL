"""The fixed image and the serving loop again, with frames from a network card through AF_XDP.

This is LFPL-O sections 3, 6 and 7 for a device that hands over whole frames in shared memory, where
src/runtime.py is the same for a socket that hands over datagrams. It is trusted in the same way, and
for the same reason is meant to be read: no libc, no allocator, and only the system calls below.

The packet memory is part of the image. Its frames are registered with the kernel at startup, and from
then on a frame does not move: the card writes it, the program's output is written back over it, and
the card sends it from where it lies. Four rings pass frames around:

    fill        frames the kernel may receive into
    rx          frames that have arrived
    tx          frames to send
    completion  frames that have been sent, which go straight back on fill

A frame is always on exactly one of those, or is the one being served, so a ring of F entries cannot
overflow, and frames are conserved with nothing having to count them.

What the source does not say, this file decides:

    Item        a chain of records, each owning the next, whose scalar fields in declaration order
                are the frame's first bytes, little endian and packed. Admission copies them in, so
                the program never sees the frame. One field of an abstract type holds the frame, and
                as with any abstract type a program can hand it on and do nothing else with it: not
                read it, not forge one, not lose it. A scalar field called `length`, of type Int16,
                is not on the wire: it is filled in with the frame's length.
    Output      each constructor is a chain of the same kind and holds the frame. One with a field
                `port` is sent: every other scalar field is written back to the frame at the place
                the item's field of that name came from, and the frame goes on tx. There is one port
                here, so a port other than 0 counts in bad_port and the frame is given back. Any
                other constructor gives the frame back without writing to it.
    Admission   counts frames and cells separately. Section 6.1's conditions, in order: a frame shorter
                than the item's scalars is malformed; then pool cells for n + R(n), or Short; then
                room on tx, or Full. A refusal puts the frame back on fill. Cells return as soon as
                egress has written them to the frame, so one transaction's cells are all there are.

System calls, at startup: rt_sigprocmask, rt_sigaction, mlockall, mprotect, socket, setsockopt,
getsockopt, mmap, ioctl, bind and bpf. In service: poll when idle, and to wake a kernel that says fill needs
it, sendto to wake one that says tx does, recvfrom on the control port, and on `S`, getsockopt and sendto.
The serving loop makes no system call for a frame while the card keeps up.

The control port answers as src/runtime.py's does. `S` answers with the counters, walking the free list
to count the pool. `P` stops publishing to tx, so that it fills, and `R` starts again. `Q` waits for what is
queued to be sent, for as long as the kernel keeps sending it, and exits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from src.rtcheck import Bound, Checker, Ctor
from src.runtime import (CELL, MAX_IMAGE_BYTES, PAGE, PARAMS, RESULTS, RUNTIME_DEPTH, BindingError,
                      StateSlot, _be16, _check_reserve, dispatch_on, fixed_image_startup, handler_slots,
                      pool_routines)

CHUNK = 2048            # bytes of packet memory for one frame: the smallest the kernel takes
BATCH = 64              # frames taken from rx between publishing to the rings
CONTROL_EVERY = 1024    # turns between looks at the control port, so a flood cannot starve it
IDLE_MILLISECONDS = 1
QUIT_PATIENCE = 1024    # how many waits of 1 ms `Q` gives the kernel to send what is queued
LENGTH_FIELD = "length"
MAX_NAME = 15           # an interface's name, without its NUL: IFNAMSIZ

AF_XDP = 44
SOCK_RAW = 3
SOL_XDP = 283
XDP_MMAP_OFFSETS, XDP_RX_RING, XDP_TX_RING, XDP_UMEM_REG = 1, 2, 3, 4
XDP_UMEM_FILL_RING, XDP_UMEM_COMPLETION_RING, XDP_STATISTICS, XDP_OPTIONS = 5, 6, 7, 8
XDP_USE_NEED_WAKEUP = 8
XDP_RING_NEED_WAKEUP = 1
XDP_OPTIONS_ZEROCOPY = 1
SIOCGIFINDEX = 0x8933
BPF_MAP_UPDATE_ELEM, BPF_OBJ_GET = 2, 7
MSG_DONTWAIT = 64
EAGAIN, ENOBUFS = 11, 105
POLLIN = 1
MAP_SHARED_POPULATE = 0x8001
RING_OFFSETS_BYTES = 32 # producer, consumer, descriptors, flags: four 64-bit offsets, for each ring

# (name, the offset that asks mmap for it, its place in xdp_mmap_offsets, bytes per entry)
RX, TX, FILL, COMPLETION = "rx", "tx", "fill", "comp"
RINGS = ((RX, 0, 0, 16), (TX, 0x80000000, 1, 16), (FILL, 0x100000000, 2, 8),
         (COMPLETION, 0x180000000, 3, 8))

# frames counts every frame taken from rx, and every one ends in forwarded, or in returned, which is
# back on fill. completed is how many of the forwarded the kernel has finished with. send_errors is what
# waking it to send has reported, and it is a lower bound: see flush_tx. kernel_dropped is the socket's
# own count of what it could not deliver, as XDP_STATISTICS gives it when asked. Each constructor of the
# output has a counter too, out_<name>, after these.
COUNTERS = ("capacity", "pool", "frames", "served", "forwarded", "dropped", "malformed", "short", "full",
            "bad_port", "send_errors", "returned", "completed", "admitted_cells", "budget_steps", "steps",
            "kernel_dropped", "zerocopy")
_SCRATCH = {8: "r10", 4: "r10d", 2: "r10w", 1: "r10b"}


@dataclass(frozen=True)
class XdpOptions:
    interface: str
    map_path: str        # the pinned socket map, from experimental/xdp_load.py
    ctl_port: int
    frames: int = 4096   # F, the frames in the packet memory, and the size of fill, rx and completion
    k_out: int = 2048    # the size of tx
    queue: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.ctl_port <= 65535:
            raise BindingError(f"section 7.3: the control port at {self.ctl_port} is not a port")
        if not 1 <= len(self.interface.encode()) <= MAX_NAME or "\0" in self.interface:
            raise BindingError(f"section 7.3: {self.interface!r} cannot name an interface")
        if not self.map_path or "\0" in self.map_path:
            raise BindingError(f"section 7.3: {self.map_path!r} cannot name a pinned map")
        for name, value in (("F", self.frames), ("K_out", self.k_out)):
            if value < 1 or value & value - 1:
                raise BindingError(f"section 8.2: {name} of {value} must be a power of two")
        if self.queue < 0:
            raise BindingError(f"section 7.3: no queue is numbered {self.queue}")


@dataclass(frozen=True)
class Scalar:
    name: str
    cell: int     # which cell of the chain, counting from the first
    offset: int   # where in that cell
    size: int
    wire: int     # where in the frame


@dataclass(frozen=True)
class Item:
    scalars: tuple[Scalar, ...]    # those on the wire, in wire order
    links: tuple[int, ...]         # for each cell but the last, where its link to the next one is
    frame: tuple[int, int]         # the cell and offset of the frame
    length: tuple[int, int] | None  # the cell and offset of the frame's length, if it is declared

    @property
    def cells(self) -> int:
        return len(self.links) + 1

    @property
    def header_bytes(self) -> int:
        return sum(scalar.size for scalar in self.scalars)


@dataclass(frozen=True)
class Output:
    name: str
    tag: int
    port: int | None               # the offset of `port` in the first cell
    writes: tuple[Scalar, ...]
    links: tuple[int, ...]
    frame: tuple[int, int]

    @property
    def cells(self) -> int:
        return len(self.links) + 1


@dataclass(frozen=True)
class XdpImage:
    handler: str
    states: tuple[StateSlot, ...]
    item: Item
    out: tuple[Output, ...]
    out_tagged: bool
    bound: Bound
    depth: int
    options: XdpOptions

    @property
    def reserve(self) -> int:
        """R(n). An item here is always n cells, so these are numbers and not a table."""
        return self.bound.a * self.item.cells + self.bound.b

    @property
    def work(self) -> int:
        """W(n)."""
        return self.bound.c * self.item.cells + self.bound.d

    @property
    def cell_count(self) -> int:
        """Section 3.4's C, with K_in and K_out taken out: a frame waits on a ring and holds no cells."""
        return self.item.cells + self.reserve

    @property
    def stack_bytes(self) -> int:
        needed = max(self.depth + 8, RUNTIME_DEPTH)
        return (needed + PAGE - 1) // PAGE * PAGE

    @property
    def bytes(self) -> int:
        return (self.cell_count * CELL + self.stack_bytes + self.options.frames * CHUNK
                + sum(s.length * s.element_size for s in self.states))

    @property
    def counters(self) -> tuple[str, ...]:
        return counter_names(ctor.name for ctor in self.out)


def counter_names(constructors: Iterable[str]) -> tuple[str, ...]:
    """What `S` answers with, in order, as 64-bit words."""
    return COUNTERS + tuple(f"out_{name}" for name in constructors)


def _chain(checker: Checker, first: Ctor) -> tuple[list[tuple[str, int, int, int, str]], list[int],
                                                   tuple[int, int]]:
    """Walk records from this constructor down to the frame: the scalars met on the way, as (name,
    cell, offset, size, type), then each cell's link, then where the frame is."""
    scalars = []
    links: list[int] = []
    frame = None
    seen = set()
    ctor = first
    while True:
        following = None
        for off, size, ty, name in ctor.fields:
            norm = checker.norm(ty)
            decl = checker._decl(norm)
            if decl is None:
                scalars.append((name, len(links), off, size, norm))
            elif decl.kind == "abstract":
                if frame is not None:
                    raise BindingError(f"section 3.1: {first.name} holds two frames, and a frame "
                                       f"is one thing that came from one place")
                frame = (len(links), off)
            elif decl.kind == "record" and following is None and decl.name not in seen:
                following = decl
                seen.add(decl.name)
                links.append(off)
            else:
                raise BindingError(f"section 3.1: {ctor.name}.{name} is {decl.name}, and this runtime "
                                   f"carries scalars, one frame, and one record to continue with")
        if following is None:
            break
        ctor = following.ctors[0]
    if frame is None:
        raise BindingError(f"section 3.1: {first.name} holds no frame, and every frame has to come back")
    return scalars, links, frame


def _item(checker: Checker, item_type: str) -> Item:
    decl = checker._decl(item_type)
    if decl is None or decl.kind != "record":
        raise BindingError(f"section 3.1: this runtime receives an item as a record, and "
                           f"{item_type} is not one")
    fields, links, frame = _chain(checker, decl.ctors[0])
    scalars = []
    length = None
    wire = 0
    for name, cell, off, size, norm in fields:
        if isinstance(norm, tuple) and norm[0] == "fin":
            # A Fin is what makes an array read need no bounds check, and only idx may make one.
            raise BindingError(f"section 4.5: {decl.name}.{name} is a Fin, and an item's fields "
                               f"come from the wire: take an integer and use idx")
        if name == LENGTH_FIELD:
            if norm != "i16" or length is not None:
                raise BindingError(f"section 3.1: {decl.name} declares `{LENGTH_FIELD}`, which "
                                   f"is the frame's length: one field, of type Int16")
            length = (cell, off)
            continue
        scalars.append(Scalar(name, cell, off, size, wire))
        wire += size
    named = [scalar.name for scalar in scalars]
    for name in named:
        if named.count(name) > 1:
            raise BindingError(f"section 3.1: two fields of the item are called {name}, and egress "
                               f"writes back by name")
    return Item(tuple(scalars), tuple(links), frame, length)


def _output(checker: Checker, ctor: Ctor, item: Item) -> Output:
    fields, links, frame = _chain(checker, ctor)
    port = None
    writes = []
    by_name = {scalar.name: scalar for scalar in item.scalars}
    for name, cell, off, size, norm in fields:
        if name == "port":
            if norm != "i8" or cell != 0:
                raise BindingError(f"section 3.1: {ctor.name}.port must be Int8, in the first cell")
            port = off
            continue
        source = by_name.get(name)
        if source is None:
            raise BindingError(f"section 3.1: {ctor.name}.{name} is written back by name, and the "
                               f"item's headers have no field of that name")
        if source.size != size:
            raise BindingError(f"section 3.1: {ctor.name}.{name} has {size} bytes, and the item's has "
                               f"{source.size}")
        if any(write.name == name for write in writes):
            raise BindingError(f"section 3.1: {ctor.name} writes {name} twice")
        writes.append(Scalar(name, cell, off, size, source.wire))
    tag = int(ctor.tag) if ctor.tag.isdigit() else 0
    return Output(ctor.name, tag, port, tuple(writes), tuple(links), frame)


def item_cells(checker: Checker) -> int:
    """n. It comes from the item's type, so it is known before the bounds are certified for it."""
    fn = next(f for f in checker.functions.values() if f.handler)
    return _item(checker, checker.norm(fn.params[-2])).cells


def bind(checker: Checker, handler_name: str, depth: int, options: XdpOptions,
         initial: dict[str, str]) -> XdpImage:
    """Read the handler's types from the checked stream, and refuse what cannot be served."""
    fn, item_type, out_type, states = handler_slots(checker, initial)
    _check_reserve(checker, fn.params[-1])
    item = _item(checker, item_type)
    out_decl = checker._decl(out_type)
    if out_decl is None or out_decl.kind not in ("tagged", "tagless", "record"):
        raise BindingError(f"section 3.1: the output {out_type} is not a data type")
    out = tuple(_output(checker, ctor, item) for ctor in out_decl.ctors)
    return XdpImage(handler_name, states, item, out, out_decl.kind == "tagged",
                    checker.bounds(handler_name), depth, options)


def _pieces(scalars: Iterable[Scalar]) -> Iterator[tuple[int, int, int]]:
    """Moves for these fields, as (wire, offset, size). Fields that lie side by side in the frame and
    in the cell are one move, which is most of them: a record's fields keep their wire order."""
    runs: list[list[int]] = []
    for scalar in sorted(scalars, key=lambda s: s.wire):
        if runs and runs[-1][0] + runs[-1][2] == scalar.wire and runs[-1][1] + runs[-1][2] == scalar.offset:
            runs[-1][2] += scalar.size
        else:
            runs.append([scalar.wire, scalar.offset, scalar.size])
    for wire, offset, size in runs:
        while size:
            width = next(w for w in (8, 4, 2, 1) if w <= size)
            yield wire, offset, width
            wire += width
            offset += width
            size -= width


def _load_frame(add: Callable[[str], None], links: tuple[int, ...], frame: tuple[int, int]) -> None:
    """r12: the first cell of an output. rbx: the frame it holds. Uses rcx."""
    add("    mov rcx, r12")
    for off in links[:frame[0]]:
        add(f"    mov rcx, [rcx+{off}]")
    add(f"    mov rbx, [rcx+{frame[1]}]")


def _free_cells(add: Callable[[str], None], output: Output, write_back: bool) -> None:
    """Give an output's cells to the pool, writing each to the frame first if asked to. r12: the first
    cell, rsi: the frame. Uses rax, rcx, rdx and r10."""
    add("    mov rcx, r12")
    add("    mov rax, [pool]")
    for cell in range(output.cells):
        if write_back:
            for wire, offset, width in _pieces(s for s in output.writes if s.cell == cell):
                add(f"    mov {_SCRATCH[width]}, [rcx+{offset}]")
                add(f"    mov [rsi+{wire}], {_SCRATCH[width]}")
        if cell < output.cells - 1:
            add(f"    mov rdx, [rcx+{output.links[cell]}]")
        add("    mov [rcx], rax")
        add("    mov rax, rcx")
        if cell < output.cells - 1:
            add("    mov rcx, rdx")
    add("    mov [pool], rax")
    add(f"    add qword [pool_count], {output.cells}")


def _bytes_of(text: str) -> str:
    return ", ".join(str(byte) for byte in text.encode()) + ", 0"


def assembly(image: XdpImage) -> str:
    """The runtime and the image, as NASM, to follow the compiled functions."""
    o = image.options
    item = image.item
    if image.bytes > MAX_IMAGE_BYTES:
        raise BindingError(f"section 8.2: {image.handler} needs an image of {image.bytes} bytes, and "
                           f"this runtime takes at most {MAX_IMAGE_BYTES}")
    lines: list[str] = []
    add = lines.append
    k = len(image.states)
    frames = o.frames
    sizes = {RX: frames, TX: o.k_out, FILL: frames, COMPLETION: frames}

    def syscall(number: int, refuse: str = "", *arguments: str) -> None:
        add(f"    mov eax, {number}")
        for register, value in zip(("rdi", "rsi", "rdx", "r10", "r8", "r9"), arguments):
            add(f"    mov {register}, {value}")
        add("    syscall")
        if refuse:
            add("    test rax, rax")
            add(f"    {refuse}")

    def set_ring_size(option: int, size: str) -> None:
        syscall(54, "jnz .refuse_socket", "[xsk_fd]", str(SOL_XDP), str(option), size, "4")

    add("section .text")
    add("global _start")
    add(f"handler_entry equ fn_{image.handler}")
    fixed_image_startup(add)
    add("    ; 5: every cell starts in the pool")
    add("    mov eax, cells")
    add("    xor esi, esi")
    add(f"    mov ecx, {image.cell_count}")
    add(".thread:")
    add("    mov [rax], rsi")
    add("    mov rsi, rax")
    add("    add rax, 16")
    add("    dec rcx")
    add("    jnz .thread")
    add("    mov [pool], rsi")
    add(f"    mov qword [pool_count], {image.cell_count}")
    add(f"    mov qword [cnt_capacity], {image.cell_count}")
    for position, state in enumerate(image.states):
        add(f"    mov qword [state_{position}], array_{state.name}")
    add("    ; 6: the control port")
    syscall(41, "js .refuse_control", "2", "2", "0")
    add("    mov [ctl_fd], rax")
    syscall(49, "jnz .refuse_control", "[ctl_fd]", "ctl_addr", "16")
    add("    ; 7: the socket, the packet memory, and the rings")
    syscall(41, "js .refuse_socket", str(AF_XDP), str(SOCK_RAW), "0")
    add("    mov [xsk_fd], rax")
    add("    mov [xsk_word], eax")
    add("    mov [waiting], eax")
    syscall(54, "jnz .refuse_socket", "[xsk_fd]", str(SOL_XDP), str(XDP_UMEM_REG), "umem_reg", "32")
    set_ring_size(XDP_UMEM_FILL_RING, "fill_size")
    set_ring_size(XDP_UMEM_COMPLETION_RING, "comp_size")
    set_ring_size(XDP_RX_RING, "rx_size")
    set_ring_size(XDP_TX_RING, "tx_size")
    syscall(55, "jnz .refuse_socket", "[xsk_fd]", str(SOL_XDP), str(XDP_MMAP_OFFSETS), "offsets",
            "offsets_len")
    for name, page_offset, position, entry in RINGS:
        at = RING_OFFSETS_BYTES * position
        add(f"    ; the {name} ring")
        add(f"    mov rsi, [offsets+{at + 16}]")
        add(f"    add rsi, {sizes[name] * entry}")
        add("    mov eax, 9")
        add("    xor edi, edi")
        add("    mov edx, 3")
        add(f"    mov r10d, {MAP_SHARED_POPULATE}")
        add("    mov r8, [xsk_fd]")
        add(f"    mov r9, {page_offset}")
        add("    syscall")
        add("    cmp rax, -4096")
        add("    ja .refuse_socket")
        for field, slot in (("prod", 0), ("cons", 8), ("descs", 16), ("flags", 24)):
            add(f"    mov rcx, [offsets+{at + slot}]")
            add("    add rcx, rax")
            add(f"    mov [{name}_{field}_at], rcx")
    add("    ; every frame goes on fill, so the kernel has somewhere to put what arrives")
    add("    mov rdi, [fill_descs_at]")
    add("    xor eax, eax")
    add("    xor ecx, ecx")
    add(".fill:")
    add("    mov [rdi+rcx*8], rax")
    add(f"    add rax, {CHUNK}")
    add("    inc ecx")
    add(f"    cmp ecx, {frames}")
    add("    jb .fill")
    add("    mov [fill_prod], ecx")
    add("    mov rax, [fill_prod_at]")
    add("    mov [rax], ecx")
    add("    ; 8: the interface")
    syscall(16, "jnz .refuse_interface", "[ctl_fd]", str(SIOCGIFINDEX), "ifreq")
    add("    mov eax, [ifreq+16]")
    add("    mov [xdp_addr+4], eax")
    syscall(49, "jnz .refuse_interface", "[xsk_fd]", "xdp_addr", "16")
    syscall(55, "jnz .refuse_interface", "[xsk_fd]", str(SOL_XDP), str(XDP_OPTIONS), "bound_flags",
            "flags_len")
    add("    mov eax, [bound_flags]")
    add(f"    and eax, {XDP_OPTIONS_ZEROCOPY}")
    add("    mov [cnt_zerocopy], rax")
    add("    ; 9: tell the XDP program where to send frames")
    syscall(321, "js .refuse_map", str(BPF_OBJ_GET), "obj_attr", "24")
    add("    mov [update_attr], eax")
    syscall(321, "jnz .refuse_map", str(BPF_MAP_UPDATE_ELEM), "update_attr", "32")
    add("    jmp serve")
    for label, status in ((".refuse_signals", 1), (".refuse_lock", 2), (".refuse_guard", 4),
                          (".refuse_control", 6), (".refuse_socket", 7), (".refuse_interface", 8),
                          (".refuse_map", 9)):
        add(f"{label}:")
        add("    mov eax, 231")
        add(f"    mov edi, {status}")
        add("    syscall")
    add("")
    add("; ---------------------------------------------------- the serving loop, section 3.3")
    add("serve:")
    add("    inc qword [turn]")
    add(f"    test qword [turn], {CONTROL_EVERY - 1}")
    add("    jnz .arrive")
    add("    call control")
    add(".arrive:")
    add("    call reap")
    add("    ; ARRIVE: how many frames the kernel has put on rx, up to a batch")
    add("    mov rax, [rx_prod_at]")
    add("    mov eax, [rax]")
    add("    sub eax, [rx_cons]")
    add("    jz .idle")
    add(f"    cmp eax, {BATCH}")
    add("    jbe .take")
    add(f"    mov eax, {BATCH}")
    add(".take:")
    add("    mov [batch], eax")
    add(".frame:")
    add("    mov eax, [rx_cons]")
    add("    inc dword [rx_cons]")
    add(f"    and eax, {frames - 1}")
    add("    shl rax, 4")
    add("    add rax, [rx_descs_at]")
    add("    mov rbx, [rax]")
    add("    mov ecx, [rax+8]")
    add("    inc qword [cnt_frames]")
    add("    ; Section 6.1, in order. A refusal puts the frame back on fill, and has touched no cell.")
    add(f"    cmp ecx, {item.header_bytes}")
    add("    jb .malformed")
    add(f"    cmp qword [pool_count], {image.cell_count}")
    add("    jb .short")
    add("    ; room on tx: what the kernel was last seen to have taken, and only if that is not enough,")
    add("    ; a look at what it has taken since, which is a cache line it is also writing")
    add("    mov eax, [tx_prod]")
    add("    sub eax, [tx_seen]")
    add(f"    cmp eax, {o.k_out}")
    add("    jb .room")
    add("    mov rax, [tx_cons_at]")
    add("    mov eax, [rax]")
    add("    mov [tx_seen], eax")
    add("    mov edx, [tx_prod]")
    add("    sub edx, eax")
    add(f"    cmp edx, {o.k_out}")
    add("    jae .full")
    add(".room:")
    add(f"    add qword [cnt_admitted_cells], {item.cells}")
    add(f"    add qword [cnt_budget_steps], {image.work}")
    add("    ; ADMIT: the headers go into cells from the pool, linked as the item's type says.")
    add("    ; rbx: the frame's address, ecx: its length, rsi: where it lies.")
    add("    lea rsi, [rbx+umem]")
    add("    mov rax, [pool]")
    add("    mov r12, rax")
    for cell in range(item.cells):
        add("    mov rdx, [rax]")
        for wire, offset, width in _pieces(s for s in item.scalars if s.cell == cell):
            add(f"    mov {_SCRATCH[width]}, [rsi+{wire}]")
            add(f"    mov [rax+{offset}], {_SCRATCH[width]}")
        if item.length is not None and item.length[0] == cell:
            add(f"    mov [rax+{item.length[1]}], cx")
        if item.frame[0] == cell:
            add("    ; the frame is one word to a program, and it is both its address and its length")
            add("    mov rdi, rcx")
            add("    shl rdi, 32")
            add("    or rdi, rbx")
            add(f"    mov [rax+{item.frame[1]}], rdi")
        if cell < item.cells - 1:
            add(f"    mov [rax+{item.links[cell]}], rdx")
            add("    mov rax, rdx")
    add("    mov [pool], rdx")
    add(f"    sub qword [pool_count], {item.cells}")
    add("    ; the item's reserve: R(n) cells, as a Free")
    add("    xor r13d, r13d")
    if image.reserve:
        add(f"    mov ecx, {image.reserve}")
        add(".reserve:")
        add("    mov rax, [pool]")
        add("    mov rdx, [rax]")
        add("    mov [pool], rdx")
        add("    mov [rax], r13")
        add("    mov r13, rax")
        add("    dec rcx")
        add("    jnz .reserve")
        add(f"    sub qword [pool_count], {image.reserve}")
    add(f"    mov {PARAMS[k + 1]}, r13")
    add(f"    mov {PARAMS[k]}, r12")
    for position in range(k):
        add(f"    mov {PARAMS[position]}, [state_{position}]")
    add("    call handler_entry")
    add("    ; DONE: the state, the output and what is left of the reserve come back")
    for position in range(k):
        add(f"    mov [state_{position}], {RESULTS[position]}")
    add(f"    mov r12, {RESULTS[k]}")
    add(f"    mov r13, {RESULTS[k + 1]}")
    add(".give_back:")
    add("    test r13, r13")
    add("    jz .given_back")
    add("    mov rax, [pool]")
    add("    mov rdx, [r13]")
    add("    mov [r13], rax")
    add("    mov [pool], r13")
    add("    inc qword [pool_count]")
    add("    mov r13, rdx")
    add("    jmp .give_back")
    add(".given_back:")
    add("    inc qword [cnt_served]")
    dispatch_on(add, [ctor.tag for ctor in image.out], image.out_tagged, "r12", ".out")
    for position, ctor in enumerate(image.out):
        add(f".out_{position}:")
        add(f"    inc qword [cnt_out_{ctor.name}]")
        _load_frame(add, ctor.links, ctor.frame)
        if ctor.port is None:
            add("    inc qword [cnt_dropped]")
            _free_cells(add, ctor, False)
            add("    jmp .frame_back")
            continue
        add(f"    cmp byte [r12+{ctor.port}], 0")
        add(f"    jne .bad_port_{position}")
        add("    ; EGRESS: the fields go back over the frame, and the frame goes on tx")
        add("    mov eax, ebx")
        add("    lea rsi, [rax+umem]")
        _free_cells(add, ctor, True)
        add("    mov eax, [tx_prod]")
        add("    mov edx, eax")
        add(f"    and edx, {o.k_out - 1}")
        add("    shl rdx, 4")
        add("    add rdx, [tx_descs_at]")
        add("    mov ecx, ebx")
        add("    mov [rdx], rcx")
        add("    shr rbx, 32")
        add("    mov [rdx+8], rbx")
        add("    inc eax")
        add("    mov [tx_prod], eax")
        add("    inc qword [cnt_forwarded]")
        add("    jmp .next_frame")
        add(f".bad_port_{position}:")
        add("    inc qword [cnt_bad_port]")
        _free_cells(add, ctor, False)
        add("    jmp .frame_back")
    for label, counter in ((".malformed", "malformed"), (".short", "short"), (".full", "full")):
        add(f"{label}:")
        add(f"    inc qword [cnt_{counter}]")
        add("    jmp .frame_back")
    add(".frame_back:")
    add("    mov eax, ebx")
    add("    mov edx, [fill_prod]")
    add("    mov ecx, edx")
    add(f"    and ecx, {frames - 1}")
    add("    mov rsi, [fill_descs_at]")
    add("    mov [rsi+rcx*8], rax")
    add("    inc edx")
    add("    mov [fill_prod], edx")
    add("    inc qword [cnt_returned]")
    add(".next_frame:")
    add("    dec dword [batch]")
    add("    jnz .frame")
    add("    ; The batch is done: the kernel may reuse its rx entries, and see what has been queued.")
    add("    mov rax, [rx_cons_at]")
    add("    mov edx, [rx_cons]")
    add("    mov [rax], edx")
    add("    mov rax, [fill_prod_at]")
    add("    mov edx, [fill_prod]")
    add("    mov [rax], edx")
    add("    call flush_tx")
    add("    jmp serve")
    add(".idle:")
    add("    call flush_tx")
    add("    ; a card in zero-copy mode that ran out of frames stops looking at fill until it is woken")
    add("    mov rax, [fill_flags_at]")
    add(f"    test dword [rax], {XDP_RING_NEED_WAKEUP}")
    add("    jz .fill_awake")
    syscall(7, "", "waiting", "1", "0")
    add(".fill_awake:")
    add("    cmp dword [reaped], 0")
    add("    jne serve")
    syscall(7, "", "waiting", "1", str(IDLE_MILLISECONDS))
    add("    call control")
    add("    jmp serve")
    add("")
    add("; What has been sent goes back on fill. Bounded, because a frame is on one ring at a time.")
    add("reap:")
    add("    mov rax, [comp_prod_at]")
    add("    mov eax, [rax]")
    add("    mov ecx, [comp_cons]")
    add("    mov edx, eax")
    add("    sub edx, ecx")
    add("    mov [reaped], edx")
    add("    jz .done")
    add("    add [cnt_completed], rdx")
    add("    mov r8, [comp_descs_at]")
    add("    mov r9, [fill_descs_at]")
    add("    mov r10d, [fill_prod]")
    add(".one:")
    add("    mov esi, ecx")
    add(f"    and esi, {frames - 1}")
    add("    mov rsi, [r8+rsi*8]")
    add("    mov edi, r10d")
    add(f"    and edi, {frames - 1}")
    add("    mov [r9+rdi*8], rsi")
    add("    inc ecx")
    add("    inc r10d")
    add("    cmp ecx, eax")
    add("    jne .one")
    add("    mov [comp_cons], ecx")
    add("    mov [fill_prod], r10d")
    add("    mov rsi, [comp_cons_at]")
    add("    mov [rsi], ecx")
    add("    mov rsi, [fill_prod_at]")
    add("    mov [rsi], r10d")
    add(".done:")
    add("    ret")
    add("")
    add("; Publish tx, and wake the kernel if it says it needs waking and there is something to send.")
    add("; EAGAIN and ENOBUFS mean it sent what it could and left the rest on tx, for the next turn. Any other")
    add("; error is counted, and that includes EBUSY, when it has taken a frame off tx and dropped it because")
    add("; the link is down. That frame is gone, and its address still comes back on the completion ring. In")
    add("; copy mode, poll sends from tx as well and has nowhere to say so, so this counts some of those frames")
    add("; and not all: the port's own tx_dropped has every one.")
    add("flush_tx:")
    add("    cmp qword [egress_paused], 0")
    add("    jne .done")
    add("    mov rax, [tx_prod_at]")
    add("    mov edx, [tx_prod]")
    add("    mov [rax], edx")
    add("    mov rax, [tx_flags_at]")
    add(f"    test dword [rax], {XDP_RING_NEED_WAKEUP}")
    add("    jz .done")
    add("    mov rax, [tx_cons_at]")
    add("    cmp edx, [rax]")
    add("    je .done")
    add("    mov eax, 44")
    add("    mov rdi, [xsk_fd]")
    add("    xor esi, esi")
    add("    xor edx, edx")
    add(f"    mov r10d, {MSG_DONTWAIT}")
    add("    xor r8d, r8d")
    add("    xor r9d, r9d")
    add("    syscall")
    add("    test rax, rax")
    add("    jns .done")
    for number in (EAGAIN, ENOBUFS):
        add(f"    cmp rax, -{number}")
        add("    je .done")
    add("    inc qword [cnt_send_errors]")
    add(".done:")
    add("    ret")
    add("")
    add("; The control port. Never blocks, and bounded by the pool's size.")
    add("control:")
    add("    mov dword [peer_len], 16")
    add("    mov eax, 45")
    add("    mov rdi, [ctl_fd]")
    add("    mov esi, control_buffer")
    add("    mov edx, 16")
    add(f"    mov r10d, {MSG_DONTWAIT}")
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
    syscall(55, "", "[xsk_fd]", str(SOL_XDP), str(XDP_STATISTICS), "statistics", "statistics_len")
    add("    ; rx_dropped and rx_ring_full are the two that count a frame the socket did not get. rx has an")
    add("    ; entry for every frame there is, so it should never be full, but if that is wrong, it shows here.")
    add("    mov rax, [statistics]")
    add("    add rax, [statistics+24]")
    add("    mov [cnt_kernel_dropped], rax")
    add("    mov eax, 44")
    add("    mov rdi, [ctl_fd]")
    add("    mov esi, counters")
    add(f"    mov edx, {8 * len(image.counters)}")
    add(f"    mov r10d, {MSG_DONTWAIT}")
    add("    mov r8d, peer")
    add("    mov r9d, 16")
    add("    syscall")
    add(".done:")
    add("    ret")
    add(".quit:")
    add("    mov qword [egress_paused], 0")
    add(f"    mov qword [patience], {QUIT_PATIENCE}")
    add(".send_what_is_queued:")
    add("    call flush_tx")
    add("    call reap")
    add("    ; completion and not tx's consumer: a card can have taken a frame and not yet sent it")
    add("    mov eax, [comp_cons]")
    add("    cmp eax, [tx_prod]")
    add("    je .leave")
    add("    dec qword [patience]")
    add("    jz .leave")
    syscall(7, "", "waiting", "1", str(IDLE_MILLISECONDS))
    add("    jmp .send_what_is_queued")
    add(".leave:")
    add("    mov eax, 231")
    add("    xor edi, edi")
    add("    syscall")
    add("")
    pool_routines(add)
    add("")
    add("section .data align=4096")
    add("data_start:")
    add("all_signals: dq -1")
    add("ignore_action: dq 1, 0, 0, 0")
    add(f"ctl_addr: dw 2\n    db {_be16(o.ctl_port)}\n    db 127, 0, 0, 1\n    dq 0")
    add(f"umem_reg: dq umem, {frames * CHUNK}\n    dd {CHUNK}, 0, 0, 0")
    add(f"fill_size: dd {sizes[FILL]}")
    add(f"comp_size: dd {sizes[COMPLETION]}")
    add(f"rx_size: dd {sizes[RX]}")
    add(f"tx_size: dd {sizes[TX]}")
    add(f"offsets_len: dd {RING_OFFSETS_BYTES * len(RINGS)}")
    add("flags_len: dd 4")
    add("statistics_len: dd 48")
    add(f"ifreq: db {_bytes_of(o.interface)}")
    add(f"    times ifreq+40-$ db 0")
    add(f"xdp_addr: dw {AF_XDP}, {XDP_USE_NEED_WAKEUP}\n    dd 0, {o.queue}, 0")
    add(f"map_path: db {_bytes_of(o.map_path)}")
    add("alignb 8")
    add("obj_attr: dq map_path, 0, 0")
    add("update_attr: dd 0, 0\n    dq queue_key, xsk_word, 0")
    add(f"queue_key: dd {o.queue}")
    add("xsk_word: dd 0")
    add(f"waiting: dd 0\n    dw {POLLIN}, 0")
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
    add(f"cells: resb {CELL * image.cell_count}")
    for state in image.states:
        if state.initial is None:
            add("alignb 16")
            add(f"array_{state.name}: resb {state.length * state.element_size}")
    add("alignb 8")
    for position in range(k):
        add(f"state_{position}: resq 1")
    for name in ("pool", "pool_count", "turn", "ctl_fd", "xsk_fd", "egress_paused", "patience"):
        add(f"{name}: resq 1")
    for name, _page, _position, _entry in RINGS:
        for field in ("prod", "cons", "descs", "flags"):
            add(f"{name}_{field}_at: resq 1")
    add(f"offsets: resq {RING_OFFSETS_BYTES * len(RINGS) // 8}")
    add("statistics: resq 6")
    add("counters:")
    for name in image.counters:
        add(f"cnt_{name}: resq 1")
    for name in ("rx_cons", "tx_prod", "tx_seen", "fill_prod", "comp_cons", "batch", "reaped", "bound_flags"):
        add(f"{name}: resd 1")
    add("peer: resb 16")
    add("peer_len: resd 1")
    add("alignb 8")
    add("control_buffer: resb 16")
    add("alignb 4096")
    add(f"umem: resb {frames * CHUNK}")
    add(f"bss_end_page: resb {PAGE}")
    add("bss_end:")
    add("section .note.GNU-stack noalloc noexec nowrite progbits")
    return "\n".join(lines) + "\n"
