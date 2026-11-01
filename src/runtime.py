"""The fixed image and the serving loop: LFPL-O sections 3, 6 and 7, as x86-64 assembly.

This is the part LFPL-O section 11.3 says must be trusted, so it is meant to be read: no
libc, no allocator, and a handful of system calls. Everything the program will ever use is
laid out at link time, locked and touched at startup.

An item arrives without being parsed. `recvmsg` scatters the datagram straight into cells,
through an iovec that names each scalar field of the item's header cell and then the payload of
each body cell, so the wire format follows from the item's declared fields and no byte of it is
copied twice. Egress gathers the same way with `sendmsg`. Refusing an item costs nothing: its
cells never left ingress, so the next datagram simply overwrites them.

What the source does not say, this file decides:

    Wire format     an item is its record's scalar fields in declaration order, little
                    endian and packed, then the bytes of its one abstract field.
    Abstract type   a chain of cells: used count at 0, 7 payload bytes, next at 8. Only the
                    last cell's used count means anything, and an empty body is null.
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
from typing import Callable, Optional

from .rtcheck import SIZES, Checker, Ctor

PAGE = 4096
CELL = 16
BODY_PAYLOAD = 7     # bytes of one chain cell
HEADER_BYTES_MAX = 15
K_IN = 1             # recvmsg takes one datagram per turn: the socket's own buffer is the ring
O_MAX = 1            # a handler returns one output
CONTROL_EVERY = 64   # turns between looks at the control port, so a flood cannot starve it
RUNTIME_DEPTH = 48   # the deepest the loop's own calls go: control, drain, free_chain, push_cell
IDLE_MICROSECONDS = 1000
COUNTERS = ("capacity", "pool", "served", "forwarded", "dropped", "too_large", "short", "full",
            "malformed", "send_errors", "recv_errors")
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
class OutCtor:
    name: str
    tag: Optional[int]       # None: the null constructor of a tagless type
    port: Optional[tuple[int, int]]  # offset, size
    scalars: tuple[tuple[int, int], ...]
    chain: Optional[int]


@dataclass(frozen=True)
class Options:
    in_port: int
    out_base: int
    ctl_port: int
    n_max: int = 216     # the largest item in cells: 1 header and 215 of 7 bytes, for a 1500 MTU
    k_out: int = 64      # the output queue, a power of two
    cells: Optional[int] = None  # None: section 3.4's C. Fewer can make admission refuse with Short


@dataclass(frozen=True)
class Image:
    handler: str
    states: tuple[StateSlot, ...]
    item_scalars: tuple[tuple[int, int], ...]
    item_chain: Optional[int]
    out_tagged: bool
    out: tuple[OutCtor, ...]
    reserve_draws: int
    steps: int
    depth: int
    options: Options

    @property
    def cell_count(self) -> int:
        """Section 3.4: C = K_in N_max + (R(N_max) + N_max) + K_out O_max N_out + S_cap."""
        n_max = self.options.n_max
        if self.options.cells is not None:
            return self.options.cells
        return K_IN * n_max + self.reserve_draws + n_max + self.options.k_out * O_MAX * n_max

    @property
    def capacity(self) -> int:
        return self.cell_count - K_IN * self.options.n_max

    @property
    def header_bytes(self) -> int:
        return sum(size for _off, size in self.item_scalars)

    @property
    def stack_bytes(self) -> int:
        needed = max(self.depth + 8, RUNTIME_DEPTH)
        return (needed + PAGE - 1) // PAGE * PAGE

    @property
    def body_cells(self) -> int:
        return self.options.n_max - 1 if self.item_chain is not None else 0


def bind(checker: Checker, handler_name: str, depth: int, options: Options,
         initial: dict[str, str]) -> Image:
    """Read the handler's types from the checked stream, and refuse what cannot be served."""
    fn = next(f for f in checker.functions.values() if f.handler)
    item_type = checker.norm(fn.params[-2])
    results = checker.tuples[checker._resolve(fn.ret)]
    out_type = checker.norm(results[-2])
    _check_reserve(checker, fn.params[-1])
    states = []
    for name, ty in checker.states:
        norm = checker.norm(ty)
        assert isinstance(norm, tuple) and norm[0] == "array"
        states.append(StateSlot(name, norm[1], SIZES[norm[2]], initial.pop(name, None)))
    if initial:
        raise BindingError(f"no state slot named {sorted(initial)[0]}")
    decl = checker._decl(item_type)
    if decl is None or decl.kind != "record":
        raise BindingError(f"section 3.1: this runtime receives an item as a record, and "
                           f"{item_type} is not one")
    scalars, chain = _split_fields(checker, decl.name, decl.ctors[0].fields)
    if sum(size for _o, size in scalars) > HEADER_BYTES_MAX:
        raise BindingError(f"section 3.1: {decl.name}'s header does not fit one cell's fields")
    out_decl = checker._decl(out_type)
    if out_decl is None or out_decl.kind not in ("tagged", "tagless", "record"):
        raise BindingError(f"section 3.1: the output {out_type} is not a data type")
    out = tuple(_out_ctor(checker, ctor) for ctor in out_decl.ctors)
    draws, steps = checker.bounds(handler_name)
    return Image(handler_name, tuple(states), tuple(scalars), chain, out_decl.kind == "tagged",
                 out, draws, steps, depth, options)


def _check_reserve(checker: Checker, reserve: str) -> None:
    """The pool and every reserve are threaded through offset 0, as the built-in Free is. A
    program may still declare its own Free, and then it has to have that shape."""
    decl = checker._decl(checker.norm(reserve))
    assert decl is not None
    full = [c for c in decl.ctors if c.tag != "null"]
    if decl.kind != "tagless" or len(full) != 1 or full[0].fields[0][:2] != (0, 8):
        raise BindingError("section 3.1: Free must be End | More(next: Free)")


def _split_fields(checker: Checker, owner: str, fields: list) -> tuple[list, Optional[int]]:
    scalars = []
    chain = None
    for off, size, ty, name in fields:
        decl = checker._decl(checker.norm(ty))
        if decl is None:
            scalars.append((off, size))
        elif decl.kind == "abstract" and chain is None:
            chain = off
        else:
            raise BindingError(f"section 3.1: {owner}.{name} is {decl.name}, and this runtime "
                               f"carries scalars and one abstract chain")
    return scalars, chain


def _out_ctor(checker: Checker, ctor: Ctor) -> OutCtor:
    port = None
    rest = []
    for entry in ctor.fields:
        if entry[3] == "port":
            if checker.norm(entry[2]) not in ("i8", "i16"):
                raise BindingError(f"section 3.1: {ctor.name}.port must be Int8 or Int16")
            port = (entry[0], entry[1])
        else:
            rest.append(entry)
    scalars, chain = _split_fields(checker, ctor.name, rest)
    tag = int(ctor.tag) if ctor.tag.isdigit() else (None if ctor.tag == "null" else 0)
    return OutCtor(ctor.name, tag, port, tuple(scalars), chain)


def _be16(port: int) -> str:
    return f"{port >> 8}, {port & 255}"


def _is_null(ctor: OutCtor, tagged: bool) -> bool:
    return ctor.tag is None and not tagged


def assembly(image: Image) -> str:
    """The runtime and the image, as NASM, to follow the compiled functions."""
    o = image.options
    if o.k_out & o.k_out - 1:
        raise BindingError("section 8.2: K_out must be a power of two")
    if image.body_cells > 1000:
        raise BindingError("section 8.2: N_max is past what one recvmsg can scatter")
    smallest = K_IN * o.n_max + image.reserve_draws + o.n_max
    if image.cell_count < smallest:
        raise BindingError(f"section 3.4: {image.cell_count} cells cannot hold ingress and one "
                           f"maximal transaction, which take {smallest}")
    if o.out_base + (1 << 8 * max((c.port[1] for c in image.out if c.port), default=0)) - 1 > 65535:
        raise BindingError(f"section 7.3: egress at {o.out_base} plus a port number can pass 65535")
    lines: list[str] = []
    add = lines.append
    item_count = len(image.item_scalars)
    iov_in = item_count + image.body_cells
    out_iov = HEADER_BYTES_MAX + max(1, image.options.n_max)
    k = len(image.states)

    add("section .text")
    add("global _start")
    add(f"handler_entry equ fn_{image.handler}")
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
    add("    ; 5: ingress takes the first N_max cells, and the rest are the pool")
    add("    mov edi, cells")
    add("    mov [slot_head], rdi")
    add("    mov rax, rdi")
    add("    call point_header_iovec")
    for position, (_off, size) in enumerate(image.item_scalars):
        add(f"    mov qword [in_iov+{16 * position + 8}], {size}")
    add("    lea rax, [rdi+16]")
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
    add(f"    mov eax, cells+{CELL * image.options.n_max}")
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
    add("    mov rdi, rax")
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
    add("    mov eax, 47")
    add("    mov rdi, [in_fd]")
    add("    mov esi, in_msg")
    add("    xor edx, edx")
    add("    syscall")
    add("    test rax, rax")
    add("    js .no_item")
    add("    ; Section 6.1, in order. A refusal counts itself and leaves the cells in ingress.")
    add("    test dword [in_msg+48], 32")
    add("    jnz .too_large")
    add(f"    cmp rax, {image.header_bytes}")
    add("    jb .malformed")
    add(f"    sub rax, {image.header_bytes}")
    add("    mov r12, rax")
    if image.item_chain is None:
        add("    test rax, rax")
        add("    jnz .too_large")
    add(f"    add rax, {BODY_PAYLOAD - 1}")
    add("    xor edx, edx")
    add(f"    mov ecx, {BODY_PAYLOAD}")
    add("    div rcx")
    add("    mov r13, rax")
    add("    lea r14, [rax+1]")
    add("    mov rax, [admission]")
    add("    imul rax, r14")
    add("    add rax, [admission+8]")
    add("    mov r15, rax")
    add("    add rax, r14")
    add("    cmp [pool_count], rax")
    add("    jb .short")
    add("    mov rax, [q_count]")
    add(f"    add rax, {O_MAX}")
    add(f"    cmp rax, {image.options.k_out}")
    add("    ja .full")
    add("    ; ADMIT: the item's cells leave ingress, and fresh ones from the pool take their place")
    add("    test r13, r13")
    add("    jz .linking")
    add("    mov rax, r13")
    add("    dec rax")
    add(f"    imul rcx, rax, {BODY_PAYLOAD}")
    add("    mov rdx, r12")
    add("    sub rdx, rcx")
    add("    shl rax, 4")
    add(f"    mov rdi, [in_iov+{16 * item_count}+rax]")
    add("    mov [rdi-1], dl")
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
    add("    mov [rdi+8], r8")
    add("    mov r8, rdi")
    add("    call pop_cell")
    add("    inc rax")
    add("    mov [rbx], rax")
    add("    jmp .link")
    add(".linked:")
    add("    mov r14, [slot_head]")
    if image.item_chain is not None:
        add(f"    mov [r14+{image.item_chain}], r8")
    add("    call pop_cell")
    add("    mov [slot_head], rax")
    add("    call point_header_iovec")
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
    add("    ; DONE: the state, the output and what is left of the reserve come back")
    for position in range(k):
        add(f"    mov [state_{position}], {RESULTS[position]}")
    add(f"    mov r12, {RESULTS[k]}")
    add(f"    mov r13, {RESULTS[k + 1]}")
    add(".give_back:")
    add("    test r13, r13")
    add("    jz .given_back")
    add("    mov rdi, r13")
    add("    mov r13, [r13]")
    add("    call push_cell")
    add("    jmp .give_back")
    add(".given_back:")
    add("    inc qword [cnt_served]")
    _dispatch(add, image, "r12", ".out")
    for position, ctor in enumerate(image.out):
        add(f".out_{position}:")
        if ctor.port is not None:
            add("    inc qword [cnt_forwarded]")
            add("    mov rax, [q_head]")
            add("    add rax, [q_count]")
            add(f"    and rax, {image.options.k_out - 1}")
            add("    mov [queue+rax*8], r12")
            add("    inc qword [q_count]")
            add("    jmp serve")
        else:
            add("    inc qword [cnt_dropped]")
            if ctor.chain is not None:
                add(f"    mov rdi, [r12+{ctor.chain}]")
                add("    call free_chain")
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
    add("; DRAIN: send what is queued, until it is empty or the socket would block")
    add("drain:")
    add("    cmp qword [egress_paused], 0")
    add("    jne .done")
    add("    cmp qword [q_count], 0")
    add("    je .done")
    add("    mov rax, [q_head]")
    add("    mov r12, [queue+rax*8]")
    _dispatch(add, image, "r12", ".send")
    for position, ctor in enumerate(image.out):
        add(f".send_{position}:")
        if ctor.port is None:
            add("    jmp .sent")  # never queued: only constructors with a port are
            continue
        port_off, port_size = ctor.port
        add(f"    movzx eax, {_WIDTH[port_size]} [r12+{port_off}]")
        add(f"    add eax, {image.options.out_base}")
        add("    xchg al, ah")
        add("    mov word [out_addr+2], ax")
        for index, (off, size) in enumerate(ctor.scalars):
            add(f"    lea rax, [r12+{off}]")
            add(f"    mov [out_iov+{16 * index}], rax")
            add(f"    mov qword [out_iov+{16 * index + 8}], {size}")
        add(f"    mov ecx, {len(ctor.scalars)}")
        add(f"    mov ebx, out_iov+{16 * len(ctor.scalars)}")
        if ctor.chain is not None:
            add(f"    mov rdi, [r12+{ctor.chain}]")
            add(".chain_%d:" % position)
            add("    test rdi, rdi")
            add(f"    jz .send_ready_{position}")
            add("    lea rax, [rdi+1]")
            add("    mov [rbx], rax")
            add(f"    mov eax, {BODY_PAYLOAD}")
            add("    cmp qword [rdi+8], 0")
            add(f"    jne .whole_{position}")
            add("    movzx eax, byte [rdi]")
            add(f".whole_{position}:")
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
        add("    jns .counted")
        add("    inc qword [cnt_send_errors]")
        add(".counted:")
        if ctor.chain is not None:
            add(f"    mov rdi, [r12+{ctor.chain}]")
            add("    call free_chain")
        add("    jmp .sent")
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
    add("    mov eax, 44")
    add("    mov rdi, [ctl_fd]")
    add("    mov esi, counters")
    add(f"    mov edx, {8 * len(COUNTERS)}")
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
    add("; rax: the header cell. The iovec names its scalar fields, so the kernel fills them.")
    add("point_header_iovec:")
    for position, (off, _size) in enumerate(image.item_scalars):
        add(f"    lea rsi, [rax+{off}]")
        add(f"    mov [in_iov+{16 * position}], rsi")
    add("    ret")
    add("")
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
    add("free_chain:")
    add("    test rdi, rdi")
    add("    jz .done")
    add("    mov rbx, [rdi+8]")
    add("    call push_cell")
    add("    mov rdi, rbx")
    add("    jmp free_chain")
    add(".done:")
    add("    ret")
    add("")
    add("section .data align=4096")
    add("data_start:")
    add("all_signals: dq -1")
    add("ignore_action: dq 1, 0, 0, 0")
    add(f"idle_time: dq 0, {IDLE_MICROSECONDS}")
    add(f"in_addr: dw 2\n    db {_be16(o.in_port)}\n    db 127, 0, 0, 1\n    dq 0")
    add(f"ctl_addr: dw 2\n    db {_be16(o.ctl_port)}\n    db 127, 0, 0, 1\n    dq 0")
    add("out_addr: dw 2\n    db 0, 0\n    db 127, 0, 0, 1\n    dq 0")
    add(f"in_msg: dq 0, 0, in_iov, {iov_in}, 0, 0, 0")
    add("out_msg: dq out_addr, 16, out_iov, 0, 0, 0, 0")
    add("; the admission table of section 6.1: R(n) = a n + b, W(n) = c n + d")
    add(f"admission: dq 0, {image.reserve_draws}, 0, {image.steps}")
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
    add(f"queue: resq {image.options.k_out}")
    add(f"in_iov: resq {2 * iov_in}")
    add(f"out_iov: resq {2 * out_iov}")
    for position in range(k):
        add(f"state_{position}: resq 1")
    for name in ("pool", "pool_count", "slot_head", "q_head", "q_count", "turn", "in_fd", "out_fd",
                 "ctl_fd", "egress_paused"):
        add(f"{name}: resq 1")
    add("counters:")
    for name in COUNTERS:
        add(f"cnt_{name}: resq 1")
    add("peer: resb 16")
    add("peer_len: resd 1")
    add("alignb 8")
    add("control_buffer: resb 16")
    add(f"bss_end_page: resb {PAGE}")
    add("bss_end:")
    add("section .note.GNU-stack noalloc noexec nowrite progbits")
    return "\n".join(lines) + "\n"


def _dispatch(add: Callable[[str], None], image: Image, register: str, stem: str) -> None:
    """Jump to the arm for the constructor this register holds. The last needs no test."""
    ctors = image.out
    for position, ctor in enumerate(ctors[:-1]):
        if image.out_tagged:
            add(f"    cmp byte [{register}], {ctor.tag}")
            add(f"    je {stem}_{position}")
        else:
            add(f"    test {register}, {register}")
            add(f"    {'jz' if _is_null(ctor, False) else 'jnz'} {stem}_{position}")
    add(f"    jmp {stem}_{len(ctors) - 1}")
