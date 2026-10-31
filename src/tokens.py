"""The conserved location universe (spec section 1).

    Lambda = Lambda_cell + Lambda_reg + Lambda_slot + {phi}

Every location is minted once, when a universe is constructed. Nothing in this
module or any other adds to it afterwards; rules only move tokens around.

Cells really are 16 bytes in one bytearray: 1 tag byte and 15 payload bytes.
A stage cannot hide state in a cell, because `CellStore.put` takes bytes and
refuses more than 15 of them. Frames larger than that must chain real cells,
which is what makes the occupancy bound (Theorem 7.2) measurable, not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Union

CELL_BYTES = 16
PAYLOAD_BYTES = 15

# Lambda_reg. rsp is not a token: it is the origin of the frame manifold.
REGS = ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")
RAX, RBX, RCX, RDX, RSI, RDI, RBP, R8, R9, R10, R11, R12, R13, R14, R15 = range(15)
REG_COUNT = len(REGS)
ALL_REGS_MASK = (1 << REG_COUNT) - 1

_LEGACY_32 = {RAX: "eax", RBX: "ebx", RCX: "ecx", RDX: "edx",
              RSI: "esi", RDI: "edi", RBP: "ebp"}
_LEGACY_16 = {RAX: "ax", RBX: "bx", RCX: "cx", RDX: "dx",
              RSI: "si", RDI: "di", RBP: "bp"}
_LEGACY_8 = {RAX: "al", RBX: "bl", RCX: "cl", RDX: "dl",
             RSI: "sil", RDI: "dil", RBP: "bpl"}


def reg_name(r: int, bytes_wide: int = 8) -> str:
    """The subregister of `r` that is `bytes_wide` bytes wide."""
    if r not in range(REG_COUNT):
        raise ValueError(f"not a register token: {r}")
    if bytes_wide == 8:
        return REGS[r]
    legacy = {4: _LEGACY_32, 2: _LEGACY_16, 1: _LEGACY_8}[bytes_wide]
    if r in legacy:
        return legacy[r]
    return REGS[r] + {4: "d", 2: "w", 1: "b"}[bytes_wide]


class ConservationError(Exception):
    """A location appeared twice, or vanished. Theorem 7.1 is violated."""


class HomeUniquenessError(Exception):
    """Two live variables share a home. Invariant H is violated."""


class PayloadOverflow(Exception):
    """A stage tried to put more than 15 bytes in a cell."""


# ---------------------------------------------------------------- locations


@dataclass(frozen=True)
class LReg:
    """reg r"""
    r: int

    def __str__(self) -> str:
        return REGS[self.r]


@dataclass(frozen=True)
class LMem:
    """mem [base + off], a field of the cell whose pointer the binder `base` holds."""
    base: int
    off: int
    size: int

    def __str__(self) -> str:
        return f"mem[b{self.base}+{self.off}]:{self.size}"


@dataclass(frozen=True)
class LSlot:
    """mem [rsp + 8k], a Lambda_slot coordinate."""
    k: int

    def __str__(self) -> str:
        return f"slot{self.k}"


@dataclass(frozen=True)
class LImm:
    n: int

    def __str__(self) -> str:
        return f"imm {self.n}"


@dataclass(frozen=True)
class LVirt:
    """virt(C, u): an unmaterialized constructor. It occupies no storage."""
    ctor: int
    fields: tuple

    def __str__(self) -> str:
        return f"virt({self.ctor},{self.fields})"


@dataclass(frozen=True)
class LHole:
    """hole(p): the cell owned through binder p, as a capability."""
    p: int

    def __str__(self) -> str:
        return f"hole(b{self.p})"


Loc = Union[LReg, LMem, LSlot, LImm, LVirt, LHole]

# ------------------------------------------------------------- token forms

# Provenance word nu (section 6.1). Words are symbolic ints minted per function;
# NULL and BOTTOM are the two distinguished words the spec names.
BOTTOM = 0  # never written
NULL = 1    # the null pointer
FIRST_WORD = 2


@dataclass(frozen=True)
class Hole:
    """The hole token, diamond_a^y: location `a`, last evacuee `y`."""
    a: Loc
    y: int = BOTTOM


@dataclass(frozen=True)
class Full:
    """a |-> v"""
    a: Loc
    v: int


@dataclass(frozen=True)
class RegTok:
    """rho_r: a free register."""
    r: int


@dataclass(frozen=True)
class Flags:
    """phi when cc is None, otherwise phi_cc(o1, o2)."""
    cc: Optional[str] = None
    o1: Optional[str] = None
    o2: Optional[str] = None

    @property
    def free(self) -> bool:
        return self.cc is None


# ------------------------------------------------------------------- cells


class CellStore:
    """Lambda_cell: C cells of real memory, minted once."""

    def __init__(self, cells: int):
        if cells <= 0:
            raise ValueError("a universe needs at least one cell")
        self.C = cells
        self.mem = bytearray(cells * CELL_BYTES)

    def put(self, cell: int, tag: int, payload: bytes = b"") -> None:
        if len(payload) > PAYLOAD_BYTES:
            raise PayloadOverflow(
                f"{len(payload)} bytes do not fit in a cell (15 payload bytes)")
        base = cell * CELL_BYTES
        self.mem[base] = tag
        self.mem[base + 1:base + 1 + len(payload)] = payload
        # Zero the remainder so a previous owner's bytes never leak into reads.
        self.mem[base + 1 + len(payload):base + CELL_BYTES] = bytes(
            PAYLOAD_BYTES - len(payload))

    def tag(self, cell: int) -> int:
        return self.mem[cell * CELL_BYTES]

    def payload(self, cell: int) -> bytes:
        base = cell * CELL_BYTES
        return bytes(self.mem[base + 1:base + CELL_BYTES])


# --------------------------------------------------- registers, slots, flags


class Silicon:
    """The register, slot and flag tokens of the function being lowered.

    `free` is Delta_reg and `pinned` holds registers named by an enclosing
    destination. Homes live in the lowerer's binder frames, so the register
    partition of Theorem 7.1 is checked by passing them to `check_partition`.
    """

    def __init__(self) -> None:
        self.free = ALL_REGS_MASK
        self.pinned = 0
        self.free_slots = 0  # slots minted so far and currently not held
        self.slots_minted = 0  # F_f: the high-water mark of slot tokens
        self.flags = Flags()

    def take(self, r: int) -> None:
        bit = 1 << r
        if not self.free & bit:
            raise ConservationError(f"rho_{REGS[r]} is not free")
        self.free &= ~bit

    def give(self, r: int) -> None:
        bit = 1 << r
        if self.free & bit:
            raise ConservationError(f"rho_{REGS[r]} returned twice")
        self.free |= bit

    def pin(self, r: int) -> None:
        bit = 1 << r
        if self.pinned & bit:
            raise ConservationError(f"{REGS[r]} pinned twice")
        self.pinned |= bit

    def unpin(self, r: int) -> None:
        bit = 1 << r
        if not self.pinned & bit:
            raise ConservationError(f"{REGS[r]} is not pinned")
        self.pinned &= ~bit

    def take_slot(self) -> int:
        """Lowest free slot; F_f grows only when every minted slot is held."""
        for k in range(self.slots_minted):
            if self.free_slots >> k & 1:
                self.free_slots &= ~(1 << k)
                return k
        k = self.slots_minted
        self.slots_minted += 1
        return k

    def give_slot(self, k: int) -> None:
        if self.free_slots >> k & 1 or k >= self.slots_minted:
            raise ConservationError(f"slot {k} returned without being held")
        self.free_slots |= 1 << k

    def check_partition(self, home_regs: Iterable[int]) -> None:
        """Delta_reg + {r | H(x) = reg r} + pinned = Lambda_reg, disjointly."""
        seen = 0
        for r in home_regs:
            bit = 1 << r
            if seen & bit:
                raise HomeUniquenessError(f"{REGS[r]} is the home of two values")
            seen |= bit
        if seen & self.free or seen & self.pinned or self.free & self.pinned:
            raise ConservationError(
                f"register partition overlaps: free={self.free:#x} "
                f"homes={seen:#x} pinned={self.pinned:#x}")
        if seen | self.free | self.pinned != ALL_REGS_MASK:
            lost = ALL_REGS_MASK & ~(seen | self.free | self.pinned)
            names = [REGS[r] for r in range(REG_COUNT) if lost >> r & 1]
            raise ConservationError(f"register tokens lost: {names}")


def check_homes(homes: Mapping[object, Loc]) -> None:
    """Invariant H: no two live variables share a storage home.

    `imm` and `virt` occupy no storage, and `hole` names a capability rather
    than a stored value, so only registers, slots and cell fields can collide.
    """
    seen: dict[Loc, object] = {}
    for var, loc in homes.items():
        if isinstance(loc, (LImm, LVirt, LHole)):
            continue
        if loc in seen:
            raise HomeUniquenessError(
                f"{var} and {seen[loc]} share the home {loc}")
        seen[loc] = var
