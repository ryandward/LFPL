"""The cells the compiler itself runs on (spec section 1).

Every cell is minted once, when a store is constructed. Nothing in this module or any other adds
to it afterwards; rules only move cells around.

Cells really are 16 bytes in one bytearray: 1 tag byte and 15 payload bytes.
A stage cannot hide state in a cell, because `CellStore.put` takes bytes and
refuses more than 15 of them. Frames larger than that must chain real cells,
which is what makes the occupancy bound (Theorem 7.2) measurable, not assumed.
"""
from __future__ import annotations

CELL_BYTES = 16
PAYLOAD_BYTES = 15


class ConservationError(Exception):
    """A location appeared twice, or vanished. Theorem 7.1 is violated."""


class PayloadOverflow(Exception):
    """A stage tried to put more than 15 bytes in a cell."""




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
