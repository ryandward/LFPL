"""The Abstract Token Machine, ring level (spec sections 2.3 and 3.2).

    <F || c2, c3, c4, c5 || S_lex, S_par, S_chk, S_low, S_egr>

F is the return edge. Each channel holds at most one cell, except the queue
EXPAND builds on c5. Scheduling is pull-driven: a stage fires only when its
output channel is empty. The methods named after rules are the only code that
moves a cell between owners, so `check_conservation` can recompute the
partition of Theorem 7.1 from those independent containers. It would not be a
real check if one owner table were the source of truth for every container.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Iterator, Optional

from .tokens import PAYLOAD_BYTES, CellStore, ConservationError


class Reject(Exception):
    """The program text is outside the language: a syntax or type diagnostic."""


class FStarved(Exception):
    """F was empty when a premise needed a cell, so C < B(D). Theorem 7.2's
    hypothesis failed. Real machine code has no such branch."""


# Content of a byte cell (spec: Content ::= byte b). The tag separates data
# from the end-of-input marker the ring carries down to each stage.
BYTE_DATA, BYTE_EOF = 0, 1


class FrameStack:
    """A stack of frames held in real cells; a frame longer than 15 bytes chains.

    `at` gives random access by frame index because the lowerer addresses
    binders by position. The Python lists only name cells the stage already
    owns: their length is the stage's frame count, which is what is bounded.
    """

    def __init__(self, ring: "Ring"):
        self.ring = ring
        self.cells: list[int] = []
        self.starts: list[int] = []

    def __len__(self) -> int:
        return len(self.starts)

    def push(self, tag: int, data: bytes, first: Optional[int] = None) -> None:
        """Push a frame. `first` is a cell the stage already holds in sigma,
        which is how a parser frame is paid by the opener token's own cell."""
        count = max(1, (len(data) + PAYLOAD_BYTES - 1) // PAYLOAD_BYTES)
        self.starts.append(len(self.cells))
        for i in range(count):
            cell = first if i == 0 and first is not None else self.ring.receive()
            self.ring.store.put(
                cell, tag, data[i * PAYLOAD_BYTES:(i + 1) * PAYLOAD_BYTES])
            self.cells.append(cell)
        self.ring.frame_moved(self, first)

    def _span(self, index: int) -> tuple[int, int]:
        start = self.starts[index]
        stop = self.starts[index + 1] if index + 1 < len(self.starts) else len(self.cells)
        return start, stop

    def at(self, index: int) -> tuple[int, bytes]:
        start, stop = self._span(index)
        data = b"".join(self.ring.store.payload(c) for c in self.cells[start:stop])
        return self.ring.store.tag(self.cells[start]), data

    def peek(self) -> tuple[int, bytes]:
        return self.at(len(self.starts) - 1)

    def rewrite(self, index: int, tag: int, data: bytes) -> None:
        start, stop = self._span(index)
        if len(data) > (stop - start) * PAYLOAD_BYTES:
            raise ConservationError("a frame cannot grow in place")
        for i, cell in enumerate(self.cells[start:stop]):
            self.ring.store.put(
                cell, tag, data[i * PAYLOAD_BYTES:(i + 1) * PAYLOAD_BYTES])

    def pop(self) -> None:
        """Pop the top frame, sending its cells to F."""
        start, _ = self._span(len(self.starts) - 1)
        for cell in self.cells[start:]:
            self.ring.release(cell)
        del self.cells[start:]
        self.starts.pop()


class Stage:
    """One coroutine on the ring: control state e_S plus cells it owns."""

    name = "stage"

    def __init__(self) -> None:
        self.sigma: set[int] = set()
        self.stacks: list[FrameStack] = []

    def attach(self, ring: "Ring") -> None:
        self.ring = ring

    def stack(self) -> FrameStack:
        frames = FrameStack(self.ring)
        self.stacks.append(frames)
        return frames

    def held_cells(self) -> Iterator[int]:
        yield from self.sigma
        for frames in self.stacks:
            yield from frames.cells

    def depth(self) -> int:
        """|K_S| in frames: the number of constructs this stage has open."""
        return 0

    def has_work(self) -> bool:
        """True if the stage can fire with no input cell (a held pushback)."""
        return False

    def fire(self, cell: Optional[int]) -> None:
        raise NotImplementedError


class Ring:
    """Runs the stages. `sink` is the Egress side of the `write` transformer.

    `stages` is [Lex, Parse, Check, Lower]. channels[0] is the Ingest -> Lex edge
    (c1, the W = 1 term of Theorem 7.2), and channels[1:] are c2, c3 and c4. The
    last stage's output is the c5 queue, which only EXPAND and PASS can fill.
    """

    def __init__(self, cells: int, source: Iterator[int], stages: list[Stage],
                 sink: Callable[[int, bytes], None],
                 on_step: Optional[Callable[["Ring"], None]] = None):
        self.store = CellStore(cells)
        self.F: list[int] = list(range(cells))  # the single mint (section 2.1)
        self.channels: list[Optional[int]] = [None] * len(stages)
        self.c5_queue: deque[int] = deque()
        self.stages = stages
        self.source = source
        self.sink = sink
        self.on_step = on_step
        self.counts = {r: 0 for r in ("ingest", "pass", "push", "pop", "drop",
                                      "expand", "egress", "frame")}
        self.peak_cells = 0
        self.peak_depth = 0
        self.peak_queue = 0
        self.ingest_finished = False
        self.current: Optional[Stage] = None
        for stage in stages:
            stage.attach(self)

    # --- primitive location transfers: every rule below is built from these

    def receive(self) -> int:
        """A cell from F. Theorem 7.2 says this is total when C >= B(D_max)."""
        if not self.F:
            raise FStarved(
                f"F is empty with {self.store.C} cells; C is below B(D_max)")
        return self.F.pop()

    def release(self, cell: int) -> None:
        self.F.append(cell)

    def frame_moved(self, frames: FrameStack, first: Optional[int]) -> None:
        """The FrameStack took `first` out of the firing stage's sigma."""
        self.counts["frame"] += 1
        if first is not None:
            assert self.current is not None
            self.current.sigma.remove(first)

    def _out(self, stage: Stage, cell: int) -> None:
        index = self.stages.index(stage) + 1
        if index == len(self.stages):
            if self.c5_queue:
                raise ConservationError(f"{stage.name} wrote to a full c5")
            self.c5_queue.append(cell)
            return
        if self.channels[index] is not None:
            raise ConservationError(f"{stage.name} wrote to a full channel")
        self.channels[index] = cell

    # --- the rules of section 3.2. `cell` is the input cell, held in sigma.

    def pass_(self, stage: Stage, cell: int, tag: int, payload: bytes = b"") -> None:
        """PASS: c_in -> c_out, contents rewritten."""
        stage.sigma.remove(cell)
        self.store.put(cell, tag, payload)
        self._out(stage, cell)
        self.counts["pass"] += 1

    def emit_fresh(self, stage: Stage, tag: int, payload: bytes = b"") -> None:
        """The half of PUSH that sends a cell received from F down the channel."""
        cell = self.receive()
        self.store.put(cell, tag, payload)
        self._out(stage, cell)
        self.counts["push"] += 1

    def drop(self, stage: Stage, cell: int) -> None:
        """DROP: c_in -> F."""
        stage.sigma.remove(cell)
        self.release(cell)
        self.counts["drop"] += 1

    def expand(self, stage: Stage, cell: Optional[int],
               contents: list[tuple[int, bytes]]) -> None:
        """EXPAND: the input cell plus cells from F, queued on c5 in order.

        This is the one place a stage takes cells it was not handed (section
        8): Egress returns each one in its own step, turning size into time.
        """
        if self.c5_queue:
            raise ConservationError(f"{stage.name} expanded onto a full queue")
        reused = False
        for tag, payload in contents:
            if cell is not None and not reused:
                out = cell
                stage.sigma.remove(cell)
                reused = True
            else:
                out = self.receive()
            self.store.put(out, tag, payload)
            self.c5_queue.append(out)
        if cell is not None and not reused:
            self.drop(stage, cell)
        self.peak_queue = max(self.peak_queue, len(self.c5_queue))
        self.counts["expand"] += 1

    def ingest(self) -> None:
        """INGEST: read_1 fills an owned cell and hands it back."""
        cell = self.receive()
        byte = next(self.source, None)
        if byte is None:
            self.store.put(cell, BYTE_EOF)
            self.ingest_finished = True
        else:
            self.store.put(cell, BYTE_DATA, bytes((byte,)))
        self.channels[0] = cell
        self.counts["ingest"] += 1

    def egress(self) -> None:
        """EGRESS: write_1 drains a cell, which goes straight back to F."""
        cell = self.c5_queue.popleft()
        self.sink(self.store.tag(cell), self.store.payload(cell))
        self.release(cell)
        self.counts["egress"] += 1

    # --- the scheduler

    def run(self) -> None:
        """Fire stages, downstream first, until nothing can move."""
        last = len(self.stages) - 1
        while True:
            fired = False
            if self.c5_queue:
                self.egress()
                self._observe()
                fired = True
            for index in range(last, -1, -1):
                stage = self.stages[index]
                blocked = (bool(self.c5_queue) if index == last
                           else self.channels[index + 1] is not None)
                if blocked:
                    continue  # the pull discipline
                # Held work goes first: a pushback precedes anything still in
                # the channel, or tokens would be reordered.
                incoming = None if stage.has_work() else self.channels[index]
                if incoming is None and not stage.has_work():
                    continue
                if incoming is not None:
                    self.channels[index] = None
                    stage.sigma.add(incoming)
                self.current = stage
                stage.fire(incoming)
                self.current = None
                self._observe()
                fired = True
            if not self.ingest_finished and self.channels[0] is None:
                self.ingest()
                self._observe()
                fired = True
            if not fired:
                return

    def _observe(self) -> None:
        in_use = self.store.C - len(self.F)
        depth = max(stage.depth() for stage in self.stages)
        self.peak_cells = max(self.peak_cells, in_use)
        self.peak_depth = max(self.peak_depth, depth)
        if self.on_step is not None:
            self.on_step(self)

    # --- Theorem 7.1

    def check_conservation(self) -> None:
        """F + channels + the c5 queue + each stage's cells = Lambda_cell."""
        seen = bytearray(self.store.C)

        def claim(cell: int, owner: str) -> None:
            if seen[cell]:
                raise ConservationError(f"cell {cell} is held twice ({owner})")
            seen[cell] = 1

        for cell in self.F:
            claim(cell, "F")
        for cell in self.channels:
            if cell is not None:
                claim(cell, "channel")
        for cell in self.c5_queue:
            claim(cell, "c5 queue")
        for stage in self.stages:
            for cell in stage.held_cells():
                claim(cell, stage.name)
        missing = seen.count(0)
        if missing:
            raise ConservationError(f"{missing} cells left the universe")
