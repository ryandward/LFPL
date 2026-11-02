"""src/machine.py: the runtime's turn in LFPL, lowered with examples/ipv4.lfpl to Verilog, frames in and out as bytes.

The machine is held to the x86 build of the same handler on live sockets, frame by frame: what x86 sends, the machine
sends to the same port, and what x86 drops or refuses, the machine drops or refuses for the same reason. Every turn is
held to the bound the emitter stated before anything ran, and afterwards every cell is in exactly one place.
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.machine import Machine, build  # noqa: E402
from src.rtcheck import Checker, RTError  # noqa: E402
from src.rtlower import rt_lines  # noqa: E402
from src.turn import OUTCOMES, TurnRefused, turn_source  # noqa: E402
from tests.reference import ipv4_model as model  # noqa: E402
from tests.test_forward import ROOT, Forwarder  # noqa: E402

SOURCE = os.path.join(ROOT, "examples", "ipv4.lfpl")
N_MAX = 217
SIMULATOR = unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog is not installed")
_BUILT: dict = {}


def machine(**options) -> Machine:
    key = tuple(sorted(options.items()))
    if key not in _BUILT:
        with open(SOURCE, "rb") as handle:
            _BUILT[key] = build(handle.read(), n_max=N_MAX, **options)
    return _BUILT[key]


@dataclass
class Run:
    outputs: list   # (port, bytes), in the order they left
    turns: list     # (outcome, cycles), in the order they ran
    final: tuple    # the pool, the ingress slot and the queue, when it was done
    words: dict     # every cell's contents, when it was done


def simulate(m: Machine, frames: list) -> Run:
    """frames: (hold egress while it goes in, the bytes)."""
    with tempfile.TemporaryDirectory() as work:
        for name, text in (("machine.v", m.top), ("serve_core.v", m.core), ("cells.mem", m.image)):
            with open(os.path.join(work, name), "w") as handle:
                handle.write(text)
        with open(os.path.join(work, "frames.txt"), "w") as handle:
            for hold, frame in frames:
                handle.write(f"{int(hold)} {len(frame)} " + " ".join(f"{byte:02x}" for byte in frame) + "\n")
        compiled = subprocess.run(["iverilog", "-g2005", f"-DCELLS={m.cells}", "-o", os.path.join(work, "sim"),
                                   os.path.join(ROOT, "tests", "tb_machine.v"), os.path.join(work, "machine.v"),
                                   os.path.join(work, "serve_core.v")], capture_output=True, text=True)
        assert compiled.returncode == 0, compiled.stderr
        done = subprocess.run(["vvp", os.path.join(work, "sim"), "+frames=frames.txt"], cwd=work, check=True,
                              capture_output=True, text=True, timeout=3000)
    run = Run([], [], (), {})
    for line in done.stdout.splitlines():
        parts = line.split()
        if parts[0] == "OUT":
            run.outputs.append((int(parts[1]), bytes.fromhex(parts[2])))
        elif parts[0] == "TURN":
            run.turns.append((OUTCOMES[int(parts[1])], int(parts[2])))
        elif parts[0] == "FINAL":
            run.final = tuple(int(part) for part in parts[1:])
        elif parts[0] == "WORD":
            run.words[int(parts[1])] = int(parts[2], 16)
    assert done.stdout.strip().endswith("DONE"), done.stdout[-500:]
    return run


def holdings(m: Machine, run: Run) -> list:
    """Every cell the pool, ingress and the queue hold, walked through the store as the machine left it."""
    held: list = []

    def at(cell: int, off: int, size: int) -> int:
        return run.words[cell] >> (8 * off) & ((1 << (8 * size)) - 1)

    def record(first: int) -> None:
        walk = m.shape.item_walk
        cells = {0: first}
        for cell, off, inner in walk.links:
            cells[inner] = at(cells[cell], off, 8)
        held.extend(cells.values())
        chunk = at(cells[walk.chain[0]], walk.chain[1], 8)
        while chunk:
            held.append(chunk)
            chunk = at(chunk, 8, 8)

    pool, slot, queue = run.final
    while pool:
        held.append(pool)
        pool = at(pool, 0, 8)
    record(slot)
    place = queue
    for part, fields in enumerate(m.queue_layout):
        held.append(place)
        for name, (off, _size) in fields.items():
            if name != "$next":
                entry = at(place, off, 8)
                if entry:
                    held.append(entry)
                    record(at(entry, m.entry_layout["frame"][0], 8))
        if "$next" in fields:
            place = at(place, fields["$next"][0], 8)
    return held


def x86(frames: list) -> list:
    """What the x86 build does with each frame: ("forwarded", port, bytes), or the counter it went up by instead."""
    results = []
    with tempfile.TemporaryDirectory() as workdir:
        router = Forwarder(bytes(1024), workdir, build_options=("--n-max", str(N_MAX)), source=SOURCE)
        try:
            for socket in router.outputs:
                socket.settimeout(0.05)
            before = router.stats()
            for number, frame in enumerate(frames, 1):
                router.raw(frame)
                now = router.settle(number)
                moved = [name for name in ("forwarded", "dropped", "malformed", "too_large", "short", "full")
                         if now[name] != before[name]]
                assert len(moved) == 1, (frame.hex(), moved)
                if moved[0] == "forwarded":
                    sent = []
                    for port, socket in enumerate(router.outputs):
                        try:
                            sent.append((port, socket.recv(4096)))
                        except TimeoutError:
                            pass
                    assert len(sent) == 1, (frame.hex(), sent)
                    results.append(("forwarded", *sent[0]))
                else:
                    results.append((moved[0],))
                before = now
        finally:
            router.close()
    return results


# What the machine calls each of x86's counters.
CALLED = {"forwarded": "queued", "dropped": "dropped", "malformed": "runt", "too_large": "too_large"}


class TheBound(unittest.TestCase):
    """Before anything runs: the turn is typed with the handler, and its bound is certified and stated."""

    def test_it_is_stated_for_a_whole_turn(self):
        m = machine()
        self.assertRegex(m.report(), r"W = \d+ steps for any frame of up to 217 cells, with 4 places in the queue; "
                                     r"cycles <= \d+\*W \+ \d+ = \d+ a turn")
        self.assertEqual(m.turn_cycles, m.timing.k * m.bound.d + m.timing.c)
        self.assertIn(f"A turn takes at most {m.turn_cycles} clock cycles", m.top)

    def test_the_readme_quotes_what_the_emitter_says(self):
        with open(os.path.join(ROOT, "README.md")) as handle:
            self.assertIn(machine().report(), handle.read())

    def test_the_turn_is_lfpl_appended_to_the_handler(self):
        with open(SOURCE, "rb") as handle:
            text, shape = turn_source(handle.read(), n_max=N_MAX)
        self.assertIn("turn serve(pool: Pool, slot: Frame, queue: Queue, length: Int16, sent: Bool)", text)
        self.assertEqual((shape.wire, shape.parts, shape.capacity), (34, 5, 34 + 7 * (N_MAX - 5)))

    def test_a_certificate_that_falls_short_is_refused(self):
        m = machine()
        checker = Checker()
        for line in rt_lines(io.BytesIO(m.turn_text.encode())):
            checker.feed(line)
        with self.assertRaisesRegex(RTError, "falls"):
            checker.verify_turn("serve", {1: N_MAX, 2: 3 + 4 * (1 + N_MAX)}, {})

    def test_a_longer_queue_costs_more_and_is_still_bounded(self):
        self.assertGreater(machine(k_out=8).bound.d, machine().bound.d)


class ShapesItDoesNotServe(unittest.TestCase):
    def refused(self, source: str, message: str) -> None:
        with self.assertRaisesRegex(TurnRefused, message):
            turn_source(source.encode(), n_max=N_MAX)

    def test_an_output_built_in_a_cell(self):
        with open(os.path.join(ROOT, "examples", "forward.lfpl")) as handle:
            self.refused(handle.read(), "reads a verdict|state slots")

    def test_a_handler_that_draws_cells(self):
        self.refused("""data Bytes;
record F { a: Int8, length: Int16, body: Bytes };
enum V = Drop | Send(port: Int8);
handler h(f: F, free: Free) : (V, F, Free) = match f with F(d, a, length, body) =>
  match free with End => (Drop, F@d {}, End) | More(c, rest) => (Send(a), F@d {}, More@c(rest)) end end;
""", r"draws R\(n\) = 0 n \+ 1")

    def test_a_name_the_turn_declares(self):
        self.refused("""data Bytes;
record F { a: Int8, length: Int16, body: Bytes };
enum V = Drop | Send(port: Int8);
fn split(x: Int8) : Int8 = x;
handler h(f: F, free: Free) : (V, F, Free) = match f with F(d, a, length, body) => (Send(split(a)), F@d {}, free) end;
""", "declares split")

    def test_a_verdict_with_more_than_a_port(self):
        self.refused("""data Bytes;
record F { a: Int8, length: Int16, body: Bytes };
enum V = Drop | Send(port: Int8, why: Int8);
handler h(f: F, free: Free) : (V, F, Free) = match f with F(d, a, length, body) => (Send(a, 1), F@d {}, free) end;
""", "port: Int8")


@SIMULATOR
class TheCorpus(unittest.TestCase):
    def test_every_frame_as_the_x86_build_does(self):
        """Every frame of the corpus, and one longer than ingress holds, through both, in the same order."""
        frames = [frame for _name, _verdict, frame in model.corpus()]
        frames.append(model.ipv4_frame(payload=bytes(range(256)) * 6)[:1519])
        m = machine()
        run = simulate(m, [(False, frame) for frame in frames])
        expected = x86(frames)
        what = [outcome for outcome, _cycles in run.turns if outcome != "idle"]
        self.assertEqual(what, [CALLED[result[0]] for result in expected])
        self.assertEqual(run.outputs, [(port, sent) for name, *rest in expected if name == "forwarded"
                                       for port, sent in [rest]])
        self.assertEqual(len([o for o in what if o == "too_large"]), 1)
        for outcome, cycles in run.turns:
            self.assertLessEqual(cycles, m.turn_cycles, outcome)
        held = holdings(m, run)
        self.assertEqual(sorted(held), list(range(1, m.cells + 1)))


@SIMULATOR
class WhatTheQueueAndThePoolRefuse(unittest.TestCase):
    FRAMES = [frame for _name, verdict, frame in model.corpus() if verdict == "forward"]

    def test_full_once_every_place_is_taken(self):
        """Egress held: four frames take the four places, the fifth finds the last one taken, and when egress goes,
        the four leave in the order they came."""
        frames = self.FRAMES[:5]
        m = machine()
        run = simulate(m, [(True, frame) for frame in frames])
        what = [outcome for outcome, _cycles in run.turns if outcome != "idle"]
        self.assertEqual(what, ["queued"] * 4 + ["full"])
        expected = [model.forward(frame, model.by_top_bits)[1] for frame in frames[:4]]
        self.assertEqual([sent for _port, sent in run.outputs], expected)
        self.assertEqual(sorted(holdings(m, run)), list(range(1, m.cells + 1)))

    def test_short_when_the_pool_runs_dry_before_the_record(self):
        """A frame of 34 bytes takes five cells for ingress's record and one for its place in the queue."""
        m = machine(cells=N_MAX + 3 + 5)
        run = simulate(m, [(False, self.FRAMES[0])])
        self.assertEqual([outcome for outcome, _cycles in run.turns], ["short"])
        self.assertEqual(run.outputs, [])
        self.assertEqual(sorted(holdings(m, run)), list(range(1, m.cells + 1)))

    def test_short_when_the_pool_runs_dry_in_the_chain(self):
        """Eight cells: six for the record and the place, and two of the three chunks. The two go back, and so do
        the three chunks, to where they were in ingress."""
        frame = model.ipv4_frame(payload=bytes(range(20)))
        self.assertEqual(len(frame), 34 + 20)
        m = machine(cells=N_MAX + 3 + 8)
        run = simulate(m, [(False, frame), (False, frame)])
        self.assertEqual([outcome for outcome, _cycles in run.turns], ["short", "short"])
        self.assertEqual(sorted(holdings(m, run)), list(range(1, m.cells + 1)))

    def test_an_earlier_frames_bytes_are_never_read(self):
        """A frame refused as too large leaves its bytes in ingress's chunks. The next frame's last chunk is cleared
        after its last byte, so a handler that reads all seven bytes of a chunk sees only its own frame."""
        source = b"""data Bytes = Last | Chunk(used: Int8, b0: Int8, b1: Int8, b2: Int8, b3: Int8, b4: Int8, b5: Int8, b6: Int8, next: Bytes);
record F { a: Int8, sum: Int8, length: Int16, body: Bytes };
enum V = Drop | Send(port: Int8);
handler h(f: F, free: Free) : (V, F, Free) = match f with F(d, a, sum, length, body) =>
  match body with
  | Last => (Drop, F@d { body = Last }, free)
  | Chunk(c, used, b0, b1, b2, b3, b4, b5, b6, next) =>
      (Send(1), F@d { sum = +(b0, +(b1, +(b2, +(b3, +(b4, +(b5, b6)))))), body = Chunk@c(used, b0, b1, b2, b3, b4, b5, b6, next) }, free)
  end end;
"""
        m = build(source, n_max=8, k_out=2)
        run = simulate(m, [(False, b"\xff" * 60), (False, bytes((0x11, 0, 1)))])
        self.assertEqual([outcome for outcome, _cycles in run.turns if outcome != "idle"], ["too_large", "queued"])
        self.assertEqual(run.outputs, [(1, bytes((0x11, 1, 1)))])
        self.assertEqual(sorted(holdings(m, run)), list(range(1, m.cells + 1)))

    def test_what_was_refused_leaves_room_for_what_comes_next(self):
        m = machine()
        frames = [(False, b"\x00" * 10), (False, model.ipv4_frame(payload=bytes(1600))), (False, self.FRAMES[1])]
        run = simulate(m, frames)
        self.assertEqual([outcome for outcome, _cycles in run.turns if outcome != "idle"],
                         ["runt", "too_large", "queued"])
        self.assertEqual([sent for _port, sent in run.outputs], [model.forward(self.FRAMES[1], model.by_top_bits)[1]])


if __name__ == "__main__":
    unittest.main()
