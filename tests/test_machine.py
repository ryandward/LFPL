"""src/machine.py: the runtime's turn in LFPL, lowered with examples/ipv4.lfpl to Verilog, frames in and out as bytes.

The machine is held to the x86 build of the same handler on live sockets, frame by frame: what x86 sends, the machine
sends to the same port, and what x86 drops or refuses, the machine drops or refuses for the same reason. Every turn is
held to the bound the emitter stated before anything ran, and afterwards every cell is in exactly one place.

What would hang is stopped by the design itself, not by a test's timeout, and WhatTheHardwareStops breaks the
generated Verilog on purpose to show it: a turn past its bound, a queue whose head no turn's outcome accounts for,
and a chain longer than any frame can have.
"""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.certify import certify_turn  # noqa: E402
from src.machine import HALTS, Machine, build  # noqa: E402
from src.rtcheck import Checker, RTError, Unbounded  # noqa: E402
from src.rtlower import rt_lines  # noqa: E402
from src.turn import OUTCOMES, TurnRefused, turn_source  # noqa: E402
from tests.reference import ipv4_model as model  # noqa: E402
from tests.test_forward import ROOT, Forwarder  # noqa: E402

SOURCE = os.path.join(ROOT, "examples", "ipv4.lfpl")
N_MAX = 217
SIMULATOR = unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog is not installed")
FORWARDED = [frame for _name, verdict, frame in model.corpus() if verdict == "forward"]
NOTHING_COUNTED = {"overruns": 0, "faults": 0, "cuts": 0}
_BUILT: dict = {}


def machine(**options) -> Machine:
    key = tuple(sorted(options.items()))
    if key not in _BUILT:
        with open(SOURCE, "rb") as handle:
            _BUILT[key] = build(handle.read(), n_max=N_MAX, **options)
    return _BUILT[key]


def sent(frame: bytes) -> bytes:
    """What ipv4.lfpl sends for a frame it forwards."""
    return model.forward(frame, model.by_top_bits)[1]


@dataclass
class Run:
    outputs: list   # (port, bytes), in the order they left
    turns: list     # (outcome, cycles), in the order they ran
    final: tuple    # the pool, the ingress slot, the queue's head and the rest of the queue, when it was done
    counted: dict   # turns stopped at their bound, turns whose queue broke the count, frames egress cut short
    words: dict     # every cell's contents, when it was done


def simulate(m: Machine, frames: list, *, top: Optional[str] = None, core: Optional[str] = None,
             image: Optional[str] = None) -> Run:
    """frames: (hold egress while it goes in, the bytes). `top`, `core` and `image` stand in for what src/machine.py
    wrote, for a test that breaks one of them on purpose."""
    with tempfile.TemporaryDirectory() as work:
        for name, text in (("machine.v", top or m.top), ("serve_core.v", core or m.core),
                           ("cells.mem", image or m.image)):
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
    run = Run([], [], (), {}, {})
    for line in done.stdout.splitlines():
        parts = line.split()
        if parts[0] == "OUT":
            run.outputs.append((int(parts[1]), bytes.fromhex(parts[2])))
        elif parts[0] == "TURN":
            code = int(parts[1])
            run.turns.append((HALTS[code] if code in HALTS else OUTCOMES[code], int(parts[2])))
        elif parts[0] == "FINAL":
            numbers = tuple(int(part) for part in parts[1:])
            run.final, run.counted = numbers[:4], dict(zip(("overruns", "faults", "cuts"), numbers[4:]))
        elif parts[0] == "WORD":
            run.words[int(parts[1])] = int(parts[2], 16)
    assert done.stdout.strip().endswith("DONE"), done.stdout[-500:]
    return run


def at(run: Run, cell: int, off: int, size: int) -> int:
    return run.words[cell] >> (8 * off) & ((1 << (8 * size)) - 1)


def record_cells(m: Machine, run: Run, first: int) -> dict:
    """The cells of the record whose first cell is `first`, by their place in it."""
    cells = {0: first}
    for cell, off, inner in m.shape.item_walk.links:
        cells[inner] = at(run, cells[cell], off, 8)
    return cells


def holdings(m: Machine, run: Run) -> list:
    """Every cell the pool, ingress and the queue hold, walked through the store as the machine left it."""
    held: list = []

    def record(first: int) -> None:
        cells = record_cells(m, run, first)
        held.extend(cells.values())
        chain = m.shape.item_walk.chain
        chunk = at(run, cells[chain[0]], chain[1], 8)
        while chunk:
            held.append(chunk)
            chunk = at(run, chunk, 8, 8)

    def entry(cell: int) -> None:
        if cell:
            held.append(cell)
            record(at(run, cell, m.entry_layout["frame"][0], 8))

    pool, slot, head, queue = run.final
    while pool:
        held.append(pool)
        pool = at(run, pool, 0, 8)
    record(slot)
    entry(head)
    place = queue
    for fields in m.queue_layout:
        held.append(place)
        for name, (off, _size) in fields.items():
            if name != "$next":
                entry(at(run, place, off, 8))
        if "$next" in fields:
            place = at(run, place, fields["$next"][0], 8)
    return held


def first_words(m: Machine) -> dict:
    """Every cell as cells.mem has it, before the first turn."""
    return {index: int(word, 16) for index, word in enumerate(m.image.split()) if index}


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
                    out = []
                    for port, socket in enumerate(router.outputs):
                        try:
                            out.append((port, socket.recv(4096)))
                        except TimeoutError:
                            pass
                    assert len(out) == 1, (frame.hex(), out)
                    results.append(("forwarded", *out[0]))
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

    def checker(self) -> Checker:
        checker = Checker()
        for line in rt_lines(io.BytesIO(machine().turn_text.encode())):
            checker.feed(line)
        checker.finish()
        return checker

    def test_it_is_stated_for_a_whole_turn(self):
        m = machine()
        self.assertRegex(m.report(), r"W = \d+ steps for any frame of up to 217 cells in and 217 out, with 4 places "
                                     r"in the queue; cycles <= \d+\*W \+ \d+ = \d+ a turn")
        self.assertEqual(m.turn_cycles, m.timing.k * m.bound.d + m.timing.c)
        self.assertIn(f"A turn takes at most {m.turn_cycles} clock cycles", m.top)
        self.assertIn(f"TURN_LIMIT = 32'd{m.turn_cycles};", m.top)

    def test_the_readme_quotes_what_the_emitter_says(self):
        with open(os.path.join(ROOT, "README.md")) as handle:
            self.assertIn(machine().report(), handle.read())

    def test_the_turn_is_lfpl_appended_to_the_handler(self):
        with open(SOURCE, "rb") as handle:
            text, shape = turn_source(handle.read(), n_max=N_MAX)
        self.assertIn("turn serve(pool: Pool, slot: Frame, head: Out, queue: Queue, length: Int16, sent: Bool)", text)
        self.assertEqual((shape.wire, shape.parts, shape.capacity, shape.n_out), (34, 5, 34 + 7 * (N_MAX - 5), N_MAX))

    def test_a_certificate_that_falls_short_is_refused(self):
        with self.assertRaisesRegex(RTError, "falls"):
            self.checker().verify_turn("serve", {1: N_MAX, 2: 1 + N_MAX}, {})

    def test_only_the_head_of_the_queue_is_charged(self):
        """A turn releases the head of the queue and no other frame, so saying how much the rest of the queue holds
        changes nothing: none of it is charged a step."""
        checker = self.checker()
        head = {1: N_MAX, 2: 1 + N_MAX}
        self.assertEqual(machine().bound, certify_turn(checker, "serve", head))
        self.assertEqual(certify_turn(checker, "serve", {**head, 3: 3 * (1 + N_MAX)}),
                         certify_turn(checker, "serve", head))

    def test_what_holds_cells_with_no_size_said_is_given_nothing(self):
        """A turn that walks a list it was given has no bound unless the list's size is said: it could be any length."""
        source = b"""data L = Nil | Cons(next: L);
fn count(l: L, pool: L, n: Int16) : (L, Int16) = match l with Nil => (pool, n) | Cons(d, t) => count(t, Cons@d(pool), +(n, 1)) end;
turn t(l: L, pool: L) : (L, Int16) = count(l, pool, 0);
"""
        checker = Checker()
        for line in rt_lines(io.BytesIO(source)):
            checker.feed(line)
        with self.assertRaisesRegex(Unbounded, "has no bound"):
            certify_turn(checker, "t", {})
        self.assertGreater(certify_turn(checker, "t", {0: 10}).d, 0)

    def test_a_free_given_back_costs_what_the_pool_does(self):
        """DRAW leaves a reserve's End arm out of a handler's bound, since a reserve of R(n) cells never reaches it.
        A turn walks the Free a handler gave back, which does end, so the walk costs what the same walk over the pool
        costs, End arm and all."""
        head = b"""data Pool = Dry | Cell(next: Pool);
data L = Nil | Cons(next: L);
handler h(l: L, free: Free) : (L, Free) = (l, free);
"""
        walks = (b"fn give_free(free: Free, pool: Pool) : Pool = match free with End => pool "
                 b"| More(c, rest) => give_free(rest, Cell@c(pool)) end;\n"
                 b"turn t(free: Free, pool: Pool) : Pool = give_free(free, pool);\n",
                 b"fn give(cells: Pool, pool: Pool) : Pool = match cells with Dry => pool "
                 b"| Cell(c, rest) => give(rest, Cell@c(pool)) end;\n"
                 b"turn t(cells: Pool, pool: Pool) : Pool = give(cells, pool);\n")
        bounds = []
        for walk in walks:
            checker = Checker()
            for line in rt_lines(io.BytesIO(head + walk)):
                checker.feed(line)
            checker.finish()
            bounds.append(certify_turn(checker, "t", {0: 10}))
        self.assertEqual(bounds[0], bounds[1])

    def test_a_longer_queue_costs_more_and_is_still_bounded(self):
        self.assertGreater(machine(k_out=8).bound.d, machine().bound.d)


class ShapesItDoesNotServe(unittest.TestCase):
    def refused(self, source: str, message: str, **options) -> None:
        with self.assertRaisesRegex(TurnRefused, message):
            turn_source(source.encode(), **{"n_max": N_MAX, **options})

    def test_an_output_built_in_a_cell(self):
        with open(os.path.join(ROOT, "examples", "forward.lfpl")) as handle:
            self.refused(handle.read(), "reads a verdict|state slots")

    def test_a_queue_of_one_place(self):
        with open(SOURCE) as handle:
            self.refused(handle.read(), "K_out of 1", k_out=1)

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

    def test_a_length_that_cannot_count_one_byte_past_ingress(self):
        """Ingress counts a frame up to one byte past what it holds, and the count has to fit in the length: one
        byte on the wire and 9362 chunks hold 65535 bytes, and 65536 does not fit in 16 bits."""
        source = """data Bytes;
record F { a: Int8, length: Int16, body: Bytes };
enum V = Drop | Send(port: Int8);
handler h(f: F, free: Free) : (V, F, Free) = match f with F(d, a, length, body) => (Send(a), F@d {}, free) end;
"""
        self.refused(source, "N_max of 9363", n_max=9363)
        self.assertEqual(turn_source(source.encode(), n_max=9362)[1].capacity, 65528)


@SIMULATOR
class TheCorpus(unittest.TestCase):
    def test_every_frame_as_the_x86_build_does(self):
        """Every frame of the corpus, the largest frame ingress holds, and one byte longer, through both, in the same
        order. Every turn is under the stated bound, and the README says how far under."""
        m = machine()
        largest = model.ipv4_frame(payload=bytes(m.shape.capacity - 34))
        self.assertEqual(len(largest), m.shape.capacity)
        frames = [frame for _name, _verdict, frame in model.corpus()]
        frames += [largest, largest + b"\x00"]
        run = simulate(m, [(False, frame) for frame in frames])
        expected = x86(frames)
        what = [outcome for outcome, _cycles in run.turns if outcome != "idle"]
        self.assertEqual(what, [CALLED[result[0]] for result in expected])
        self.assertEqual(run.outputs, [(port, out) for name, *rest in expected if name == "forwarded"
                                       for port, out in [rest]])
        self.assertEqual(what[-2:], ["queued", "too_large"])
        self.assertEqual(run.counted, NOTHING_COUNTED)
        for outcome, cycles in run.turns:
            self.assertLessEqual(cycles, m.turn_cycles, outcome)
        held = holdings(m, run)
        self.assertEqual(sorted(held), list(range(1, m.cells + 1)))
        admitted = max(cycles for outcome, cycles in run.turns if outcome != "idle")
        released = max(cycles for outcome, cycles in run.turns if outcome == "idle")
        with open(os.path.join(ROOT, "README.md")) as handle:
            self.assertIn(f"The longest admission measured took {admitted} cycles, and the longest release "
                          f"{released}, against the bound of {m.turn_cycles}", " ".join(handle.read().split()))


@SIMULATOR
class WhatTheQueueAndThePoolRefuse(unittest.TestCase):
    def test_full_once_every_place_is_taken(self):
        """Egress held: four frames take the four places, the fifth finds the last one taken, and when egress goes,
        the four leave in the order they came."""
        frames = FORWARDED[:5]
        m = machine()
        run = simulate(m, [(True, frame) for frame in frames])
        what = [outcome for outcome, _cycles in run.turns if outcome != "idle"]
        self.assertEqual(what, ["queued"] * 4 + ["full"])
        self.assertEqual([out for _port, out in run.outputs], [sent(frame) for frame in frames[:4]])
        self.assertEqual(run.counted, NOTHING_COUNTED)
        self.assertEqual(sorted(holdings(m, run)), list(range(1, m.cells + 1)))

    def test_short_when_the_pool_runs_dry_before_the_record(self):
        """A frame of 34 bytes takes five cells for ingress's record and one for its place in the queue."""
        m = machine(cells=N_MAX + len(machine().queue_layout) + 5)
        run = simulate(m, [(False, FORWARDED[0])])
        self.assertEqual([outcome for outcome, _cycles in run.turns], ["short"])
        self.assertEqual(run.outputs, [])
        self.assertEqual(sorted(holdings(m, run)), list(range(1, m.cells + 1)))

    def test_short_when_the_pool_runs_dry_in_the_chain(self):
        """Eight cells: six for the record and the place, and two of the three chunks. The two go back, and so do
        the three chunks, to where they were in ingress."""
        frame = model.ipv4_frame(payload=bytes(range(20)))
        self.assertEqual(len(frame), 34 + 20)
        m = machine(cells=N_MAX + len(machine().queue_layout) + 8)
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
        frames = [(False, b"\x00" * 10), (False, model.ipv4_frame(payload=bytes(1600))), (False, FORWARDED[1])]
        run = simulate(m, frames)
        self.assertEqual([outcome for outcome, _cycles in run.turns if outcome != "idle"],
                         ["runt", "too_large", "queued"])
        self.assertEqual([out for _port, out in run.outputs], [sent(FORWARDED[1])])


@SIMULATOR
class WhatTheHardwareStops(unittest.TestCase):
    """Each test breaks one thing in the Verilog src/machine.py wrote, and the machine itself stops what follows."""

    def test_a_turn_that_never_finishes_is_stopped_at_its_bound(self):
        """A core that never raises done: the turn is stopped one cycle past the bound and counted, and every cell
        is as cells.mem has it, the last of the pool too, which this frame had taken."""
        m = machine(cells=N_MAX + len(machine().queue_layout) + 8)
        core = m.core.replace("done <= 1'b1;", "done <= 1'b0;")
        self.assertNotEqual(core, m.core)
        run = simulate(m, [(False, model.ipv4_frame(payload=bytes(14)))], core=core)
        self.assertEqual(run.turns, [("stopped", m.turn_cycles + 1)])
        self.assertEqual(run.counted, {**NOTHING_COUNTED, "overruns": 1})
        self.assertEqual(run.outputs, [])
        self.assertEqual(run.final, (m.start[0], m.start[1], 0, m.start[2]))
        self.assertEqual(run.words, first_words(m))

    def test_after_a_stop_the_next_frame_is_served(self):
        """With the bound set below what a long frame takes, its turn is stopped, and the frame after it is served
        from the starting image and goes out."""
        m = machine()
        top = m.top.replace(f"TURN_LIMIT = 32'd{m.turn_cycles};", "TURN_LIMIT = 32'd1000;")
        self.assertNotEqual(top, m.top)
        run = simulate(m, [(False, model.ipv4_frame(payload=bytes(900))), (False, FORWARDED[0])], top=top)
        self.assertEqual([outcome for outcome, _cycles in run.turns], ["stopped", "queued", "idle"])
        self.assertEqual([out for _port, out in run.outputs], [sent(FORWARDED[0])])
        self.assertEqual(run.counted, {**NOTHING_COUNTED, "overruns": 1})
        self.assertEqual(sorted(holdings(m, run)), list(range(1, m.cells + 1)))

    def test_the_mutant_that_hung_is_stopped(self):
        """tools/mutate_bounds.py's "every argument of a turn is its first" gives the turn the pool for each argument.
        Each of its turns finished within the bound, and egress sent the same head forever. Wired that way by hand,
        no frame goes out, every turn is stopped by the queue's count, and every cell is in one place."""
        m = machine()
        top = m.top.replace(".arg1(slot), .arg2(head), .arg3(queue)", ".arg1(pool), .arg2(pool), .arg3(pool)")
        top = top.replace(".arg4(length), .arg5(sent)", ".arg4(pool), .arg5(pool[0])")
        self.assertEqual(top.count("(pool"), 6)
        run = simulate(m, [(True, frame) for frame in FORWARDED[:5]], top=top)
        self.assertEqual(run.outputs, [])
        self.assertEqual([outcome for outcome, _cycles in run.turns], ["faulted"] * 5)
        self.assertEqual(run.counted, {**NOTHING_COUNTED, "faults": 5})
        self.assertEqual(sorted(holdings(m, run)), list(range(1, m.cells + 1)))

    def test_egress_cuts_a_chain_longer_than_it_may_read(self):
        """With egress let read two chunks, a frame of six goes out cut after its 48th byte, which ends it, and is
        counted; its release gives every cell back."""
        m = machine()
        top = m.top.replace(f"CHUNKS_OUT = 16'd{N_MAX - 5};", "CHUNKS_OUT = 16'd2;")
        self.assertNotEqual(top, m.top)
        frame = model.ipv4_frame(payload=bytes(range(36)))
        run = simulate(m, [(False, frame), (False, FORWARDED[0])], top=top)
        self.assertEqual([out for _port, out in run.outputs], [sent(frame)[:34 + 14], sent(FORWARDED[0])])
        self.assertEqual(run.counted, {**NOTHING_COUNTED, "cuts": 1})
        self.assertEqual(sorted(holdings(m, run)), list(range(1, m.cells + 1)))

    def test_ingress_writes_no_cell_past_n_max_whatever_its_chain(self):
        """Ingress's chain made a chunk longer than N_max cells allow, by linking its last chunk to the first cell of
        the pool: a frame one byte longer than ingress holds fills the N_max cells and leaves that one alone."""
        m = machine()
        words = m.image.split()
        pool = m.start[0]
        words[N_MAX] = f"{int(words[N_MAX], 16) | pool << 64:032x}"  # ingress is cells 1 to N_MAX, its last chunk last
        frame = model.ipv4_frame(payload=bytes(range(256)) * 6)[:m.shape.capacity + 1]
        run = simulate(m, [(False, frame)], image="\n".join(words) + "\n")
        self.assertEqual([outcome for outcome, _cycles in run.turns], ["too_large"])
        self.assertEqual(run.words[pool], first_words(m)[pool])
        self.assertEqual(at(run, N_MAX, 1, 7), int.from_bytes(frame[-8:-1], "little"))

    def test_ingress_counts_one_byte_past_what_it_holds(self):
        """A frame longer than N_max cells hold ends at the byte past them: the record's length says so, and the
        rest of the frame is read and thrown away."""
        m = machine()
        run = simulate(m, [(False, model.ipv4_frame(payload=bytes(1600)))])
        self.assertEqual([outcome for outcome, _cycles in run.turns], ["too_large"])
        length = m.shape.item_walk.length
        cells = record_cells(m, run, run.final[1])
        self.assertEqual(at(run, cells[length[0]], length[1], 2), m.shape.capacity + 1)


if __name__ == "__main__":
    unittest.main()
