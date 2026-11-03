"""src/machine.py: the runtime's turn in LFPL, lowered with a handler to Verilog, frames in and out as bytes.

The machine is held to the x86 build of the same handler on live sockets, frame by frame, for examples/ipv4.lfpl,
examples/firewall.lfpl and examples/udp_verify.lfpl: what x86 sends, the machine sends to the same port, what x86 drops
or refuses, the machine drops or refuses for the same reason, and each verdict is counted the same. Every turn is held
to the bound the emitter stated for its frame before anything ran, and afterwards every cell and every slot is in
exactly one place.

What would hang is stopped by the design itself, not by a test's timeout, and WhatTheHardwareStops breaks the
generated Verilog on purpose to show it: a turn past its bound, a queue whose head no turn's outcome accounts for,
and a slot that says it holds more than a slot can.
"""
import collections
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.certify import certify, certify_turn  # noqa: E402
from src.machine import HALTS, INGRESS, Machine, build  # noqa: E402
from src.rtcheck import Checker, RTError, Unbounded  # noqa: E402
from src.rtlower import rt_lines  # noqa: E402
from src.rtverilog import CELL_BYTES, Emitter  # noqa: E402
from src.turn import OUTCOMES, TurnRefused, turn_source  # noqa: E402
from tests.reference import ipv4_model as model  # noqa: E402
from tests.test_forward import ROOT, Forwarder  # noqa: E402
from tests.test_slots import TheMachine as UDPFrames, frame as udp  # noqa: E402

PROGRAMS = ("ipv4", "firewall", "udp_verify")
SOURCE = os.path.join(ROOT, "examples", "ipv4.lfpl")
SIMULATOR = unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog is not installed")
FORWARDED = [frame for _name, verdict, frame in model.corpus() if verdict == "forward"]
NOTHING_COUNTED = {"overruns": 0, "faults": 0}
_BUILT: dict = {}


def machine(program: str = "ipv4", **options) -> Machine:
    key = (program, tuple(sorted(options.items())))
    if key not in _BUILT:
        with open(os.path.join(ROOT, "examples", f"{program}.lfpl"), "rb") as handle:
            _BUILT[key] = build(handle.read(), **options)
    return _BUILT[key]


def sent(frame: bytes) -> bytes:
    """What ipv4.lfpl sends for a frame it forwards."""
    return model.forward(frame, model.by_top_bits)[1]


@dataclass
class Run:
    outputs: list   # (port, bytes), in the order they left
    turns: list     # (outcome, cycles), in the order they ran
    clocks: list    # the clock each turn was seen to end in, in the same order
    verdicts: list  # the tag of each turn's verdict, for a turn that decided a frame, and None for any other
    final: tuple    # the pool, the spare slots, ingress's record, the queue's head and the rest of the queue
    counted: dict   # turns stopped at their bound, turns whose queue broke the count
    decided: tuple  # how many turns decided each value of the verdict, by its tag
    words: dict     # every cell's contents, and every slot cell's, when it was done
    unknown: int = 0  # bytes sent that nothing had written, which egress cannot send unless a slot lies about itself


def simulate(m: Machine, frames: list, *, top: Optional[str] = None, core: Optional[str] = None,
             image: Optional[str] = None) -> Run:
    """frames: (hold egress while it goes in, the bytes). `top`, `core` and `image` stand in for what src/machine.py
    wrote, for a test that breaks one of them on purpose."""
    per_slot = m.shape.slot_bytes // CELL_BYTES
    with tempfile.TemporaryDirectory() as work:
        for name, text in (("machine.v", top or m.top), ("serve_core.v", core or m.core),
                           ("cells.mem", image or m.image)):
            with open(os.path.join(work, name), "w") as handle:
                handle.write(text)
        with open(os.path.join(work, "frames.txt"), "w") as handle:
            for hold, frame in frames:
                handle.write(f"{int(hold)} {len(frame)} " + " ".join(f"{byte:02x}" for byte in frame) + "\n")
        compiled = subprocess.run(["iverilog", "-g2005", f"-DCELLS={m.cells}", f"-DPARTS={m.shape.parts}",
                                   f"-DVERDICTS={len(m.shape.verdicts)}", f"-DSLOT_FIRST={m.first_slot * per_slot}",
                                   f"-DSLOT_LAST={(m.first_slot + m.slots) * per_slot - 1}",
                                   "-o", os.path.join(work, "sim"),
                                   os.path.join(ROOT, "tests", "tb_machine.v"), os.path.join(work, "machine.v"),
                                   os.path.join(work, "serve_core.v")], capture_output=True, text=True)
        assert compiled.returncode == 0, compiled.stderr
        done = subprocess.run(["vvp", os.path.join(work, "sim"), "+frames=frames.txt"], cwd=work, check=True,
                              capture_output=True, text=True, timeout=3000)
    run = Run([], [], [], [], (), {}, (), {})
    for line in done.stdout.splitlines():
        parts = line.split()
        if parts[0] == "OUT":
            run.unknown += parts[2].count("x") // 2
            run.outputs.append((int(parts[1]), bytes.fromhex(parts[2].replace("x", "0"))))
        elif parts[0] == "TURN":
            code = int(parts[1])
            outcome = HALTS[code] if code in HALTS else OUTCOMES[code]
            run.turns.append((outcome, int(parts[2])))
            run.clocks.append(int(parts[3]))
            run.verdicts.append(int(parts[4]) if outcome in ("queued", "dropped") else None)
        elif parts[0] == "FINAL":
            numbers = tuple(int(part) for part in parts[1:])
            run.final, run.counted = numbers[:5], dict(zip(("overruns", "faults"), numbers[5:7]))
            run.decided = numbers[7:]
        elif parts[0] == "WORD":  # a slot's cell nothing has written is unknown, and None here
            run.words[int(parts[1])] = None if "x" in parts[2] else int(parts[2], 16)
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


def slot_of(m: Machine, run: Run, first: int) -> tuple:
    """The number of the slot a record holds, and how many bytes it says it holds."""
    cell, off = m.shape.item_walk.slot
    value = at(run, record_cells(m, run, first)[cell], off, 8)
    return value & 0xFFFFFFFF, value >> 32


def holdings(m: Machine, run: Run) -> tuple:
    """Every cell the pool, the spares, ingress and the queue hold, and every slot, walked through the store as the
    machine left it."""
    cells: list = []
    slots: list = []

    def record(first: int) -> None:
        cells.extend(record_cells(m, run, first).values())
        slots.append(slot_of(m, run, first)[0])

    def entry(cell: int) -> None:
        if cell:
            cells.append(cell)
            record(at(run, cell, m.entry_layout["frame"][0], 8))

    pool, spares, ingress, head, queue = run.final
    while pool:
        cells.append(pool)
        pool = at(run, pool, 0, 8)
    while spares:
        cells.append(spares)
        slots.append(at(run, spares, m.spare_layout["spare"][0], 8) & 0xFFFFFFFF)
        spares = at(run, spares, m.spare_layout["rest"][0], 8)
    record(ingress)
    entry(head)
    place = queue
    for fields in m.queue_layout:
        cells.append(place)
        for name, (off, _size) in fields.items():
            if name != "$next":
                entry(at(run, place, off, 8))
        if "$next" in fields:
            place = at(run, place, fields["$next"][0], 8)
    return cells, slots


def conserved(test: unittest.TestCase, m: Machine, run: Run) -> None:
    cells, slots = holdings(m, run)
    test.assertEqual(sorted(cells), list(range(1, m.cells + 1)), "every cell in one place")
    test.assertEqual(sorted(slots), list(range(m.first_slot, m.first_slot + m.slots)), "and every slot")


def first_words(m: Machine) -> dict:
    """Every cell as cells.mem has it, before the first turn."""
    return {index: int(word, 16) for index, word in enumerate(m.image.split()) if index}


def x86(program: str, frames: list) -> tuple:
    """What the x86 build does with each frame: ("forwarded", port, bytes), or the counter it went up by instead; and
    how many frames it counted for each value of the verdict."""
    results = []
    with tempfile.TemporaryDirectory() as workdir:
        router = Forwarder(bytes(1024), workdir, source=os.path.join(ROOT, "examples", f"{program}.lfpl"))
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
            assert (now["pool"], now["slots"]) == (now["capacity"], now["slot_capacity"])
            counted = now["verdicts"]
        finally:
            router.close()
    return results, counted


def egress(m: Machine, sent: bytes) -> int:
    """T10's E(f), in the specification's section 10: the clocks egress takes to send a frame of B bytes, L of them in
    its slot, from the clock it leaves IDLE for it to the clock it asks for its release. A byte a clock, two clocks to
    read each cell of the slot it reaches, three for each cell of the record that holds scalars, two for each of the
    record's links and its slot field, and three more: the head's cell, read and taken, and the clock after the last
    byte, which egress waits out."""
    walk = m.shape.item_walk
    slot_cells = -(-(len(sent) - m.shape.wire) // 16)
    return len(sent) + 2 * slot_cells + 2 * (len(walk.links) + 1) + 3 * len({cell for cell, _o, _s in walk.scalars}) + 3


def ingress(m: Machine, frame: bytes) -> int:
    """T10's I(f), in the specification's section 10: the clocks the writer takes for a frame whose bytes come a clock
    each, from the clock it leaves IDLE for it to the clock it starts the frame's turn. Its own links and its slot field
    once, a clock a byte, and a clock each for the slot's length, the record's length and the turn's start."""
    return len(frame) + 2 * (len(m.shape.item_walk.links) + 1) + 3


def receiving(m: Machine, run: Run, frames: list) -> list:
    """Every frame the writer took straight after a turn that queued nothing, so that nothing else ran between:
    the clocks from that turn's end to its own, less its turn and the three clocks of the controller around it,
    against `ingress`. What differs, as (frame, measured, said)."""
    wrong = []
    admitted = iter(frames)
    for index in range(len(run.turns)):
        outcome, cycles = run.turns[index]
        if outcome == "idle":
            continue
        frame = next(admitted)
        before = run.turns[index - 1][0] if index else None
        if before is not None and before != "queued" and len(frame) <= m.shape.capacity:
            measured = run.clocks[index] - run.clocks[index - 1] - cycles - 3
            if measured != ingress(m, frame):
                wrong.append((frame.hex()[:16], measured, ingress(m, frame)))
    return wrong


def sending(m: Machine, run: Run, frames: list) -> list:
    """Every frame that was queued and sent at once, as egress was ready and nothing else was waiting: the clocks
    from its admission's end to its release's end, less the release and four clocks of the controller, which is
    egress's alone, against `egress`. The four are the clock in IDLE that chooses egress, and around the release
    the clock in TURN, the one serve_core samples start in and the one done is seen in. What differs, as
    (frame, measured, said)."""
    wrong, sent = [], iter(run.outputs)
    for index in range(len(run.turns) - 1):
        (outcome, _cycles), (following, release) = run.turns[index], run.turns[index + 1]
        if outcome == "queued" and following == "idle":
            _port, out = next(sent)
            measured = run.clocks[index + 1] - run.clocks[index] - release - 4
            if measured != egress(m, out):
                wrong.append((out.hex()[:16], measured, egress(m, out)))
    return wrong


def admitted_lengths(run: Run, frames: list) -> list:
    """Each turn, with the length of the frame it was for, or 0 for one that only released."""
    lengths, given = [], iter(frames)
    for outcome, _cycles in run.turns:
        lengths.append(0 if outcome == "idle" else len(next(given)))
    return lengths


# What the machine calls each of x86's counters.
CALLED = {"forwarded": "queued", "dropped": "dropped", "malformed": "runt", "too_large": "too_large"}


class TheBound(unittest.TestCase):
    """Before anything runs: the turn is typed with the handler, and its bound is certified and stated for each frame."""

    def checker(self, program: str = "ipv4") -> Checker:
        checker = Checker()
        for line in rt_lines(io.BytesIO(machine(program).turn_text.encode())):
            checker.feed(line)
        checker.finish()
        return checker

    def test_it_is_stated_for_a_whole_turn(self):
        m = machine()
        self.assertRegex(m.report(), r"W = \d+ steps for any frame of up to 2082 bytes, with 4 places in the queue; "
                                     r"clocks <= \d+ a turn, certified")
        self.assertEqual(m.turn_cycles(64), m.timing.clocks.d)
        self.assertIn(f"LIMIT_BASE = 32'd{m.timing.clocks.d};", m.top)
        self.assertIn(f"LIMIT_PER = 32'd{m.timing.clocks.c};", m.top)

    def test_a_small_frames_bound_is_lower_than_a_large_ones(self):
        """udp_verify walks its payload, and its bound is paid for by the slot's bytes, so it grows with the frame.
        The other two never read their payload, and their bound is the same for every frame."""
        udp = machine("udp_verify")
        self.assertGreater(udp.timing.clocks.c, 0)
        self.assertLess(udp.turn_cycles(64), udp.turn_cycles(1518))
        self.assertLess(udp.bound.c * udp.n_of(64) + udp.bound.d, udp.bound.c * udp.n_of(1518) + udp.bound.d)
        self.assertRegex(udp.report(), r"W = \d+ n \+ \d+ steps, n being a frame's 6 cells and the bytes of its slot")
        for program in ("ipv4", "firewall"):
            self.assertEqual(machine(program).turn_cycles(64), machine(program).turn_cycles(1518), program)

    def test_the_readme_quotes_what_the_emitter_says(self):
        with open(os.path.join(ROOT, "README.md")) as handle:
            self.assertIn(machine().report(), handle.read())

    def test_the_readme_quotes_the_udp_check(self):
        udp = machine("udp_verify")
        with open(os.path.join(ROOT, "README.md")) as handle:
            readme = handle.read()
        self.assertIn(udp.report(), readme)
        self.assertIn(f"A frame of 64 bytes is bounded at {udp.turn_cycles(64)} clocks in the machine, and one of 1518 "
                      f"at {udp.turn_cycles(1518)}", " ".join(readme.split()))

    def test_the_turn_is_lfpl_appended_to_the_handler(self):
        with open(SOURCE, "rb") as handle:
            text, shape = turn_source(handle.read())
        self.assertIn("turn serve(pool: Pool, spares: Spares, ingress: Frame, head: Out, queue: Queue, event: Int16)", text)
        self.assertEqual((shape.wire, shape.parts, shape.slot_bytes, shape.capacity), (34, 5, 2048, 34 + 2048))

    def test_a_certificate_that_falls_short_is_refused(self):
        with self.assertRaisesRegex(RTError, "falls"):
            self.checker().verify_turn("serve", {2: machine().shape.n_max, 3: 1 + machine().shape.n_max}, {})

    def test_only_the_head_of_the_queue_is_charged(self):
        """A turn releases the head of the queue and no other frame, so saying how much the rest of the queue holds
        changes nothing: none of it is charged a step."""
        checker = self.checker()
        n = machine().shape.n_max
        head = {INGRESS: n, INGRESS + 1: 1 + n}
        self.assertEqual(machine().bound, certify_turn(checker, "serve", head, per_turn=INGRESS))
        self.assertEqual(certify_turn(checker, "serve", {**head, INGRESS + 2: 3 * (1 + n)}, per_turn=INGRESS),
                         certify_turn(checker, "serve", head, per_turn=INGRESS))

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
        self.assertEqual(certify_turn(checker, "t", {0: 10}, per_turn=0).d, certify_turn(checker, "t", {0: 0}).d)

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

    HANDLER = ("ipv4", "fold", "halves", "well_formed", "patched", "next_hop")  # and what it calls

    def test_the_decision_touches_no_block_ram(self):
        """Ingress's record is registers, and the handler reads and writes nothing else: none of its states, nor
        any of what it calls, drives a port of the cell store."""
        core = machine().core
        ports = core[core.index("What each state drives on the memory ports"):core.index("// The machine.")]
        driving = {name.rsplit("_s", 1)[0] for name in re.findall(r"^\s+(\w+_s\d+): begin", ports, re.M)}
        self.assertTrue(driving)
        self.assertEqual(driving & set(self.HANDLER), set())
        self.assertIn("serve", driving)

    def test_the_readme_says_what_the_decision_costs(self):
        """Its steps, as x86 counts them, and its clocks in the machine, certified with the clocks charged there."""
        m = machine()
        lines = list(rt_lines(io.BytesIO(m.turn_text.encode())))
        emitter = Emitter(m.pointer_bits, header=m.start[2])
        for line in lines:
            emitter.feed(line)
        turn = next(f for f in emitter.functions.values() if f.turn)
        n = m.shape.n_max
        emitter.render(turn, "serve_core", m.bound, m.cells, {}, sizes={INGRESS: n, INGRESS + 1: 1 + n}, slots=m.slots,
                       per_turn=INGRESS)
        costs = collections.Counter(state.line for fn in emitter._reachable(turn) for state in fn.states)
        steps, clocks = Checker(), Checker(costs)
        for checker in (steps, clocks):
            for line in lines:
                checker.feed(line)
            checker.finish()
        said = f"its {certify(steps, 'ipv4', n).d} steps are at most {certify(clocks, 'ipv4', n).d} clocks"
        with open(os.path.join(ROOT, "README.md")) as handle:
            self.assertIn(said, " ".join(handle.read().split()))

    def test_a_longer_queue_costs_more_and_is_still_bounded(self):
        self.assertGreater(machine(k_out=8).bound.d, machine().bound.d)


class ShapesItDoesNotServe(unittest.TestCase):
    def refused(self, source: str, message: str, **options) -> None:
        with self.assertRaisesRegex(TurnRefused, message):
            turn_source(source.encode(), **options)

    def test_an_output_built_in_a_cell(self):
        with open(os.path.join(ROOT, "examples", "forward.lfpl")) as handle:
            self.refused(handle.read(), "reads a verdict|state slots")

    def test_a_queue_of_one_place(self):
        with open(SOURCE) as handle:
            self.refused(handle.read(), "K_out of 1", k_out=1)

    def test_a_handler_that_draws_cells(self):
        self.refused("""slot Bytes;
record F { a: Int8, length: Int16, body: Bytes };
enum V = Drop | Send(port: Int8);
handler h(f: F, free: Free) : (V, F, Free) = match f with F(d, a, length, body) =>
  match free with End => (Drop, F@d {}, End) | More(c, rest) => (Send(a), F@d {}, More@c(rest)) end end;
""", r"draws R\(n\) = 0 n \+ 1")

    def test_a_name_the_turn_declares(self):
        self.refused("""slot Bytes;
record F { a: Int8, length: Int16, body: Bytes };
enum V = Drop | Send(port: Int8);
fn release(x: Int8) : Int8 = x;
handler h(f: F, free: Free) : (V, F, Free) = match f with F(d, a, length, body) => (Send(release(a)), F@d {}, free) end;
""", "declares release")

    def test_a_verdict_with_more_than_a_port(self):
        self.refused("""slot Bytes;
record F { a: Int8, length: Int16, body: Bytes };
enum V = Drop | Send(port: Int8, why: Int8);
handler h(f: F, free: Free) : (V, F, Free) = match f with F(d, a, length, body) => (Send(a, 1), F@d {}, free) end;
""", "port: Int8")

    def test_a_record_whose_payload_is_a_chain(self):
        self.refused("""data Bytes;
record F { a: Int8, length: Int16, body: Bytes };
enum V = Drop | Send(port: Int8);
handler h(f: F, free: Free) : (V, F, Free) = match f with F(d, a, length, body) => (Send(a), F@d {}, free) end;
""", "payload is one slot, and F has a chain")


def corpus_of(program: str) -> list:
    """Every frame of the IPv4 corpus, the UDP frames of tests/test_slots.py, the largest frame ingress holds, and one
    byte longer."""
    m = machine(program)
    largest = udp((bytes(range(256)) * 9)[:m.shape.capacity - 42])  # a UDP datagram, exactly what fits
    assert len(largest) == m.shape.capacity
    frames = [frame for _name, _verdict, frame in model.corpus()]
    frames += [packet for _name, packet, length in UDPFrames("test_every_verdict_is_the_models").cases() if length is None]
    return frames + [largest, largest + b"\x00"]


@SIMULATOR
class TheCorpus(unittest.TestCase):
    """Each of the three programs, every frame through both, in the same order."""

    def held_to_x86(self, program: str) -> Run:
        m = machine(program)
        frames = corpus_of(program)
        run = simulate(m, [(False, frame) for frame in frames])
        expected, counted = x86(program, frames)
        what = [outcome for outcome, _cycles in run.turns if outcome != "idle"]
        self.assertEqual(what, [CALLED[result[0]] for result in expected])
        self.assertEqual(run.outputs, [(port, out) for name, *rest in expected if name == "forwarded"
                                       for port, out in [rest]])
        self.assertEqual(what[-2:], ["queued", "too_large"])
        self.assertEqual(run.decided, tuple(counted), "each verdict counted as x86 counts it")
        self.assertEqual(run.unknown, 0, "and every byte sent was the frame's")
        self.assertEqual(run.counted, NOTHING_COUNTED)
        for (outcome, cycles), length in zip(run.turns, admitted_lengths(run, frames)):
            self.assertLessEqual(cycles, m.turn_cycles(length), (outcome, length))
        conserved(self, m, run)
        self.assertEqual(sending(m, run, frames), [], "egress takes what T10's E(f) says, to the clock")
        self.assertEqual(receiving(m, run, frames), [], "and the writer what its I(f) says")
        return run

    def test_ipv4_as_the_x86_build_does(self):
        run = self.held_to_x86("ipv4")
        m = machine()
        admitted = max(cycles for outcome, cycles in run.turns if outcome != "idle")
        released = max(cycles for outcome, cycles in run.turns if outcome == "idle")
        with open(os.path.join(ROOT, "README.md")) as handle:
            self.assertIn(f"The longest admission measured took {admitted} cycles, and the longest release "
                          f"{released}, against the bound of {m.turn_cycles(64)}", " ".join(handle.read().split()))

    def test_firewall_as_the_x86_build_does(self):
        self.held_to_x86("firewall")

    def test_udp_verify_as_the_x86_build_does(self):
        self.held_to_x86("udp_verify")


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
        conserved(self, m, run)

    def test_short_when_the_pool_runs_dry_before_the_record(self):
        """A frame takes five cells for the copy of ingress's record and one for its place in the queue."""
        m = machine(cells=machine().shape.parts + len(machine().queue_layout) + 4 + 5)
        run = simulate(m, [(False, FORWARDED[0])])
        self.assertEqual([outcome for outcome, _cycles in run.turns], ["short"])
        self.assertEqual(run.outputs, [])
        conserved(self, m, run)

    def test_short_with_no_spare_slot(self):
        """Started with no spares, a frame cannot take ingress's slot into the queue, and the cells it took go back."""
        m = machine()
        top = m.top.replace(f"spares <= {m.pointer_bits}'d{m.start[1]};", "spares <= {P{1'b0}};")
        self.assertNotEqual(top, m.top)
        run = simulate(m, [(False, FORWARDED[0])], top=top)
        self.assertEqual([outcome for outcome, _cycles in run.turns], ["short"])
        self.assertEqual(run.outputs, [])
        cells, _slots = holdings(m, run)
        self.assertEqual(sorted(cells + list(range(m.start[1], m.start[1] + m.slots - 1))), list(range(1, m.cells + 1)))

    def test_an_earlier_frames_bytes_are_never_read(self):
        """A frame refused as too large leaves its bytes in ingress's slot. The next frame is one byte, so a handler
        that reads its slot's fourth byte reads past the length, and 0, whatever the cell holds."""
        source = b"""slot Bytes;
record F { a: Int8, sum: Int8, length: Int16, body: Bytes };
enum V = Drop | Send(port: Int8);
handler h(f: F, free: Free) : (V, F, Free) = match f with F(d, a, sum, length, body) =>
  let x = body[idx(3, Bytes)] in (Send(1), F@d { sum = x }, free) end;
"""
        m = build(source, k_out=2)
        run = simulate(m, [(False, b"\xff" * (m.shape.capacity + 10)), (False, bytes((0x11, 0x22, 0x01)))])
        self.assertEqual([outcome for outcome, _cycles in run.turns if outcome != "idle"], ["too_large", "queued"])
        self.assertEqual(run.outputs, [(1, bytes((0x11, 0x00, 0x01)))])
        ingress_slot = m.first_slot * (m.shape.slot_bytes // CELL_BYTES)
        self.assertEqual(run.words[ingress_slot] >> 24 & 0xFF, 0xFF, "the cell still holds the old frame's byte")
        conserved(self, m, run)

    def test_what_was_refused_leaves_room_for_what_comes_next(self):
        m = machine()
        frames = [(False, b"\x00" * 10), (False, model.ipv4_frame(payload=bytes(2100))), (False, FORWARDED[1])]
        run = simulate(m, frames)
        self.assertEqual([outcome for outcome, _cycles in run.turns if outcome != "idle"],
                         ["runt", "too_large", "queued"])
        self.assertEqual([out for _port, out in run.outputs], [sent(FORWARDED[1])])


@SIMULATOR
class WhatTheHardwareStops(unittest.TestCase):
    """Each test breaks one thing in the Verilog src/machine.py wrote, and the machine itself stops what follows."""

    def test_a_turn_that_never_finishes_is_stopped_at_its_bound(self):
        """A core that never raises done: the turn is stopped one cycle past its frame's bound and counted, and every
        cell is as cells.mem has it, the last of the pool too, which this frame had taken."""
        m = machine("udp_verify", cells=machine("udp_verify").shape.parts + len(machine().queue_layout) + 4 + 7)
        core = m.core.replace("done <= 1'b1;", "done <= 1'b0;")
        self.assertNotEqual(core, m.core)
        frame = model.ipv4_frame(payload=bytes(14))
        run = simulate(m, [(False, frame)], core=core)
        self.assertEqual(run.turns, [("stopped", m.turn_cycles(len(frame)) + 1)])
        self.assertEqual(run.counted, {**NOTHING_COUNTED, "overruns": 1})
        self.assertEqual(run.outputs, [])
        self.assertEqual(run.final, (m.start[0], m.start[1], m.start[2], 0, m.start[3]))
        self.assertEqual({cell: word for cell, word in run.words.items() if cell <= m.cells}, first_words(m))

    def test_after_a_stop_the_next_frame_is_served(self):
        """With the bound set below what a long frame takes to check, its turn is stopped, and the frame after it is
        served from the starting image and goes out."""
        m = machine("udp_verify")
        top = (m.top.replace(f"LIMIT_BASE = 32'd{m.timing.clocks.d};", "LIMIT_BASE = 32'd1000;")
               .replace(f"LIMIT_PER = 32'd{m.timing.clocks.c};", "LIMIT_PER = 32'd0;"))
        self.assertNotEqual(top, m.top)
        cases = UDPFrames("test_every_verdict_is_the_models").cases()
        longest, short = cases[12][1], cases[0][1]
        run = simulate(m, [(False, longest), (False, short)], top=top)
        self.assertEqual([outcome for outcome, _cycles in run.turns], ["stopped", "queued", "idle"])
        self.assertEqual(len(run.outputs), 1)
        self.assertEqual(run.counted, {**NOTHING_COUNTED, "overruns": 1})
        conserved(self, m, run)

    def test_the_mutant_that_hung_is_stopped(self):
        """tools/mutate_bounds.py's "every argument of a turn is its first" gives the turn the pool for each argument.
        Each of its turns finished within the bound, and egress sent the same head forever. Wired that way by hand,
        egress sends no more than the turns said they queued, the last turn is stopped past its bound or refused for a
        queue nothing accounts for, and every cell is in one place. test_a_head_no_outcome_accounts_for_is_a_fault holds
        the machine to the second of those on its own."""
        m = machine()
        top = m.top.replace(".arg1(spares), .arg2(ingress),", ".arg1(pool), .arg2(pool),")
        top = top.replace(".arg3(head), .arg4(queue), .arg5({sent, length[14:0]})",
                          ".arg3(pool), .arg4(pool), .arg5(pool[15:0])")
        self.assertEqual(top.count("(pool"), 6)
        run = simulate(m, [(True, frame) for frame in FORWARDED[:5]], top=top)
        what = [outcome for outcome, _cycles in run.turns]
        self.assertLessEqual(len(run.outputs), what.count("queued"))
        self.assertIn(what[-1], ("faulted", "stopped"), "a turn past its bound, or a queue nothing accounts for")
        self.assertEqual(run.counted, {"overruns": what.count("stopped"), "faults": what.count("faulted")})
        conserved(self, m, run)

    def test_more_queued_than_the_queue_holds_is_a_fault(self):
        """A turn whose outcome always says Queued: four frames fill the queue, and the fifth, which the turn
        refused as Full, would be a fifth output in four places. The machine takes none of that turn's results,
        and starts again."""
        m = machine()
        top = m.top.replace("    wire [63:0] r_outcome;", "    wire [63:0] r_said;\n    wire [63:0] r_outcome = 64'd1;")
        top = top.replace(".res5(r_outcome)", ".res5(r_said)")
        self.assertNotIn(".res5(r_outcome)", top)
        run = simulate(m, [(True, frame) for frame in FORWARDED[:5]], top=top)
        self.assertEqual([outcome for outcome, _cycles in run.turns], ["queued"] * 4 + ["faulted"])
        self.assertEqual(run.outputs, [])
        self.assertEqual(run.counted, {**NOTHING_COUNTED, "faults": 1})
        conserved(self, m, run)

    def test_a_head_no_outcome_accounts_for_is_a_fault(self):
        """A turn that queues a frame and says it dropped it: the head of the queue is an output no outcome accounts
        for. The machine takes none of that turn's results, so egress sends nothing, and every cell starts again."""
        m = machine()
        top = m.top.replace("    wire [63:0] r_outcome;",
                            "    wire [63:0] r_said;\n"
                            "    wire [63:0] r_outcome = r_said[7:0] == QUEUED ? {r_said[63:8], DROPPED} : r_said;")
        top = top.replace(".res5(r_outcome)", ".res5(r_said)")
        self.assertNotIn(".res5(r_outcome)", top)
        run = simulate(m, [(False, FORWARDED[0])], top=top)
        self.assertEqual([outcome for outcome, _cycles in run.turns], ["faulted"])
        self.assertEqual(run.outputs, [])
        self.assertEqual(run.counted, {**NOTHING_COUNTED, "faults": 1})
        conserved(self, m, run)

    def test_egress_sends_no_more_than_a_slot_holds(self):
        """A record whose slot says it holds 60000 bytes, as one a broken turn could hand egress: what goes out is the
        record's bytes and a slot's, 2048, and no more, and the frame ends there."""
        m = machine()
        top = m.top.replace(f"wire [15:0] said = host_dout[{8 * (m.shape.item_walk.slot[1] + 4)} +: 16];",
                            "wire [15:0] said = 16'd60000;")
        self.assertNotEqual(top, m.top)
        run = simulate(m, [(False, FORWARDED[0])], top=top)
        self.assertEqual([len(out) for _port, out in run.outputs], [m.shape.wire + m.shape.slot_bytes])
        self.assertEqual(run.outputs[0][1][:len(sent(FORWARDED[0]))], sent(FORWARDED[0]), "the frame, then the rest")

    def test_ingress_writes_no_byte_past_its_slot(self):
        """A frame longer than ingress holds fills the slot and stops: the next slot's first cell, a spare, is as it
        was, the byte past the slot's last is not wrapped onto its first, and the turn refuses the frame as too large."""
        m = machine()
        frame = model.ipv4_frame(payload=bytes(range(1, 256)) * 9)[:m.shape.capacity + 1]
        run = simulate(m, [(False, frame)])
        self.assertEqual([outcome for outcome, _cycles in run.turns], ["too_large"])
        per_slot = m.shape.slot_bytes // CELL_BYTES
        first, last = m.first_slot * per_slot, (m.first_slot + 1) * per_slot - 1
        self.assertEqual(run.words[first], int.from_bytes(frame[m.shape.wire:m.shape.wire + 16], "little"),
                         "the slot's first cell holds the frame's first bytes past the record")
        self.assertEqual(run.words[last], int.from_bytes(frame[-17:-1], "little"), "the slot's last cell holds the frame")
        self.assertIsNone(run.words[last + 1], "and the cell after it is untouched")

    def test_ingress_counts_one_byte_past_what_it_holds(self):
        """A frame longer than a slot holds ends at the byte past it: the record's length says so, its slot says it
        holds a slot's worth, and the rest of the frame is read and thrown away."""
        m = machine()
        run = simulate(m, [(False, model.ipv4_frame(payload=bytes(2100)))])
        self.assertEqual([outcome for outcome, _cycles in run.turns], ["too_large"])
        length = m.shape.item_walk.length
        cells = record_cells(m, run, run.final[2])
        self.assertEqual(at(run, cells[length[0]], length[1], 2), m.shape.capacity + 1)
        self.assertEqual(slot_of(m, run, run.final[2])[1], m.shape.slot_bytes)


if __name__ == "__main__":
    unittest.main()
