"""Random handlers that walk their item, timed against the bound the compiler certified for them.

LFPL-O's Theorem 5.1 says a handler given R(n) cells cannot run out of them, and halts within W(n)
steps. tests/test_checksum.py holds one handler to that. This holds handlers to it that nobody picked
by hand: each is a random pipeline of list walks, with a Python model beside each walk that shares
nothing with the compiler, built with a counter added at every step the checker charged for.

So for each datagram, three things can fail: the bytes that come out against the model, the steps
that ran against W(n), and the End arm of a draw, which leaves its copy out, so that an item given too
few cells has visibly different bytes. Every program here is linear by construction, so a refusal
is a false refusal, and fails the test too.
"""
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_forward import ROOT, Forwarder  # noqa: E402

COUNT = int(os.environ.get("LFPL_FUZZ", "60")) // 2
SEED = int(os.environ.get("LFPL_FUZZ_SEED", "0"))
DATAGRAMS = 12
PORT = 2
WIDTH = 7
MOST_CELLS_OUT = 150  # well inside what one message can gather, so nothing here is dropped for size

FIELDS = "used, b0, b1, b2, b3, b4, b5, b6"
C2 = "v, c0, c1, c2, c3, c4, c5, c6"
PRELUDE = f"""data Bytes = Last | Chunk(used: Int8, b0: Int8, b1: Int8, b2: Int8, b3: Int8, b4: Int8,
                          b5: Int8, b6: Int8, next: Bytes);
record P {{ port: Int8, body: Bytes }};
fn mix(x: Int8) : Int8 = ^(+(x, 7), 85);
fn rev(l: Bytes, acc: Bytes) : Bytes = match l with
  | Last => acc
  | Chunk(d, {FIELDS}, next) => rev(next, Chunk@d({FIELDS}, acc))
  end;
fn dup(l: Bytes, acc: Bytes, free: Free) : (Bytes, Free) = match l with
  | Last => (acc, free)
  | Chunk(d, {FIELDS}, next) => match free with
      | End => dup(next, Chunk@d({FIELDS}, acc), End)
      | More(c, rest) => dup(next, Chunk@d({FIELDS}, Chunk@c({FIELDS}, acc)), rest)
      end
  end;
fn thin(l: Bytes, acc: Bytes, free: Free) : (Bytes, Free) = match l with
  | Last => (acc, free)
  | Chunk(d, {FIELDS}, next) =>
      if ==(&(b0, 1), 0) then thin(next, Chunk@d({FIELDS}, acc), free)
      else thin(next, acc, More@d(free))
  end;
fn bump(l: Bytes, acc: Bytes, s: Int8) : Bytes = match l with
  | Last => acc
  | Chunk(d, {FIELDS}, next) =>
      if <(b1, 128) then bump(next, Chunk@d(used, +(b0, s), b1, b2, b3, b4, b5, b6, acc), +(s, 1))
      else bump(next, Chunk@d(used, ^(b0, mix(mix(s))), b1, b2, b3, b4, b5, b6, acc), mix(*(s, 3)))
  end;
fn pairs(l: Bytes, acc: Bytes) : Bytes = match l with
  | Last => acc
  | Chunk(d, {FIELDS}, next) => match next with
      | Last => Chunk@d({FIELDS}, acc)
      | Chunk(e, {C2}, rest) => pairs(rest, Chunk@e({FIELDS}, Chunk@d({C2}, acc)))
      end
  end;
fn split(l: Bytes, a: Bytes, b: Bytes) : (Bytes, Bytes) = match l with
  | Last => (a, b)
  | Chunk(d, {FIELDS}, next) => split(next, b, Chunk@d({FIELDS}, a))
  end;
fn weave(x: Bytes, y: Bytes, acc: Bytes) : Bytes = match x with
  | Last => rev(y, acc)
  | Chunk(d, {FIELDS}, next) => weave(y, next, Chunk@d({FIELDS}, acc))
  end;
fn dear(l: Bytes, acc: Bytes) : Bytes = match l with
  | Last => acc
  | Chunk(d, {FIELDS}, next) =>
      let m = if <(b1, 128) then +(+(+(b0, 1), 2), 3) else b0 in
      let k = if <(b2, 128) then mix(m) else m in
      dear(next, Chunk@d(used, k, b1, b2, b3, b4, b5, b6, acc))
  end;
fn first(l: Bytes) : (Int8, Bytes) = match l with
  | Last => (0, Last)
  | Chunk(d, {FIELDS}, next) => (b0, Chunk@d({FIELDS}, next))
  end;
fn choose(l: Bytes) : Bytes =
  let (k, m) = first(l) in
  let j = if ==(&(k, 1), 0) then m else rev(m, Last) in
  rev(j, Last);
fn sums(l: Bytes, acc: Bytes) : Bytes = match l with
  | Last => acc
  | Chunk(d, {FIELDS}, next) => match next with
      | Last => Chunk@d({FIELDS}, acc)
      | Chunk(e, {C2}, rest) =>
          sums(rest, Chunk@e(used, +(b0, c0), b1, b2, b3, b4, b5, b6, Chunk@d({C2}, acc)))
      end
  end;
fn late(l: Bytes) : Bytes = match l with
  | Last => Last
  | Chunk(d, {FIELDS}, next) =>
      let r = rev(next, Last) in
      Chunk@d(used, +(b0, 1), b1, b2, b3, b4, b5, b6, r)
  end;
fn early(l: Bytes) : Bytes = match l with
  | Last => Last
  | Chunk(d, {FIELDS}, next) =>
      let r = if <(b1, 128) then next else rev(next, Last) in
      Chunk@d(used, +(b0, 1), b1, b2, b3, b4, b5, b6, r)
  end;
fn touch(l: Bytes) : Bytes = match l with
  | Last => Last
  | Chunk(d, {FIELDS}, next) => Chunk@d(used, ^(b0, 255), b1, b2, b3, b4, b5, b6, next)
  end;
fn incr(l: Bytes) : Bytes @l = match l with
  | Last => Last
  | Chunk(d, {FIELDS}, next) => Chunk@d(used, +(b0, 1), b1, b2, b3, b4, b5, b6, incr(next))
  end;
data Marks = Stop(x: Int8) | Odd(x: Int8, y: Int8, rest: Marks) | Even(x: Int8, y: Int8, rest: Marks);
fn mark(l: Bytes, acc: Marks) : Marks = match l with
  | Last => acc
  | Chunk(d, {FIELDS}, next) =>
      if ==(&(b0, 1), 0) then mark(next, Even@d(used, b0, acc)) else mark(next, Odd@d(used, b0, acc))
  end;
fn unmark(t: Marks, acc: Bytes, free: Free) : (Bytes, Free) = match t with
  | Stop(d, x) => (acc, More@d(free))
  | Odd(d, x, y, rest) => unmark(rest, Chunk@d(x, +(y, 1), 0, 0, 0, 0, 0, 0, acc), free)
  | Even(d, x, y, rest) => unmark(rest, Chunk@d(x, y, 0, 0, 0, 0, 0, 0, acc), free)
  end;
fn marks(x: Bytes, free: Free) : (Bytes, Free) = match free with
  | End => (x, End)
  | More(c, rest) => unmark(mark(x, Stop@c(0)), Last, rest)
  end;
"""


def _mix(x: int) -> int:
    return ((x + 7) & 255) ^ 85


def model_rev(cells: list) -> list:
    return cells[::-1]


def model_dup(cells: list) -> list:
    out: list = []
    for cell in cells:
        out = [cell, cell] + out
    return out


def model_thin(cells: list) -> list:
    return [cell for cell in reversed(cells) if not cell[1] & 1]


def model_bump(cells: list) -> list:
    out: list = []
    s = 0
    for used, b0, *others in cells:
        if others[0] < 128:
            b0, s = (b0 + s) & 255, (s + 1) & 255
        else:
            b0, s = b0 ^ _mix(_mix(s)), _mix((s * 3) & 255)
        out.insert(0, (used, b0, *others))
    return out


def model_pairs(cells: list) -> list:
    out: list = []
    for at in range(0, len(cells), 2):
        out = cells[at:at + 2] + out
    return out


def model_weave(cells: list) -> list:
    a: list = []
    b: list = []
    for cell in cells:
        a, b = b, [cell] + a
    out: list = []
    while a:
        head, *a = a
        a, b = b, a
        out.insert(0, head)
    return b[::-1] + out


def model_dear(cells: list) -> list:
    out: list = []
    for used, b0, b1, b2, *others in cells:
        m = (b0 + 6) & 255 if b1 < 128 else b0
        k = _mix(m) if b2 < 128 else m
        out.insert(0, (used, k, b1, b2, *others))
    return out


def model_choose(cells: list) -> list:
    return list(cells) if cells and cells[0][1] & 1 else cells[::-1]


def model_sums(cells: list) -> list:
    out: list = []
    for at in range(0, len(cells), 2):
        pair = cells[at:at + 2]
        if len(pair) == 2:
            (used, b0, *others), second = pair
            pair = [(used, (b0 + second[1]) & 255, *others), second]
        out = pair + out
    return out


def model_late(cells: list) -> list:
    if not cells:
        return cells
    (used, b0, *others), *rest = cells
    return [(used, (b0 + 1) & 255, *others), *rest[::-1]]


def model_early(cells: list) -> list:
    if not cells:
        return cells
    (used, b0, b1, *others), *rest = cells
    return [(used, (b0 + 1) & 255, b1, *others), *(rest if b1 < 128 else rest[::-1])]


def model_touch(cells: list) -> list:
    if not cells:
        return cells
    (used, b0, *others), *rest = cells
    return [(used, b0 ^ 255, *others), *rest]


def model_incr(cells: list) -> list:
    return [(used, (b0 + 1) & 255, *others) for used, b0, *others in cells]


def model_marks(cells: list) -> list:
    out = []
    for used, b0, *_others in cells:
        out.append((used, (b0 + (b0 & 1)) & 255, *bytes(WIDTH - 1)))
    return out


STEPS = {
    "rev": ("let {out} = rev({x}, Last) in\n", model_rev, 1),
    "dear": ("let {out} = dear({x}, Last) in\n", model_dear, 1),
    "choose": ("let {out} = choose({x}) in\n", model_choose, 1),
    "sums": ("let {out} = sums({x}, Last) in\n", model_sums, 1),
    "late": ("let {out} = late({x}) in\n", model_late, 1),
    "early": ("let {out} = early({x}) in\n", model_early, 1),
    "touch": ("let {out} = touch({x}) in\n", model_touch, 1),
    "incr": ("let {out} = incr({x}) in\n", model_incr, 1),
    "marks": ("let ({out}, {free_out}) = marks({x}, {free}) in\n", model_marks, 1),
    "dup": ("let ({out}, {free_out}) = dup({x}, Last, {free}) in\n", model_dup, 2),
    "thin": ("let ({out}, {free_out}) = thin({x}, Last, {free}) in\n", model_thin, 1),
    "bump": ("let {out} = bump({x}, Last, 0) in\n", model_bump, 1),
    "pairs": ("let {out} = pairs({x}, Last) in\n", model_pairs, 1),
    "weave": ("let ({u}, {v}) = split({x}, Last, Last) in\nlet {out} = weave({u}, {v}, Last) in\n",
              model_weave, 1),
}


def pipeline_text(names: list) -> str:
    x, free = "body", "free"
    lines = ""
    for at, name in enumerate(names):
        out, free_out, u, v = f"x{at}", f"f{at}", f"u{at}", f"v{at}"
        lines += STEPS[name][0].format(out=out, x=x, free=free, free_out=free_out, u=u, v=v)
        x, free = out, free_out if "free_out" in STEPS[name][0] else free
    return (PRELUDE + "handler h(p: P, free: Free) : (P, Free) = match p with P(d, port, body) =>\n"
            + lines + f"(P@d(port, {x}), {free})\nend;\n")


def cells_of(payload: bytes) -> list:
    cells = []
    for at in range(0, len(payload), WIDTH):
        chunk = payload[at:at + WIDTH]
        cells.append((len(chunk), *chunk, *bytes(WIDTH - len(chunk))))
    return cells


def wire_of(cells: list) -> bytes:
    return b"".join(bytes(cell[1:1 + cell[0]]) for cell in cells)


def refusal(text: str) -> str:
    """Build a program that must not build: what the compiler said, with no binary left behind."""
    with tempfile.TemporaryDirectory() as workdir:
        path, exe = os.path.join(workdir, "refused.lfpl"), os.path.join(workdir, "refused")
        with open(path, "w") as handle:
            handle.write(text)
        done = subprocess.run(
            [sys.executable, "-m", "src.lfplo", path, "-o", exe, "--in-port", "41000",
             "--out-port", "41100", "--ctl-port", "41001"],
            cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 1, done.stdout + done.stderr
        assert not os.path.exists(exe)
    return done.stderr


def linear(text: str) -> tuple:
    """`28 n + 4`, `8 n` or `6`, as the compiler reports them, into a slope and a constant."""
    match = re.fullmatch(r"(?:(\d+) n)?(?: \+ )?(\d+)?", text)
    assert match is not None, text
    return int(match.group(1) or 0), int(match.group(2) or 0)


def bounds_in(report: str) -> tuple:
    match = re.search(r"R\(n\) = (.+?) cells, W\(n\) = (.+?) steps", report)
    assert match is not None, report
    return linear(match.group(1)), linear(match.group(2))


class ReportedBounds(unittest.TestCase):
    def test_the_report_reads_back(self):
        self.assertEqual(linear("28 n + 4"), (28, 4))
        self.assertEqual(linear("8 n"), (8, 0))
        self.assertEqual(linear("6"), (0, 6))


class TheModels(unittest.TestCase):
    """The models are what the handlers are judged by, so the arithmetic in them gets looked at."""

    A, B, C = (1, 2, *bytes(6)), (1, 3, *bytes(6)), (1, 4, *bytes(6))

    def test_each_walk_does_what_its_source_says(self):
        a, b, c = self.A, self.B, self.C
        self.assertEqual(model_rev([a, b, c]), [c, b, a])
        self.assertEqual(model_dup([a, b]), [b, b, a, a])
        self.assertEqual(model_thin([a, b, c]), [c, a])
        self.assertEqual(model_pairs([a, b, c]), [c, a, b])
        self.assertEqual(model_sums([a, b, c]), [c, (1, 5, *bytes(6)), b])
        self.assertEqual(model_late([a, b, c]), [(1, 3, *bytes(6)), c, b])
        self.assertEqual(model_early([a, b, c]), [(1, 3, *bytes(6)), b, c])
        self.assertEqual(model_touch([a, b]), [(1, 2 ^ 255, *bytes(6)), b])
        self.assertEqual(model_incr([a, b]), [(1, 3, *bytes(6)), (1, 4, *bytes(6))])
        # b1 and b2 are zero, so both ifs take their first arm: 2 + 6 = 8, and mix(8) = 15 ^ 85.
        self.assertEqual(model_dear([a]), [(1, 15 ^ 85, *bytes(6))])
        # a's first byte is even, so it is reversed once. b's is odd, so it is reversed twice.
        self.assertEqual(model_choose([a, b]), [b, a])
        self.assertEqual(model_choose([b, a]), [b, a])
        self.assertEqual(model_marks([a, b]), [a, (1, 4, *bytes(6))])

    def test_weave_alternates_and_finishes_the_longer_side_reversed(self):
        a, b, c = self.A, self.B, self.C
        self.assertEqual(model_weave([]), [])
        self.assertEqual(model_weave([a]), [a])
        # split makes x = [a], y = [b]. weave puts a on the accumulator, then b on top of it.
        self.assertEqual(model_weave([a, b]), [b, a])
        # split makes x = [b], y = [c, a]. weave moves b and then c, and x is empty, so what is
        # left of y goes on the accumulator: that is a, in front.
        self.assertEqual(model_weave([a, b, c]), [a, c, b])


def payload_of(length: int, fill: str, rng: random.Random) -> bytes:
    """Random bytes take every `if` both ways, so the cheap arms' slack pays for the dear arms' in
    the total, and a bound that only covers the average passes. Zeros and ones take one way all the
    way along, which is what shows it."""
    if fill == "zeros":
        return bytes(length)
    if fill == "ones":
        return b"\xff" * length
    return bytes(rng.randrange(256) for _ in range(length))


class RandomWalks(unittest.TestCase):
    def _serve(self, names: list, workdir: str, rng: random.Random, seed: int) -> float:
        """Build one pipeline and run its datagrams. The answer is the largest share of W(n) that
        any of its items used."""
        path = os.path.join(workdir, "walk.lfpl")
        with open(path, "w") as handle:
            handle.write(pipeline_text(names))
        router = Forwarder(bytes(1024), workdir, build_options=("--count-steps",), source=path)
        try:
            (a, b), (c, d) = bounds_in(router.report)
            said = f"seed {seed}, {names}, {router.report}"
            growth = 1
            for name in names:
                growth *= STEPS[name][2]
            longest = WIDTH * (MOST_CELLS_OUT // growth)
            before = router.stats()
            share = 0.0
            for number in range(1, DATAGRAMS + 1):
                if number <= 3:
                    payload = payload_of(longest, ("zeros", "ones", "random")[number - 1], rng)
                else:
                    payload = payload_of(rng.choice((0, 1, 7, 8, 30, longest)),
                                         rng.choice(("zeros", "ones", "random", "random")), rng)
                cells = cells_of(payload)
                n = 1 + len(cells)
                for name in names:
                    cells = STEPS[name][1](cells)
                router.raw(bytes((PORT,)) + payload)
                answer = router.outputs[PORT].recv(4096)
                self.assertEqual(answer, wire_of(cells), f"{said}, payload {payload.hex()}")
                now = router.settle(number)
                steps = now["steps"] - before["steps"]
                self.assertEqual(now["served"] - before["served"], 1, said)
                self.assertEqual(now["admitted_cells"] - before["admitted_cells"], n, said)
                self.assertEqual(now["budget_steps"] - before["budget_steps"], c * n + d, said)
                self.assertLessEqual(steps, c * n + d, f"{said}, {n} cells")
                share = max(share, steps / (c * n + d))
                before = now
            self.assertEqual(before["pool"], before["capacity"], said)
            self.assertEqual(a > 0, "dup" in names, said)
            if "dup" not in names:  # otherwise a header cell's share of a can pay for a draw
                self.assertEqual(b > 0, "marks" in names, said)
            return share
        finally:
            router.close()

    def test_every_pipeline_is_accepted_correct_and_inside_its_bound(self):
        """And the bound is worth having: a pipeline that keeps its cells and has one way through
        for some long item spends nearly all of W(n) on it. One that lets go of cells on the way was
        charged for them anyway."""
        loosest_exact = loosest = 1.0
        for seed in range(1000 * SEED, 1000 * SEED + COUNT):
            rng = random.Random(seed)
            names = [rng.choice(tuple(STEPS)) for _ in range(rng.randrange(1, 5))]
            with tempfile.TemporaryDirectory() as workdir:
                share = self._serve(names, workdir, rng, seed)
            loosest = min(loosest, share)
            if not {"thin", "marks", "choose", "early"} & set(names):
                loosest_exact = min(loosest_exact, share)
        print(f"walking handlers: {COUNT} ran; the loosest W(n) was {loosest:.0%} used, "
              f"and {loosest_exact:.0%} among those with one path", file=sys.stderr)
        self.assertGreater(loosest_exact, 0.75, "the bound is true and useless")
        self.assertGreater(loosest, 0.25, "the bound is true and useless")


class TheDearestWayThrough(unittest.TestCase):
    """Pipelines chosen so that one mistaken rule is what the bound comes down to, each timed on the
    payload that takes its dearest path along the whole of a long item."""

    def assert_covered(self, names: list, payload: bytes) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            path = os.path.join(workdir, "walk.lfpl")
            with open(path, "w") as handle:
                handle.write(pipeline_text(names))
            router = Forwarder(bytes(1024), workdir, build_options=("--count-steps",), source=path)
            try:
                _, (c, d) = bounds_in(router.report)
                before = router.stats()
                router.raw(bytes((PORT,)) + payload)
                cells = cells_of(payload)
                for name in names:
                    cells = STEPS[name][1](cells)
                self.assertEqual(router.outputs[PORT].recv(4096), wire_of(cells))
                ran = router.settle(1)["steps"] - before["steps"]
            finally:
                router.close()
        self.assertLessEqual(ran, c * (1 + len(cells_of(payload))) + d, f"{names}, {router.report}")

    def test_a_rebuilt_cell_can_only_claim_what_the_chain_it_leaves_in_place_has(self):
        self.assert_covered(["touch", "rev"], b"x" * 700)

    def test_a_join_leaves_what_its_dearest_edge_left(self):
        self.assert_covered(["dear"], bytes(700))

    def test_a_value_out_of_a_join_carries_what_every_edge_vouches_for(self):
        """An odd first byte is the arm whose value came from a walk, with a walk still to come."""
        self.assert_covered(["choose"], b"\xff" * 700)

    def test_the_arm_with_the_mixes_in_it(self):
        self.assert_covered(["bump"], b"\xff" * 700)


class OneFunctionAtSeveralAmounts(unittest.TestCase):
    """Each call of a finished function gets its own numbers. One set per function would refuse
    these, each of which is plainly linear."""

    def bound(self, body: str) -> tuple:
        with tempfile.TemporaryDirectory() as workdir:
            path = os.path.join(workdir, "walk.lfpl")
            with open(path, "w") as handle:
                handle.write(PRELUDE + "handler h(p: P, free: Free) : (P, Free) = "
                             f"match p with P(d, port, body) => {body} end;\n")
            router = Forwarder(bytes(1024), workdir, source=path)
            router.close()
        return bounds_in(router.report)

    def test_a_walk_over_what_the_same_walk_returned(self):
        once = self.bound("(P@d(port, rev(body, Last)), free)")[1]
        twice = self.bound("(P@d(port, rev(rev(body, Last), Last)), free)")[1]
        self.assertGreater(once[0], 0)
        self.assertEqual(twice[0], 2 * once[0])

    def test_the_copy_of_a_copy_draws_for_what_the_first_one_made(self):
        cells = self.bound("let (a, f) = dup(body, Last, free) in let (b, g) = dup(a, Last, f) in "
                           "(P@d(port, b), g)")[0]
        self.assertEqual(cells, (3, 0))


class TooLargeForAnyMachine(unittest.TestCase):
    """Every program here has a linear bound, and something about it is too large to certify or to
    serve. A bound is a promise that the memory exists and the arithmetic works, so that is a refusal."""

    def test_a_linear_program_past_the_declared_size(self):
        """Each call gets its own copy of what it calls, so layers of calls multiply the rows."""
        layers = "fn t0(x: Bytes) : Bytes = rev(rev(x, Last), Last);\n" + "".join(
            f"fn t{k}(x: Bytes) : Bytes = t{k - 1}(t{k - 1}(x));\n" for k in range(1, 9))
        text = (PRELUDE + layers + "handler h(p: P, free: Free) : (P, Free) = match p with "
                "P(d, port, body) => (P@d(port, t8(body)), free) end;\n")
        self.assertRegex(refusal(text), r"section 8\.2: .*inequalities")

    def test_a_reserve_past_what_an_image_can_hold(self):
        """Each copy doubles the cells behind it, so twenty copies want a million per item cell."""
        self.assertRegex(refusal(pipeline_text(["dup"] * 20)), r"section 8\.2: .*image of \d+ bytes")

    def test_work_past_what_the_admission_table_can_hold(self):
        """Each layer calls the one below twice. No cells are involved, so nothing else stops it."""
        layers = "fn s0(x: Int64) : Int64 = +(x, 1);\n" + "".join(
            f"fn s{k}(x: Int64) : Int64 = s{k - 1}(s{k - 1}(x));\n" for k in range(1, 70))
        text = (PRELUDE + layers + "handler h(p: P, free: Free) : (P, Free) = match p with "
                "P(d, port, body) => (P@d(low8(s69(pack<8>(port))), body), free) end;\n")
        self.assertRegex(refusal(text), r"section 8\.2: .*W\(N_max\)")


class TheEndArm(unittest.TestCase):
    """Section 5.3's DRAW leaves the End arm out of the bound, because a reserve of R(n) cells does
    not reach it. That rests on every Free in a program being the reserve. rtcheck refuses to bound a
    handler that makes one, and the second half of this class shows the language gets there first:
    a program that made one could not finish."""

    def serve(self, text: str) -> Forwarder:
        workdir = tempfile.TemporaryDirectory()
        self.addCleanup(workdir.cleanup)
        path = os.path.join(workdir.name, "end.lfpl")
        with open(path, "w") as handle:
            handle.write(text)
        router = Forwarder(bytes(1024), workdir.name, build_options=("--count-steps",), source=path)
        self.addCleanup(router.close)
        return router

    def test_what_only_the_end_arm_does_is_not_in_the_bound_and_does_not_run(self):
        """Its reversal would show in the bytes, and in the steps."""
        router = self.serve(PRELUDE + """
handler h(p: P, free: Free) : (P, Free) = match p with P(d, port, body) =>
  match free with
  | End => (P@d(port, rev(body, Last)), End)
  | More(c, rest) => (P@d(port, body), More@c(rest))
  end
end;
""")
        (a, b), (c, d) = bounds_in(router.report)
        self.assertEqual((a, b, c), (0, 1, 0))
        before = router.stats()
        router.raw(bytes((PORT,)) + b"0123456789")
        self.assertEqual(router.outputs[PORT].recv(4096), b"0123456789")
        now = router.settle(1)
        self.assertLessEqual(now["steps"] - before["steps"], d)

    def test_a_walk_with_no_linear_bound_is_fine_where_it_cannot_run(self):
        """The arm has to type, and that is all it has to do."""
        router = self.serve(PRELUDE + """
fn quad(l: Bytes, acc: Bytes) : Bytes = match l with
  | Last => acc
  | Chunk(d, used, b0, b1, b2, b3, b4, b5, b6, next) =>
      quad(next, rev(Chunk@d(used, b0, b1, b2, b3, b4, b5, b6, acc), Last))
  end;
handler h(p: P, free: Free) : (P, Free) = match p with P(d, port, body) =>
  match free with
  | End => (P@d(port, quad(body, Last)), End)
  | More(c, rest) => (P@d(port, body), More@c(rest))
  end
end;
""")
        self.assertEqual(bounds_in(router.report)[0], (0, 1))
        router.raw(bytes((PORT,)) + b"abc")
        self.assertEqual(router.outputs[PORT].recv(4096), b"abc")

    HANDLER = """
handler h(p: P, free: Free) : (P, Free) = match p with P(d, port, body) =>
  match none(port) with
  | End => (P@d(port, body), free)
  | More(c, rest) => (P@d(port, Chunk@c(1, 0, 0, 0, 0, 0, 0, 0, body)), {merged})
  end
end;
"""

    def test_a_second_free_cannot_be_walked_away(self):
        self.assertRegex(refusal(PRELUDE + "fn none(x: Int8) : Free = End;\n"
                                 "fn append(a: Free, b: Free) : Free = match a with End => b "
                                 "| More(c, r) => append(r, More@c(b)) end;\n"
                                 + self.HANDLER.format(merged="append(rest, free)")), "rule R3")

    def test_a_second_free_cannot_be_walked_away_on_the_back_of_something_else(self):
        """Something else falling satisfies R3, and then it runs out with two Frees still in hand."""
        self.assertRegex(refusal(PRELUDE + """fn none(x: Int8) : Free = End;
fn merge(a: Free, b: Free, l: Bytes) : (Free, Bytes) = match l with
  | Last => (a, Last)
  | Chunk(d, used, b0, b1, b2, b3, b4, b5, b6, next) => match a with
      | End => merge(End, b, next)
      | More(c, r) => merge(r, More@c(b), next)
      end
  end;
""" + self.HANDLER.format(merged="free")), "linear value still in r1")

    def test_a_second_free_cannot_simply_be_left(self):
        self.assertRegex(refusal(PRELUDE + "fn none(x: Int8) : Free = End;\n"
                                 + self.HANDLER.format(merged="free")), "linear variable rest ")

    def test_a_second_free_cannot_leave_inside_the_output(self):
        self.assertRegex(refusal("""record Q { port: Int8 };
data Out = Nothing | Both(port: Int8, extra: Free);
handler h(q: Q, free: Free) : (Out, Free) = match q with Q(d, port) => (Both@d(port, End), free) end;
"""), "makes a Free")

    def test_a_second_free_cannot_leave_as_a_second_result(self):
        self.assertRegex(refusal(PRELUDE + """handler h(p: P, free: Free) : (P, Free, Free) =
  match p with P(d, port, body) => (P@d(port, body), free, End) end;
"""), "section 3.1")


if __name__ == "__main__":
    unittest.main()
