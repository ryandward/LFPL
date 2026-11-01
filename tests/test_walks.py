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


STEPS = {
    "rev": ("let {out} = rev({x}, Last) in\n", model_rev, 1),
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

    def test_weave_alternates_and_finishes_the_longer_side_reversed(self):
        a, b, c = self.A, self.B, self.C
        self.assertEqual(model_weave([]), [])
        self.assertEqual(model_weave([a]), [a])
        # split makes x = [a], y = [b]. weave puts a on the accumulator, then b on top of it.
        self.assertEqual(model_weave([a, b]), [b, a])
        # split makes x = [b], y = [c, a]. weave moves b and then c, and x is empty, so what is
        # left of y goes on the accumulator: that is a, in front.
        self.assertEqual(model_weave([a, b, c]), [a, c, b])


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
                payload = bytes(rng.randrange(256) for _ in range(rng.choice((0, 1, 7, 8, 30, longest))))
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
            self.assertEqual(b, 0, said)
            return share
        finally:
            router.close()

    def test_every_pipeline_is_accepted_correct_and_inside_its_bound(self):
        """And the bound is worth having: a pipeline that drops no cells on the way spends nearly
        all of W(n) on a long item. One that does was charged for cells it let go of."""
        loosest_exact = loosest = 1.0
        for seed in range(1000 * SEED, 1000 * SEED + COUNT):
            rng = random.Random(seed)
            names = [rng.choice(tuple(STEPS)) for _ in range(rng.randrange(1, 5))]
            with tempfile.TemporaryDirectory() as workdir:
                share = self._serve(names, workdir, rng, seed)
            loosest = min(loosest, share)
            if "thin" not in names and "bump" not in names:
                loosest_exact = min(loosest_exact, share)
        print(f"walking handlers: {COUNT} ran; the loosest W(n) was {loosest:.0%} used, "
              f"and {loosest_exact:.0%} among those with one path", file=sys.stderr)
        self.assertGreater(loosest_exact, 0.75, "the bound is true and useless")


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
    """Every program here has a linear bound, and the numbers in it are too large to be served. A
    bound is a promise that the memory exists and that the arithmetic works, so that is a refusal."""

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
    not reach it. That rests on every Free in a program being the reserve, which is what the second
    half of this class pins: if any of these were ever accepted, a program's own End would be
    reachable, and its cost would be missing from the bound."""

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
"""), "carries scalars and one chain")

    def test_a_second_free_cannot_leave_as_a_second_result(self):
        self.assertRegex(refusal(PRELUDE + """handler h(p: P, free: Free) : (P, Free, Free) =
  match p with P(d, port, body) => (P@d(port, body), free, End) end;
"""), "section 3.1")


if __name__ == "__main__":
    unittest.main()
