"""A handler that walks its item: examples/checksum.lfpl, on a live socket, with a certified bound.

Everything here is judged from outside a running binary, against a checksum written independently
in Python, and against the number of steps the binary really runs. A bound the checker accepts is
only a claim until the code it describes has been timed against it.
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import lfplo  # noqa: E402
from src.lfplo import Refused, certify  # noqa: E402
from src.rtcheck import Checker, RTError  # noqa: E402
from src.rtlower import rt_lines  # noqa: E402
from src.runtime import Options  # noqa: E402
from src.simplex import minimize  # noqa: E402
from tests.test_forward import ROOT, Forwarder  # noqa: E402

SOURCE = os.path.join(ROOT, "examples", "checksum.lfpl")
PORT = 3
HEADER = 3
PER_CELL = 7
LARGEST = 215 * PER_CELL  # 216 cells: the header, then 215 of 7 bytes


def internet_checksum(data: bytes) -> bytes:
    """RFC 1071, big-endian words, sharing nothing with the handler's way of doing it."""
    total = 0
    for at in range(0, len(data), 2):
        total += (data[at] << 8) + (data[at + 1] if at + 1 < len(data) else 0)
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return struct.pack(">H", ~total & 0xFFFF)


def cells_in(length: int) -> int:
    """How many cells an item of this payload takes, worked out independently of the runtime."""
    return 1 + (length + PER_CELL - 1) // PER_CELL


def packet(body: bytes, stale: int = 0xABCD) -> bytes:
    return struct.pack("<BH", PORT, stale) + body


def bound_of(report: str) -> tuple[int, int]:
    found = re.search(r"W\(n\) = (\d+) n \+ (\d+) steps", report)
    assert found is not None, report
    return int(found.group(1)), int(found.group(2))


def payload(length: int) -> bytes:
    return bytes((31 * i + length) & 255 for i in range(length))


class Serving(unittest.TestCase):
    def start(self, *options: str, limit_lock: bool = False, source: str = SOURCE) -> Forwarder:
        workdir = tempfile.TemporaryDirectory()
        self.addCleanup(workdir.cleanup)
        router = Forwarder(bytes(1024), workdir.name, limit_lock=limit_lock,
                           build_options=options, source=source)
        self.addCleanup(router.close)
        return router


class TheBound(Serving):
    def test_no_reserve_and_work_linear_in_the_item(self):
        report = self.start().report
        self.assertIn("R(n) = 0 cells", report)
        slope, constant = bound_of(report)
        self.assertGreater(slope, 0)
        self.assertGreater(constant, 0)

    def test_admission_works_out_n_and_w_of_n_from_the_length_on_the_wire(self):
        router = self.start()
        slope, constant = bound_of(router.report)
        before = router.stats()
        for count, length in enumerate((0, 1, 6, 7, 8, 13, 14, 15, 100, LARGEST), 1):
            router.raw(packet(payload(length)))
            router.outputs[PORT].recv(4096)
            now = router.settle(count)
            n = cells_in(length)
            self.assertEqual(now["admitted_cells"] - before["admitted_cells"], n, length)
            self.assertEqual(now["budget_steps"] - before["budget_steps"], slope * n + constant, length)
            before = now

    def test_the_bound_holds_for_what_really_runs_and_is_tight(self):
        router = self.start("--count-steps")
        slope, constant = bound_of(router.report)
        before = router.stats()
        ran = {}
        for count, length in enumerate(list(range(0, 50)) + [700, LARGEST], 1):
            router.raw(packet(payload(length)))
            self.assertEqual(router.outputs[PORT].recv(4096)[2:], payload(length))
            now = router.settle(count)
            steps = now["steps"] - before["steps"]
            before = now
            n = cells_in(length)
            self.assertLessEqual(steps, slope * n + constant, f"{length} bytes, {n} cells")
            ran[n] = steps
        # Each further cell costs exactly the slope, so the slope is not an overestimate. The
        # surplus is one cell's worth at most: the header cell carries a share it does not use.
        self.assertEqual(ran[3] - ran[2], slope)
        self.assertEqual(ran[216] - ran[3], slope * 213)
        self.assertLessEqual(slope * 216 + constant - ran[216], slope + constant)

    def test_numbers_lower_than_the_code_needs_are_refused(self):
        """The certificate is the checker's own inequalities with the solver's numbers in them. So
        a lower number than the code needs is refused, however it was arrived at."""
        checker = Checker()
        with open(SOURCE, "rb") as handle:
            for line in rt_lines(handle):
                checker.feed(line)
        checker.finish()
        good = certify(checker, "checksum", 216)
        program = checker.linear_program("checksum", 216)
        cells = minimize(program.objective(), program.constraints("cells"))
        steps = minimize(program.objective(), program.constraints("steps"))
        self.assertEqual(checker.verify("checksum", cells, steps), good)
        for name, forged in (("a lower slope", {**steps, program.item: steps[program.item] - 1}),
                             ("a fraction less", {**steps, program.item: steps[program.item] - Fraction(1, 7)}),
                             ("nothing at all", {}),
                             ("a negative number", {**steps, program.entry: Fraction(-1)})):
            with self.subTest(name), self.assertRaisesRegex(RTError, "certificate"):
                checker.verify("checksum", cells, forged)

    def test_a_solver_with_a_wrong_answer_leaves_no_binary(self):
        """This is the whole of the claim that the solver need not be trusted, so it goes through
        the driver: a build whose solver lies must fail like any other failed check."""
        real = lfplo.minimize
        lfplo.minimize = lambda objective, constraints: {unknown: Fraction(0) for unknown in objective}
        try:
            with tempfile.TemporaryDirectory() as workdir:
                exe = os.path.join(workdir, "checksum")
                with open(exe, "w") as handle:
                    handle.write("a binary from an earlier build")
                with self.assertRaisesRegex(Refused, "certificate"):
                    lfplo.build(SOURCE, exe, Options(41000, 41100, 41001))
                self.assertFalse(os.path.exists(exe))
        finally:
            lfplo.minimize = real


class TheChecksum(Serving):
    def test_every_short_length_and_the_longest(self):
        router = self.start()
        for length in list(range(0, 64)) + [1000, LARGEST]:
            body = payload(length)
            router.raw(packet(body))
            with self.subTest(length=length):
                self.assertEqual(router.outputs[PORT].recv(4096), internet_checksum(body) + body)

    def test_the_examples_of_the_rfc_and_of_an_ip_header(self):
        router = self.start()
        for body, expected in ((bytes.fromhex("0001f203f4f5f6f7"), "220d"),
                               (bytes.fromhex("450000730000400040110000c0a80001c0a800c7"), "b861")):
            router.raw(packet(body))
            self.assertEqual(router.outputs[PORT].recv(64)[:2].hex(), expected)

    def test_what_a_sender_put_in_the_field_does_not_matter(self):
        router = self.start()
        for stale in (0, 0xFFFF, 0x1234):
            router.raw(packet(b"same payload", stale))
            self.assertEqual(router.outputs[PORT].recv(64)[:2], internet_checksum(b"same payload"))

    def test_a_short_item_cannot_read_what_a_longer_one_left_in_its_cells(self):
        """A cell comes back with its old bytes. If the bytes past the used count were not cleared
        at admission, a one-byte payload would be summed with the last item's."""
        router = self.start()
        for round_ in range(30):
            router.raw(packet(b"\xff" * 700))
            router.outputs[PORT].recv(4096)
            router.raw(packet(b"\x01"))
            self.assertEqual(router.outputs[PORT].recv(64), internet_checksum(b"\x01") + b"\x01", round_)

    def test_a_checksum_verifies_the_way_a_receiver_would_check_it(self):
        router = self.start()
        body = payload(601)
        router.raw(packet(body))
        answer = router.outputs[PORT].recv(4096)
        self.assertEqual(internet_checksum(answer), b"\0\0")


class Refusals(Serving):
    """What tests/test_forward.py shows of forward.lfpl, on a handler whose work depends on n."""

    def test_an_item_too_large_is_refused_with_no_effect(self):
        router = self.start()
        before = router.stats()
        router.raw(packet(b"z" * 20000))
        after = router.settle(1)
        self.assertEqual(after["too_large"] - before["too_large"], 1)
        self.assertEqual(after["served"], before["served"])
        self.assertEqual(after["admitted_cells"], before["admitted_cells"])
        self.assertEqual(after["pool"], after["capacity"])

    def test_the_largest_item_fits_and_one_more_byte_does_not(self):
        router = self.start()
        router.raw(packet(payload(LARGEST)))
        self.assertEqual(len(router.outputs[PORT].recv(4096)), 2 + LARGEST)
        before = router.settle(1)
        router.raw(packet(payload(LARGEST + 1)))
        after = router.settle(2)
        self.assertEqual(after["too_large"] - before["too_large"], 1)

    def test_a_malformed_packet_is_counted_and_harmless(self):
        router = self.start()
        for data in (b"", b"\x03", b"\x03\x00"):
            router.raw(data)
        after = router.settle(3)
        self.assertEqual(after["malformed"], 3)
        self.assertEqual(after["pool"], after["capacity"])

    def test_a_full_queue_refuses_and_nothing_admitted_is_lost(self):
        router = self.start("--k-out", "8")
        router.stats(b"P")
        for i in range(12):
            router.raw(packet(b"%d" % i))
        paused = router.settle(12)
        self.assertEqual((paused["served"], paused["forwarded"], paused["full"]), (8, 8, 4))
        router.stats(b"R")
        for i in range(8):
            self.assertEqual(router.outputs[PORT].recv(64)[2:], b"%d" % i)
        router.outputs[PORT].settimeout(0.1)
        with self.assertRaises(TimeoutError):
            router.outputs[PORT].recv(64)
        drained = router.stats()
        self.assertEqual(drained["pool"], drained["capacity"])

    def test_a_short_pool_refuses_by_the_n_of_each_item_and_recovers(self):
        """Admission needs n cells from the pool, and n comes from this datagram's length. So with
        two cells left, an item of three is refused, and an item of two behind it is served."""
        router = self.start("--n-max", "8", "--cells", "40")
        router.stats(b"P")
        largest = b"z" * (7 * 7)
        for _ in range(3):
            router.raw(packet(largest))
        paused = router.settle(3)
        self.assertEqual((paused["capacity"], paused["pool"]), (32, 8))
        router.raw(packet(b"z" * 7 * 7))  # 8 cells of the 8 left
        self.assertEqual(router.settle(4)["pool"], 0)
        router.raw(packet(b"y"))
        self.assertEqual(router.settle(5)["short"], 1)
        router.stats(b"R")
        for _ in range(4):
            self.assertEqual(router.outputs[PORT].recv(128)[2:], largest)
        now = router.stats(b"P")
        self.assertEqual(now["pool"], 32)
        for _ in range(6):
            router.raw(packet(b"z" * 7 * 4))  # 5 cells each: six of them would take 30
        self.assertEqual(router.settle(11)["short"], 1)
        for _ in range(2):
            router.raw(packet(b"z" * 7 * 2))  # 3 cells, with 2 left
        self.assertEqual(router.settle(13)["short"], 3)
        router.raw(packet(b"z"))              # 2 cells, with 2 left
        after = router.settle(14)
        self.assertEqual((after["short"], after["pool"]), (3, 0))
        router.stats(b"R")
        for _ in range(7):
            router.outputs[PORT].recv(128)
        self.assertEqual(router.stats()["pool"], 32)

    def test_every_cell_comes_back_and_memory_never_moves_under_a_flood(self):
        router = self.start()
        start_rss = router.rss()
        for i in range(30000):
            router.raw(packet(b"p" * (i % 300)))
        now = router.settle(30000)
        self.assertGreater(now["served"], 1000)
        self.assertEqual(now["pool"], now["capacity"], now)
        self.assertEqual(router.rss(), start_rss)
        self.assertIsNone(router.process.poll())

    def test_it_refuses_to_start_when_its_image_cannot_be_locked(self):
        if os.geteuid() == 0:
            self.skipTest("root may lock regardless of the limit")
        router = self.start(limit_lock=True)
        self.assertEqual(router.process.wait(timeout=3), 2)

    def test_it_shuts_down_cleanly_when_told_to(self):
        router = self.start()
        router.raw(packet(b"x"))
        router.outputs[PORT].recv(64)
        router.ctl.sendto(b"Q", ("127.0.0.1", router.ctl_port))
        self.assertEqual(router.process.wait(timeout=3), 0)

    def test_nothing_is_linked_in(self):
        router = self.start()
        listing = subprocess.run(["objdump", "-p", router.exe], capture_output=True, text=True).stdout
        self.assertNotIn("NEEDED", listing)
        self.assertNotIn("INTERP", listing)


CHAIN = """data Bytes = Last | Chunk(used: Int8, b0: Int8, b1: Int8, b2: Int8, b3: Int8, b4: Int8, b5: Int8,
                          b6: Int8, next: Bytes);
record P { port: Int8, body: Bytes };
"""


class WhatTheChainTypeOpensUp(Serving):
    """A program that can see the chain can also write it, so the runtime cannot trust it."""

    def program(self, text: str) -> str:
        workdir = tempfile.TemporaryDirectory()
        self.addCleanup(workdir.cleanup)
        path = os.path.join(workdir.name, "program.lfpl")
        with open(path, "w") as handle:
            handle.write(CHAIN + text)
        return path

    def test_a_forged_used_count_names_no_byte_outside_its_cell(self):
        router = self.start(source=self.program("""
fn forge(l: Bytes, acc: Bytes) : Bytes = match l with
  | Last => acc
  | Chunk(d, used, b0, b1, b2, b3, b4, b5, b6, next) => forge(next, Chunk@d(200, b0, b1, b2, b3, b4, b5, b6, acc))
  end;
handler h(p: P, free: Free) : (P, Free) =
  match p with P(d, port, body) => (P@d(port, forge(forge(body, Last), Last)), free) end;
"""))
        router.raw(bytes((PORT,)) + b"abc")
        self.assertEqual(router.outputs[PORT].recv(4096), b"abc\0\0\0\0")
        router.raw(bytes((PORT,)) + b"0123456789")
        self.assertEqual(router.outputs[PORT].recv(4096), b"0123456" + b"789\0\0\0\0")
        self.assertEqual(router.settle(2)["pool"], router.stats()["capacity"])

    def test_a_chain_grown_past_what_a_message_can_gather_is_dropped_and_returned(self):
        router = self.start(source=self.program("""
fn more(free: Free, acc: Bytes) : (Free, Bytes) = match free with
  | End => (End, acc)
  | More(c, rest) => (rest, Chunk@c(1, 65, 0, 0, 0, 0, 0, 0, acc))
  end;
fn grow(l: Bytes, acc: Bytes, free: Free) : (Bytes, Free) = match l with
  | Last => (acc, free)
  | Chunk(d, used, b0, b1, b2, b3, b4, b5, b6, next) =>
      let (f1, a1) = more(free, Chunk@d(used, b0, b1, b2, b3, b4, b5, b6, acc)) in
      let (f2, a2) = more(f1, a1) in let (f3, a3) = more(f2, a2) in let (f4, a4) = more(f3, a3) in
      let (f5, a5) = more(f4, a4) in let (f6, a6) = more(f5, a5) in let (f7, a7) = more(f6, a6) in
      let (f8, a8) = more(f7, a7) in let (f9, a9) = more(f8, a8) in let (fa, aa) = more(f9, a9) in
      let (fb, ab) = more(fa, aa) in let (fc, ac) = more(fb, ab) in
      grow(next, ac, fc)
  end;
handler h(p: P, free: Free) : (P, Free) =
  match p with P(d, port, body) => let (a, f) = grow(body, Last, free) in (P@d(port, a), f) end;
"""))
        self.assertIn("R(n) = 12 n", router.report)
        router.raw(bytes((PORT,)) + b"x" * 700)  # 100 body cells become 1300, past what sendmsg takes
        now = router.settle(1)
        self.assertEqual((now["served"], now["send_errors"]), (1, 1))
        self.assertEqual(now["pool"], now["capacity"])
        router.raw(bytes((PORT,)) + b"y")      # one body cell: 13 cells, which fit
        answer = router.outputs[PORT].recv(4096)
        self.assertEqual(answer, b"A" * 12 + b"y")
        self.assertEqual(router.settle(2)["pool"], now["capacity"])

    TWICE = """
fn twice(l: Bytes, acc: Bytes, free: Free) : (Bytes, Free) = match l with
  | Last => (acc, free)
  | Chunk(d, used, b0, b1, b2, b3, b4, b5, b6, next) => match free with
      | End => twice(next, Chunk@d(used, b0, b1, b2, b3, b4, b5, b6, acc), End)
      | More(c, rest) =>
          twice(next, Chunk@d(used, b0, b1, b2, b3, b4, b5, b6,
                              Chunk@c(used, b0, b1, b2, b3, b4, b5, b6, acc)), rest)
      end
  end;
handler h(p: P, free: Free) : (P, Free) =
  match p with P(d, port, body) => let (a, f) = twice(body, Last, free) in (P@d(port, a), f) end;
"""

    def test_at_the_size_section_3_4_gives_the_pool_is_never_short(self):
        """With a queue full of the largest outputs this handler can make, the next item is refused
        for the queue. If it were refused for the pool, the image is smaller than the formula means:
        an output can hold every cell of its item and of its reserve, not just N_max of them."""
        router = self.start("--k-out", "2", source=self.program(self.TWICE))
        router.stats(b"P")
        largest = b"z" * LARGEST
        for _ in range(6):
            router.raw(bytes((PORT,)) + largest)
        now = router.settle(6)
        self.assertEqual((now["served"], now["full"], now["short"]), (2, 4, 0))
        router.stats(b"R")
        for _ in range(2):
            self.assertEqual(len(router.outputs[PORT].recv(8192)), 2 * LARGEST)

    def test_the_largest_output_of_a_handler_that_grows_its_item_goes_out_whole(self):
        router = self.start(source=self.program(self.TWICE))
        router.raw(bytes((PORT,)) + b"z" * LARGEST)
        self.assertEqual(len(router.outputs[PORT].recv(8192)), 2 * LARGEST)
        now = router.settle(1)
        self.assertEqual((now["send_errors"], now["pool"]), (0, now["capacity"]))

    def test_a_reserve_proportional_to_the_item_is_given_and_never_runs_out(self):
        """R(n) = n: each body cell is copied into a drawn cell, so if admission worked out n wrong,
        or handed over fewer cells, the End arm would leave a copy out."""
        router = self.start(source=self.program(self.TWICE))
        self.assertIn("R(n) = 1 n cells", router.report)
        for length in (0, 1, 7, 8, 21, 70):
            body = bytes(range(length))
            router.raw(bytes((PORT,)) + body)
            cells = [body[i:i + 7] for i in range(0, length, 7)]
            expected = b"".join(cell + cell for cell in reversed(cells))
            self.assertEqual(router.outputs[PORT].recv(4096), expected, length)
        now = router.settle(6)
        self.assertEqual(now["pool"], now["capacity"])


if __name__ == "__main__":
    unittest.main()
