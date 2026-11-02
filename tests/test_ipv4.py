"""examples/ipv4.lfpl on live UDP sockets, against a model that shares nothing with it.

A datagram is a frame here. The model works in network byte order, as the RFCs write it, and so does the handler now,
through its field types, so what they share is the RFCs. What this adds to tests/test_spans.py is the wire: the bytes
a record that spans takes in and lets out, in network order, with every cell back in the pool afterwards.
"""
import io
import os
import re
import socket
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rtlower import rt_lines  # noqa: E402
from tests.reference import ipv4_model as model  # noqa: E402
from tests.test_forward import ROOT, Forwarder  # noqa: E402

SOURCE = "examples/ipv4.lfpl"
LARGEST_FRAME = ("--n-max", "217")  # the 34 bytes of its headers in 5 cells, and 1480 more at 7 bytes a cell


def serve(test: unittest.TestCase, source: str = SOURCE, options: tuple = LARGEST_FRAME) -> Forwarder:
    workdir = tempfile.TemporaryDirectory()
    test.addCleanup(workdir.cleanup)
    path = source if os.path.exists(os.path.join(ROOT, source)) else os.path.join(workdir.name, "program.lfpl")
    if path != source:
        with open(path, "w") as handle:
            handle.write(source)
    router = Forwarder(bytes(1024), workdir.name, build_options=options, source=path)
    test.addCleanup(router.close)
    return router


class TheExample(unittest.TestCase):
    def test_every_kind_of_frame_as_the_model_says(self):
        router = serve(self, options=(*LARGEST_FRAME, "--count-steps"))
        corpus = model.corpus()
        sent = refused = short = 0
        for number, (name, verdict, frame) in enumerate(corpus, 1):
            want_verdict, want_frame = model.forward(frame, model.by_top_bits)
            self.assertEqual(want_verdict, verdict, name)
            router.raw(frame)
            if want_frame is not None:
                hop = model.by_top_bits(frame[model.DESTINATION_AT:model.DESTINATION_AT + 4])
                self.assertEqual(router.outputs[hop].recv(4096).hex(), want_frame.hex(), name)
                sent += 1
            elif len(frame) < model.HEADER_BYTES:
                short += 1
            else:
                refused += 1
            if number % 64 == 0:
                router.settle(number)
        now = router.settle(len(corpus))
        self.assertEqual((now["forwarded"], now["dropped"], now["malformed"]), (sent, refused, short))
        self.assertEqual(now["pool"], now["capacity"])
        self.assertLessEqual(now["steps"], now["budget_steps"])
        self.assertGreater(now["steps"], 0)
        for port in router.outputs:
            port.setblocking(False)
            with self.assertRaises(BlockingIOError):
                port.recv(4096)
        self.assertIsNone(router.process.poll())

    def test_its_bound_counts_the_cell_it_draws_and_nothing_else(self):
        router = serve(self)
        self.assertRegex(router.report, r"R\(n\) = 1 cells, W\(n\) = \d+ steps")

    def test_the_readme_quotes_what_the_compiler_says(self):
        router = serve(self)
        with open(os.path.join(ROOT, "README.md")) as handle:
            self.assertIn(router.report.strip(), handle.read())

    def test_a_frame_past_n_max_is_too_large(self):
        router = serve(self, options=("--n-max", "216"))
        largest = model.ipv4_frame(payload=bytes(1480))
        router.raw(largest)
        router.raw(model.ipv4_frame(payload=bytes(1473)))
        now = router.settle(2)
        self.assertEqual((now["too_large"], now["forwarded"]), (1, 1))
        router.outputs[0].recv(4096)
        self.assertEqual(router.settle(2)["pool"], now["capacity"])

    def test_n_max_has_to_hold_the_record(self):
        with tempfile.TemporaryDirectory() as workdir:
            with self.assertRaises(Exception) as caught:
                Forwarder(bytes(1024), workdir, build_options=("--n-max", "4"), source=SOURCE)
            self.assertIn("alone is 5 cells", caught.exception.stderr)

    def test_it_is_one_record_in_five_cells(self):
        with open(os.path.join(ROOT, SOURCE), "rb") as handle:
            source = handle.read()
            self.assertEqual(len(re.findall(rb"^record ", source, re.M)), 1)
            handle.seek(0)
            lines = list(rt_lines(handle))
        self.assertEqual(len([line for line in lines if re.match(r"type t\d+ record", line)]), 5)

    def test_what_stays_in_the_frame_is_not_written(self):
        """Each cell's build, innermost first. Only the addresses, the TTL and the checksum are stored, and a
        link is the value it was. So forwarding touches 6 fields of 15, and the verdict is a cell of its own."""
        with open(os.path.join(ROOT, SOURCE), "rb") as handle:
            lines = list(rt_lines(handle))
        at = next(i for i, line in enumerate(lines) if line.startswith("handler"))
        builds = [line.strip() for line in lines[at:] if line.strip().startswith("build")]
        built = {}
        for line in lines[:at]:
            parts = line.split()
            if parts and parts[0] == "ctor":
                built[parts[1]] = parts[-1]
        frame = [line for line in builds if built[line.split()[2].rstrip(",")].startswith(("$", "Frame"))]
        self.assertEqual(len(frame), 5, builds)
        last, ip2, ip1, ethernet, first = frame
        self.assertRegex(last, r", =, =, =$")
        self.assertRegex(ip2, r", sub\(\[0:1\], #1\), =, r\d+, =, r\d+!$")
        self.assertRegex(ip1, r", =, =, =, =, =, r\d+!$")
        self.assertRegex(ethernet, r", #\d+, r\d+, =, r\d+!$")
        self.assertRegex(first, r", #\d+, r\d+, r\d+!$")


# A program of each shape the wire format has to get right, with what it does in Python beside it.
SCALARS = """record Hdr { a: Be16, b: Be32, c: Be64, e: Int16, g: Int32, h: Int8 };
data Out = Gone | Send(port: Int8, m: Hdr);
handler h(m: Hdr, free: Free) : (Out, Free) = match m with Hdr(d, a, b, c, e, g, h) =>
  match free with
  | End => (Gone, More@d(End))
  | More(p, rest) => (Send@p(1, Hdr@d(+(a, 1), +(b, 1), +(c, 1), +(e, 1), +(g, 1), +(h, 1))), rest)
  end end;
"""
LENGTHS = """data Bytes;
record Hdr { a: Be16, length: Int16, body: Bytes };
data Out = Send(port: Int8, a: Be16, seen: Be16, body: Bytes);
handler h(m: Hdr, free: Free) : (Out, Free) = match m with Hdr(d, a, length, body) =>
  (Send@d(2, a, length, body), free) end;
"""
ALL_THE_WAY = """data Bytes;
record F { a: Int64, b: Int64, c: Be16, e: Be16, g: Int64, h: Int64, body: Bytes };
data Out = Gone(body: Bytes) | Send(port: Int8, f: F);
handler h(m: F, free: Free) : (Out, Free) = match free with
  | End => match m with F(d, a, b, c, e, g, h, body) => let (p, rest) = peel(d) in (Gone@p(body), More@rest(End)) end
  | More(p, rest) => match m with F(d, a, b, c, e, g, h, body) =>
      if ==(c, 0) then (Gone@p(body), More@d(rest))
      else (Send@p(3, F@d(h, g, e, c, b, a, body)), rest) end end;
"""


class TheWire(unittest.TestCase):
    def test_each_field_in_the_order_it_was_declared(self):
        router = serve(self, SCALARS, ())
        router.raw(struct.pack(">HIQ", 0x0102, 0x03040506, 0x0708090A0B0C0D0E) + struct.pack("<HIB", 0x1112, 0x13141516, 0x17))
        self.assertEqual(router.outputs[1].recv(64).hex(),
                         (struct.pack(">HIQ", 0x0103, 0x03040507, 0x0708090A0B0C0D0F)
                          + struct.pack("<HIB", 0x1113, 0x13141517, 0x18)).hex())
        now = router.settle(1)
        self.assertEqual(now["pool"], now["capacity"])

    def test_a_carry_goes_where_the_integer_says(self):
        router = serve(self, SCALARS, ())
        router.raw(struct.pack(">HIQ", 0x00FF, 0x0000FFFF, 0xFF) + struct.pack("<HIB", 0xFF, 0xFF, 0xFF))
        self.assertEqual(router.outputs[1].recv(64).hex(),
                         (struct.pack(">HIQ", 0x0100, 0x00010000, 0x100) + struct.pack("<HIB", 0x100, 0x100, 0)).hex())

    def test_length_is_how_long_the_datagram_was_and_is_not_on_the_wire(self):
        router = serve(self, LENGTHS, ())
        for size in (0, 1, 7, 8, 100, 1000):
            body = bytes(i & 255 for i in range(size))
            router.raw(struct.pack(">H", 0xABCD) + body)
            self.assertEqual(router.outputs[2].recv(4096),
                             struct.pack(">HH", 0xABCD, 2 + size) + body, size)
        now = router.settle(6)
        self.assertEqual(now["pool"], now["capacity"])

    def test_too_short_for_what_is_on_the_wire_is_malformed(self):
        router = serve(self, LENGTHS, ())
        router.raw(b"\x01")
        now = router.settle(1)
        self.assertEqual(now["malformed"], 1)

    def test_a_record_that_spans_comes_back_with_its_fields_exchanged_and_its_cells_counted(self):
        router = serve(self, ALL_THE_WAY, ())
        self.assertRegex(router.report, r"R\(n\) = 1 cells")
        before = router.stats()
        wire = struct.pack("<QQ", 1, 2) + struct.pack(">HH", 3, 4) + struct.pack("<QQ", 5, 6)
        for size in (0, 1, 7, 8, 700):
            body = bytes(i & 255 for i in range(size))
            router.raw(wire + body)
            self.assertEqual(router.outputs[3].recv(4096).hex(),
                             (struct.pack("<QQ", 6, 5) + struct.pack(">HH", 4, 3) + struct.pack("<QQ", 2, 1) + body).hex())
        now = router.settle(5)
        self.assertEqual(now["pool"], now["capacity"])
        # 5 cells of record, and however many of body, for each item. Admission worked it out from each length.
        cells = sum(5 + (size + 6) // 7 for size in (0, 1, 7, 8, 700))
        self.assertEqual(now["admitted_cells"] - before["admitted_cells"], cells)

    def test_a_verdict_that_keeps_only_some_of_the_cells_gives_the_rest_back(self):
        router = serve(self, ALL_THE_WAY, ())
        wire = struct.pack("<QQ", 1, 2) + struct.pack(">HH", 0, 4) + struct.pack("<QQ", 5, 6)
        router.raw(wire + bytes(100))
        now = router.settle(1)
        self.assertEqual((now["dropped"], now["pool"]), (1, now["capacity"]))


if __name__ == "__main__":
    unittest.main()
