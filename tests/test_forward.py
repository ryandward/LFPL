"""The milestone: the section 4.7 handler, compiled unchanged, serving real sockets.

A packet on the wire is [ttl: u8][dst: u32 little-endian][body bytes]. The control port
answers `S` with the counters and `Q` by shutting down cleanly. Everything here happens
to a running binary, from outside it.
"""
import os
import resource
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = "examples/forward.lfpl"
COUNTERS = ("capacity", "pool", "served", "forwarded", "dropped",
            "too_large", "short", "full", "malformed", "send_errors", "recv_errors")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def egress_block() -> tuple[int, list]:
    """Egress ports are a base plus the handler's port number, so 8 in a row, held open."""
    while True:
        base = free_port()
        held = []
        try:
            for number in range(8):
                out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                held.append(out)
                out.bind(("127.0.0.1", base + number))
                out.settimeout(0.5)
            return base, held
        except OSError:
            for out in held:
                out.close()


class Forwarder:
    """Builds examples/forward.lfpl with a route table, and runs it."""

    def __init__(self, routes: bytes, workdir: str, limit_lock: bool = False):
        self.out_base, self.outputs = egress_block()
        self.in_port, self.ctl_port = free_port(), free_port()
        table = os.path.join(workdir, "routes.bin")
        with open(table, "wb") as handle:
            handle.write(routes)
        exe = os.path.join(workdir, "forward")
        subprocess.run([sys.executable, "-m", "src.lfplo", SOURCE, "-o", exe,
                        "--in-port", str(self.in_port), "--out-port", str(self.out_base),
                        "--ctl-port", str(self.ctl_port), "--state", f"routes={table}"],
                       cwd=ROOT, check=True, capture_output=True, text=True)
        self.exe = exe
        limits = (lambda: resource.setrlimit(resource.RLIMIT_MEMLOCK, (0, 0))) if limit_lock else None
        self.process = subprocess.Popen([exe], preexec_fn=limits)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ctl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ctl.settimeout(2.0)

    def send(self, ttl: int, dst: int, body: bytes = b"") -> None:
        self.tx.sendto(struct.pack("<BI", ttl, dst) + body, ("127.0.0.1", self.in_port))

    def raw(self, data: bytes) -> None:
        self.tx.sendto(data, ("127.0.0.1", self.in_port))

    def stats(self) -> dict:
        for _ in range(20):
            self.ctl.sendto(b"S", ("127.0.0.1", self.ctl_port))
            try:
                reply = self.ctl.recv(256)
            except TimeoutError:
                continue
            return dict(zip(COUNTERS, struct.unpack(f"<{len(COUNTERS)}Q", reply)))
        raise AssertionError("the control port never answered")

    def settle(self, served: int) -> dict:
        """Wait until the binary has taken in this many items, or has stopped moving."""
        deadline = time.time() + 5
        last = None
        while time.time() < deadline:
            now = self.stats()
            handled = now["served"] + now["too_large"] + now["short"] + now["full"] + now["malformed"]
            if handled >= served or (last is not None and now == last):
                return now
            last = now
            time.sleep(0.05)
        return self.stats()

    def rss(self) -> int:
        with open(f"/proc/{self.process.pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        raise AssertionError("no VmRSS")

    def close(self) -> int:
        if self.process.poll() is None:
            self.ctl.sendto(b"Q", ("127.0.0.1", self.ctl_port))
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                return -999
        for sock in self.outputs + [self.tx, self.ctl]:
            sock.close()
        return self.process.returncode


class ForwardingHandler(unittest.TestCase):
    ROUTES = bytes((i % 8) for i in range(1024))

    def setUp(self):
        self.workdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.workdir.cleanup)
        self.router = Forwarder(self.ROUTES, self.workdir.name)
        self.addCleanup(self._finish)
        self.closed = False
        time.sleep(0.05)

    def _finish(self):
        if not self.closed:
            self.router.close()

    def test_a_packet_comes_out_on_its_route_with_ttl_decremented(self):
        body = bytes(range(256)) * 3
        self.router.send(64, 3, body)
        packet = self.router.outputs[3].recv(4096)
        self.assertEqual(packet[0], 63)
        self.assertEqual(struct.unpack_from("<I", packet, 1)[0], 3)
        self.assertEqual(packet[5:], body)

    def test_bodies_of_every_small_length_survive(self):
        for length in list(range(0, 40)) + [1000, 1400]:
            body = bytes((length + i) & 255 for i in range(length))
            self.router.send(9, 5, body)
            with self.subTest(length=length):
                self.assertEqual(self.router.outputs[5].recv(4096), struct.pack("<BI", 8, 5) + body)

    def test_the_table_decides_the_port_and_idx_wraps_the_address(self):
        for dst in (0, 1, 7, 8, 1023, 1024, 1025, 0xFFFFFFFF):
            self.router.send(2, dst, b"x")
            with self.subTest(dst=dst):
                packet = self.router.outputs[self.ROUTES[dst & 1023]].recv(64)
                self.assertEqual(packet, struct.pack("<BI", 1, dst) + b"x")

    def test_an_expired_ttl_is_dropped_and_its_cells_come_back(self):
        before = self.router.stats()
        for ttl in (0, 1):
            self.router.send(ttl, 2, b"gone" * 100)
        after = self.router.settle(before["served"] + 2)
        self.assertEqual(after["dropped"] - before["dropped"], 2)
        self.assertEqual(after["forwarded"], before["forwarded"])
        self.assertEqual(after["pool"], after["capacity"])
        self.router.outputs[2].settimeout(0.1)
        with self.assertRaises(TimeoutError):
            self.router.outputs[2].recv(64)

    def test_an_item_too_large_is_refused_at_admission_with_no_effect(self):
        before = self.router.stats()
        self.router.send(9, 1, b"z" * 20000)
        after = self.router.settle(before["served"] + 1)
        self.assertEqual(after["too_large"] - before["too_large"], 1)
        self.assertEqual(after["served"], before["served"])
        self.assertEqual(after["pool"], after["capacity"])
        self.router.send(9, 1, b"still alive")
        self.assertEqual(self.router.outputs[1].recv(64)[5:], b"still alive")

    def test_a_malformed_packet_is_counted_and_harmless(self):
        before = self.router.stats()
        for data in (b"", b"\x05", b"\x05\x00\x00\x00"):
            self.router.raw(data)
        after = self.router.settle(before["served"] + 3)
        self.assertEqual(after["malformed"] - before["malformed"], 3)
        self.assertEqual(after["pool"], after["capacity"])

    def test_every_cell_comes_back_and_memory_never_moves_under_a_flood(self):
        start_rss = self.router.rss()
        for i in range(30000):
            self.router.send(i % 5, i, b"p" * (i % 300))
        now = self.router.settle(30000)
        self.assertGreater(now["served"], 1000)
        self.assertEqual(now["pool"], now["capacity"], now)
        self.assertEqual(self.router.rss(), start_rss)
        self.assertIsNone(self.router.process.poll())

    def test_it_shuts_down_cleanly_when_told_to(self):
        self.closed = True
        self.assertEqual(self.router.close(), 0)


class Startup(unittest.TestCase):
    def test_it_refuses_to_start_when_its_image_cannot_be_locked(self):
        """Section 7.2: a distinct exit status, before it reads any input."""
        if os.geteuid() == 0:
            self.skipTest("root may lock regardless of the limit")
        with tempfile.TemporaryDirectory() as workdir:
            router = Forwarder(bytes(1024), workdir, limit_lock=True)
            try:
                self.assertEqual(router.process.wait(timeout=3), 2)
            finally:
                router.close()


class Binary(unittest.TestCase):
    def test_nothing_is_linked_in(self):
        with tempfile.TemporaryDirectory() as workdir:
            router = Forwarder(bytes(1024), workdir)
            try:
                listing = subprocess.run(["objdump", "-p", router.exe], capture_output=True,
                                         text=True).stdout
                self.assertNotIn("NEEDED", listing)
                self.assertNotIn("INTERP", listing)
            finally:
                router.close()


if __name__ == "__main__":
    unittest.main()
