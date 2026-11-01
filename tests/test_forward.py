"""The milestone: the section 4.7 handler, compiled unchanged, serving real sockets.

A packet on the wire is [ttl: u8][dst: u32 little-endian][body bytes]. The control port
answers `S` with the counters and `Q` by shutting down cleanly. Everything here happens
to a running binary, from outside it.
"""
import os
import re
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
                out.settimeout(10)  # a test that waits for nothing sets its own, shorter one
            return base, held
        except OSError:
            for out in held:
                out.close()


class Forwarder:
    """Builds examples/forward.lfpl with a route table, and runs it."""

    def __init__(self, routes: bytes, workdir: str, limit_lock: bool = False,
                 build_options: tuple = ()):
        self.out_base, self.outputs = egress_block()
        self.in_port, self.ctl_port = free_port(), free_port()
        table = os.path.join(workdir, "routes.bin")
        with open(table, "wb") as handle:
            handle.write(routes)
        exe = os.path.join(workdir, "forward")
        subprocess.run([sys.executable, "-m", "src.lfplo", SOURCE, "-o", exe,
                        "--in-port", str(self.in_port), "--out-port", str(self.out_base),
                        "--ctl-port", str(self.ctl_port), "--state", f"routes={table}",
                        *build_options],
                       cwd=ROOT, check=True, capture_output=True, text=True)
        self.exe = exe
        limits = (lambda: resource.setrlimit(resource.RLIMIT_MEMLOCK, (0, 0))) if limit_lock else None
        self.process = subprocess.Popen([exe], preexec_fn=limits)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ctl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ctl.settimeout(2.0)
        if not limit_lock:
            self.stats()  # it answers, so its sockets are bound and a first packet is not lost

    def send(self, ttl: int, dst: int, body: bytes = b"") -> None:
        self.tx.sendto(struct.pack("<BI", ttl, dst) + body, ("127.0.0.1", self.in_port))

    def raw(self, data: bytes) -> None:
        self.tx.sendto(data, ("127.0.0.1", self.in_port))

    def stats(self, command: bytes = b"S") -> dict:
        for _ in range(20):
            self.ctl.sendto(command, ("127.0.0.1", self.ctl_port))
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


class Overload(unittest.TestCase):
    """Section 6.4. Loopback sends never block, so `P` pauses egress to make the pressure real."""

    ROUTES = bytes((i % 8) for i in range(1024))

    def start(self, *options: str) -> Forwarder:
        workdir = tempfile.TemporaryDirectory()
        self.addCleanup(workdir.cleanup)
        router = Forwarder(self.ROUTES, workdir.name, build_options=options)
        self.addCleanup(router.close)
        return router

    def test_a_full_queue_refuses_and_nothing_admitted_is_lost(self):
        router = self.start("--k-out", "8")
        router.stats(b"P")
        for i in range(12):
            router.send(9, 3, b"%d" % i)
        paused = router.settle(12)
        self.assertEqual((paused["served"], paused["forwarded"], paused["full"]), (8, 8, 4))
        self.assertLess(paused["pool"], paused["capacity"])
        router.stats(b"R")
        bodies = [router.outputs[3].recv(64)[5:] for _ in range(8)]
        self.assertEqual(bodies, [b"%d" % i for i in range(8)], "out of order, or lost")
        router.outputs[3].settimeout(0.1)
        with self.assertRaises(TimeoutError):
            router.outputs[3].recv(64)
        drained = router.stats()
        self.assertEqual(drained["pool"], drained["capacity"])

    def test_a_short_pool_refuses_and_recovers(self):
        # 8 cells of ingress and 32 in the pool, and a queue that could hold far more than that.
        router = self.start("--n-max", "8", "--cells", "40")
        largest = b"z" * (7 * 7)
        router.stats(b"P")
        for _ in range(7):
            router.send(9, 2, largest)
        paused = router.settle(7)
        self.assertEqual((paused["capacity"], paused["served"], paused["short"]), (32, 4, 3))
        self.assertEqual(paused["pool"], 0)
        router.stats(b"R")
        for _ in range(4):
            self.assertEqual(router.outputs[2].recv(128)[5:], largest)
        router.send(9, 2, largest)
        self.assertEqual(router.outputs[2].recv(128)[5:], largest)
        drained = router.settle(8)
        self.assertEqual(drained["pool"], drained["capacity"])

    def test_the_largest_item_fits_and_one_more_cell_is_too_large(self):
        router = self.start("--n-max", "8", "--cells", "40")
        router.send(9, 1, b"y" * (7 * 7))
        self.assertEqual(len(router.outputs[1].recv(128)), 5 + 7 * 7)
        before = router.stats()
        router.send(9, 1, b"y" * (7 * 7 + 1))
        after = router.settle(before["served"] + before["too_large"] + 1)
        self.assertEqual(after["too_large"] - before["too_large"], 1)


class Validation(unittest.TestCase):
    """Section 8.4: every check runs on every build, and a failure leaves no binary."""

    def build_with_damage(self, damage, exe: str, table: str):
        from src import lfplo
        from src.runtime import Options

        real = lfplo._generate

        def damaged(lines, asm_path, options, initial):
            result = real(lines, asm_path, options, initial)
            with open(asm_path) as handle:
                text = handle.read()
            with open(asm_path, "w") as handle:
                handle.write(damage(text))
            return result

        lfplo._generate = damaged
        try:
            return lfplo.build(os.path.join(ROOT, SOURCE), exe, Options(41000, 41100, 41001),
                               {"routes": table})
        finally:
            lfplo._generate = real

    def test_a_wrong_handler_leaves_no_binary_and_removes_a_stale_one(self):
        from src.lfplo import Refused

        with tempfile.TemporaryDirectory() as workdir:
            exe, table = os.path.join(workdir, "forward"), os.path.join(workdir, "routes.bin")
            with open(table, "wb") as handle:
                handle.write(bytes(1024))
            self.build_with_damage(lambda text: text, exe, table)
            self.assertTrue(os.path.exists(exe), "the control: an undamaged build must succeed")
            with self.assertRaisesRegex(Refused, "translation validation"):
                self.build_with_damage(lambda text: text.replace("and rbx, 1023", "and rbx, 1022"),
                                       exe, table)
            self.assertFalse(os.path.exists(exe))

    def test_a_stack_the_compiler_did_not_reserve_leaves_no_binary(self):
        from src.lfplo import Refused

        with tempfile.TemporaryDirectory() as workdir:
            exe, table = os.path.join(workdir, "forward"), os.path.join(workdir, "routes.bin")
            with open(table, "wb") as handle:
                handle.write(bytes(1024))
            with self.assertRaisesRegex(Refused, "stack"):
                self.build_with_damage(
                    lambda text: text.replace("F_forward equ 0", "F_forward equ 800"), exe, table)
            self.assertFalse(os.path.exists(exe))

    def test_the_symbolic_check_sees_every_harmful_change_to_the_handler(self):
        """Damage each instruction every way that still parses. Whatever survives must be on
        this list, each entry with the reason it does not change what the handler means."""
        from src import lfplo
        from src.rtlower import rt_lines
        from src.runtime import Options
        from src.symcheck import check_translation

        harmless = {
            ("mov byte [rsi], 0", "word"),      # also zeroes offset 1, which Drop leaves as padding
            ("movzx ebx, byte [rsi]", "word"),  # the extra byte never reaches the low 8 bits kept
            ("movzx ebx, bl", "delete"),        # only bl is stored, so the upper bits never matter
            ("mov byte [rsi+2], bl", "word"),   # offset 3 of Fwd is padding too
        }
        with open(os.path.join(ROOT, SOURCE), "rb") as handle:
            source = handle.read()
        with tempfile.TemporaryDirectory() as workdir:
            asm_path, table = os.path.join(workdir, "p.asm"), os.path.join(workdir, "r.bin")
            with open(table, "wb") as handle:
                handle.write(bytes(1024))
            with open(os.path.join(ROOT, SOURCE), "rb") as handle:
                lfplo._generate(rt_lines(handle), asm_path, Options(1, 2, 3), {"routes": table})
            with open(asm_path) as handle:
                asm = handle.read()
        self.assertEqual(check_translation(source, asm), [])
        lines = asm.split("\n")
        start = lines.index("fn_forward_body:")
        end = next(i for i, line in enumerate(lines) if line.startswith("F_forward equ"))
        caught = 0
        for number in range(start + 1, end):
            line = lines[number]
            if not line.startswith("    "):
                continue
            for name, new in _damaged(line):
                text = "\n".join(lines[:number] + ([] if new is None else [new]) + lines[number + 1:])
                if check_translation(source, text):
                    caught += 1
                else:
                    self.assertIn((line.strip(), name), harmless, f"{line.strip()} -> {new}")
        self.assertGreater(caught, 50)


def _damaged(line: str):
    """Every small way to get one instruction wrong."""
    yield "delete", None
    for found in re.finditer(r"(?<![\w])(\d+)(?![\w])", line):
        value = int(found.group(1))
        for other in {value + 1, value - 1 if value else value + 2}:
            yield f"{value}", line[:found.start()] + str(other) + line[found.end():]
    registers = {"rdi": "rsi", "rsi": "rdx", "rdx": "rcx", "rcx": "rax", "rax": "rdi", "rbx": "rbp",
                 "rbp": "rbx", "bpl": "bl", "bl": "bpl", "ebx": "ebp", "ebp": "ebx"}
    for found in re.finditer(r"\b(" + "|".join(registers) + r")\b", line):
        yield found.group(1), line[:found.start()] + registers[found.group(1)] + line[found.end():]
    for old, new in (("ja ", "jb "), ("jbe ", "ja "), ("sub ", "add "), ("byte", "word")):
        if old in line and "rsp" not in line:
            yield new.strip(), line.replace(old, new, 1)


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
