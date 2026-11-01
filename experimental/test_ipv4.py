"""The IPv4 forwarder, in LFPL-O and in C, against each other and against the model of the workload.

The tests that need root put a virtual Ethernet pair in a network namespace, load the XDP program on one
end, run a forwarder on it, and speak raw frames on the other end. With no root they skip, and what runs
is what a build can show: the refusals of binding, and the shape of the binary.

Nothing here measures speed. A virtual pair says what a frame turns into, and that the rings conserve
frames, but how fast it goes says nothing about a card. experimental/speed.py is for that, and it refuses to
give a rate on a virtual pair.
"""
import dataclasses
import glob
import json
import os
import re
import resource
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import uuid

from src.certify import certify
from src.rtcheck import Checker
from src.rtlower import rt_lines
from src.runtime import BindingError
from experimental import ipv4_model as model
from experimental import runtime_xdp, speed, xdp_load

import io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "experimental", "ipv4.lfpl")
BASELINE = os.path.join(ROOT, "experimental", "ipv4_forward.c")
CONSTRUCTORS = ("Malformed", "Expired", "Send")
LFPLO_COUNTERS = runtime_xdp.counter_names(CONSTRUCTORS)
C_COUNTERS = ("frames", "forwarded", "malformed", "expired", "full", "completed", "kernel_dropped", "zerocopy",
              "send_errors")
WORK = 85  # W(5), which is what the report gives: 17 steps a cell, and an item is always 5 cells
DEAREST = 81  # the most steps any frame in the corpus takes
FORWARDER_END, TESTER_END = "vfwd", "vpeer"
CTL_PORT = 47100
WAIT = 5.0
AS_ROOT = os.geteuid() == 0

needs_root = unittest.skipUnless(AS_ROOT, "needs root, for a network namespace and an XDP program")


def run(*command: str) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def build_lfplo(workdir: str, name: str, pin: str, *options: str, interface: str = FORWARDER_END,
                ctl_port: int = CTL_PORT, source: str = SOURCE) -> tuple:
    exe = os.path.join(workdir, name)
    asm = exe + ".asm"
    built = subprocess.run([sys.executable, "-m", "experimental.build", source, "-o", exe, "--asm", asm,
                            "--xdp", interface, "--map", os.path.join(pin, xdp_load.MAP_FILE),
                            "--ctl-port", str(ctl_port), *options],
                           cwd=ROOT, capture_output=True, text=True)
    if built.returncode:
        raise AssertionError(f"{source} did not build:\n{built.stderr}")
    return exe, asm, built.stdout


def build_baseline(workdir: str) -> str:
    exe = os.path.join(workdir, "ipv4_forward")
    # -Werror, so that what the baseline says it does is not undercut by what the compiler told us.
    run("cc", "-O2", "-Wall", "-Wextra", "-Werror", "-o", exe, BASELINE)
    return exe


class VirtualPair:
    """A namespace holding one virtual Ethernet pair, with nothing else talking on it."""

    def __init__(self) -> None:
        self.name = f"lfpl{uuid.uuid4().hex[:8]}"
        self.path = f"/run/netns/{self.name}"
        self.pin = f"/sys/fs/bpf/{self.name}"
        run("ip", "netns", "add", self.name)
        with self.inside():
            # before any link is up, or IPv6 starts its own conversation on it
            run("sysctl", "-q", "-w", "net.ipv6.conf.all.disable_ipv6=1", "net.ipv6.conf.default.disable_ipv6=1")
            run("ip", "link", "set", "lo", "up")
            run("ip", "link", "add", FORWARDER_END, "type", "veth", "peer", "name", TESTER_END)
            run("ip", "link", "set", FORWARDER_END, "up")
            run("ip", "link", "set", TESTER_END, "up")

    def inside(self):
        return xdp_load.in_namespace(self.path)

    def close(self) -> None:
        # Whatever a failed test left running in here would keep the namespace alive after it was deleted.
        left = subprocess.run(["ip", "netns", "pids", self.name], capture_output=True, text=True).stdout.split()
        for pid in left:
            os.kill(int(pid), signal.SIGKILL)
        subprocess.run(["ip", "netns", "del", self.name], capture_output=True)
        shutil.rmtree(self.pin, ignore_errors=True)


class Running:
    """A forwarder on the pair, with the XDP program loaded for it, and the far end's socket."""

    def __init__(self, pair: VirtualPair, command: list, names: tuple, *, ctl_port: int = CTL_PORT):
        self.pair = pair
        self.names = names
        self.ctl_address = ("127.0.0.1", ctl_port)
        with pair.inside():
            xdp_load.attach(FORWARDER_END, pair.pin, "drv", 1)
            self.process = subprocess.Popen(command, stderr=subprocess.PIPE)
            self.tester = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
            self.tester.bind((TESTER_END, 0))
            self.tester.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 24)
            self.control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.control.settimeout(0.05)
        try:
            self.discard_stray()
            self.wait_until_serving()
        except BaseException:  # a forwarder left running would keep the namespace alive
            self.stop()
            raise

    def discard_stray(self) -> None:
        self.tester.setblocking(False)
        try:
            while True:
                self.tester.recv(65536)
        except BlockingIOError:
            pass

    def ask(self, command: bytes = b"S") -> dict:
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            self.control.sendto(command, self.ctl_address)
            try:
                answer = self.control.recv(4096)
            except OSError:
                if self.process.poll() is not None:
                    raise AssertionError(f"the forwarder exited with {self.process.returncode}: "
                                         f"{self.process.stderr.read().decode()}")
                continue
            if len(answer) != 8 * len(self.names):
                raise AssertionError(f"the control port gave {len(answer)} bytes")
            return dict(zip(self.names, struct.unpack(f"<{len(self.names)}Q", answer)))
        raise AssertionError("the control port did not answer")

    def wait_until_serving(self) -> None:
        """A frame sent now reaches the socket, or the kernel's own stack if the socket is not yet in
        the map. So send frames until one is counted: the stack ignores what is not addressed to it."""
        self.ask()
        before = self.ask()["frames"]
        deadline = time.monotonic() + WAIT
        while self.ask()["frames"] == before and time.monotonic() < deadline:
            self.tester.send(b"\xff" * 14)
            time.sleep(0.005)
        if self.ask()["frames"] == before:
            raise AssertionError("no frame reached the forwarder")
        settled = self.ask()["frames"]
        while True:
            time.sleep(0.02)
            now = self.ask()["frames"]
            if now == settled:
                break
            settled = now
        self.discard_stray()

    def settle(self, frames: int) -> dict:
        """The counters once `frames` frames have been taken from rx, and everything queued sent."""
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            now = self.ask()
            if now["frames"] >= frames and now["completed"] == now["forwarded"]:
                return now
            time.sleep(0.001)
        raise AssertionError(f"waited for {frames} frames, and the counters say {self.ask()}")

    def receive(self, timeout: float = WAIT) -> bytes:
        self.tester.settimeout(timeout)
        while True:
            data, address = self.tester.recvfrom(65536)
            if address[2] != socket.PACKET_OUTGOING:
                return data

    def nothing_more_arrives(self, wait: float = 0.05) -> bool:
        try:
            self.receive(wait)
        except TimeoutError:
            return True
        return False

    def stop(self) -> int:
        try:
            self.control.sendto(b"Q", self.ctl_address)
            status = self.process.wait(timeout=WAIT)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            status = -1
        with self.pair.inside():
            xdp_load.detach(FORWARDER_END, self.pair.pin, "drv")
        self.tester.close()
        self.control.close()
        self.process.stderr.close()
        return status


def verdict_of_lfplo(before: dict, after: dict) -> str:
    moved = [name for name in ("out_Send", "out_Expired", "out_Malformed", "malformed", "short", "full")
             if after[name] != before[name]]
    verdicts = {"out_Send": model.FORWARD, "out_Expired": model.EXPIRED, "out_Malformed": model.MALFORMED,
                "malformed": model.MALFORMED}
    return verdicts.get(moved[0], f"counters moved: {moved}") if len(moved) == 1 else f"counters moved: {moved}"


def verdict_of_c(before: dict, after: dict) -> str:
    moved = [name for name in ("forwarded", "expired", "malformed", "full") if after[name] != before[name]]
    verdicts = {"forwarded": model.FORWARD, "expired": model.EXPIRED, "malformed": model.MALFORMED}
    return verdicts.get(moved[0], f"counters moved: {moved}") if len(moved) == 1 else f"counters moved: {moved}"


def exchange(running: Running, frame: bytes, judge) -> tuple:
    """Send one frame: what became of it, as the forwarder's counters say, and what came out."""
    before = running.ask()
    running.tester.send(frame)
    after = running.settle(before["frames"] + 1)
    verdict = judge(before, after)
    return verdict, running.receive() if verdict == model.FORWARD else None


@needs_root
class TheTwoAgree(unittest.TestCase):
    """Every frame in the corpus through each forwarder, one at a time. Each gives a verdict by its
    counters, and a frame on the far end of the pair."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.work, True)
        cls.pair = VirtualPair()
        cls.addClassCleanup(cls.pair.close)
        cls.cases = model.corpus()
        lfplo, _asm, _report = build_lfplo(cls.work, "ipv4", cls.pair.pin)
        baseline = build_baseline(cls.work)
        cls.results = {}
        cls.counters = {}
        cls.final = {}
        cls.leftover = {}
        cls.status = {}
        for name, command, names, judge in (
                ("lfplo", [lfplo], LFPLO_COUNTERS, verdict_of_lfplo),
                ("c", [baseline, "--ifname", FORWARDER_END, "--map", os.path.join(cls.pair.pin, xdp_load.MAP_FILE),
                       "--ctl-port", str(CTL_PORT), "--match"], C_COUNTERS, verdict_of_c)):
            running = Running(cls.pair, command, names)
            try:
                start = running.ask()  # what finding out it was serving took, which can differ
                cls.results[name] = [exchange(running, frame, judge) for _name, _verdict, frame in cls.cases]
                cls.leftover[name] = not running.nothing_more_arrives()
                now = running.ask()
                cls.counters[name] = {counter: now[counter] - start[counter] for counter in now}
                cls.final[name] = now
            finally:
                cls.status[name] = running.stop()

    def against_the_model(self, forwarder: str) -> None:
        for (name, _verdict, frame), (verdict, sent) in zip(self.cases, self.results[forwarder]):
            with self.subTest(forwarder=forwarder, frame=name):
                want_verdict, want_sent = model.forward(frame, model.by_top_bits)
                self.assertEqual(verdict, want_verdict)
                self.assertEqual(sent, want_sent)

    def test_lfplo_against_the_model(self):
        self.against_the_model("lfplo")

    def test_c_against_the_model(self):
        self.against_the_model("c")

    def test_lfplo_against_c(self):
        for (name, _verdict, _frame), one, other in zip(self.cases, self.results["lfplo"], self.results["c"]):
            with self.subTest(frame=name):
                self.assertEqual(one, other)

    def test_every_frame_sent_was_valid_after_forwarding(self):
        forwarded = 0
        for verdict, sent in self.results["lfplo"]:
            if verdict == model.FORWARD:
                forwarded += 1
                self.assertEqual(model.one_complement_sum(sent[model.ETHERNET_HEADER:model.HEADER_BYTES]), 0xFFFF)
        self.assertGreater(forwarded, 100)

    def test_nothing_came_out_that_was_not_asked_for(self):
        self.assertEqual(self.leftover, {"lfplo": False, "c": False})

    def test_the_counters_agree_with_each_other(self):
        lfplo, c = self.counters["lfplo"], self.counters["c"]
        self.assertEqual(lfplo["forwarded"], c["forwarded"])
        self.assertEqual(lfplo["out_Expired"], c["expired"])
        self.assertEqual(lfplo["malformed"] + lfplo["out_Malformed"], c["malformed"])

    def test_each_counter_has_the_frames_it_should(self):
        now = self.counters["lfplo"]
        short = sum(1 for _name, _verdict, frame in self.cases if len(frame) < model.HEADER_BYTES)
        self.assertEqual(now["frames"], len(self.cases))
        self.assertEqual(now["malformed"], short, "admission refuses what is too short to read")
        self.assertEqual(now["dropped"], now["out_Malformed"] + now["out_Expired"])
        self.assertEqual(now["budget_steps"], WORK * now["served"])

    def test_lfplos_counters_balance(self):
        now = self.counters["lfplo"]
        self.assertEqual(self.final["lfplo"]["pool"], self.final["lfplo"]["capacity"], "every cell came back")
        self.assertEqual(now["frames"], now["forwarded"] + now["returned"], "every frame is accounted for")
        self.assertEqual(now["completed"], now["forwarded"], "and was sent")
        self.assertEqual(now["served"], now["out_Send"] + now["out_Expired"] + now["out_Malformed"])
        self.assertEqual(now["frames"], now["served"] + now["malformed"] + now["short"] + now["full"])
        self.assertEqual(now["kernel_dropped"], 0)
        self.assertEqual(now["bad_port"], 0)
        self.assertEqual(now["send_errors"], 0)
        self.assertEqual(now["admitted_cells"], 5 * now["served"])

    def test_both_shut_down_clean(self):
        self.assertEqual(self.status, {"lfplo": 0, "c": 0})


@needs_root
class TheTable(unittest.TestCase):
    """The C baseline's DIR-24-8 table, on what a slow and obvious longest-prefix match says."""

    # Hops under 64, with every length the table treats differently: its two levels meet at /24.
    ROUTES = [((0, 0, 0, 0), 0, 1), ((10, 0, 0, 0), 8, 2), ((10, 1, 0, 0), 16, 3), ((10, 1, 2, 0), 24, 4),
              ((10, 1, 2, 128), 25, 5), ((10, 1, 2, 200), 32, 6), ((10, 1, 2, 192), 26, 7),
              ((10, 1, 3, 0), 25, 8), ((192, 168, 0, 0), 23, 9), ((192, 168, 1, 1), 32, 10),
              ((172, 16, 0, 0), 12, 11), ((172, 16, 255, 255), 32, 12), ((224, 0, 0, 0), 4, 13)]

    def run_table(self, routes: list, destinations: list) -> None:
        work = tempfile.mkdtemp()
        pair = VirtualPair()
        try:
            table = os.path.join(work, "routes")
            with open(table, "w") as handle:
                # out of order on purpose: the table has to sort out which is longer
                for address, length, hop in reversed(routes):
                    handle.write(f"{'.'.join(map(str, address))}/{length} {hop}\n")
            look_up = model.longest_prefix([(bytes(a), n, h) for a, n, h in routes])
            running = Running(pair, [build_baseline(work), "--ifname", FORWARDER_END, "--ctl-port", str(CTL_PORT),
                                     "--map", os.path.join(pair.pin, xdp_load.MAP_FILE), "--routes", table], C_COUNTERS)
            try:
                for destination in destinations:
                    frame = model.ipv4_frame(bytes(destination))
                    got = exchange(running, frame, verdict_of_c)
                    self.assertEqual(got, model.forward(frame, look_up), destination)
            finally:
                running.stop()
        finally:
            pair.close()
            shutil.rmtree(work, ignore_errors=True)

    def test_every_route_at_its_edges(self):
        destinations = []
        for address, length, _hop in self.ROUTES:
            value = int.from_bytes(bytes(address), "big")
            span = 1 << (32 - length)
            for point in (value, value + 1, value + span // 2, value + span - 1, value + span, value - 1):
                destinations.append(tuple((point & 0xFFFFFFFF).to_bytes(4, "big")))
        self.run_table(self.ROUTES, destinations)

    def test_where_there_is_no_route_it_is_refused(self):
        routes = self.ROUTES[1:]
        self.run_table(routes, [(9, 255, 255, 255), (11, 0, 0, 0), (10, 0, 0, 0), (0, 0, 0, 0), (255, 255, 255, 255)])


@needs_root
class TheRings(unittest.TestCase):
    """What a small image does when the rings are put under pressure."""

    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.pair = VirtualPair()
        self.addCleanup(shutil.rmtree, self.work, True)
        self.addCleanup(self.pair.close)

    def start(self, *options: str) -> Running:
        exe, _asm, _report = build_lfplo(self.work, "ipv4", self.pair.pin, *options)
        running = Running(self.pair, [exe], LFPLO_COUNTERS)
        self.addCleanup(running.stop)
        return running

    def test_more_frames_than_there_are_come_through(self):
        """Each path recycles its frame, or a frame lost on any of them would stop this."""
        running = self.start("--frames", "16", "--k-out", "8")
        start = running.ask()
        frames = [model.ipv4_frame(), model.ipv4_frame(ttl=1), model.ipv4_frame(checksum=0), b"\x00" * 20]
        wanted = 25
        for frame in frames * wanted:
            verdict, sent = model.forward(frame, model.by_top_bits)
            got = exchange(running, frame, verdict_of_lfplo)
            self.assertEqual(got, (verdict, sent))
        now = running.ask()
        self.assertEqual(now["frames"] - start["frames"], len(frames) * wanted)
        self.assertEqual(now["forwarded"] - start["forwarded"], wanted)
        self.assertEqual(now["kernel_dropped"], 0, "the kernel never ran out of frames")
        self.assertEqual(now["pool"], now["capacity"])

    def test_a_full_tx_ring_refuses_and_gives_the_frame_back(self):
        running = self.start("--frames", "64", "--k-out", "8")
        start = running.ask("P".encode())
        self.assertEqual(start["full"], 0)
        frames = [model.ipv4_frame(identification=number) for number in range(20)]
        for frame in frames:
            running.tester.send(frame)
        deadline = time.monotonic() + WAIT
        while running.ask()["frames"] < start["frames"] + 20 and time.monotonic() < deadline:
            time.sleep(0.001)
        now = running.ask()
        self.assertEqual(now["frames"] - start["frames"], 20)
        self.assertEqual(now["forwarded"] - start["forwarded"], 8, "the ring holds 8")
        self.assertEqual(now["full"], 12, "and the rest are refused")
        self.assertEqual(now["served"] - start["served"], 8, "a refusal does not run the handler")
        self.assertEqual(now["returned"] - start["returned"], 12)
        self.assertEqual(now["pool"], now["capacity"], "a refusal takes no cells")
        self.assertTrue(running.nothing_more_arrives(), "while paused, nothing goes out")
        running.ask(b"R")
        for frame in frames[:8]:
            self.assertEqual(running.receive(), model.forward(frame, model.by_top_bits)[1], "the first 8, in order")
        settled = running.settle(now["frames"])
        self.assertEqual(settled["completed"], settled["forwarded"])
        self.assertEqual(settled["frames"], settled["forwarded"] + settled["returned"])
        self.assertEqual(exchange(running, frames[0], verdict_of_lfplo), model.forward(frames[0], model.by_top_bits),
                         "and it serves again")

    def test_out_of_frames_the_kernel_drops_and_says_so(self):
        """Every frame on tx and none on fill. Refusal has moved to the ring: the kernel has nowhere to
        put a frame, and the counter it keeps is the only thing that can say so."""
        running = self.start("--frames", "16", "--k-out", "16")
        start = running.ask(b"P")
        # What the serving loop's start-up frames left: frames have to be queued until fill runs out.
        for number in range(40):
            running.tester.send(model.ipv4_frame(identification=number))
            time.sleep(0.001)
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline:
            now = running.ask()
            if now["frames"] - start["frames"] + now["kernel_dropped"] - start["kernel_dropped"] >= 40:
                break
            time.sleep(0.001)
        now = running.ask()
        self.assertEqual(now["frames"] - start["frames"] + now["kernel_dropped"] - start["kernel_dropped"], 40)
        self.assertEqual(now["forwarded"] - start["forwarded"], 16 - start["forwarded"] % 16, "all there are")
        self.assertGreater(now["kernel_dropped"], 0)
        running.ask(b"R")
        settled = running.settle(now["frames"])
        self.assertEqual(settled["completed"], settled["forwarded"])
        frame = model.ipv4_frame(identification=0x7777)
        running.discard_stray()
        self.assertEqual(exchange(running, frame, verdict_of_lfplo), model.forward(frame, model.by_top_bits),
                         "with its frames back, it serves again")

    def test_the_kernel_is_woken_between_batches_while_frames_keep_coming(self):
        """poll wakes the kernel too, but poll is only for when nothing has arrived. With frames waiting
        on rx there is no idle turn, so if the wake at the end of a batch is missing, tx is never sent
        and everything past its first 8 frames is refused. Stopped, the forwarder lets rx fill first,
        which makes that the case for the whole run without needing a fast sender."""
        running = self.start("--frames", "4096", "--k-out", "8")
        start = running.ask()
        running.process.send_signal(signal.SIGSTOP)
        burst = 640  # 10 batches
        try:
            for number in range(burst):
                running.tester.send(model.ipv4_frame(identification=number))
        finally:
            running.process.send_signal(signal.SIGCONT)
        deadline = time.monotonic() + WAIT
        while running.ask()["frames"] < start["frames"] + burst and time.monotonic() < deadline:
            time.sleep(0.001)
        now = running.settle(start["frames"] + burst)
        self.assertEqual(now["frames"] - start["frames"], burst)
        self.assertEqual(now["kernel_dropped"], 0)
        self.assertEqual(now["forwarded"] + now["full"], burst, "each was sent, or refused for want of room")
        self.assertGreaterEqual(now["forwarded"], 5 * 8, "the ring emptied again and again, with no idle turn")

    def dropped_by_the_port(self) -> int:
        with self.pair.inside():
            shown = subprocess.run(["ip", "-j", "-s", "link", "show", "dev", FORWARDER_END],
                                   capture_output=True, text=True, check=True).stdout
        return json.loads(shown)[0]["stats64"]["tx"]["dropped"]

    def test_a_frame_the_kernel_drops_for_a_link_that_is_down_is_counted(self):
        """Copy mode takes the frame off tx, finds no carrier, and says EBUSY. It gives the frame's address
        back as it would for one it sent, so the error is all the runtime can know. But poll sends from tx too,
        and when that drops one nobody is told, so send_errors is at least 1 and not exactly 3. The port's own
        count has all of them, and a card's zero-copy path has no such drop."""
        running = self.start("--frames", "64", "--k-out", "8")
        running.ask(b"P")
        for number in range(3):
            running.tester.send(model.ipv4_frame(identification=number))
        deadline = time.monotonic() + WAIT
        while running.ask()["forwarded"] < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
        dropped = self.dropped_by_the_port()
        with self.pair.inside():
            run("ip", "link", "set", TESTER_END, "down")
        running.ask(b"R")
        deadline = time.monotonic() + WAIT
        while running.ask()["completed"] < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
        now = running.ask()
        self.assertGreaterEqual(now["send_errors"], 1)
        self.assertEqual(now["completed"], 3, "and they all came back")
        self.assertEqual(self.dropped_by_the_port() - dropped, 3)
        with self.pair.inside():
            run("ip", "link", "set", TESTER_END, "up")
        running.tester.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)  # the far end's socket saw it go down
        running.discard_stray()
        frame = model.ipv4_frame(identification=0x7777)
        self.assertEqual(exchange(running, frame, verdict_of_lfplo), model.forward(frame, model.by_top_bits),
                         "with the link back, it serves again")
        self.assertEqual(running.ask()["frames"], running.ask()["forwarded"] + running.ask()["returned"])

    def test_quitting_sends_what_is_queued(self):
        running = self.start("--frames", "64", "--k-out", "8")
        running.ask(b"P")
        frames = [model.ipv4_frame(identification=number) for number in range(5)]
        for frame in frames:
            running.tester.send(frame)
        deadline = time.monotonic() + WAIT
        while running.ask()["forwarded"] < 5 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertTrue(running.nothing_more_arrives(), "still paused")
        running.control.sendto(b"Q", running.ctl_address)
        self.assertEqual(running.process.wait(timeout=WAIT), 0)
        for frame in frames:
            self.assertEqual(running.receive(), model.forward(frame, model.by_top_bits)[1])

    def test_a_flood_loses_nothing_it_took(self):
        running = self.start()
        frames = [model.ipv4_frame(bytes((number & 255, 1, 2, 3)), identification=number & 0xFFFF,
                                   ttl=(number % 5), payload=bytes(number % 300)) for number in range(5000)]
        start = running.ask()
        for frame in frames:
            try:
                running.tester.send(frame)
            except OSError:
                pass
        time.sleep(0.2)
        previous = None
        while True:
            now = running.ask()
            if now == previous:
                break
            previous = now
            time.sleep(0.05)
        self.assertGreater(now["frames"], start["frames"])
        self.assertEqual(now["frames"], now["forwarded"] + now["returned"])
        self.assertEqual(now["completed"], now["forwarded"])
        self.assertEqual(now["pool"], now["capacity"])
        self.assertEqual(now["frames"], now["served"] + now["malformed"] + now["short"] + now["full"])


@needs_root
class OtherHandlers(unittest.TestCase):
    """experimental/ipv4.lfpl has no state and no reserve, and its fields fit its cells exactly. What this
    runtime promises for other shapes has to be shown on other programs."""

    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.pair = VirtualPair()
        self.addCleanup(shutil.rmtree, self.work, True)
        self.addCleanup(self.pair.close)

    def start(self, program: str, constructors: tuple, *options: str) -> Running:
        source = os.path.join(self.work, "program.lfpl")
        with open(source, "w") as handle:
            handle.write(program)
        exe, _asm, _report = build_lfplo(self.work, "program", self.pair.pin, *options, source=source)
        running = Running(self.pair, [exe], runtime_xdp.counter_names(constructors))
        self.addCleanup(running.stop)
        return running

    def test_a_gap_in_the_cell_does_not_shift_the_frame(self):
        """a and b lie side by side in the frame, and 3 bytes apart in the cell, by natural alignment.
        Bytes between the item's fields and after them are not the program's to change."""
        running = self.start("""
data Frame;
record Tail { c: Int16, frame: Frame };
record Item { a: Int8, b: Int32, rest: Tail };
record Tx { c: Int16, frame: Frame };
record Out { port: Int8, a: Int8, b: Int32, rest: Tx };
handler h(i: Item, free: Free) : (Out, Free) =
  match i with Item(d1, a, b, tail) => match tail with Tail(d2, c, frame) =>
    (Out@d1(0, +(a, 1), +(b, 1), Tx@d2(+(c, 1), frame)), free) end end;
""", ("Out",))
        frame = bytes((0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70)) + b"after the fields"
        start = running.ask()
        running.tester.send(frame.ljust(40, b"."))
        running.settle(start["frames"] + 1)
        sent = running.receive()
        self.assertEqual(sent[0], 0x11, "a, one higher")
        self.assertEqual(int.from_bytes(sent[1:5], "little"), 0x50403020 + 1, "b, one higher")
        self.assertEqual(int.from_bytes(sent[5:7], "little"), 0x7060 + 1, "c, one higher")
        self.assertEqual(sent[7:], frame.ljust(40, b".")[7:])

    def test_a_frame_shorter_than_a_shorter_item_is_measured_against_that_item(self):
        running = self.start("""
data Frame;
record Tail { c: Int32, d: Int32, frame: Frame };
record Item { a: Int32, b: Int32, rest: Tail };
record Out { frame: Frame };
handler h(i: Item, free: Free) : (Out, Free) =
  match i with Item(d1, a, b, tail) => match tail with Tail(d2, c, d, frame) =>
    (Out@d1(frame), More@d2(free)) end end;
""", ("Out",))
        start = running.ask()
        running.tester.send(bytes(15))
        running.tester.send(bytes(16))
        now = running.settle(start["frames"] + 2)
        self.assertEqual(now["malformed"] - start["malformed"], 1, "16 bytes of headers: 15 is too short")
        self.assertEqual(now["served"] - start["served"], 1, "and 16 is enough")

    def test_a_state_array_comes_through_and_chooses_the_port(self):
        """The table the workload will have: a read-only array, read with get. That needs no change in
        the language, and this is the runtime carrying it, with the port it chooses checked."""
        table = os.path.join(self.work, "routes.bin")
        with open(table, "wb") as handle:
            handle.write(bytes(1 if address % 2 else 0 for address in range(1024)))
        running = self.start("""
data Frame;
record Item { ttl: Int8, dst: Int16, frame: Frame };
data Out = Late(frame: Frame) | Send(port: Int8, ttl: Int8, frame: Frame);
state routes : Array[1024] Int8;
handler h(routes: Array[1024] Int8, i: Item, free: Free) : (Array[1024] Int8, Out, Free) =
  match i with Item(d, ttl, dst, frame) =>
    if <=(ttl, 1) then (routes, Late@d(frame), free)
    else let (r2, port) = get(routes, idx(dst, 1024)) in (r2, Send@d(port, -(ttl, 1), frame), free)
  end;
""", ("Late", "Send"), "--state", f"routes={table}")
        start = running.ask()
        running.tester.send(bytes((9, 4, 0)) + bytes(20))
        now = running.settle(start["frames"] + 1)
        self.assertEqual(running.receive()[0], 8, "an even address: port 0, and the TTL goes down")
        self.assertEqual(now["forwarded"] - start["forwarded"], 1)
        running.tester.send(bytes((9, 5, 0)) + bytes(20))
        now = running.settle(start["frames"] + 2)
        self.assertEqual(now["bad_port"] - start["bad_port"], 1, "an odd address: port 1, which is not there")
        self.assertEqual(now["returned"] - start["returned"], 1)
        running.tester.send(bytes((1, 4, 0)) + bytes(20))
        now = running.settle(start["frames"] + 3)
        self.assertEqual(now["out_Late"] - start["out_Late"], 1)
        self.assertEqual(now["pool"], now["capacity"])

    def test_a_reserve_is_there_when_the_output_needs_it(self):
        """The item is one cell and the output two, so R(n) = 1, and that cell has to be there, every
        time, and come back."""
        running = self.start("""
data Frame;
record Item { a: Int16, frame: Frame };
record Tx { frame: Frame };
data Out = Gone(frame: Frame) | Send(port: Int8, a: Int16, rest: Tx);
handler h(i: Item, free: Free) : (Out, Free) =
  match i with Item(d, a, frame) =>
    match free with
    | End => (Gone@d(frame), End)
    | More(c, rest) => (Send@d(0, +(a, 1), Tx@c(frame)), rest)
    end
  end;
""", ("Gone", "Send"))
        self.assertEqual(running.ask()["capacity"], 2)
        for number in range(40):
            start = running.ask()
            running.tester.send(bytes((number, 0)) + bytes(30))
            now = running.settle(start["frames"] + 1)
            self.assertEqual(running.receive()[0], number + 1)
            self.assertEqual(now["pool"], 2, "both back")
            self.assertEqual(now["short"], 0)


@needs_root
class WhatItCosts(unittest.TestCase):
    """W(n) against steps that ran, and memory against what the image promised. No time is measured."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.work, True)
        cls.pair = VirtualPair()
        cls.addClassCleanup(cls.pair.close)
        cls.exe, cls.asm, cls.report = build_lfplo(cls.work, "ipv4_steps", cls.pair.pin, "--count-steps")

    def test_no_frame_takes_more_steps_than_the_bound_and_the_dearest_comes_close(self):
        running = Running(self.pair, [self.exe], LFPLO_COUNTERS)
        try:
            dearest = 0
            for _name, _verdict, frame in model.corpus():
                before = running.ask()
                running.tester.send(frame)
                after = running.settle(before["frames"] + 1)
                if after["served"] != before["served"]:
                    self.assertLessEqual(after["steps"] - before["steps"], after["budget_steps"] - before["budget_steps"])
                    dearest = max(dearest, after["steps"] - before["steps"])
                else:
                    self.assertEqual(after["steps"], before["steps"], "a refused frame runs nothing")
                running.discard_stray()
            self.assertLessEqual(dearest, WORK)
            self.assertEqual(dearest, DEAREST, "README.md quotes this, and it decides the bound's worth")
        finally:
            running.stop()

    def test_the_packet_memory_is_locked_touched_and_does_not_grow(self):
        running = Running(self.pair, [self.exe], LFPLO_COUNTERS)
        try:
            def status() -> dict:
                with open(f"/proc/{running.process.pid}/status") as handle:
                    return {line.split(":")[0]: line.split(":")[1].strip() for line in handle}
            before = status()
            for _ in range(2):
                for frame in (model.ipv4_frame(), model.ipv4_frame(ttl=1), b"\x00" * 14):
                    running.tester.send(frame)
                time.sleep(0.01)
            after = status()
            image = 4096 * 2048
            self.assertGreaterEqual(int(before["VmLck"].split()[0]) * 1024, image)
            self.assertGreaterEqual(int(before["VmRSS"].split()[0]) * 1024, image, "every page already touched")
            self.assertEqual(before["VmRSS"], after["VmRSS"])
            self.assertEqual(before["VmLck"], after["VmLck"])
        finally:
            running.stop()


class TheTwoAreOneShape(unittest.TestCase):
    """experimental/workload.md says the two forwarders are alike but for what they are, and a speed comparison
    means nothing if that has drifted."""

    @classmethod
    def setUpClass(cls):
        with open(BASELINE) as handle:
            cls.c = handle.read()

    def define(self, name: str) -> int:
        return int(re.search(rf"^#define {name} (\d+)u", self.c, re.M).group(1))

    def test_the_packet_memory_and_the_rings(self):
        defaults = runtime_xdp.XdpOptions("eth0", "/x", 9000)
        self.assertEqual(self.define("FRAME_SIZE"), runtime_xdp.CHUNK)
        self.assertEqual(self.define("FRAMES"), defaults.frames)
        self.assertIn("#define RING_SIZE FRAMES", self.c)
        self.assertEqual(self.define("TX_SIZE"), defaults.k_out)

    def test_the_batch_and_the_looks_at_the_control_port(self):
        self.assertEqual(self.define("BATCH"), runtime_xdp.BATCH)
        self.assertEqual(int(re.search(r"\+\+turn % (\d+)", self.c).group(1)), runtime_xdp.CONTROL_EVERY)

    def test_the_header_the_item_reads(self):
        with open(SOURCE, "rb") as handle:
            checker = Checker()
            for line in rt_lines(handle):
                checker.feed(line)
        checker.finish()
        fn = next(f for f in checker.functions.values() if f.handler)
        self.assertEqual(self.define("HEADER_BYTES"), runtime_xdp._item(checker, checker.norm(fn.params[-2])).header_bytes)
        self.assertEqual(self.define("HEADER_BYTES"), model.HEADER_BYTES)


class TheHarnessesArithmetic(unittest.TestCase):
    """experimental/speed.py's reasoning, which can be checked without a card."""

    def test_a_size_on_the_wire_is_four_bytes_more_than_pktgen_calls_it(self):
        self.assertEqual(speed.pktgen_size(64), 60)
        self.assertEqual(speed.pktgen_size(1500), 1496)

    def test_rates_are_per_second_of_the_window(self):
        rate = speed.rates(2.0, 2000, 1000, 500, 0.75)
        self.assertEqual((rate.offered, rate.forwarded, rate.back, rate.core_busy), (1000, 500, 250, 0.75))

    def test_the_median_round_is_a_round_that_happened(self):
        rounds = [("c", 64, speed.Rates(1000, forwarded, forwarded, 1.0), {"round": forwarded})
                  for forwarded in (900, 700, 800)]
        rounds.append(("lfplo", 64, speed.Rates(1000, 500, 500, 1.0), {"round": 1}))
        results = {row[0]: row for row in speed.median_rounds(rounds)}
        self.assertEqual(results["c"][2].forwarded, 800)
        self.assertEqual(results["c"][3], {"round": 800}, "its counters, and not another's")
        self.assertEqual(results["c"][4], 200 / 800, "how far apart they were")
        self.assertEqual(results["lfplo"][4], 0)

    def test_a_forwarder_that_keeps_up_was_not_the_limit(self):
        self.assertFalse(speed.Rates(1000, 1000, 1000, 0.5).saturated)
        self.assertFalse(speed.Rates(1000, 995, 995, 0.5).saturated, "what reading the counters skews")
        self.assertTrue(speed.Rates(1000, 900, 900, 1.0).saturated)

    def result(self, name: str, size: int, offered: float, forwarded: float) -> tuple:
        return (name, size, speed.Rates(offered, forwarded, forwarded, 1.0), {})

    def test_a_ratio_is_given_only_where_the_ceiling_was_reached(self):
        lines = speed.comparisons([self.result("c", 64, 1000, 800), self.result("lfplo", 64, 1000, 400),
                                   self.result("c", 1500, 100, 100), self.result("lfplo", 1500, 100, 100)])
        self.assertIn("LFPL-O forwards 50.0% of what the C baseline does", lines[0])
        self.assertIn("both keep up", lines[1])
        self.assertNotIn("%  of", lines[1])

    def test_when_only_lfplo_falls_behind_it_does_not_claim_a_ratio(self):
        lines = speed.comparisons([self.result("c", 64, 1000, 1000), self.result("lfplo", 64, 1000, 400),
                                   self.result("c", 1500, 100, 100), self.result("lfplo", 1500, 100, 90)])
        self.assertIn("the ceiling is above that", lines[0])
        self.assertNotIn("forwards", lines[0])

    def test_frames_refused_mean_the_run_was_not_the_workload(self):
        self.assertEqual(speed.check_the_counters("c", {"frames": 10000, "malformed": 3, "expired": 0}, 64), [])
        self.assertTrue(speed.check_the_counters("c", {"frames": 10000, "malformed": 5000, "expired": 0}, 64))
        self.assertTrue(speed.check_the_counters("lfplo", {"frames": 10000, "out_Expired": 5000}, 64))

    def fake_cpus(self, directory: str, cores: dict) -> str:
        for cpu, (package, core) in cores.items():
            topology = os.path.join(directory, f"cpu{cpu}", "topology")
            os.makedirs(topology)
            with open(os.path.join(topology, "physical_package_id"), "w") as handle:
                handle.write(f"{package}\n")
            with open(os.path.join(topology, "core_id"), "w") as handle:
                handle.write(f"{core}\n")
        return directory

    def test_two_threads_of_one_core_are_refused(self):
        with tempfile.TemporaryDirectory() as work:
            sysfs = self.fake_cpus(work, {0: (0, 0), 6: (0, 0), 1: (0, 1), 7: (1, 0)})
            with self.assertRaisesRegex(speed.Refused, "two threads of one physical core"):
                speed.check_cores(0, 6, sysfs)
            speed.check_cores(0, 1, sysfs)
            speed.check_cores(0, 7, sysfs)  # the same core number, in another package
            with self.assertRaisesRegex(speed.Refused, "not there"):
                speed.check_cores(0, 99, sysfs)

    def test_cores_are_chosen_apart_and_the_forwarder_keeps_its_own(self):
        with tempfile.TemporaryDirectory() as work:
            sysfs = self.fake_cpus(work, {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 0), 4: (0, 1), 5: (0, 2)})
            self.assertEqual(speed.assign_cores(None, None, None, [0, 1, 2, 3, 4, 5], sysfs), (5, 4, 3),
                             "three threads of three cores, from the far end")
            self.assertEqual(speed.assign_cores(0, None, None, [0, 1, 2, 3, 4, 5], sysfs), (0, 5, 4))
            self.assertEqual(speed.assign_cores(None, None, None, [0, 1, 3, 4], sysfs), (4, 3, 3),
                             "with two cores, the interrupts share with the generator")
            with self.assertRaisesRegex(speed.Refused, "shares a core with the forwarder"):
                speed.assign_cores(0, 1, 3, [0, 1, 2, 3, 4, 5], sysfs)
            with self.assertRaisesRegex(speed.Refused, "two physical cores"):
                speed.assign_cores(None, None, None, [0, 3], sysfs)

    def test_only_a_port_with_a_device_can_give_a_rate(self):
        with tempfile.TemporaryDirectory() as work:
            os.makedirs(os.path.join(work, "enp1s0", "device"))
            os.makedirs(os.path.join(work, "veth0"))
            self.assertTrue(speed.has_a_device("enp1s0", work))
            self.assertFalse(speed.has_a_device("veth0", work))


@needs_root
class TheHarness(unittest.TestCase):
    """experimental/speed.py on a virtual pair, where the point is that it gives no rate, and that a run that
    went wrong is not reported as one."""

    # Anything a rate could be written as. Its plumbing mode prints counts, and no count per anything.
    RATE = re.compile(r"/s\b|pps|per second|packets per|[0-9] ?[kKM]pps", re.I)

    def setUp(self):
        self.pair = VirtualPair()
        self.addCleanup(self.pair.close)
        self.addCleanup(self.sweep)
        self.had_pktgen = os.path.exists("/proc/net/pktgen")

    def sweep(self) -> None:
        """If the harness did not put things back, the test has failed, and this is not to be left on the machine."""
        for path in glob.glob("/run/netns/lfplspeed*"):
            subprocess.run(["ip", "netns", "del", os.path.basename(path)], capture_output=True)
        for path in glob.glob("/sys/fs/bpf/lfplspeed*"):
            shutil.rmtree(path, ignore_errors=True)
        # A deleted namespace goes away a little later, and pktgen is in use until it has.
        deadline = time.monotonic() + WAIT
        while not self.had_pktgen and os.path.exists("/proc/net/pktgen") and time.monotonic() < deadline:
            if subprocess.run(["rmmod", "pktgen"], capture_output=True).returncode:
                time.sleep(0.1)

    def speed(self, *arguments: str) -> subprocess.CompletedProcess:
        with self.pair.inside():
            return subprocess.run([sys.executable, os.path.join(ROOT, "experimental", "speed.py"), *arguments],
                                  cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_on_a_virtual_pair_it_refuses_and_touches_nothing(self):
        done = self.speed("--generator", TESTER_END, "--forwarder", FORWARDER_END)
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn("has no device behind it (veth)", done.stderr)
        self.assertEqual(done.stdout, "")
        self.assertNotRegex(done.stderr, r"\d+ ?(pps|/s)")
        with self.pair.inside():
            self.assertGreater(socket.if_nametoindex(TESTER_END), 0, "the port is still where it was")

    def test_plumbing_mode_runs_every_step_and_gives_no_rate(self):
        before = set(os.listdir("/run/netns"))
        done = self.speed("--generator", TESTER_END, "--forwarder", FORWARDER_END, "--plumbing")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertNotRegex(done.stdout, self.RATE)
        self.assertIn("they are not a measure of anything", done.stdout)
        found = re.findall(r"(\S+) at (\d+) bytes: .* the forwarder took (\d+) and forwarded (\d+)", done.stdout)
        self.assertEqual({(name, int(size)) for name, size, _took, _sent in found},
                         {(name, size) for name in ("c", "lfplo", "c-table") for size in (64, 1500)})
        for name, size, took, forwarded in found:
            with self.subTest(forwarder=name, size=size):
                # Some are refused as Full, because the kernel's copy is slower than the forwarder. If they
                # had been refused as anything else, this run would have failed: see check_the_counters.
                self.assertGreater(int(forwarded), 0, "the generator's frames got through")
                self.assertLessEqual(int(forwarded), int(took))
        for name, size, back in re.findall(r"(\S+) at (\d+) bytes: .* its port had (\d+) back", done.stdout):
            with self.subTest(forwarder=name, size=size):
                self.assertGreater(int(back), 0, "and came back to the generator's port")
        # and it left things as it found them
        with self.pair.inside():
            self.assertGreater(socket.if_nametoindex(TESTER_END), 0)
            links = subprocess.run(["ip", "-d", "link", "show", FORWARDER_END], capture_output=True, text=True).stdout
        self.assertNotIn("xdp", links.split("\n")[0], "the program is detached")
        self.assertEqual(set(os.listdir("/run/netns")), before)
        self.assertEqual(glob.glob("/sys/fs/bpf/lfplspeed*"), [])
        self.assertEqual(os.path.exists("/proc/net/pktgen"), self.had_pktgen, "and the module as it was")

    def assert_left_as_found(self) -> None:
        with self.pair.inside():
            self.assertGreater(socket.if_nametoindex(TESTER_END), 0)
            links = subprocess.run(["ip", "-d", "link", "show"], capture_output=True, text=True).stdout
        self.assertNotIn(" xdp", links, "no program is left attached")
        self.assertNotRegex(links, "PROMISC", "and no port left promiscuous")
        self.assertEqual(glob.glob("/sys/fs/bpf/lfplspeed*"), [])
        self.assertEqual(glob.glob("/run/netns/lfplspeed*"), [])
        self.assertEqual(os.path.exists("/proc/net/pktgen"), self.had_pktgen)

    def test_ports_that_are_not_cabled_together_are_a_failure_and_not_a_rate(self):
        """The generator sends and the forwarder hears nothing. That must not come out as a rate of zero."""
        with self.pair.inside():
            run("ip", "link", "add", "vgen", "type", "veth", "peer", "name", "vgenpeer")
            run("ip", "link", "set", "vgen", "up")
            run("ip", "link", "set", "vgenpeer", "up")
        done = self.speed("--generator", "vgen", "--forwarder", FORWARDER_END, "--plumbing")
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("the forwarder took none", done.stderr)
        self.assert_left_as_found()

    def test_a_port_with_no_carrier_is_a_failure_and_says_so(self):
        with self.pair.inside():
            run("ip", "link", "set", FORWARDER_END, "down")
        # Its peer is the generator here, and it has no carrier until the far end is up.
        done = self.speed("--generator", TESTER_END, "--forwarder", "vnone", "--plumbing")
        self.assertEqual(done.returncode, 2, "there is no such port")
        done = self.speed("--generator", TESTER_END, "--forwarder", FORWARDER_END, "--plumbing", "--carrier-wait", "0.5",
                          "--seconds", "0.1")
        self.assertEqual(done.returncode, 0, "the harness brings up its own forwarder port")
        with self.pair.inside():
            run("ip", "link", "add", "vlone", "type", "veth", "peer", "name", "vlonepeer")
            run("ip", "link", "set", "vlone", "up")
        done = self.speed("--generator", "vlone", "--forwarder", FORWARDER_END, "--plumbing", "--carrier-wait", "0.5")
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("has no carrier", done.stderr)
        self.assert_left_as_found()

    def test_told_to_stop_it_puts_things_back(self):
        """A dropped session sends SIGHUP, and that must not strand the port in a namespace."""
        with self.pair.inside():
            process = subprocess.Popen([sys.executable, os.path.join(ROOT, "experimental", "speed.py"), "--generator", TESTER_END,
                                        "--forwarder", FORWARDER_END, "--plumbing"], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        seen = ""
        for line in process.stdout:
            seen += line
            if "lfplo at 64" in line:  # well into the run, with pktgen going
                process.send_signal(signal.SIGHUP)
                break
        rest, _ = process.communicate(timeout=60)
        self.assertEqual(process.returncode, 128 + signal.SIGHUP, seen + rest)
        self.assert_left_as_found()

    def test_what_pktgen_sends_is_the_workload_at_the_size_it_says(self):
        loaded = not self.had_pktgen
        if loaded:
            run("modprobe", "pktgen")
            self.addCleanup(lambda: subprocess.run(["rmmod", "pktgen"], capture_output=True))
        with self.pair.inside():
            tap = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
            tap.bind((FORWARDER_END, 0))
            tap.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.addCleanup(tap.close)
        tap.settimeout(WAIT)
        for wire_bytes in (64, 1500):
            with self.subTest(wire_bytes=wire_bytes):
                generator = speed.Pktgen(self.pair, TESTER_END, sorted(os.sched_getaffinity(0))[-1], False)
                generator.configure(wire_bytes, "02:00:00:00:00:aa", False)
                generator.start()
                try:
                    while True:
                        frame = tap.recv(65536)
                        if len(frame) >= model.ETHERNET_HEADER and frame[:6].hex() == "0200000000aa":
                            break
                finally:
                    generator.stop()
                    generator.forget()
                self.assertEqual(len(frame), wire_bytes - 4)
                self.assertEqual(int.from_bytes(frame[model.ETHERNET_HEADER + 2:model.ETHERNET_HEADER + 4], "big"),
                                 wire_bytes - 4 - model.ETHERNET_HEADER)
                self.assertGreater(frame[model.TTL_AT], 1)
                verdict, sent = model.forward(frame, model.by_top_bits)
                self.assertEqual(verdict, model.FORWARD)
                self.assertEqual(sent[:6], model.macs_of(0)[0])
                self.drain(tap)

    @staticmethod
    def drain(tap: socket.socket) -> None:
        tap.setblocking(False)
        try:
            while True:
                tap.recv(65536)
        except BlockingIOError:
            pass
        tap.settimeout(WAIT)


@needs_root
class TheLoader(unittest.TestCase):
    def setUp(self):
        self.pair = VirtualPair()
        self.addCleanup(self.pair.close)

    def seen_by_the_stack(self, frame: bytes) -> bool:
        with self.pair.inside():
            tap = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
            sender = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3))
        with tap, sender:
            tap.bind((FORWARDER_END, 0))
            sender.bind((TESTER_END, 0))
            tap.settimeout(0.2)
            sender.send(frame)
            try:
                while True:
                    data, address = tap.recvfrom(65536)
                    if address[2] != socket.PACKET_OUTGOING and data == frame:
                        return True
            except TimeoutError:
                return False

    def test_with_no_socket_registered_frames_go_on_to_the_stack(self):
        frame = model.ipv4_frame(identification=0x4242)
        with self.pair.inside():
            xdp_load.attach(FORWARDER_END, self.pair.pin, "drv", 1)
        self.assertTrue(self.seen_by_the_stack(frame), "an attached program passes what no socket wants")
        self.assertTrue(os.path.exists(os.path.join(self.pair.pin, xdp_load.MAP_FILE)))
        with self.pair.inside():
            xdp_load.detach(FORWARDER_END, self.pair.pin, "drv")
        self.assertFalse(os.path.exists(self.pair.pin), "and detaching leaves nothing behind")
        self.assertTrue(self.seen_by_the_stack(frame))

    def test_a_program_already_there_is_not_replaced(self):
        self.addCleanup(shutil.rmtree, self.pair.pin + "-other", True)
        with self.pair.inside():
            xdp_load.attach(FORWARDER_END, self.pair.pin, "drv", 1)
            with self.assertRaisesRegex(xdp_load.LoadError, "attaching XDP"):
                xdp_load.attach(FORWARDER_END, self.pair.pin + "-other", "drv", 1)
            xdp_load.detach(FORWARDER_END, self.pair.pin, "drv")
        self.assertFalse(os.path.exists(os.path.join(self.pair.pin + "-other", xdp_load.MAP_FILE)),
                         "a failed attach leaves no map behind for a runtime to find")

    def test_the_program_for_returning_frames_drops_them(self):
        frame = model.ipv4_frame(identification=0x4343)
        with self.pair.inside():
            xdp_load.attach_drop(FORWARDER_END, "drv")
        self.assertFalse(self.seen_by_the_stack(frame), "dropped before the stack sees it")
        with self.pair.inside():
            xdp_load.detach_program(FORWARDER_END, "drv")
        self.assertTrue(self.seen_by_the_stack(frame))

    def test_an_interface_that_is_not_there_is_an_error_with_a_message(self):
        with self.pair.inside():
            done = subprocess.run([sys.executable, os.path.join(ROOT, "experimental", "xdp_load.py"), "attach", "nosuchif",
                                   "--pin", self.pair.pin], capture_output=True, text=True)
        self.assertEqual(done.returncode, 1)
        self.assertIn("xdp_load:", done.stderr)


@needs_root
class RefusalsAtStartup(unittest.TestCase):
    """Any failure before service is an exit with a number that says which step it was."""

    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.pair = VirtualPair()
        self.addCleanup(shutil.rmtree, self.work, True)
        self.addCleanup(self.pair.close)

    def status_of(self, exe: str) -> int:
        with self.pair.inside():
            return subprocess.run([exe], capture_output=True, timeout=WAIT).returncode

    def test_no_interface(self):
        exe, _asm, _report = build_lfplo(self.work, "x", self.pair.pin, "--frames", "16", interface="nosuchif")
        self.assertEqual(self.status_of(exe), 8)

    def test_no_pinned_map(self):
        exe, _asm, _report = build_lfplo(self.work, "x", self.pair.pin, "--frames", "16")
        self.assertEqual(self.status_of(exe), 9)

    def test_the_control_port_taken(self):
        exe, _asm, _report = build_lfplo(self.work, "x", self.pair.pin, "--frames", "16")
        with self.pair.inside():
            holder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(holder.close)
        holder.bind(("127.0.0.1", CTL_PORT))
        self.assertEqual(self.status_of(exe), 6)

    def test_a_queue_the_interface_does_not_have(self):
        exe, _asm, _report = build_lfplo(self.work, "x", self.pair.pin, "--frames", "16", "--queue", "4000")
        self.assertEqual(self.status_of(exe), 8)

    def test_an_item_that_comes_back_on_a_port_that_is_not_there(self):
        """Port 1, on a runtime with port 0. The frame comes back to fill and it is counted."""
        source = os.path.join(self.work, "wrong_port.lfpl")
        with open(SOURCE) as handle:
            text = handle.read()
        self.assertIn("(Send@d1(0,", text)
        with open(source, "w") as handle:
            handle.write(text.replace("(Send@d1(0,", "(Send@d1(1,"))
        exe, _asm, _report = build_lfplo(self.work, "wrong", self.pair.pin, source=source)
        running = Running(self.pair, [exe], LFPLO_COUNTERS)
        self.addCleanup(running.stop)
        start = running.ask()
        running.tester.send(model.ipv4_frame())
        now = running.settle(start["frames"] + 1)
        self.assertEqual(now["bad_port"] - start["bad_port"], 1)
        self.assertEqual(now["forwarded"], start["forwarded"])
        self.assertEqual(now["returned"] - start["returned"], 1)
        self.assertEqual(now["pool"], now["capacity"])
        self.assertTrue(running.nothing_more_arrives())


class TheCommandLine(unittest.TestCase):
    def refused(self, *arguments: str) -> str:
        done = subprocess.run([sys.executable, "-m", "experimental.build", SOURCE, "-o", os.path.join(tempfile.gettempdir(), "x"),
                               *arguments], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        return done.stderr

    XDP = ("--xdp", "eth0", "--map", "/sys/fs/bpf/x/xsks", "--ctl-port", "9000")

    def test_a_zero_is_not_taken_for_the_default(self):
        self.assertIn("K_out of 0", self.refused(*self.XDP, "--k-out", "0"))
        self.assertIn("F of 0", self.refused(*self.XDP, "--frames", "0"))

    def test_what_a_build_for_xdp_needs(self):
        self.assertIn("required: --map", self.refused("--xdp", "eth0", "--ctl-port", "9000"))


class TheBaselinesRoutes(unittest.TestCase):
    """A table missing routes would call good frames malformed, and look like a slower forwarder."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.work, True)
        cls.exe = build_baseline(cls.work)

    def refusal(self, routes: str) -> str:
        path = os.path.join(self.work, "routes")
        with open(path, "w") as handle:
            handle.write(routes)
        done = subprocess.run([self.exe, "--ifname", "nosuchif", "--map", "/nosuchmap", "--ctl-port", "9000",
                               "--routes", path], capture_output=True, text=True, timeout=WAIT)
        self.assertEqual(done.returncode, 1)
        return done.stderr

    def test_a_line_that_does_not_parse_is_refused_wherever_it_is(self):
        for routes in ("# a comment\n10.0.0.0/8 1\n", "10.0.0.0/8 1\nnonsense\n", "10.0.0.0/8\n", "10.0.0.0/8 1 2\n"):
            with self.subTest(routes=routes):
                self.assertIn("does not parse", self.refusal(routes))

    def test_what_is_out_of_range_is_refused(self):
        for routes in ("256.0.0.0/8 1\n", "10.0.0.0/33 1\n", "10.0.0.0/8 64\n", "10.0.0.-1/8 1\n"):
            with self.subTest(routes=routes):
                self.assertIn("out of range", self.refusal(routes))

    def test_a_good_file_gets_past_the_table(self):
        self.assertNotIn("routes", self.refusal("10.0.0.0/8 1\n0.0.0.0/0 2\n10.1.2.3/32 3\n"))


class WithoutRoot(unittest.TestCase):
    """What a binary shows without being allowed to serve."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, cls.work, True)
        cls.exe, cls.asm_path, cls.report = build_lfplo(cls.work, "ipv4", "/sys/fs/bpf/none", "--frames", "16")
        with open(cls.asm_path) as handle:
            cls.asm = handle.read()

    def test_the_report_gives_the_bounds(self):
        self.assertIn(f"handler ipv4: R(n) = 0 cells, W(n) = {WORK // 5} n steps", self.report)
        self.assertIn("5 cells", self.report)

    def test_the_binary_is_static_with_no_libc(self):
        with open(self.exe, "rb") as handle:
            data = handle.read()
        offset, = struct.unpack_from("<Q", data, 32)
        size, count = struct.unpack_from("<HH", data, 54)
        kinds = [struct.unpack_from("<I", data, offset + size * index)[0] for index in range(count)]
        self.assertNotIn(2, kinds, "PT_DYNAMIC")
        self.assertNotIn(3, kinds, "PT_INTERP")
        self.assertNotIn("extern", self.asm)

    def service(self) -> str:
        return self.asm[self.asm.index("\nserve:\n"):self.asm.index("section .data")]

    def test_serving_a_frame_makes_no_system_call(self):
        loop = self.service()
        path = loop[loop.index("\n.frame:\n"):loop.index("\n.next_frame:\n")]
        self.assertNotIn("syscall", path)
        for call in re.findall(r"call (\S+)", path):
            self.assertEqual(call, "handler_entry")

    def test_the_serving_loop_uses_only_the_system_calls_it_says(self):
        numbers = set(re.findall(r"mov eax, (\d+)\n(?:.*\n){0,7}?\s*syscall", self.service()))
        self.assertEqual({int(number) for number in numbers}, {7, 44, 45, 55, 231})

    def test_the_packet_memory_is_in_the_image(self):
        bss = self.asm[self.asm.index("section .bss"):]
        self.assertIn(f"umem: resb {16 * 2048}", bss)
        self.assertLess(bss.index("umem:"), bss.index("bss_end:"), "so startup locks and touches it")

    def test_where_frames_go_comes_only_from_the_program(self):
        """The handle comes back in the output, and nothing in between keeps a copy."""
        loop = self.service()
        egress = loop[loop.index("call handler_entry"):loop.index("\n.malformed:\n")]
        self.assertNotIn("[rx_", egress)

    @unittest.skipIf(AS_ROOT, "root can create the socket and lock anything")
    def test_without_the_capability_it_refuses_to_start(self):
        ran = subprocess.run([self.exe], capture_output=True, timeout=WAIT)
        self.assertIn(ran.returncode, (2, 7), "the lock, for a small limit, or the socket")

    @unittest.skipIf(AS_ROOT, "root can lock anything")
    def test_without_room_to_lock_the_image_it_refuses_to_start(self):
        def no_lock() -> None:
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (0, 0))
        ran = subprocess.run([self.exe], capture_output=True, timeout=WAIT, preexec_fn=no_lock)
        self.assertEqual(ran.returncode, 2)


ITEM = """
data Frame;
record Tail { b: Int16, frame: Frame };
record Item { a: Int32, rest: Tail };
"""
HANDLER = "handler h(i: Item, free: Free) : (%s, Free) ="


def bound_image(source: str):
    """A program through the checker and into a binding, or a BindingError. No assembler, no socket."""
    checker = Checker()
    for line in rt_lines(io.BytesIO(source.encode())):
        checker.feed(line)
    checker.finish()
    certify(checker, "h", runtime_xdp.item_cells(checker))
    options = runtime_xdp.XdpOptions("eth0", "/sys/fs/bpf/x/xsks", 9000)
    return runtime_xdp.bind(checker, "h", 8, options, {})


class TheProgramsBytes(unittest.TestCase):
    def test_the_program_is_what_the_kernel_reads_it_as(self):
        """Written out, against what `llvm-objdump` printed for the same program from C."""
        words = xdp_load.program_bytes(7)
        self.assertEqual(words.hex(" ", 8).split(), [
            "6112100000000000",  # r2 = *(u32 *)(r1 + 16)
            "1811000007000000",  # r1 = map fd 7, the first half of a two-slot load
            "0000000000000000",
            "b703000002000000",  # r3 = 2, which is XDP_PASS
            "8500000033000000",  # call 51, bpf_redirect_map
            "9500000000000000",  # exit
        ])


class Binding(unittest.TestCase):
    """What the runtime can carry, read off the handler's types."""

    GOOD_BODY = """ match i with Item(d1, a, tail) => match tail with Tail(d2, b, frame) =>
        (Out@d1(0, a, Tx@d2(b, frame)), free) end end;"""
    GOOD_OUT = """
record Tx { b: Int16, frame: Frame };
record Out { port: Int8, a: Int32, rest: Tx };
"""

    def refuse(self, pattern: str, source: str) -> None:
        with self.assertRaisesRegex(BindingError, pattern):
            bound_image(source)

    def test_a_chain_of_records_is_laid_out_in_wire_order(self):
        image = bound_image(ITEM + self.GOOD_OUT + HANDLER % "Out" + self.GOOD_BODY)
        self.assertEqual(image.item.cells, 2)
        self.assertEqual(image.item.header_bytes, 6)
        self.assertEqual([(s.name, s.cell, s.wire) for s in image.item.scalars], [("a", 0, 0), ("b", 1, 4)])
        self.assertEqual(image.item.frame[0], 1)
        self.assertEqual(image.item.length, None)
        out = image.out[0]
        self.assertEqual([(s.name, s.wire) for s in out.writes], [("a", 0), ("b", 4)])
        self.assertIsNotNone(out.port)

    def test_length_is_not_on_the_wire(self):
        source = """
data Frame;
record Tail { length: Int16, b: Int16, frame: Frame };
record Item { a: Int32, rest: Tail };
record Out { b: Int16, frame: Frame };
""" + HANDLER % "Out" + """ match i with Item(d1, a, tail) => match tail with Tail(d2, n, b, frame) =>
        (Out@d1(b, frame), More@d2(free)) end end;"""
        image = bound_image(source)
        self.assertEqual(image.item.header_bytes, 6)
        self.assertEqual(image.item.length, (1, 0))
        self.assertEqual([(s.name, s.wire) for s in image.item.scalars], [("a", 0), ("b", 4)])

    def test_length_must_be_int16(self):
        self.refuse("frame's length", """
data Frame;
record Item { length: Int32, frame: Frame };
record Out { frame: Frame };
""" + HANDLER % "Out" + """ match i with Item(d, n, frame) => (Out@d(frame), free) end;""")

    def test_an_item_must_be_a_record(self):
        self.refuse("receives an item as a record", """
data Frame;
data Item = A(frame: Frame) | B(frame: Frame);
record Out { frame: Frame };
""" + HANDLER % "Out" + """ match i with | A(d, frame) => (Out@d(frame), free)
                                          | B(d, frame) => (Out@d(frame), free) end;""")

    def test_an_item_holds_a_frame(self):
        self.refuse("holds no frame", """
record Item { a: Int32 };
record Out { a: Int32 };
""" + HANDLER % "Out" + """ match i with Item(d, a) => (Out@d(a), free) end;""")

    def test_every_constructor_of_an_output_holds_the_frame(self):
        # A frame the type gives no way to return. The null constructor could not be built here, but a
        # runtime reading types should not have to know that.
        self.refuse("holds no frame", """
data Frame;
record Item { frame: Frame };
data Out = None | Some(frame: Frame);
""" + HANDLER % "Out" + """ match i with Item(d, frame) => (Some@d(frame), free) end;""")

    def test_an_item_with_two_frames_is_refused(self):
        self.refuse("two frames", """
data Frame;
record Item { x: Frame, y: Frame };
record Out { a: Frame, b: Frame };
""" + HANDLER % "Out" + """ match i with Item(d, x, y) => (Out@d(x, y), free) end;""")

    def test_a_fin_cannot_come_from_the_wire(self):
        self.refuse("Fin", """
data Frame;
record Item { i: Fin 8, frame: Frame };
record Out { frame: Frame };
""" + HANDLER % "Out" + """ match i with Item(d, x, frame) => (Out@d(frame), free) end;""")

    def test_an_output_writes_only_what_the_item_has(self):
        self.refuse("no field of that name", ITEM + """
record Tx { b: Int16, frame: Frame };
record Out { port: Int8, other: Int32, rest: Tx };
""" + HANDLER % "Out" + self.GOOD_BODY)

    def test_an_output_writes_a_field_at_its_own_size(self):
        self.refuse("has 2 bytes, and the item's has 4", ITEM + """
record Tx { a: Int16, frame: Frame };
record Out { port: Int8, b: Int16, rest: Tx };
""" + HANDLER % "Out" + """ match i with Item(d1, a, tail) => match tail with Tail(d2, b, frame) =>
        (Out@d1(0, b, Tx@d2(b, frame)), free) end end;""")

    def test_an_output_writes_a_field_once(self):
        self.refuse("writes a twice", ITEM + """
record Tx { a: Int32, frame: Frame };
record Out { port: Int8, a: Int32, rest: Tx };
""" + HANDLER % "Out" + """ match i with Item(d1, a, tail) => match tail with Tail(d2, b, frame) =>
        (Out@d1(0, a, Tx@d2(a, frame)), free) end end;""")

    def test_port_is_int8_and_in_the_first_cell(self):
        self.refuse("port must be Int8", ITEM + """
record Tx { b: Int16, frame: Frame };
record Out { port: Int16, a: Int32, rest: Tx };
""" + HANDLER % "Out" + self.GOOD_BODY)
        self.refuse("port must be Int8", ITEM + """
record Tx { port: Int8, frame: Frame };
record Out { a: Int32, rest: Tx };
""" + HANDLER % "Out" + """ match i with Item(d1, a, tail) => match tail with Tail(d2, b, frame) =>
        (Out@d1(a, Tx@d2(0, frame)), free) end end;""")

    def test_a_sum_inside_a_chain_is_refused(self):
        self.refuse("carries scalars, one frame, and one record", ITEM + """
data Maybe = Nothing | Just(frame: Frame);
record Out { port: Int8, rest: Maybe };
""" + HANDLER % "Out" + """ match i with Item(d1, a, tail) => match tail with Tail(d2, b, frame) =>
        (Out@d1(0, Just@d2(frame)), free) end end;""")

    def test_a_chain_that_never_ends_is_refused(self):
        self.refuse("carries scalars, one frame, and one record", """
data Frame;
record Loop { a: Int8, next: Loop };
record Item { next: Loop, frame: Frame };
record Out { frame: Frame };
fn drain(l: Loop, acc: Free) : Free = match l with Loop(d, a, next) => drain(next, More@d(acc)) end;
""" + HANDLER % "Out" + """ match i with Item(d, next, frame) => (Out@d(frame), drain(next, free)) end;""")

    def test_an_image_past_what_the_code_can_reach_is_refused(self):
        image = bound_image(ITEM + self.GOOD_OUT + HANDLER % "Out" + self.GOOD_BODY)
        too_many = dataclasses.replace(image.options, frames=1 << 20)
        with self.assertRaisesRegex(BindingError, "needs an image of"):
            runtime_xdp.assembly(dataclasses.replace(image, options=too_many))

    def test_options_are_checked_at_the_boundary(self):
        for arguments, pattern in ((("eth0", "/x", 0), "not a port"), (("", "/x", 9000), "cannot name an interface"),
                                   (("a" * 16, "/x", 9000), "cannot name an interface"),
                                   (("eth0", "", 9000), "cannot name a pinned map"),
                                   (("eth0", "/x", 9000, 100), "power of two"),
                                   (("eth0", "/x", 9000, 64, 3), "power of two")):
            with self.assertRaisesRegex(BindingError, pattern):
                runtime_xdp.XdpOptions(*arguments)


if __name__ == "__main__":
    unittest.main()
