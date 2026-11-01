"""Measure the forwarders on a real card: experimental/workload.md, in packets per second.

    sudo python3 experimental/speed.py --generator PORT --forwarder PORT [--forwarder-core N] [--generator-core N]
                                [--generator-irq-core N] [--seconds S] [--rounds N]
                                [--random-destinations] [--skb]

The two ports are cabled to each other. The generator's port moves into a network namespace, so that
the kernel cannot see both ends and take a short cut, and pktgen sends from it. The forwarder is on the
other port, and sends each frame back out the port it came in on, so what it forwards arrives at the
generator's port and is counted there. Each forwarder is run at 64 and at 1500 bytes on the wire, which
to pktgen is 60 and 1496, because the card adds the 4 bytes of the frame check sequence.

    c         experimental/ipv4_forward.c, with --match, built with cc -O3 -march=native
    lfplo     experimental/ipv4.lfpl
    c-table   the baseline again with a DIR-24-8 table, to see what a real table costs

A rate is only worth having if it came from a card. A virtual pair's rate says how fast the kernel copies
memory, and that must not be mistaken for speed, so on a port with no device behind it this refuses
to run. `--plumbing` runs every step anyway, briefly, to show that the harness itself works, and then
says how many frames moved and gives no rate. Nothing else in this repository reports speed.

If offered and forwarded come out equal, the forwarder was not the limit, and the report says so: that
rate is the generator's or the link's, and it is no ceiling. The core's utilization is printed for that
reason, but dividing by it would be a guess, so this does not. A window in which nothing moved is a
failure of the run and not a rate of zero.

Frames come back to the generator's port as fast as they are forwarded. So that receiving them does not take
the core pktgen is sending from, an XDP program on that port drops them, and its interrupts go to a third
core. The card has counted them by then, so what is reported as back is still what crossed the cable, if
the driver counts as most do. Compare it with forwarded.

The rounds go through every forwarder and size in turn, so that anything drifting falls on all of them,
and the report gives the median round and how far apart the rounds were.

Run it with nothing else on those cores. It changes the ports' channels, flow control, state and interrupt
affinity, and it puts back each of them on the way out, even if it is told to stop.
"""
import argparse
import dataclasses
import glob
import json
import os
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experimental import runtime_xdp, xdp_load  # noqa: E402

FCS_BYTES = 4
WIRE_SIZES = (64, 1500)
PLUMBING_SECONDS = 1.0
SETTLE_SECONDS = 2.0
GENERATOR_BURST = 32
GENERATOR_CLONES = 1000
CTL_PORT = 47200
EXPECTED_DESTINATION = "10.0.0.1"
LFPLO_COUNTERS = runtime_xdp.counter_names(("Malformed", "Expired", "Send"))
C_COUNTERS = ("frames", "forwarded", "malformed", "expired", "full", "completed", "kernel_dropped", "zerocopy",
              "send_errors")
ROUTES = "0.0.0.0/2 0\n64.0.0.0/2 1\n128.0.0.0/2 2\n192.0.0.0/2 3\n10.0.0.0/8 0\n"
CPU_SYSFS = "/sys/devices/system/cpu"


class Refused(Exception):
    """This harness will not run like this. Says why, and nothing has been changed yet."""


class Failed(Exception):
    """A step failed while running. Whatever was changed is being put back."""


def pktgen_size(wire_bytes: int) -> int:
    """What pktgen calls a frame of this size on the wire. It does not count the frame check sequence."""
    return wire_bytes - FCS_BYTES


def run(*command: str, check: bool = True) -> subprocess.CompletedProcess:
    done = subprocess.run(command, capture_output=True, text=True)
    if check and done.returncode:
        raise Failed(f"{' '.join(command)}: {done.stderr.strip() or done.returncode}")
    return done


def best_effort(*command: str) -> None:
    """For putting things back: one failure should not stop the rest, and should not go unsaid."""
    done = run(*command, check=False)
    if done.returncode:
        print(f"  note: could not put this back: {' '.join(command)}: {done.stderr.strip()}", flush=True)


def has_a_device(port: str, sysfs: str = "/sys/class/net") -> bool:
    """A real card has a device behind it. veth, tun and dummy have none, and that is what is refused."""
    return os.path.exists(os.path.join(sysfs, port, "device"))


def driver_of(port: str) -> str:
    for line in run("ethtool", "-i", port).stdout.splitlines():
        if line.startswith("driver:"):
            return line.split(":", 1)[1].strip()
    raise Failed(f"ethtool does not say what drives {port}")


def physical_core(cpu: int, sysfs: str = CPU_SYSFS) -> tuple[int, int]:
    topology = os.path.join(sysfs, f"cpu{cpu}", "topology")
    try:
        with open(os.path.join(topology, "physical_package_id")) as package, \
                open(os.path.join(topology, "core_id")) as core:
            return int(package.read()), int(core.read())
    except OSError as failure:
        raise Refused(f"cpu {cpu} is not there, or is offline: {failure}") from failure


def check_cores(generator: int, forwarder: int, sysfs: str = CPU_SYSFS) -> None:
    if physical_core(generator, sysfs) == physical_core(forwarder, sysfs):
        raise Refused(f"cpus {generator} and {forwarder} are two threads of one physical core, so what "
                      f"each does would slow the other, and the rate would be neither's")


def assign_cores(forwarder: Optional[int], generator: Optional[int], irq: Optional[int], allowed: list,
                 sysfs: str = CPU_SYSFS) -> tuple[int, int, int]:
    """The cores asked for, and the rest from the far end, because cpu 0 takes most of what else runs. The
    interrupts of the generator's port can share with pktgen if there is no third core, and never with
    the forwarder."""
    used = {physical_core(cpu, sysfs) for cpu in (forwarder, generator, irq) if cpu is not None}

    def spare() -> Optional[int]:
        for cpu in sorted(allowed, reverse=True):
            if physical_core(cpu, sysfs) not in used:
                used.add(physical_core(cpu, sysfs))
                return cpu
        return None

    forwarder = spare() if forwarder is None else forwarder
    generator = spare() if generator is None else generator
    if forwarder is None or generator is None:
        raise Refused("this needs two physical cores, and only one is allowed here")
    check_cores(generator, forwarder, sysfs)
    if irq is None:
        irq = spare()
        irq = generator if irq is None else irq
    elif physical_core(irq, sysfs) == physical_core(forwarder, sysfs):
        raise Refused(f"cpu {irq} shares a core with the forwarder, and would give it the work of receiving")
    return forwarder, generator, irq


@dataclasses.dataclass(frozen=True)
class Rates:
    """What one window of a run showed, per second. forwarded comes from the forwarder's own counter,
    offered from the generator's port, and back from that port's receive counter."""
    offered: float
    forwarded: float
    back: float
    core_busy: float

    @property
    def saturated(self) -> bool:
        """Did the forwarder fall behind what it was offered? Reading the counters takes milliseconds
        at each end of the window, so 1% is slack in timing and not a shortfall."""
        return self.forwarded < self.offered * 0.99


def rates(seconds: float, offered: int, forwarded: int, back: int, core_busy: float) -> Rates:
    return Rates(offered / seconds, forwarded / seconds, back / seconds, core_busy)


def median_rounds(rounds: list) -> list:
    """From (name, size, rates, settled) for each round, one for each name and size: the round with the median
    rate, and how far apart the rounds were, as a fraction of it."""
    grouped: dict = {}
    for name, size, measured, settled in rounds:
        grouped.setdefault((name, size), []).append((measured, settled))
    results = []
    for (name, size), seen in grouped.items():
        forwarded = [measured.forwarded for measured, _settled in seen]
        middle = statistics.median_low(forwarded)
        measured, settled = next(pair for pair in seen if pair[0].forwarded == middle)
        results.append((name, size, measured, settled, (max(forwarded) - min(forwarded)) / middle))
    return results


class Namespace:
    def __init__(self) -> None:
        self.name = f"lfplspeed{uuid.uuid4().hex[:6]}"
        self.path = f"/run/netns/{self.name}"
        run("ip", "netns", "add", self.name)

    def inside(self):
        return xdp_load.in_namespace(self.path)

    def close(self) -> None:
        best_effort("ip", "netns", "del", self.name)


class Pktgen:
    """One pktgen thread, on one core, sending from one port, in that port's namespace."""

    def __init__(self, namespace, port: str, core: int, real_card: bool) -> None:
        self.namespace = namespace
        self.port = port
        self.core = core
        self.real_card = real_card
        self.starter: Optional[subprocess.Popen] = None

    def _write(self, path: str, text: str) -> None:
        """pktgen says what it made of a line in the device's own file, and a wrong line still 'succeeds'."""
        try:
            with self.namespace.inside(), open(path, "w") as handle:
                handle.write(text + "\n")
        except OSError as failure:
            raise Failed(f"pktgen would not take {text!r} in {path}: {failure}") from failure

    def _set(self, line: str) -> None:
        self._write(f"/proc/net/pktgen/{self.device}", line)
        with self.namespace.inside(), open(f"/proc/net/pktgen/{self.device}") as handle:
            result = handle.read().rsplit("Result:", 1)[-1].strip()
        if not result.startswith("OK"):
            raise Failed(f"pktgen did not take {line!r}: {result}")

    @property
    def device(self) -> str:
        return f"{self.port}@{self.core}"

    def configure(self, wire_bytes: int, destination_mac: str, random_destinations: bool) -> None:
        thread = f"/proc/net/pktgen/kpktgend_{self.core}"
        self._write(thread, "rem_device_all")
        self._write(thread, f"add_device {self.device}")
        self._set("count 0")
        self._set(f"pkt_size {pktgen_size(wire_bytes)}")
        self._set("delay 0")
        self._set("flag NO_TIMESTAMP")
        self._set(f"dst_mac {destination_mac}")
        if random_destinations:
            self._set("flag IPDST_RND")
            self._set("dst_min 1.0.0.0")
            self._set("dst_max 223.255.255.255")
        else:
            self._set(f"dst_min {EXPECTED_DESTINATION}")
            self._set(f"dst_max {EXPECTED_DESTINATION}")
        self._set("src_min 192.168.0.1")
        self._set("src_max 192.168.0.1")
        # Each of these sends one buffer many times, so a new destination for each frame can have neither,
        # and a virtual port cannot share a buffer between sends, so it refuses both.
        if self.real_card and not random_destinations:
            self._set(f"clone_skb {GENERATOR_CLONES}")
            self._set(f"burst {GENERATOR_BURST}")

    def start(self) -> None:
        # Writing `start` blocks for as long as it runs, so it gets a process of its own.
        self.starter = subprocess.Popen(["nsenter", f"--net={self.namespace.path}", "sh", "-c",
                                         "echo start > /proc/net/pktgen/pgctrl"],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def running(self) -> bool:
        """pktgen stops by itself if its port loses its carrier, and `start` then returns."""
        return self.starter is not None and self.starter.poll() is None

    def stop(self) -> None:
        if self.starter is None:
            return
        try:
            with self.namespace.inside(), open("/proc/net/pktgen/pgctrl", "w") as handle:
                handle.write("stop\n")
        except OSError:
            self.starter.kill()
        try:
            self.starter.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.starter.kill()
            self.starter.wait()
        self.starter = None

    def forget(self) -> None:
        try:
            self._write(f"/proc/net/pktgen/kpktgend_{self.core}", "rem_device_all")
        except Failed:
            pass


def link(port: str, namespace: Optional[Namespace] = None) -> dict:
    """What ip says of a port. sysfs shows the namespace it was mounted in and not the caller's, so
    anything that has to be right in a namespace comes from here."""
    where = ("-n", namespace.name) if namespace is not None else ()
    return json.loads(run("ip", *where, "-j", "-s", "link", "show", "dev", port).stdout)[0]


def packets(port: str, direction: str, namespace: Optional[Namespace] = None) -> int:
    return int(link(port, namespace)["stats64"][direction]["packets"])


def wait_for_carrier(port: str, namespace: Optional[Namespace], seconds: float) -> None:
    """Attaching XDP resets the port on many drivers. pktgen would stop at once if it started before the
    link came back, so it does not start until then."""
    deadline = time.monotonic() + seconds
    while "LOWER_UP" not in link(port, namespace)["flags"]:
        if time.monotonic() > deadline:
            raise Failed(f"{port} has no carrier after {seconds:g} s: are the ports cabled together?")
        time.sleep(0.1)


def core_ticks(cpu: int) -> tuple[int, int]:
    """(busy, total) in ticks. The driver's half of the work is softirq time, on the same core as the
    forwarder, and the process's own times would leave it out."""
    with open("/proc/stat") as handle:
        for line in handle:
            fields = line.split()
            if fields[0] == f"cpu{cpu}":
                numbers = [int(field) for field in fields[1:9]]  # guest time is already in user
                return sum(numbers) - numbers[3] - numbers[4], sum(numbers)
    raise Failed(f"/proc/stat has no cpu {cpu}")


class Forwarder:
    def __init__(self, name: str, command: list, counters: tuple, port: str, pin_dir: str, core: int, mode: str):
        self.name = name
        self.counters = counters
        self.port = port
        self.pin_dir = pin_dir
        self.mode = mode
        xdp_load.attach(port, pin_dir, mode, 1)
        try:
            self.process = subprocess.Popen(["taskset", "-c", str(core), *command], stderr=subprocess.PIPE)
        except OSError:
            xdp_load.detach(port, pin_dir, mode)
            raise
        self.control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.control.settimeout(0.2)
        try:
            self.ask()
        except Failed:
            self.stop()
            raise

    def ask(self) -> dict:
        for _attempt in range(50):
            self.control.sendto(b"S", ("127.0.0.1", CTL_PORT))
            try:
                answer = self.control.recv(4096)
            except OSError:
                if self.process.poll() is not None:
                    raise Failed(f"{self.name} exited with {self.process.returncode}: "
                                 f"{self.process.stderr.read().decode()}") from None
                continue
            return dict(zip(self.counters, struct.unpack(f"<{len(answer) // 8}Q", answer)))
        raise Failed(f"{self.name} did not answer on its control port")

    def stop(self) -> None:
        if self.process.poll() is None:
            self.control.sendto(b"Q", ("127.0.0.1", CTL_PORT))
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.control.close()
        self.process.stderr.close()
        xdp_load.detach(self.port, self.pin_dir, self.mode)


def irqs_of(port: str) -> list[str]:
    return sorted(os.path.basename(path) for path in glob.glob(f"/sys/class/net/{port}/device/msi_irqs/*"))


def pin_irqs(port: str, core: int, saved: list) -> None:
    for irq in irqs_of(port):
        path = f"/proc/irq/{irq}/smp_affinity_list"
        try:
            with open(path) as handle:
                saved.append((path, handle.read().strip()))
            with open(path, "w") as handle:
                handle.write(str(core))
        except OSError as failure:
            print(f"  note: irq {irq} of {port} stays where it is: {failure}", flush=True)


def one_queue(port: str, undo: list) -> None:
    """Frames on any other queue would miss the socket, which is bound to queue 0. A card has either combined
    channels or separate ones, and ethtool shows `n/a` for what it does not have."""
    channels = run("ethtool", "-l", port, check=False)
    if channels.returncode:
        print(f"  note: {port} does not report channels", flush=True)
        return
    current = channels.stdout[channels.stdout.find("Current"):]
    for kind, label in (("combined", "Combined:"), ("rx", "RX:"), ("tx", "TX:")):
        for line in current.splitlines():
            if line.startswith(label):
                value = line.split(":")[1].strip()
                if value.isdigit() and int(value) > 1:
                    run("ethtool", "-L", port, kind, "1")
                    undo.append(("ethtool", "-L", port, kind, value))


def no_pause(port: str, undo: list) -> None:
    """Pause frames would have a forwarder that has fallen behind hold back the generator, and then it would
    look as if it had kept up. So a card that will not turn them off has to be said out loud."""
    shown = run("ethtool", "-a", port, check=False)
    now = {}
    for line in shown.stdout.splitlines():
        key, _, value = line.partition(":")
        if key in ("RX", "TX"):
            now[key.lower()] = value.strip()
    if shown.returncode or len(now) != 2:
        print(f"  note: {port} does not report flow control", flush=True)
        return
    if run("ethtool", "-A", port, "rx", "off", "tx", "off", check=False).returncode:
        print(f"  note: {port} would not turn off flow control, so a rate on it may be held back", flush=True)
        return
    undo.append(("ethtool", "-A", port, "rx", now["rx"], "tx", now["tx"]))


class Run:
    """All the state a run changes, so that it can be put back whatever goes wrong."""

    def __init__(self, args: argparse.Namespace, real_card: bool) -> None:
        self.args = args
        self.real_card = real_card
        self.undo: list = []
        self.irq_undo: list = []
        self.namespace: Optional[Namespace] = None
        self.generator_moved = False
        self.generator_was_up = False
        self.drop_attached = False
        self.pin = f"/sys/fs/bpf/lfplspeed{uuid.uuid4().hex[:6]}"
        self.loaded_pktgen = False
        self.work = tempfile.mkdtemp()
        self.mode = "skb" if args.skb else "drv"

    def prepare(self) -> None:
        args = self.args
        if not os.path.exists("/proc/net/pktgen"):
            run("modprobe", "pktgen")
            self.loaded_pktgen = True
        self.namespace = Namespace()
        with self.namespace.inside():
            run("ip", "link", "set", "lo", "up")
            run("sysctl", "-q", "-w", "net.ipv6.conf.all.disable_ipv6=1", "net.ipv6.conf.default.disable_ipv6=1")
        for port in (args.forwarder, args.generator):
            one_queue(port, self.undo)
            no_pause(port, self.undo)
        pin_irqs(args.forwarder, args.forwarder_core, self.irq_undo)
        pin_irqs(args.generator, args.generator_irq_core, self.irq_undo)
        if "UP" not in link(args.forwarder)["flags"]:
            self.undo.append(("ip", "link", "set", args.forwarder, "down"))
        setting = f"net.ipv6.conf.{args.forwarder}.disable_ipv6"
        ipv6 = run("sysctl", "-n", setting, check=False).stdout.strip()
        if ipv6 == "0":
            self.undo.append(("sysctl", "-q", "-w", f"{setting}=0"))
            run("sysctl", "-q", "-w", f"{setting}=1")
        run("ip", "link", "set", args.forwarder, "up")
        self.generator_was_up = "UP" in link(args.generator)["flags"]
        run("ip", "link", "set", args.generator, "netns", self.namespace.name)
        self.generator_moved = True
        with self.namespace.inside():
            run("ip", "link", "set", args.generator, "up")
            # What comes back is addressed to the next hop, and has to be counted anyway.
            run("ip", "link", "set", args.generator, "promisc", "on")
            xdp_load.attach_drop(args.generator, self.mode)
        self.drop_attached = True

    def restore(self) -> None:
        args = self.args
        if self.generator_moved:
            if self.drop_attached:
                try:
                    with self.namespace.inside():
                        xdp_load.detach_program(args.generator, self.mode)
                except (xdp_load.LoadError, OSError) as failure:
                    print(f"  note: could not detach from {args.generator}: {failure}", flush=True)
            best_effort("ip", "-n", self.namespace.name, "link", "set", args.generator, "promisc", "off")
            best_effort("ip", "-n", self.namespace.name, "link", "set", args.generator, "netns", str(os.getpid()))
            if self.generator_was_up:
                best_effort("ip", "link", "set", args.generator, "up")
        if self.namespace is not None:
            self.namespace.close()
        for command in reversed(self.undo):
            best_effort(*command)
        for path, value in reversed(self.irq_undo):
            try:
                with open(path, "w") as handle:
                    handle.write(value)
            except OSError as failure:
                print(f"  note: could not put back {path}: {failure}", flush=True)
        if self.loaded_pktgen:
            best_effort("rmmod", "pktgen")
        shutil.rmtree(self.pin, ignore_errors=True)
        shutil.rmtree(self.work, ignore_errors=True)

    def build(self) -> dict:
        args = self.args
        map_path = os.path.join(self.pin, xdp_load.MAP_FILE)
        lfplo = os.path.join(self.work, "lfplo")
        report = subprocess.run([sys.executable, "-m", "experimental.build", "experimental/ipv4.lfpl", "-o", lfplo,
                                 "--xdp", args.forwarder, "--map", map_path, "--ctl-port", str(CTL_PORT)],
                                cwd=ROOT, capture_output=True, text=True)
        if report.returncode:
            raise Failed(f"experimental/ipv4.lfpl did not build: {report.stderr}")
        print(f"  {report.stdout.strip()}", flush=True)
        baseline = os.path.join(self.work, "ipv4_forward")
        run("cc", "-O3", "-march=native", "-Wall", "-Wextra", "-o", baseline,
            os.path.join(ROOT, "experimental", "ipv4_forward.c"))
        routes = os.path.join(self.work, "routes")
        with open(routes, "w") as handle:
            handle.write(ROUTES)
        common = [baseline, "--ifname", args.forwarder, "--map", map_path, "--ctl-port", str(CTL_PORT)]
        return {"c": (common + ["--match"], C_COUNTERS),
                "lfplo": ([lfplo], LFPLO_COUNTERS),
                "c-table": (common + ["--routes", routes], C_COUNTERS)}

    def measure(self, name: str, command: list, counters: tuple, wire_bytes: int) -> tuple:
        args = self.args
        mac = link(args.forwarder)["address"]
        generator = Pktgen(self.namespace, args.generator, args.generator_core, self.real_card)
        forwarder = Forwarder(name, command, counters, args.forwarder, self.pin, args.forwarder_core, self.mode)
        try:
            wait_for_carrier(args.forwarder, None, args.carrier_wait)
            wait_for_carrier(args.generator, self.namespace, args.carrier_wait)
            generator.configure(wire_bytes, mac, args.random_destinations)
            generator.start()
            try:
                time.sleep(args.warmup)
                sent_before = packets(args.generator, "tx", self.namespace)
                back_before = packets(args.generator, "rx", self.namespace)
                before = forwarder.ask()
                busy_before, total_before = core_ticks(args.forwarder_core)
                started = time.monotonic()
                time.sleep(args.seconds)
                after = forwarder.ask()
                seconds = time.monotonic() - started
                busy_after, total_after = core_ticks(args.forwarder_core)
                sent_after = packets(args.generator, "tx", self.namespace)
                back_after = packets(args.generator, "rx", self.namespace)
                if not generator.running():
                    raise Failed(f"pktgen stopped itself before the window closed, for {name} at {wire_bytes} "
                                 f"bytes, which is what it does if {args.generator} loses its carrier")
            finally:
                generator.stop()
            time.sleep(SETTLE_SECONDS if self.real_card else 0.2)
            settled = forwarder.ask()
        finally:
            forwarder.stop()
            generator.forget()
        where = f"for {name} at {wire_bytes} bytes"
        if sent_after == sent_before:
            raise Failed(f"{args.generator} sent nothing in the window {where}")
        if after["frames"] == before["frames"]:
            raise Failed(f"{args.generator} sent {sent_after - sent_before} frames in the window {where}, and the "
                         f"forwarder took none: are the ports cabled together, and is it all on queue 0?")
        if after["forwarded"] == before["forwarded"]:
            raise Failed(f"the forwarder took {after['frames'] - before['frames']} frames in the window {where}, "
                         f"and forwarded none of them")
        busy = (busy_after - busy_before) / max(total_after - total_before, 1)
        measured = rates(seconds, sent_after - sent_before, after["forwarded"] - before["forwarded"],
                         back_after - back_before, busy)
        return measured, settled, sent_after, back_after


def link_speed(port: str) -> str:
    try:
        with open(f"/sys/class/net/{port}/speed") as handle:
            return f"{int(handle.read())} Mb/s"
    except (OSError, ValueError):
        return "unknown speed"


def print_rates(results: list, driver: str, args: argparse.Namespace) -> None:
    print(f"\ndriver {driver}, {link_speed(args.forwarder)}, {'generic XDP, which copies' if args.skb else 'native XDP'}, "
          f"forwarder on cpu {args.forwarder_core}, generator on cpu {args.generator_core}, its interrupts on cpu "
          f"{args.generator_irq_core}, the median of {args.rounds} rounds of {args.seconds:g} s")
    print(f"{'forwarder':<10} {'frame':>6} {'offered/s':>14} {'forwarded/s':>14} {'back/s':>14} {'apart':>6} {'core':>6} "
          f"{'zero-copy':>10} {'full':>12} {'dropped':>12}  limit")
    for name, wire_bytes, measured, settled, spread in results:
        limit = "the forwarder" if measured.saturated else "not the forwarder: this is no ceiling"
        print(f"{name:<10} {wire_bytes:>6} {measured.offered:>14,.0f} {measured.forwarded:>14,.0f} "
              f"{measured.back:>14,.0f} {spread:>6.1%} {measured.core_busy:>6.0%} "
              f"{'yes' if settled['zerocopy'] else 'no':>10} "
              f"{settled['full']:>12,} {settled['kernel_dropped']:>12,}  {limit}")
    print("apart: how far the fastest and slowest rounds were, of the median. full: refused because the transmit ring "
          "had no room. dropped: the socket's own count of what it could not take. A card in zero-copy mode counts "
          "that in its own statistics instead: see ethtool -S.")
    print()
    for line in comparisons(results):
        print(line)


def comparisons(results: list) -> list:
    """LFPL-O against the ceiling, but only where the ceiling was reached. If the C baseline kept up
    with everything it was offered, its rate is the generator's, and a ratio to it compares nothing."""
    lines = []
    for wire_bytes in WIRE_SIZES:
        by_name = {row[0]: row[2] for row in results if row[1] == wire_bytes}
        c, lfplo = by_name["c"], by_name["lfplo"]
        if c.saturated:
            lines.append(f"at {wire_bytes} bytes, LFPL-O forwards {lfplo.forwarded / c.forwarded:.1%} of what the "
                         f"C baseline does")
        elif lfplo.saturated:
            lines.append(f"at {wire_bytes} bytes, LFPL-O falls behind at {lfplo.forwarded:,.0f}/s, and the C baseline "
                         f"keeps up with all {c.offered:,.0f}/s it is offered, so the ceiling is above that")
        else:
            lines.append(f"at {wire_bytes} bytes, both keep up with what they are offered, so this compares only "
                         f"how busy the core was: {c.core_busy:.0%} for C and {lfplo.core_busy:.0%} for LFPL-O")
    return lines


def check_the_counters(name: str, settled: dict, wire_bytes: int) -> list:
    """pktgen sends valid frames with a TTL to spare. If any is refused, the run was not the workload."""
    problems = []
    refused = sum(settled.get(counter, 0) for counter in ("malformed", "out_Malformed", "expired", "out_Expired"))
    # a stray frame from the port's own startup can be refused, so only a real number counts
    if refused > settled["frames"] // 1000:
        problems.append(f"{name} at {wire_bytes} bytes refused {refused} of {settled['frames']} frames")
    return problems


def parse(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generator", required=True, help="the port pktgen sends from")
    parser.add_argument("--forwarder", required=True, help="the port the forwarder serves")
    parser.add_argument("--generator-core", type=int)
    parser.add_argument("--forwarder-core", type=int, help="the forwarder, and its port's interrupts")
    parser.add_argument("--generator-irq-core", type=int, help="the interrupts of the generator's port")
    parser.add_argument("--seconds", type=float, default=10.0, help="the length of the window that is measured")
    parser.add_argument("--rounds", type=int, default=3, help="how many times to go through them all")
    parser.add_argument("--warmup", type=float, default=2.0, help="seconds of traffic before the window opens")
    parser.add_argument("--carrier-wait", type=float, default=30.0, help="seconds to wait for a port's link")
    parser.add_argument("--random-destinations", action="store_true",
                        help="a new destination for each frame, which also slows the generator, and shows "
                             "what the table costs. Without it, every lookup hits the same entry")
    parser.add_argument("--skb", action="store_true", help="generic XDP, for a card whose driver has no XDP")
    parser.add_argument("--plumbing", action="store_true",
                        help="run every step briefly, to test this harness, and report no rate")
    return parser.parse_args(argv)


def refusal(args: argparse.Namespace) -> Optional[str]:
    """Everything that can be said before anything is changed."""
    if os.geteuid() != 0:
        return "this needs root, for namespaces, pktgen and an XDP program"
    if args.generator == args.forwarder:
        return "one port cannot be cabled to itself"
    if args.rounds < 1 or args.seconds <= 0:
        return "a measurement needs at least one round, of some length"
    for port in (args.generator, args.forwarder):
        try:
            socket.if_nametoindex(port)
        except OSError:
            return f"there is no port {port} here"
    if not args.plumbing:
        for port in (args.generator, args.forwarder):
            if not has_a_device(port):
                return (f"{port} has no device behind it ({driver_of(port)}), so a rate from it would be the "
                        f"speed of the kernel's memcpy. Use two ports of a real card. --plumbing tests this "
                        f"harness on a virtual pair, and gives no rate")
    return None


def _leave(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)  # so that `finally` runs, and the ports are put back


def main(argv: list) -> int:
    args = parse(argv)
    try:
        reason = refusal(args)
        if reason is not None:
            raise Refused(reason)
        args.forwarder_core, args.generator_core, args.generator_irq_core = assign_cores(
            args.forwarder_core, args.generator_core, args.generator_irq_core, list(os.sched_getaffinity(0)))
    except Refused as failure:
        print(f"speed: refused: {failure}", file=sys.stderr)
        return 2
    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, _leave)
    real = has_a_device(args.forwarder) and has_a_device(args.generator)
    driver = driver_of(args.forwarder)
    if args.plumbing:
        args.seconds = PLUMBING_SECONDS
        args.warmup = min(args.warmup, 0.2)
        args.rounds = 1
    state = Run(args, real)
    rounds = []
    moved = []
    problems: list = []
    try:
        print(f"preparing {args.generator} and {args.forwarder}", flush=True)
        state.prepare()
        forwarders = state.build()
        for number in range(args.rounds):
            for name, (command, counters) in forwarders.items():
                for wire_bytes in WIRE_SIZES:
                    print(f"  round {number + 1}: {name} at {wire_bytes} bytes", flush=True)
                    measured, settled, sent, back = state.measure(name, command, counters, wire_bytes)
                    rounds.append((name, wire_bytes, measured, settled))
                    problems += check_the_counters(name, settled, wire_bytes)
                    moved.append((name, wire_bytes, sent, settled["frames"], settled["forwarded"], back))
    except (Failed, xdp_load.LoadError, OSError) as failure:
        print(f"speed: failed: {failure}", file=sys.stderr)
        return 1
    finally:
        state.restore()
    if args.plumbing:
        for name, wire_bytes, sent, frames, forwarded, back in moved:
            print(f"{name} at {wire_bytes} bytes: the generator's port had sent {sent} frames by then, the forwarder "
                  f"took {frames} and forwarded {forwarded}, and its port had {back} back")
        print("plumbing only: these counts say each step worked, and they are not a measure of anything")
    else:
        print_rates(median_rounds(rounds), driver, args)
    for problem in problems:
        print(f"speed: {problem}: this run was not the workload", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
