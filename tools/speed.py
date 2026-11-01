"""Measure the forwarders on a real card: docs/workload.md, in packets per second.

    sudo python3 tools/speed.py --generator PORT --forwarder PORT [--generator-core N] [--forwarder-core N]
                                [--seconds S] [--random-destinations] [--skb]

The two ports are cabled to each other. The generator's port moves into a network namespace, so that
the kernel cannot see both ends and take a short cut, and pktgen sends from it. The forwarder is on the
other port, and sends each frame back out the port it came in on, so what it forwards arrives at the
generator's port and is counted there. Each forwarder is run at 64 and at 1500 bytes on the wire, which
to pktgen is 60 and 1496, because the card adds the 4 bytes of the frame check sequence.

    c         baseline/ipv4_forward.c, with --match, built with cc -O3 -march=native
    lfplo     examples/ipv4.lfpl
    c-table   the baseline again with a DIR-24-8 table, to see what a real table costs

A rate is only worth having if it came from a card. A virtual pair's rate says how fast the kernel copies
memory, and that must not be mistaken for speed, so on a port with no device behind it this refuses
to run. `--plumbing` runs every step anyway, briefly, to show that the harness itself works, and then
says how many frames moved and gives no rate. Nothing else in this repository reports speed.

If offered and forwarded come out equal, the forwarder was not the limit, and the report says so: that
rate is the generator's or the link's, and it is no ceiling. The core's utilization is printed for that
reason, but dividing by it would be a guess, so this does not.

Run it with nothing else on the two cores. It changes the ports' channels, flow control and interrupt
affinity, and it restores what it can on the way out.
"""
import argparse
import dataclasses
import glob
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import runtime_xdp  # noqa: E402
from tools import xdp_load  # noqa: E402

FCS_BYTES = 4
WIRE_SIZES = (64, 1500)
PLUMBING_SECONDS = 1.0
SETTLE_SECONDS = 2.0
GENERATOR_BURST = 32
GENERATOR_CLONES = 1000
CTL_PORT = 47200
EXPECTED_DESTINATION = "10.0.0.1"
LFPLO_COUNTERS = runtime_xdp.counter_names(("Malformed", "Expired", "Send"))
C_COUNTERS = ("frames", "forwarded", "malformed", "expired", "full", "completed", "kernel_dropped", "zerocopy")
ROUTES = "0.0.0.0/2 0\n64.0.0.0/2 1\n128.0.0.0/2 2\n192.0.0.0/2 3\n10.0.0.0/8 0\n"
CLOCK_TICKS = os.sysconf("SC_CLK_TCK")


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


def has_a_device(port: str, sysfs: str = "/sys/class/net") -> bool:
    """A real card has a device behind it. veth, tun and dummy have none, and that is what is refused."""
    return os.path.exists(os.path.join(sysfs, port, "device"))


def driver_of(port: str) -> str:
    for line in run("ethtool", "-i", port).stdout.splitlines():
        if line.startswith("driver:"):
            return line.split(":", 1)[1].strip()
    raise Failed(f"ethtool does not say what drives {port}")


def physical_core(cpu: int, sysfs: str = "/sys/devices/system/cpu") -> tuple[int, int]:
    topology = os.path.join(sysfs, f"cpu{cpu}", "topology")
    try:
        with open(os.path.join(topology, "physical_package_id")) as package, \
                open(os.path.join(topology, "core_id")) as core:
            return int(package.read()), int(core.read())
    except OSError as failure:
        raise Refused(f"cpu {cpu} is not there, or is offline: {failure}") from failure


def default_cores(sysfs: str = "/sys/devices/system/cpu") -> tuple[int, int]:
    """Two threads of different physical cores, from the far end: cpu 0 takes most of what else runs."""
    allowed = sorted(os.sched_getaffinity(0), reverse=True)
    chosen: dict = {}
    for cpu in allowed:
        chosen.setdefault(physical_core(cpu, sysfs), cpu)
    cores = list(chosen.values())
    if len(cores) < 2:
        raise Refused("this needs two physical cores, and only one is allowed here")
    return cores[1], cores[0]


def check_cores(generator: int, forwarder: int, sysfs: str = "/sys/devices/system/cpu") -> None:
    if physical_core(generator, sysfs) == physical_core(forwarder, sysfs):
        raise Refused(f"cpus {generator} and {forwarder} are two threads of one physical core, so what "
                      f"each does would slow the other, and the rate would be neither's")


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


def rates(seconds: float, offered: int, forwarded: int, back: int, ticks: int) -> Rates:
    return Rates(offered / seconds, forwarded / seconds, back / seconds, ticks / CLOCK_TICKS / seconds)


class Namespace:
    def __init__(self) -> None:
        self.name = f"lfplspeed{uuid.uuid4().hex[:6]}"
        self.path = f"/run/netns/{self.name}"
        run("ip", "netns", "add", self.name)

    def inside(self):
        return xdp_load.in_namespace(self.path)

    def close(self) -> None:
        run("ip", "netns", "del", self.name, check=False)


class Pktgen:
    """One pktgen thread, on one core, sending from one port, in that port's namespace."""

    def __init__(self, namespace: Namespace, port: str, core: int, real_card: bool) -> None:
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
        # A virtual port cannot share one buffer between sends, so both of these are refused there.
        if self.real_card:
            if not random_destinations:
                self._set(f"clone_skb {GENERATOR_CLONES}")
            self._set(f"burst {GENERATOR_BURST}")

    def start(self) -> None:
        # Writing `start` blocks for as long as it runs, so it gets a process of its own.
        self.starter = subprocess.Popen(["nsenter", f"--net={self.namespace.path}", "sh", "-c",
                                         "echo start > /proc/net/pktgen/pgctrl"],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

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
        self.starter.stderr.close()
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


def cpu_ticks(pid: int) -> int:
    with open(f"/proc/{pid}/stat") as handle:
        fields = handle.read().rsplit(")", 1)[1].split()
    return int(fields[11]) + int(fields[12])


class Forwarder:
    def __init__(self, name: str, command: list, counters: tuple, port: str, pin_dir: str, core: int, mode: str):
        self.name = name
        self.counters = counters
        self.port = port
        self.pin_dir = pin_dir
        self.mode = mode
        xdp_load.attach(port, pin_dir, mode, 1)
        self.process = subprocess.Popen(command, stderr=subprocess.PIPE,
                                        preexec_fn=lambda: os.sched_setaffinity(0, {core}))
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
    """Frames on any other queue would miss the socket, which is bound to queue 0."""
    channels = run("ethtool", "-l", port, check=False)
    if channels.returncode:
        print(f"  note: {port} does not report channels", flush=True)
        return
    current = None
    in_current = False
    for line in channels.stdout.splitlines():
        if line.startswith("Current"):
            in_current = True
        if in_current and line.startswith("Combined:"):
            current = line.split(":")[1].strip()
    if current in (None, "n/a", "1"):
        return
    run("ethtool", "-L", port, "combined", "1")
    undo.append(("ethtool", "-L", port, "combined", current))


class Run:
    """All the state a run changes, so that it can be put back whatever goes wrong."""

    def __init__(self, args: argparse.Namespace, real_card: bool) -> None:
        self.args = args
        self.real_card = real_card
        self.undo: list = []
        self.irq_undo: list = []
        self.namespace: Optional[Namespace] = None
        self.generator_moved = False
        self.pin = f"/sys/fs/bpf/lfplspeed{uuid.uuid4().hex[:6]}"
        self.loaded_pktgen = False
        self.work = tempfile.mkdtemp()

    def prepare(self) -> None:
        args = self.args
        if not os.path.exists("/proc/net/pktgen"):
            run("modprobe", "pktgen")
            self.loaded_pktgen = True
        self.namespace = Namespace()
        with self.namespace.inside():
            run("ip", "link", "set", "lo", "up")
            run("sysctl", "-q", "-w", "net.ipv6.conf.all.disable_ipv6=1", "net.ipv6.conf.default.disable_ipv6=1")
        one_queue(args.forwarder, self.undo)
        one_queue(args.generator, self.undo)
        for port in (args.forwarder, args.generator):
            # Pause frames would have the forwarder hold back the generator, and hide what it can't do.
            run("ethtool", "-A", port, "rx", "off", "tx", "off", check=False)
        pin_irqs(args.forwarder, args.forwarder_core, self.irq_undo)
        pin_irqs(args.generator, args.generator_core, self.irq_undo)
        run("ip", "link", "set", args.generator, "netns", self.namespace.name)
        self.generator_moved = True
        with self.namespace.inside():
            run("ip", "link", "set", args.generator, "up")
            # What comes back is addressed to the next hop, and has to be counted anyway.
            run("ip", "link", "set", args.generator, "promisc", "on")
        run("sysctl", "-q", "-w", f"net.ipv6.conf.{args.forwarder}.disable_ipv6=1", check=False)
        run("ip", "link", "set", args.forwarder, "up")

    def restore(self) -> None:
        args = self.args
        if self.generator_moved:
            run("ip", "-n", self.namespace.name, "link", "set", args.generator, "promisc", "off", check=False)
            run("ip", "-n", self.namespace.name, "link", "set", args.generator, "netns", str(os.getpid()), check=False)
            run("ip", "link", "set", args.generator, "up", check=False)
        if self.namespace is not None:
            self.namespace.close()
        for command in reversed(self.undo):
            run(*command, check=False)
        for path, value in reversed(self.irq_undo):
            try:
                with open(path, "w") as handle:
                    handle.write(value)
            except OSError:
                pass
        if self.loaded_pktgen:
            run("rmmod", "pktgen", check=False)
        shutil.rmtree(self.pin, ignore_errors=True)
        shutil.rmtree(self.work, ignore_errors=True)

    def build(self) -> dict:
        args = self.args
        map_path = os.path.join(self.pin, xdp_load.MAP_FILE)
        lfplo = os.path.join(self.work, "lfplo")
        report = subprocess.run([sys.executable, "-m", "src.lfplo", "examples/ipv4.lfpl", "-o", lfplo,
                                 "--xdp", args.forwarder, "--map", map_path, "--ctl-port", str(CTL_PORT)],
                                cwd=ROOT, capture_output=True, text=True)
        if report.returncode:
            raise Failed(f"examples/ipv4.lfpl did not build: {report.stderr}")
        print(f"  {report.stdout.strip()}", flush=True)
        baseline = os.path.join(self.work, "ipv4_forward")
        run("cc", "-O3", "-march=native", "-Wall", "-Wextra", "-o", baseline,
            os.path.join(ROOT, "baseline", "ipv4_forward.c"))
        routes = os.path.join(self.work, "routes")
        with open(routes, "w") as handle:
            handle.write(ROUTES)
        common = [baseline, "--ifname", args.forwarder, "--map", map_path, "--ctl-port", str(CTL_PORT)]
        return {"c": (common + ["--match"], C_COUNTERS),
                "lfplo": ([lfplo], LFPLO_COUNTERS),
                "c-table": (common + ["--routes", routes], C_COUNTERS)}

    def measure(self, name: str, command: list, counters: tuple, wire_bytes: int) -> tuple:
        args = self.args
        mode = "skb" if args.skb else "drv"
        mac = link(args.forwarder)["address"]
        generator = Pktgen(self.namespace, args.generator, args.generator_core, self.real_card)
        forwarder = Forwarder(name, command, counters, args.forwarder, self.pin, args.forwarder_core, mode)
        try:
            generator.configure(wire_bytes, mac, args.random_destinations)
            generator.start()
            try:
                time.sleep(args.warmup)
                sent_before = packets(args.generator, "tx", self.namespace)
                back_before = packets(args.generator, "rx", self.namespace)
                before = forwarder.ask()
                ticks_before = cpu_ticks(forwarder.process.pid)
                started = time.monotonic()
                time.sleep(args.seconds)
                after = forwarder.ask()
                seconds = time.monotonic() - started
                ticks_after = cpu_ticks(forwarder.process.pid)
                sent_after = packets(args.generator, "tx", self.namespace)
                back_after = packets(args.generator, "rx", self.namespace)
            finally:
                generator.stop()
            time.sleep(SETTLE_SECONDS if self.real_card else 0.2)
            settled = forwarder.ask()
        finally:
            forwarder.stop()
            generator.forget()
        measured = rates(seconds, sent_after - sent_before, after["forwarded"] - before["forwarded"],
                         back_after - back_before, ticks_after - ticks_before)
        return measured, settled, sent_after


def print_rates(results: list, driver: str, args: argparse.Namespace) -> None:
    print(f"\ndriver {driver}, {'generic XDP, which copies twice' if args.skb else 'native XDP'}, forwarder on cpu "
          f"{args.forwarder_core}, generator on cpu {args.generator_core}, {args.seconds:g} s a window")
    print(f"{'forwarder':<10} {'frame':>6} {'offered/s':>14} {'forwarded/s':>14} {'back/s':>14} {'core':>6} "
          f"{'zero-copy':>10} {'full':>12} {'dropped':>12}  limit")
    for name, wire_bytes, measured, settled in results:
        limit = "the forwarder" if measured.saturated else "not the forwarder: this is no ceiling"
        print(f"{name:<10} {wire_bytes:>6} {measured.offered:>14,.0f} {measured.forwarded:>14,.0f} "
              f"{measured.back:>14,.0f} {measured.core_busy:>6.0%} {'yes' if settled['zerocopy'] else 'no':>10} "
              f"{settled['full']:>12,} {settled['kernel_dropped']:>12,}  {limit}")
    print("full: refused because the transmit ring had no room. dropped: the socket's own count of what it could "
          "not take. A card in zero-copy mode counts that in its own statistics instead: see ethtool -S.")
    print()
    for line in comparisons(results):
        print(line)


def comparisons(results: list) -> list:
    """LFPL-O against the ceiling, but only where the ceiling was reached. If the C baseline kept up
    with everything it was offered, its rate is the generator's, and a ratio to it compares nothing."""
    lines = []
    for wire_bytes in WIRE_SIZES:
        by_name = {name: measured for name, size, measured, _settled in results if size == wire_bytes}
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
    parser.add_argument("--forwarder-core", type=int)
    parser.add_argument("--seconds", type=float, default=10.0, help="the length of the window that is measured")
    parser.add_argument("--warmup", type=float, default=2.0, help="seconds of traffic before the window opens")
    parser.add_argument("--random-destinations", action="store_true",
                        help="a new destination for each frame, which also slows the generator")
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


def main(argv: list) -> int:
    args = parse(argv)
    try:
        reason = refusal(args)
        if reason is not None:
            raise Refused(reason)
        if args.generator_core is None or args.forwarder_core is None:
            generator_core, forwarder_core = default_cores()
            args.generator_core = generator_core if args.generator_core is None else args.generator_core
            args.forwarder_core = forwarder_core if args.forwarder_core is None else args.forwarder_core
        check_cores(args.generator_core, args.forwarder_core)
    except Refused as failure:
        print(f"speed: refused: {failure}", file=sys.stderr)
        return 2
    real = has_a_device(args.forwarder) and has_a_device(args.generator)
    driver = driver_of(args.forwarder)
    if args.plumbing:
        args.seconds = PLUMBING_SECONDS
        args.warmup = min(args.warmup, 0.2)
    state = Run(args, real)
    results = []
    moved = []
    problems: list = []
    try:
        print(f"preparing {args.generator} and {args.forwarder}", flush=True)
        state.prepare()
        forwarders = state.build()
        for name, (command, counters) in forwarders.items():
            for wire_bytes in WIRE_SIZES:
                print(f"  {name} at {wire_bytes} bytes", flush=True)
                measured, settled, sent = state.measure(name, command, counters, wire_bytes)
                results.append((name, wire_bytes, measured, settled))
                problems += check_the_counters(name, settled, wire_bytes)
                moved.append((name, wire_bytes, sent, settled["frames"], settled["forwarded"]))
    except (Failed, xdp_load.LoadError, OSError) as failure:
        print(f"speed: failed: {failure}", file=sys.stderr)
        return 1
    finally:
        state.restore()
    if args.plumbing:
        for name, wire_bytes, sent, frames, forwarded in moved:
            print(f"{name} at {wire_bytes} bytes: the generator's port had sent {sent} frames by then, the forwarder "
                  f"took {frames} and forwarded {forwarded}")
        print("plumbing only: these counts say each step worked, and they are not a measure of anything")
    else:
        print_rates(results, driver, args)
    for problem in problems:
        print(f"speed: {problem}: this run was not the workload", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
