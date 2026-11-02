"""The Verilog backend, against the x86 binary on the same bytes, and against its own arithmetic.

The question is whether the RT layer is a physical abstraction or x86 under another name. So the same
stream goes down both backends, each datagram goes through both, and every output has to agree. That
is tested parity on these inputs, and not a proof of equivalence.

The other half is that the emitter says what the machine will take before anything runs. rtverilog
derives k and c from the states it wrote, and these tests hold the simulation to them exactly. A run
over the bound, or with a different cost per cell, is a bug in the emitter and not latency.

Needs iverilog and vvp, and for the synthesis tests yosys and verilator. A missing tool skips what
needs it, and says so.
"""
import ast
import io
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ring import Reject  # noqa: E402
from src.rtcheck import RTError, Unbounded  # noqa: E402
from src.rtlower import Limit, rt_lines  # noqa: E402
from src.rtverilog import CELL_BITS, Translation, translate  # noqa: E402
from tests.test_checksum import (LARGEST, PORT, SOURCE, cells_in, internet_checksum,  # noqa: E402
                                 packet, payload)
from tests.reference import hgen, ipv4_model  # noqa: E402
from tests.test_forward import ROOT, Forwarder  # noqa: E402
from tests.test_generated import DECLARED  # noqa: E402
from tests.test_walks import STEPS, WIDTH, cells_of, payload_of, pipeline_text, wire_of  # noqa: E402

TESTBENCH = os.path.join(ROOT, "tests", "tb_checksum.v")
SIMULATOR = unittest.skipUnless(shutil.which("iverilog") and shutil.which("vvp"), "iverilog is not installed")
COUNT = int(os.environ.get("LFPL_FUZZ", "60"))
SEED = int(os.environ.get("LFPL_FUZZ_SEED", "0"))
RANDOM_PAYLOADS = 240
NAMED = (0, 7, 8, 1000, LARGEST)  # what the live UDP run showed


def translate_file(path: str, **options) -> Translation:
    with open(path, "rb") as handle:
        return translate(rt_lines(handle), **options)


def translate_text(text: str, **options) -> Translation:
    return translate(rt_lines(io.BytesIO(text.encode())), **options)


def simulate(translation: Translation, items: list, sum_bytes: int) -> tuple:
    """Each item is a datagram and how many cells to give it. The results, and the last line."""
    lines = []
    for datagram, reserve in items:
        lines += [f"{len(datagram) & 255:02x}", f"{len(datagram) >> 8:02x}",
                  f"{reserve & 255:02x}", f"{reserve >> 8:02x}"]
        lines += [f"{byte:02x}" for byte in datagram]
    lines += ["ff", "ff"]
    with tempfile.TemporaryDirectory() as workdir:
        design = os.path.join(workdir, "design.v")
        with open(design, "w") as handle:
            handle.write(translation.verilog)
        vectors = os.path.join(workdir, "vectors.hex")
        with open(vectors, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        sim = os.path.join(workdir, "sim")
        compiled = subprocess.run(["iverilog", "-g2005", f"-DTOP={translation.module}",
                                   f"-DPOINTER_BITS={translation.pointer_bits}", f"-DSTREAM_BYTES={len(lines)}",
                                   f"-DSUM_BYTES={sum_bytes}", "-o", sim, TESTBENCH, design],
                                  capture_output=True, text=True)
        assert compiled.returncode == 0, compiled.stderr
        done = subprocess.run(["vvp", sim, f"+vectors={vectors}"], check=True, capture_output=True,
                              text=True, cwd=workdir)
    results = []
    for line in done.stdout.splitlines():
        if line.startswith("ITEM"):
            fields = dict(part.split("=") for part in line.split()[2:])
            results.append({key: value if key == "out" else int(value) for key, value in fields.items()})
    last = done.stdout.strip().splitlines()[-1]
    assert last.startswith("DONE"), done.stdout[-500:]
    return results, dict(part.split("=") for part in last.split()[1:])


def on_x86(source: str, datagrams: list) -> list:
    """The native binary, built to count its steps, on a live socket: its answer and its steps."""
    results = []
    with tempfile.TemporaryDirectory() as workdir:
        router = Forwarder(bytes(1024), workdir, build_options=("--count-steps",), source=source)
        try:
            before = router.stats()
            for count, datagram in enumerate(datagrams, 1):
                router.raw(datagram)
                answer = router.outputs[PORT].recv(16384)
                now = router.settle(count)
                results.append((answer, now["steps"] - before["steps"]))
                before = now
        finally:
            router.close()
    return results


class TargetIndependence(unittest.TestCase):
    def test_it_loads_nothing_of_the_x86_backend_or_runtime(self):
        """Through anything it imports, too: this is why `certify` has a module of its own."""
        check = ("import sys, src.rtverilog; print(sorted(m for m in sys.modules "
                 "if m in ('src.x86', 'src.runtime', 'src.lfplo', 'src.validate', 'src.symcheck')))")
        done = subprocess.run([sys.executable, "-c", check], cwd=ROOT, check=True, capture_output=True, text=True)
        self.assertEqual(done.stdout.strip(), "[]")

    def test_what_it_imports_by_name(self):
        with open(os.path.join(ROOT, "src", "rtverilog.py")) as handle:
            tree = ast.parse(handle.read())
        modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.level}
        self.assertEqual(modules, {"certify", "ring", "rtcheck", "rtlower"})

    STREAM = """
type t16 abstract Bytes
type t17 record Hdr
ctor c18 t17 - Hdr
field c18 0 1 i8 ttl
field c18 8 8 t16 body
type t30 tagless Free
ctor c31 t30 null End
ctor c32 t30 - More
field c32 0 8 t30 next
tuple t50 t17 t30
{handler}
"""

    def test_it_refuses_a_stream_rtcheck_refuses_before_writing_any_of_it(self):
        for name, handler, why in (
                ("a linear value left behind", """handler f0 h(r0 : t17, r1 : t30) : t50
    const r3, #0 : t30
    ret r0, r3
end""", "linear value still in r1"),
                ("a store that was skipped", """handler f0 h(r0 : t17, r1 : t30) : t50
    open r0, c18
    take r2, r0[8:8]
    build r0, c18, none, =, =
    ret r0, r1
end""", "STAY"),
                ("an instruction that does not exist", """handler f0 h(r0 : t17, r1 : t30) : t50
    frobnicate r0
    ret r0, r1
end""", "unknown instruction")):
            with self.subTest(name), self.assertRaisesRegex(RTError, why):
                translate(self.STREAM.format(handler=handler).splitlines())

    def test_it_refuses_a_handler_whose_bound_does_not_pass_its_certificate(self):
        with self.assertRaisesRegex(Unbounded, "no linear bound"):
            translate_text("""data Bytes = Last | Chunk(used: Int8, b0: Int8, b1: Int8, b2: Int8, b3: Int8, b4: Int8,
                          b5: Int8, b6: Int8, next: Bytes);
record P { port: Int8, body: Bytes };
fn rev(l: Bytes, acc: Bytes) : Bytes = match l with
  | Last => acc
  | Chunk(d, used, b0, b1, b2, b3, b4, b5, b6, next) => rev(next, Chunk@d(used, b0, b1, b2, b3, b4, b5, b6, acc))
  end;
fn quad(l: Bytes, acc: Bytes) : Bytes = match l with
  | Last => acc
  | Chunk(d, used, b0, b1, b2, b3, b4, b5, b6, next) =>
      quad(next, rev(Chunk@d(used, b0, b1, b2, b3, b4, b5, b6, acc), Last))
  end;
handler h(p: P, free: Free) : (P, Free) =
  match p with P(d, port, body) => (P@d(port, quad(body, Last)), free) end;
""")

    def test_a_refusal_leaves_no_verilog_and_removes_a_stale_one(self):
        with tempfile.TemporaryDirectory() as workdir:
            path, output = os.path.join(workdir, "spin.lfpl"), os.path.join(workdir, "spin.v")
            with open(path, "w") as handle:
                handle.write("fn spin(x: Int64) : Int64 = spin(x);\n")
            with open(output, "w") as handle:
                handle.write("// an earlier build")
            done = subprocess.run([sys.executable, "-m", "src.rtverilog", path, "-o", output], cwd=ROOT,
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 1)
            self.assertIn("rule R3", done.stderr)
            self.assertFalse(os.path.exists(output))


class TheEmittersOwnArithmetic(unittest.TestCase):
    """These run nothing. They are what the simulation is then held to."""

    def test_the_checksum_is_one_cycle_per_step_and_one_read_outside_the_walks(self):
        timing = translate_file(SOURCE).timing
        self.assertEqual((timing.k, timing.c), (1, 1))

    def test_each_kind_says_what_it_takes(self):
        kinds = {kind: (steps, low, high) for kind, steps, low, high, _count in translate_file(SOURCE).timing.kinds}
        self.assertEqual(kinds["pack"], (1, 1, 1))
        self.assertEqual(kinds["take"], (1, 1, 1))
        self.assertEqual(kinds["build"], (1, 1, 1))
        self.assertEqual(kinds["tail"], (1, 1, 1))
        self.assertEqual(kinds["br.null"], (1, 1, 1))
        self.assertEqual(kinds["drop"], (0, 0, 0))
        self.assertEqual(kinds["open"], (0, 0, 0))
        self.assertEqual(kinds["call"], (1, 1, 2), "the first names a field of a cell that has not been read")

    def test_the_walks_add_up_to_the_steps_per_cell(self):
        translation = translate_file(SOURCE)
        walks = {name: (str(iteration), str(leaving)) for name, iteration, leaving in translation.timing.iterations}
        self.assertEqual(walks, {"add": ("24", "2"), "reverse": ("4", "2")})
        self.assertEqual(24 + 4, translation.timing.per_cell(translation.bound))

    def test_the_report_gives_the_bound_before_anything_runs(self):
        report = translate_file(SOURCE).report()
        self.assertIn("cycles(n) <= 1*W(n) + 1 = 28 n + 5", report)
        self.assertRegex(report, r"br\.null\s+1\s+1\s+2")

    def test_a_tag_is_a_read_and_then_a_compare_so_a_walk_of_them_doubles_k(self):
        timing = translate_text(pipeline_text(["marks"])).timing
        self.assertEqual(timing.k, 2)
        kinds = {kind: (steps, low, high) for kind, steps, low, high, _count in timing.kinds}
        self.assertEqual(kinds["br.tag"], (1, 1, 2))

    def test_two_cells_in_one_instruction_cost_a_read_and_a_latch(self):
        """There is one read port. `sums` adds a byte of one cell to a byte of the next."""
        timing = translate_text(pipeline_text(["sums"])).timing
        kinds = {kind: high for kind, _steps, _low, high, _count in timing.kinds}
        self.assertEqual(kinds["op"], 3, "latch the byte at the port, read the other cell, and add")
        self.assertEqual(timing.k, 3)


    def test_c_counts_what_a_helper_does_outside_a_walk(self):
        """The handler reads its header, and `late` has to read its cell again after calling rev."""
        timing = translate_text(pipeline_text(["late"])).timing
        self.assertEqual((timing.k, timing.c), (1, 2))

    LOOK = """data Bytes = Last | Chunk(used: Int8, b0: Int8, b1: Int8, b2: Int8, b3: Int8, b4: Int8,
                          b5: Int8, b6: Int8, next: Bytes);
record P { port: Int8, body: Bytes };
state t : Array[8] Int8;
fn look(t: Array[8] Int8, x: Int8) : (Array[8] Int8, Int8) = get(t, idx(x, 8));
fn walk(t: Array[8] Int8, l: Bytes, acc: Bytes) : (Array[8] Int8, Bytes) = match l with
  | Last => (t, acc)
  | Chunk(d, used, b0, b1, b2, b3, b4, b5, b6, next) =>
      let (t2, v) = look(t, b0) in walk(t2, next, Chunk@d(used, v, b1, b2, b3, b4, b5, b6, acc))
  end;
handler h(t: Array[8] Int8, p: P, free: Free) : (Array[8] Int8, P, Free) =
  match p with P(d, port, body) => let (t2, a) = walk(t, body, Last) in (t2, P@d(port, a), free) end;
"""

    def test_k_counts_what_a_walk_calls(self):
        """`look` is not recursive, and it runs once a cell. An array read is two states."""
        self.assertEqual(translate_text(self.LOOK).timing.k, 2)

    TAGGED = """
type t16 tagged Two
ctor c17 t16 0 P
field c17 1 1 i8 x
ctor c18 t16 1 Q
field c18 1 1 i8 y
type t30 tagless Free
ctor c31 t30 null End
ctor c32 t30 - More
field c32 0 8 t30 next
tuple t50 t16 t30
fn f0 same(r0 : t16) : t16
    ret r0
end
{handler}
"""

    def test_a_register_written_again_is_not_the_cell_it_named(self):
        """Nothing Lower writes reuses a register like this, and rtcheck allows it, so the emitter
        has to get it right. Neither callee reads, so what is at the port is still the first word,
        and the second tag test is of a value that may be another cell."""
        for name, again in (("by a call", "call r0, f0, r0"), ("by a move", "call r5, f0, r0\n    move r0, r5")):
            text = translate(self.TAGGED.format(handler=f"""handler f1 h(r0 : t16, r1 : t30) : t50
    br.tag r0, 0, L1, L2
L1: build r0, c17, write, =
    jump L3
L2: {again}
    br.tag r0, 1, L4, L5
L4: build r0, c18, write, =
    jump L3
L5: open r0, c17
    build r0, c17, write, =
    jump L3
L3: ret r0, r1
end""").splitlines()).verilog
            with self.subTest(name):
                self.assertEqual(text.count("read the cell in r0 (a state RT does not charge for)"), 2)

    def test_a_field_put_back_where_it_came_from_has_no_write_enable(self):
        """RESTORE, which is the other half of RTL-3."""
        text = translate("""
type t16 tagless List
ctor c17 t16 null Nil
ctor c18 t16 - Cons
field c18 0 8 i64 head
field c18 8 8 t16 tail
type t30 tagless Free
ctor c31 t30 null End
ctor c32 t30 - More
field c32 0 8 t30 next
type t40 record Item
ctor c41 t40 - Item
field c41 0 1 i8 port
field c41 8 8 t16 body
tuple t50 t40 t30
handler f0 h(r0 : t40, r1 : t30) : t50
    open r0, c41
    take r2, r0[8:8]
    build r0, c41, none, #1, r2!
    ret r0, r1
end
""".splitlines()).verilog
        arm = re.search(r"build r0, c41, none, #1, r2!\n(.*?)\n\s*end", text, re.S).group(1)
        self.assertIn("b_we = 16'h0001;", arm)
        self.assertEqual(arm.count("b_din"), 1)


@SIMULATOR
class TheChecksum(unittest.TestCase):
    """The five datagrams of the live run, 240 random ones, and those that take each `if` one way all
    along. The payloads come from the generators the live tests use, and no byte is retyped."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(SEED)
        named = [payload(length) for length in NAMED]
        short = [payload(length) for length in range(0, 64)]
        randomly = [payload_of(rng.randrange(LARGEST + 1), "random", rng) for _ in range(RANDOM_PAYLOADS)]
        dearest = [payload_of(length, fill, rng) for fill in ("zeros", "ones") for length in (1, 6, 7, 8, 700, LARGEST)]
        cls.payloads = named + short + randomly + dearest
        cls.datagrams = [packet(body) for body in cls.payloads]
        cls.translation = translate_file(SOURCE)
        cls.bound = cls.translation.bound
        cls.results, cls.summary = simulate(cls.translation, [(d, 0) for d in cls.datagrams], 2)
        cls.native = on_x86(SOURCE, cls.datagrams)

    def test_every_output_is_the_x86_binarys_bit_for_bit(self):
        self.assertEqual(len(self.results), len(self.datagrams))
        self.assertGreaterEqual(len(self.datagrams), 200 + len(NAMED))
        for body, result, (answer, _steps) in zip(self.payloads, self.results, self.native):
            with self.subTest(length=len(body)):
                self.assertEqual(result["out"], answer.hex())
                self.assertEqual(answer, internet_checksum(body) + body, "and both are the right answer")

    def test_a_run_never_exceeds_the_bound_and_steps_are_what_the_checker_said(self):
        timing = self.translation.timing
        for body, result, (_answer, steps) in zip(self.payloads, self.results, self.native):
            n = cells_in(len(body))
            self.assertEqual(result["n"], n)
            self.assertLessEqual(steps, self.bound.c * n + self.bound.d)
            self.assertLessEqual(result["cycles"], timing.bound(n, self.bound), f"{len(body)} bytes: a bug, not latency")

    def test_cycles_are_steps_plus_the_one_read_the_emitter_counted(self):
        """No input is special: this handler has one path for each n, so this is an equality, and a
        difference at any length is something the emitter does not know about its own machine."""
        for body, result, (_answer, steps) in zip(self.payloads, self.results, self.native):
            self.assertEqual(result["cycles"], self.translation.timing.k * steps + self.translation.timing.c, len(body))

    def test_each_cell_costs_what_the_emitter_said(self):
        per_cell = self.translation.timing.per_cell(self.bound)
        by_n = {result["n"]: result["cycles"] for result in self.results}
        low = min(by_n)
        for n, cycles in by_n.items():
            self.assertEqual(cycles - by_n[low], per_cell * (n - low), f"{n} cells")

    def test_one_read_and_one_write_per_walk_of_a_cell_and_the_header(self):
        """The block RAM's traffic follows from the program: each body cell is opened twice and its
        link written twice, and the header is read once and written once."""
        for result in self.results:
            self.assertEqual(result["reads"], 2 * (result["n"] - 1) + 1)
            self.assertEqual(result["writes"], 2 * (result["n"] - 1) + 1)

    def test_no_clock_read_and_wrote_at_once(self):
        self.assertEqual(self.summary["both_ports"], "0")
        self.assertEqual(self.summary["same_address"], "0")

    def test_every_cell_comes_back(self):
        for result in self.results:
            self.assertEqual(result["back"], result["n"])

    def test_the_table(self):
        print("\n" + cycle_table(self.bound, self.translation, self.payloads, self.results, self.native), file=sys.stderr)


def cycle_table(bound, translation, payloads, results, native) -> str:
    rows = ["payload    n   W(n)  steps  cycles  cycles/steps  bound k*W(n)+c"]
    seen = set()
    for body, result, (_answer, steps) in zip(payloads, results, native):
        if len(body) in NAMED and len(body) not in seen:
            seen.add(len(body))
            n = result["n"]
            rows.append(f"{len(body):>7} {n:>4} {bound.c * n + bound.d:>6} {steps:>6} {result['cycles']:>7} "
                        f"{result['cycles'] / steps:>13.4f} {translation.timing.bound(n, bound):>15}")
    return "\n".join(rows)


@SIMULATOR
class RandomWalks(unittest.TestCase):
    """The pipelines tests/test_walks.py runs on x86, which are the machine's other instructions:
    joins, tagged types, tail recursion modulo cons, a reserve, and a walk calling a helper."""

    def run_pipeline(self, names: list, rng: random.Random, said: str) -> None:
        growth = 1
        for name in names:
            growth *= STEPS[name][2]
        longest = WIDTH * (150 // growth)
        bodies = [payload_of(longest, fill, rng) for fill in ("zeros", "ones", "random")]
        bodies += [payload_of(rng.choice((0, 1, 7, 8, 30)), "random", rng) for _ in range(4)]
        datagrams = [bytes((PORT,)) + body for body in bodies]
        with tempfile.TemporaryDirectory() as workdir:
            path = os.path.join(workdir, "walk.lfpl")
            with open(path, "w") as handle:
                handle.write(pipeline_text(names))
            translation = translate_file(path)
            bound = translation.bound
            items = [(d, bound.a * cells_in(len(d) - 1) + bound.b) for d in datagrams]
            results, summary = simulate(translation, items, 0)
            native = on_x86(path, datagrams)
        said = f"{said}, {names}, W(n) = {bound.c} n + {bound.d}"
        self.assertEqual((summary["both_ports"], summary["same_address"]), ("0", "0"), said)
        for body, result, (answer, steps), (_d, reserve) in zip(bodies, results, native, items):
            cells = cells_of(body)
            for name in names:
                cells = STEPS[name][1](cells)
            n = cells_in(len(body))
            self.assertEqual(result["out"], answer.hex(), f"{said}, {body.hex()}")
            self.assertEqual(answer, wire_of(cells), said)
            self.assertEqual(result["back"], n + reserve, said)
            self.assertLessEqual(result["cycles"], translation.timing.bound(n, bound), f"{said}: a bug, not latency")
            self.assertGreaterEqual(result["cycles"], steps, f"{said}: a state RT charges for did not run")

    def test_every_walk_by_itself(self):
        """So that each is run whatever the seeds choose: each has a path no other walk has."""
        for name in STEPS:
            with self.subTest(name):
                self.run_pipeline([name], random.Random(SEED), name)

    def test_what_an_at_function_returns_is_the_cell_it_was_given(self):
        """By itself its result only goes back where it came from, with no store, so nothing reads
        it. A walk over the result has to find the first cell, and not where the last iteration got to."""
        self.run_pipeline(["incr", "rev"], random.Random(SEED), "an @ function, then a walk")

    def test_every_pipeline_agrees_with_the_x86_binary_and_keeps_to_its_bound(self):
        for seed in range(1000 * SEED, 1000 * SEED + COUNT // 3):
            rng = random.Random(seed)
            names = [rng.choice(tuple(STEPS)) for _ in range(rng.randrange(1, 5))]
            self.run_pipeline(names, rng, f"seed {seed}")


class Layout:
    """What the wire format of section 12 says about a handler, from its RT stream: the same things
    src/runtime.py reads from the checker, read here from the text."""

    def __init__(self, lines: list) -> None:
        self.ctors: dict = {}
        self.by_type: dict = {}
        self.kinds: dict = {}
        self.tuples: dict = {}
        self.aliases: dict = {}
        for line in lines:
            head, *parts = line.split() or [""]
            if head == "type":
                self.kinds[parts[0]] = parts[1]
            elif head == "alias":
                self.aliases[parts[0]] = parts[1]
            elif head == "tuple":
                self.tuples[parts[0]] = parts[1:]
            elif head == "ctor":
                self.ctors[parts[0]] = (parts[2], [])
                self.by_type.setdefault(parts[1], []).append(parts[0])
            elif head == "field":
                self.ctors[parts[0]][1].append((int(parts[1]), int(parts[2]), parts[3], parts[4]))
            elif head == "handler":
                match = re.match(r"handler \S+ \w+\((.*)\) : (\S+)", line)
                params = [part.partition(" : ")[2] for part in match.group(1).split(", ")]
                self.item = self.by_type[self._resolve(params[-2])][0]
                self.out_type = self._resolve(self.tuples[self._resolve(match.group(2))][-2])

    def _resolve(self, name: str) -> str:
        while name in self.aliases:
            name = self.aliases[name]
        return name

    @staticmethod
    def _is_scalar(type_text: str) -> bool:
        return type_text in ("i8", "i16", "i32", "i64", "bool", "be16", "be32", "be64")

    def _record_of(self, type_text: str):
        """The constructor of a field that is a record, whose cell it points to, or None."""
        name = self._resolve(type_text)
        return self.by_type[name][0] if self.kinds.get(name) == "record" else None

    @staticmethod
    def _is_length(size: int, type_text: str, name: str) -> bool:
        return name == "length" and type_text == "i16"

    def shape(self, ctor: str) -> tuple:
        """How many bytes of this record are on the wire, and how many cells it is."""
        wire, cells = 0, 1
        for _offset, size, type_text, name in self.ctors[ctor][1]:
            inner = self._record_of(type_text)
            if inner is not None:
                inner_wire, inner_cells = self.shape(inner)
                wire += inner_wire
                cells += inner_cells
            elif self._is_scalar(type_text) and not self._is_length(size, type_text, name):
                wire += size
        return wire, cells

    def image(self, datagram: bytes, reserve: int) -> tuple:
        """The cells src/runtime.py would have made of a datagram: (address, enables, data)."""
        words: dict = {}
        chain_field = None
        at = 0
        next_address = 2

        def fill(ctor: str, address: int) -> None:
            nonlocal at, next_address, chain_field
            word, enables = 0, 0
            for offset, size, type_text, name in self.ctors[ctor][1]:
                inner = self._record_of(type_text)
                if self._is_length(size, type_text, name):
                    value = len(datagram)
                elif inner is not None:
                    value = next_address
                    next_address += 1
                    fill(inner, value)
                elif self._is_scalar(type_text):
                    value = int.from_bytes(datagram[at:at + size], "little")
                    at += size
                else:
                    chain_field = (address, offset, size)
                    continue
                word |= value << (8 * offset)
                enables |= ((1 << size) - 1) << offset
            words[address] = [enables, word]

        fill(self.item, 1)
        body = datagram[at:]
        chunks = [body[i:i + WIDTH] for i in range(0, len(body), WIDTH)]
        first_chunk = next_address
        loads = []
        for index, chunk in enumerate(chunks):
            following = first_chunk + index + 1 if index + 1 < len(chunks) else 0
            word = len(chunk) | (int.from_bytes(chunk, "little") << 8) | (following << 64)
            loads.append((first_chunk + index, 0xffff, word))
        if chain_field is not None:
            address, offset, size = chain_field
            words[address][1] |= (first_chunk if chunks else 0) << (8 * offset)
            words[address][0] |= ((1 << size) - 1) << offset
        loads = [(address, enables, word) for address, (enables, word) in sorted(words.items())] + loads
        first_free = first_chunk + len(chunks)
        for index in range(reserve):
            following = first_free + index + 1 if index + 1 < reserve else 0
            loads.append((first_free + index, 0x00ff, following))
        return loads, 1, (first_free if reserve else 0), first_free - 1 + reserve

    def gather(self, out: int, free: int, words: dict) -> tuple:
        """What egress would do with an output: where it goes, its bytes, and how many cells came back."""
        names = self.by_type[self.out_type]
        kind = self.kinds[self.out_type]
        back = 0
        pointer = free
        while pointer:
            back += 1
            pointer = int(words[pointer][-16:], 16)  # a Free's next, at offset 0
        if kind == "tagged":
            tag = int(words[out][-2:], 16)
            ctor = next(c for c in names if self.ctors[c][0] == str(tag))
        elif kind == "record":
            ctor = names[0]
        else:
            ctor = next(c for c in names if (self.ctors[c][0] == "null") == (out == 0))
        if out:
            back += 1

        def field(word: str, offset: int, size: int) -> bytes:
            digits = word[-2 * (offset + size):len(word) - 2 * offset]
            return bytes.fromhex(digits)[::-1]

        port, payload_out = None, b""
        chain = 0

        def walk(ctor: str, pointer: int) -> None:
            nonlocal port, payload_out, chain, back
            for offset, size, type_text, name in self.ctors[ctor][1]:
                inner = self._record_of(type_text)
                if name == "port" and pointer == out:
                    port = field(words[pointer], offset, size)[0]
                elif self._is_length(size, type_text, name):
                    pass
                elif inner is not None:
                    back += 1
                    walk(inner, int.from_bytes(field(words[pointer], offset, size), "little"))
                elif self._is_scalar(type_text):
                    payload_out += field(words[pointer], offset, size)
                else:
                    chain = int.from_bytes(field(words[pointer], offset, size), "little")

        walk(ctor, out)
        while chain:
            word = words[chain]
            back += 1
            used = min(field(word, 0, 1)[0], WIDTH)
            payload_out += field(word, 1, used) if used else b""
            chain = int.from_bytes(field(word, 8, 8), "little")
        return port, payload_out, back


def simulate_images(translation: Translation, layout: Layout, datagrams: list) -> tuple:
    bound = translation.bound
    lines, cells = [], []
    for datagram in datagrams:
        wire_bytes, record_cells = layout.shape(layout.item)
        n = record_cells - 1 + cells_in(max(0, len(datagram) - wire_bytes))
        reserve = bound.a * n + bound.b
        loads, item, free, words = layout.image(datagram, reserve)
        cells.append((n, reserve))
        lines.append(f"{item:x} {free:x} {len(loads):x} {words:x}")
        lines += [f"{address:x} {enables:x} {word:x}" for address, enables, word in loads]
    lines.append("0 0 0 0")
    with tempfile.TemporaryDirectory() as workdir:
        design = os.path.join(workdir, "design.v")
        with open(design, "w") as handle:
            handle.write(translation.verilog)
        for name, text in translation.memories.items():
            with open(os.path.join(workdir, f"{name}.mem"), "w") as handle:
                handle.write(text)
        images = os.path.join(workdir, "images.txt")
        with open(images, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        sim = os.path.join(workdir, "sim")
        compiled = subprocess.run(["iverilog", "-g2005", f"-DTOP={translation.module}",
                                   f"-DPOINTER_BITS={translation.pointer_bits}", "-o", sim,
                                   os.path.join(ROOT, "tests", "tb_image.v"), design],
                                  capture_output=True, text=True)
        assert compiled.returncode == 0, compiled.stderr
        done = subprocess.run(["vvp", sim, f"+images={images}"], check=True, capture_output=True, text=True,
                              cwd=workdir)
    results, words = [], {}
    for line in done.stdout.splitlines():
        parts = line.split()
        if parts[0] == "ITEM":
            words = {}
            fields = dict(part.split("=") for part in parts[2:])
            results.append({"cycles": int(fields["cycles"]), "out": int(fields["out"]),
                            "free": int(fields["free"]), "words": words})
        elif parts[0] == "WORD":
            words[int(parts[1])] = parts[2]
    last = done.stdout.strip().splitlines()[-1]
    assert last.startswith("DONE"), done.stdout[-500:]
    return results, dict(part.split("=") for part in last.split()[1:]), cells


@SIMULATOR
class OtherHandlers(unittest.TestCase):
    """Arrays, tagged outputs, comparisons, and helpers with tuples: what LFPL-O section 9.5 means by
    the same corpus. Every output is held to the x86 binary's, on the same bytes, on a live socket."""

    def compare(self, source: str, initial: dict, options: tuple, datagrams: list, said: str) -> Translation:
        with open(source, "rb") as handle:
            lines = list(rt_lines(handle))
        translation = translate(lines, initial=initial)
        layout = Layout(lines)
        results, summary, cells = simulate_images(translation, layout, datagrams)
        self.assertEqual((summary["both_ports"], summary["same_address"]), ("0", "0"), said)
        with tempfile.TemporaryDirectory() as workdir:
            router = Forwarder(bytes(1024), workdir, build_options=("--count-steps", *options), source=source)
            try:
                before = router.stats()
                for number, (datagram, result, (n, reserve)) in enumerate(zip(datagrams, results, cells), 1):
                    router.raw(datagram)
                    port, sent, back = layout.gather(result["out"], result["free"], result["words"])
                    context = f"{said}, datagram {datagram.hex()}"
                    self.assertEqual(back, n + reserve, context)
                    if port is not None and port < len(router.outputs):
                        self.assertEqual(sent, router.outputs[port].recv(4096), context)
                    now = router.settle(number)
                    if port is None:
                        self.assertEqual(now["dropped"] - before["dropped"], 1, context)
                    steps = now["steps"] - before["steps"]
                    self.assertLessEqual(result["cycles"], translation.timing.bound(n, translation.bound),
                                         f"{context}: a bug, not latency")
                    self.assertGreaterEqual(result["cycles"], steps, context)
                    before = now
            finally:
                router.close()
        return translation

    def test_the_ipv4_example(self):
        """One record across five cells, in network order, as x86 runs it and as the machine does."""
        frames = [frame for _name, _verdict, frame in ipv4_model.corpus() if 34 <= len(frame) < 300]
        self.compare(os.path.join(ROOT, "examples", "ipv4.lfpl"), {}, ("--n-max", "217"), frames[::4], "ipv4")

    def test_the_forwarding_handler(self):
        routes = bytes((i * 7) % 8 for i in range(1024))
        rng = random.Random(SEED)
        datagrams = [bytes((rng.choice((0, 1, 2, 64, 255)),)) + rng.randrange(1 << 32).to_bytes(4, "little")
                     + payload_of(rng.choice((0, 1, 7, 8, 100)), "random", rng) for _ in range(60)]
        with tempfile.TemporaryDirectory() as workdir:
            table = os.path.join(workdir, "routes.bin")
            with open(table, "wb") as handle:
                handle.write(routes)
            translation = self.compare(os.path.join(ROOT, "examples", "forward.lfpl"), {"routes": routes},
                                       ("--state", f"routes={table}"), datagrams, "forward")
        kinds = {kind: high for kind, _steps, _low, high, _count in translation.timing.kinds}
        self.assertEqual(kinds["aget"], 2, "an array is its own block RAM: a read, then a latch")
        self.assertEqual((translation.timing.k, translation.timing.c), (1, 2))

    def test_pack_puts_its_first_operand_highest(self):
        with tempfile.TemporaryDirectory() as workdir:
            path = os.path.join(workdir, "pack.lfpl")
            with open(path, "w") as handle:
                handle.write("""record Hd { a: Int8, b: Int16 };
record Out { port: Int8, x: Int64 };
handler h(m: Hd, free: Free) : (Out, Free) = match m with Hd(d, a, b) => (Out@d(1, pack<8,16>(a, b)), free) end;
""")
            with open(path, "rb") as handle:
                lines = list(rt_lines(handle))
            datagram = bytes((0xab, 0xcd, 0xef))
            results, _summary, _cells = simulate_images(translate(lines), Layout(lines), [datagram])
            port, sent, _back = Layout(lines).gather(results[0]["out"], results[0]["free"], results[0]["words"])
            self.assertEqual((port, sent), (1, (0xabefcd).to_bytes(8, "little")))
            self.compare(path, {}, (), [datagram, bytes(3), b"\xff\xff\xff"], "pack")

    def test_a_walk_that_calls_an_array_read(self):
        """So k is 2, and the bound holds by a wide margin. `look`'s table says each byte's new value."""
        table = bytes((i * 11 + 3) & 255 for i in range(8))
        with tempfile.TemporaryDirectory() as workdir:
            path, state = os.path.join(workdir, "look.lfpl"), os.path.join(workdir, "t.bin")
            with open(path, "w") as handle:
                handle.write(TheEmittersOwnArithmetic.LOOK)
            with open(state, "wb") as handle:
                handle.write(table)
            rng = random.Random(SEED)
            datagrams = [bytes((PORT,)) + payload_of(length, "random", rng) for length in (0, 1, 7, 8, 100, 700)]
            self.compare(path, {"t": table}, ("--state", f"t={state}"), datagrams, "look")

    def test_every_generated_handler(self):
        ran = 0
        for seed in range(1000 * SEED, 1000 * SEED + COUNT // 3):
            try:
                handler = hgen.H(seed)
                handler.build()
            except hgen.Skip:
                continue
            with tempfile.TemporaryDirectory() as workdir:
                path = os.path.join(workdir, "program.lfpl")
                with open(path, "w") as handle:
                    handle.write(handler.text())
                options = []
                initial = {}
                for name, data in handler.state_bytes().items():
                    table = os.path.join(workdir, f"{name}.bin")
                    with open(table, "wb") as handle:
                        handle.write(data)
                    options += ["--state", f"{name}={table}"]
                    initial[name] = data
                try:
                    with open(path, "rb") as handle:
                        list(rt_lines(handle))
                except (Reject, Limit) as failure:
                    self.assertTrue(any(reason in str(failure) for reason in DECLARED), f"seed {seed}: {failure}")
                    continue
                datagrams = [handler.datagram()[0] for _ in range(12)]
                self.compare(path, initial, tuple(options), datagrams, f"seed {seed}\n{handler.text()}")
            ran += 1
        print(f"generated handlers on both targets: {ran} of {COUNT // 3}", file=sys.stderr)
        self.assertGreater(ran, COUNT // 9)


TOOLS = unittest.skipUnless(shutil.which("yosys") and shutil.which("verilator"), "yosys or verilator is not installed")


@TOOLS
class Synthesis(unittest.TestCase):
    """LFPL-O section 9.5. A smaller image, so that it fits the parts these map to."""

    @classmethod
    def setUpClass(cls):
        cls.workdir = tempfile.TemporaryDirectory()
        cls.translation = translate_file(SOURCE, cells=432)
        cls.design = os.path.join(cls.workdir.name, "checksum.v")
        with open(cls.design, "w") as handle:
            handle.write(cls.translation.verilog)

    @classmethod
    def tearDownClass(cls):
        cls.workdir.cleanup()

    def yosys(self, script: str) -> str:
        path = os.path.join(self.workdir.name, "run.ys")
        with open(path, "w") as handle:
            handle.write(f"read_verilog -sv {self.design}\nhierarchy -check -top checksum\n" + script)
        done = subprocess.run(["yosys", "-q", "-l", path + ".log", path], capture_output=True, text=True,
                              cwd=self.workdir.name)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        with open(path + ".log") as handle:
            return handle.read()

    def test_verilator_has_nothing_to_say(self):
        done = subprocess.run(["verilator", "--lint-only", "-Wall", self.design], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("%Warning", done.stderr)

    def test_every_register_comes_from_a_clock_and_there_are_no_latches(self):
        log = self.yosys("proc\ncheck -assert\nselect -assert-none t:$dlatch t:$adlatch t:$dlatchsr t:$sr\n")
        self.assertNotIn("Latch inferred", log)

    def test_there_are_no_implicit_wires(self):
        self.assertIn("`default_nettype none", self.translation.verilog)
        done = subprocess.run(["iverilog", "-g2005", "-Wall", "-tnull", self.design], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertNotIn("implicit", done.stderr)

    def test_the_cell_store_is_one_memory_of_cells_wide_words_with_byte_enables(self):
        log = self.yosys("proc\nopt\nmemory -nomap\ndump t:$mem_v2\n")
        self.assertRegex(log, rf"parameter \\WIDTH {CELL_BITS}\b")
        self.assertRegex(log, rf"parameter \\SIZE {432 + 1}\b")
        self.assertRegex(log, r"parameter \\RD_PORTS 1\b")
        self.assertRegex(log, r"parameter \\WR_PORTS 1\b")
        self.assertRegex(log, r"parameter \\RD_CLK_ENABLE 1'1")
        self.assertRegex(log, r"parameter \\RD_TRANSPARENCY_MASK 1'0", "on paper, a read returns the old word")
        self.assertRegex(log, r"parameter \\RD_COLLISION_X_MASK 1'1",
                         "yosys has proved from the state decode that a read and a write never meet")
        enables = re.search(r"connect \\WR_EN (.*)", log).group(1)
        bits = re.findall(r"\[(\d+)\]", enables)
        self.assertEqual(len(bits), CELL_BITS)
        for lane in range(16):
            self.assertEqual(len(set(bits[8 * lane:8 * lane + 8])), 1, "one enable for each byte")

    def test_it_maps_to_block_ram_and_not_to_flip_flops(self):
        log = self.yosys("synth_ice40 -top checksum\nstat\n")
        rams = int(re.search(r"(\d+)\s+SB_RAM40_4K", log).group(1))
        flops = sum(int(count) for count in re.findall(r"(\d+)\s+SB_DFF\w*", log.rpartition("Number of cells")[2]))
        self.assertGreater(rams, 0)
        self.assertLess(flops, CELL_BITS * 433 // 10, "a store in flip-flops would be 55,424 of them")

    def test_it_spends_no_logic_on_a_read_during_a_write(self):
        log = self.yosys("synth_ice40 -top checksum -run begin:map_ram\n"
                         "debug memory_libmap -lib +/ice40/brams.txt\n")
        self.assertIn("mapping memory checksum.cells via $__ICE40_RAM4K_", log)
        self.assertNotIn("emulate read-first", log)
        self.assertNotIn("emulate transparency", log)

    def test_xilinx_maps_it_to_block_ram_too(self):
        log = self.yosys("synth_xilinx -top checksum -noiopad\nstat\n")
        self.assertRegex(log, r"\d+\s+RAMB(18|36)E1")

    def test_a_state_uses_one_port_of_the_cell_store(self):
        """What yosys proves, on the text: the case that drives the ports."""
        text = translate_file(SOURCE).verilog
        block = text.partition("always @* begin")[2].partition("endcase")[0]
        arms = re.findall(r"(\w+): begin[^\n]*\n(.*?)\n\s*end", block, re.S)
        self.assertGreater(len(arms), 5)
        for name, body in arms:
            if name != "IDLE":
                self.assertFalse("a_en" in body and "b_we" in body, name)
        self.assertIn("a_en = host_en && host_we == 16'd0;", block)

    def test_a_field_that_stays_has_no_write_enable(self):
        """RTL-3, and the first specification's Theorem 6.1. Eight fields say `=`, and one is stored."""
        text = translate_file(SOURCE).verilog
        arm = re.search(r"build r2, c6, none, =, =, =, =, =, =, =, =, r3\n(.*?)\n\s*end", text, re.S).group(1)
        self.assertIn("b_we = 16'hff00;", arm)
        self.assertEqual(arm.count("b_din"), 1)


if __name__ == "__main__":
    unittest.main()
