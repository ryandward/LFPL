"""Slots, and a Fin counted down: LFPL-O sections 4.12 and 4.6, and what rtcheck and the Verilog backend make of them.

examples/udp_verify.lfpl is what they were added for. It checks the UDP checksum of every datagram over the whole of its
payload, which is a slot the frame owns, read a byte at a time at a Fin that counts down. These tests hold each piece to
its rule on its own, then compile that program, certify its bound, and run its handler's Verilog on slots built directly
in memory, against a model of what it has to decide that is written from RFC 768 and not from the program.
"""
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

from src.certify import certify  # noqa: E402
from src.lfplo import Refused, build  # noqa: E402
from src.ring import Reject  # noqa: E402
from src.rtcheck import Checker, RTError, Unbounded, check_stream  # noqa: E402
from src.rtlower import rt_lines  # noqa: E402
from src.runtime import Options  # noqa: E402
from src.rtverilog import CELL_BYTES, translate  # noqa: E402
from src.symcheck import check_translation  # noqa: E402
from src.x86 import Backend  # noqa: E402
from tests.reference import ipv4_model  # noqa: E402
from tests.test_forward import Forwarder  # noqa: E402
from tests.test_rtverilog import SIMULATOR, TOOLS, Layout, image_lines, run_images  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UDP_VERIFY = os.path.join(ROOT, "examples", "udp_verify.lfpl")
SLOT = """slot Bytes;
"""


def rt(source: str) -> list[str]:
    return list(rt_lines(io.BytesIO(source.encode())))


def bound_of(source: str, handler: str = "h", n_max: int = 8):
    checker = check_stream(rt(source))
    return certify(checker, handler, n_max)


class TheLanguage(unittest.TestCase):
    """A slot is a linear type that owns a slot's bytes: it is used exactly once, as every linear value is, and the two
    things that read it, a byte and its length, borrow it and leave it where it was."""

    def refused(self, source: str, message: str) -> None:
        with self.assertRaisesRegex(Reject, message):
            rt(source)

    def test_slot_is_a_word_only_where_a_declaration_starts(self):
        """src/turn.lfpl names a parameter slot, and so may any program."""
        rt(SLOT + "fn f(slot: Int8) : Int8 = +(slot, 1);\n")

    def test_one_never_used_is_refused(self):
        self.refused(SLOT + "fn lose(b: Bytes) : Int8 = 0;\n", "linear variable b is never used")

    def test_and_so_is_one_used_twice(self):
        self.refused(SLOT + "fn twice(b: Bytes) : (Bytes, Bytes) = (b, b);\n", "used twice")

    def test_a_byte_read_borrows_it(self):
        rt(SLOT + "fn first(b: Bytes) : (Bytes, Int8) = let x = b[idx(0, Bytes)] in (b, x);\n")

    def test_and_so_does_its_length(self):
        rt(SLOT + "fn size(b: Bytes) : (Bytes, Int16) = let n = len(b) in (b, n);\n")

    def test_so_a_read_alone_is_not_its_use(self):
        self.refused(SLOT + "fn lose(b: Bytes) : Int8 = b[idx(0, Bytes)];\n", "linear variable b is never used")
        self.refused(SLOT + "fn lose(b: Bytes) : Int16 = len(b);\n", "linear variable b is never used")

    def test_a_read_after_it_was_used_is_refused(self):
        self.refused(SLOT + "fn late(b: Bytes) : (Bytes, Int8) = let c = b in (c, b[idx(0, Bytes)]);\n",
                     "slot b is read after it was used")

    def test_a_byte_is_read_at_a_fin_of_the_slots_size(self):
        self.refused(SLOT + "fn at(b: Bytes, i: Fin 16) : (Bytes, Int8) = let x = b[i] in (b, x);\n",
                     "read at a Fin 2048, its size, and this is Fin 16")
        self.refused(SLOT + "fn at(b: Bytes, i: Int16) : (Bytes, Int8) = let x = b[i] in (b, x);\n",
                     "read at a Fin 2048, its size, and this is Int16")

    def test_only_a_slot_has_bytes_to_read(self):
        self.refused("record R { a: Int64 };\nfn at(x: R, i: Fin 2048) : Int8 = x[i];\n",
                     "a byte read reads a slot, and x is R")
        self.refused("record R { a: Int64 };\nfn size(x: R) : Int16 = len(x);\n", "len reads a slot, and x is R")

    def test_len_names_one_slot(self):
        self.refused(SLOT + "fn size(b: Bytes, c: Bytes) : Int16 = len(b, c);\n", "len takes one slot")
        self.refused(SLOT + "fn size() : Int16 = len(3);\n", "len takes one slot, by its name")
        self.refused(SLOT + "fn me(b: Bytes) : Bytes = b;\nfn size(b: Bytes) : Int16 = len(me(b));\n",
                     "len names a slot")

    def test_nothing_takes_one_apart(self):
        self.refused(SLOT + "fn m(b: Bytes) : Bytes = match b with End => b end;\n", "a slot is bytes")

    def test_nor_makes_one(self):
        self.refused(SLOT + "fn make(d: Diamond) : Bytes = Bytes@d();\n", "unknown constructor Bytes")
        self.refused(SLOT + "fn make() : Bytes = Bytes;\n", "Bytes is not a null constructor")

    def test_a_record_holds_one(self):
        lines = rt(SLOT + "record R { a: Int8, body: Bytes };\n"
                   "fn swap(r: R) : R = match r with R(d, a, body) => R@d(+(a, 1), body) end;\n")
        slot = next(line.split()[1] for line in lines if line.endswith(" slot Bytes 2048"))
        self.assertTrue(any(re.fullmatch(rf"field c\d+ 8 8 {slot} body", line) for line in lines), lines)

    def test_a_fin_bounded_by_a_slot_is_a_fin_of_its_size(self):
        rt(SLOT + "fn same(i: Fin Bytes) : Fin 2048 = i;\nfn made(x: Int16) : Fin 2048 = idx(x, Bytes);\n")

    def test_a_bound_by_name_names_a_slot(self):
        self.refused("record R { a: Int8 };\nfn f(i: Fin R) : Int8 = 0;\n", "Fin R: a bound by name is a slot's size")
        self.refused("record R { a: Int8 };\nfn f(x: Int8) : Fin 16 = idx(x, R);\n", "idx by R: a bound by name")


COUNT = SLOT + """fn walk(b: Bytes, i: Fin Bytes, acc: Int64) : (Bytes, Int64) =
  match i with
  | 0 => (b, acc)
  | down(j) => let x = b[j] in walk(b, j, +(acc, pack<8>(x)))
  end;
"""


class TheCountdown(unittest.TestCase):
    """`| 0 =>` and `| down(j) =>`, with j one less, are the arms of a match on a Fin: and rule R3's other measure is a
    Fin parameter that every recursive call passes one less of, whatever the linear arguments do."""

    def refused(self, source: str, message: str) -> None:
        with self.assertRaisesRegex(Reject, message):
            rt(source)

    def test_it_terminates_by_r3_with_every_linear_argument_passed_on_as_it_was(self):
        lines = rt(COUNT)
        self.assertIn("end decreasing 1", lines)
        self.assertTrue(any(re.fullmatch(r"    br\.down r1, r\d+, L\d+, L\d+", line) for line in lines), lines)

    def test_a_read_comes_before_the_slot_is_passed_on(self):
        """Arguments are evaluated in the order they are written, and the first has used the slot."""
        self.refused(COUNT.replace("let x = b[j] in walk(b, j, +(acc, pack<8>(x)))", "walk(b, j, +(acc, pack<8>(b[j])))"),
                     "slot b is read after it was used")

    def test_a_fin_passed_on_as_it_is_shows_nothing(self):
        self.refused(COUNT.replace("walk(b, j,", "walk(b, i,"), "rule R3")

    def test_nor_does_one_less_of_another_parameter(self):
        self.refused("fn w(i: Fin 8, k: Fin 8) : Int8 = match i with | 0 => 0 | down(j) => w(k, j) end;\n", "rule R3")

    def test_its_arms_come_in_either_order(self):
        rt("fn w(i: Fin 8) : Int8 = match i with | down(j) => w(j) | 0 => 0 end;\n")

    def test_two_less_is_less(self):
        rt("fn w(i: Fin 8) : Int8 = match i with | 0 => 0 | down(j) =>\n"
           "  match j with | 0 => 1 | down(k) => w(k) end end;\n")

    def test_and_a_let_is_a_renaming(self):
        rt("fn w(i: Fin 8) : Int8 = match i with | 0 => 0 | down(j) => let k = j in w(k) end;\n")

    def test_but_a_value_made_since_is_not_known_to_be_less(self):
        self.refused("fn w(i: Fin 8) : Int8 = match i with | 0 => 0 | down(j) => w(idx(pack<8>(j), 8)) end;\n",
                     "rule R3")

    def test_it_has_both_arms(self):
        self.refused("fn w(i: Fin 8) : Int8 = match i with | 0 => 0 end;\n", "non-exhaustive match on Fin 8")
        self.refused("fn w(i: Fin 8) : Int8 = match i with | 0 => 0 | 0 => 1 end;\n", "duplicate match arm")
        self.refused("fn w(i: Fin 8) : Int8 = match i with | 0 => 0 | Nil => 1 end;\n",
                     "two arms, 0 and down\\(j\\), and Nil is neither")
        self.refused("fn w(i: Fin 8) : Int8 = match i with | 1 => 0 end;\n", "0 or down\\(j\\) for a Fin")
        self.refused("fn w(i: Fin 8) : Int8 = match i with | 0 => 0 | down => 1 end;\n", "pattern needs 1 binders")

    def test_a_fin_becomes_an_integer_by_pack_if_every_value_fits(self):
        rt("fn n(i: Fin 2048) : Int64 = pack<16>(i);\nfn m(i: Fin 256) : Int64 = pack<8>(i);\n")
        self.refused("fn n(i: Fin 2048) : Int64 = pack<8>(i);\n", "2047 does not fit 8 bits")


WALK = SLOT + """fn walk(b: Bytes, i: Fin Bytes, acc: Int64) : (Bytes, Int64) =
  let x = b[i] in match i with | 0 => (b, +(acc, pack<8>(x))) | down(j) => walk(b, j, +(acc, pack<8>(x))) end;
"""


class Within(unittest.TestCase):
    """`match within(s, x) with | inside(i) => ... | outside => ... end`: i is x, as a Fin below the slot's length, or
    x is not below it. It borrows the slot, and it is matched where it is written, since it is no value."""

    def refused(self, source: str, message: str) -> None:
        with self.assertRaisesRegex(Reject, message):
            rt(source)

    def test_it_makes_a_fin_below_the_length(self):
        lines = rt(WALK + """fn sum(b: Bytes, x: Int16) : (Bytes, Int64) =
  match within(b, x) with | inside(i) => walk(b, i, 0) | outside => (b, 0) end;
""")
        self.assertTrue(any(re.fullmatch(r"    br\.within r0, r\d+, r\d+, L\d+, L\d+", line) for line in lines), lines)

    def test_its_arms_come_in_either_order(self):
        rt(WALK + "fn sum(b: Bytes, x: Int16) : (Bytes, Int64) = match within(b, x) with | outside => (b, 0) "
                  "| inside(i) => walk(b, i, 0) end;\n")

    def test_it_is_matched_where_it_is_written(self):
        self.refused(SLOT + "fn f(b: Bytes) : Bytes = let w = within(b, 3) in b;\n", "matched where it is written")

    def test_it_names_a_slot_first(self):
        self.refused(SLOT + "fn f(x: Int16) : Int8 = match within(x, 3) with | inside(i) => 1 | outside => 0 end;\n",
                     "within reads a slot, and x is Int16")
        self.refused(SLOT + "fn me(b: Bytes) : Bytes = b;\nfn f(b: Bytes) : Bytes = match within(me(b), 3) with "
                            "| inside(i) => b | outside => b end;\n", "names a slot first")

    def test_it_has_both_arms_and_only_those(self):
        self.refused(SLOT + "fn f(b: Bytes) : Bytes = match within(b, 3) with | inside(i) => b end;\n",
                     "non-exhaustive match on within")
        self.refused(SLOT + "fn f(b: Bytes) : Bytes = match within(b, 3) with | inside(i) => b | down(j) => b end;\n",
                     "two arms, inside\\(i\\) and outside, and down is neither")
        self.refused(SLOT + "fn f(b: Bytes) : Bytes = match within(b, 3) with | outside => b | outside => b end;\n",
                     "duplicate match arm")

    def test_what_it_makes_is_the_slots_own_fin(self):
        self.refused(SLOT + "fn f(b: Bytes) : Bytes = match within(b, 3) with | inside(i) => let k = pack<8>(i) in b "
                            "| outside => b end;\n", "2047 does not fit 8 bits")


def counting(bound: int, uses: str = "count(i, a)", from_a_cell: bool = False) -> str:
    """A handler that counts a Fin i of `bound` down to 0, as `uses` says: one made by idx, or one read from its item."""
    field = f", i: Fin {bound}" if from_a_cell else ""
    binder = ", i" if from_a_cell else ""
    made = "" if from_a_cell else f"let i = idx(a, {bound}) in "
    return f"""record Frame {{ a: Int8{field} }};
enum Verdict = Drop | Send(port: Int8);
fn count(i: Fin {bound}, acc: Int8) : Int8 = match i with | 0 => acc | down(j) => count(j, +(acc, 1)) end;
fn both(i: Fin {bound}) : (Fin {bound}, Fin {bound}) = (i, i);
handler h(f: Frame, free: Free) : (Verdict, Frame, Free) =
  match f with Frame(d, a{binder}) => {made}(Send({uses}), Frame@d(a{binder}), free) end;
"""


class TheBound(unittest.TestCase):
    """A Fin carries an amount per unit of its value. idx pays for the most it can be, and each step down releases one
    unit's amount: so a countdown's work is bounded by where it starts, which is the most idx could make."""

    def test_its_work_is_its_starting_index(self):
        bounds = {size: bound_of(counting(size)) for size in (16, 32, 64)}
        for size, bound in bounds.items():
            self.assertEqual((bound.a, bound.b, bound.c), (0, 0, 0), size)
        step = (bounds[32].d - bounds[16].d) // 16
        self.assertGreater(step, 0)
        self.assertEqual(bounds[32].d - bounds[16].d, 16 * step, "each unit of the index is one step down")
        self.assertEqual(bounds[64].d - bounds[32].d, 32 * step)
        self.assertGreaterEqual(bounds[16].d, 15 * step, "it can start at 15")

    def test_one_counted_down_twice_is_paid_for_twice(self):
        """The share rule: what the first count takes, the register no longer has, and the second takes the rest.
        However the one Fin gets to the second count: passed again, copied into another register, or returned twice."""
        once = bound_of(counting(32))
        step = (once.d - bound_of(counting(16)).d) // 16
        for said, uses in (("passed twice", "count(i, count(i, a))"),
                           ("copied", "let k = if ==(a, 0) then i else i in count(k, count(i, a))"),
                           ("returned twice", "let (x, y) = both(i) in count(x, count(y, a))")):
            with self.subTest(said):
                lines = rt(counting(32, uses))
                if said == "copied":
                    self.assertTrue(any(line.strip().startswith("copy ") for line in lines), lines)
                twice = certify(check_stream(lines), "h", 8)
                self.assertGreaterEqual(twice.d - once.d, 31 * step)

    def test_a_fin_that_carries_nothing_cannot_pay_for_one(self):
        """One read from a cell carries nothing, and is still a number: so the countdown cannot be paid for."""
        with self.assertRaises(Unbounded):
            bound_of(counting(16, from_a_cell=True))
        bound_of(counting(16, "count(idx(pack<8>(i), 16), a)", from_a_cell=True))  # and one idx makes of it can

    def test_a_slot_carries_no_cells(self):
        """The item is its record's cells. The slot is not one of them, and walking it is paid for by the Fin."""
        bound = bound_of(SLOT + """record Frame { a: Int8, body: Bytes };
enum Verdict = Drop | Send(port: Int8);
fn walk(b: Bytes, i: Fin Bytes, acc: Int8) : (Bytes, Int8) =
  match i with | 0 => (b, acc) | down(j) => let x = b[j] in walk(b, j, +(acc, x)) end;
handler h(f: Frame, free: Free) : (Verdict, Frame, Free) =
  match f with Frame(d, a, body) => let (rest, sum) = walk(body, idx(a, Bytes), 0) in
    (Send(sum), Frame@d(a, rest), free) end;
""")
        self.assertEqual((bound.a, bound.b, bound.c), (0, 0, 0))
        self.assertGreater(bound.d, 2047, "the walk can start at 2047")

    def test_slots_are_counted_apart_from_cells(self):
        """An item holds the slots its type says, and a handler gives back each and draws no more: no rule makes one."""
        two = SLOT + """record Inner { b: Int8, second: Bytes };
record Frame { a: Int8, first: Bytes, inner: Inner };
enum Verdict = Drop | Send(port: Int8);
handler h(f: Frame, free: Free) : (Verdict, Frame, Free) =
  match f with Frame(d, a, first, inner) => (Send(a), Frame@d(a, first, inner), free) end;
"""
        self.assertEqual(check_stream(rt(two)).held_slots("h"), 2)
        self.assertEqual(check_stream(rt(counting(16))).held_slots("h"), 0)

    def walking(self, uses: str) -> object:
        return bound_of(WALK + """record Frame { a: Int8, b16: Int16, body: Bytes };
enum Verdict = Drop | Send(port: Int8);
handler h(f: Frame, free: Free) : (Verdict, Frame, Free) =
  match f with Frame(d, a, b16, body) => """ + uses + """ end;
""")

    def test_a_walk_from_within_is_paid_for_by_the_slots_bytes(self):
        """Below the length, so the walk is as long as the frame, and its bound grows with n, the item's cells and
        its slot's bytes. idx can only say it is below 2048, so its walk is charged as the longest there can be."""
        within = self.walking("""match within(body, b16) with
    | inside(i) => let (rest, s) = walk(body, i, 0) in (Send(low8(s)), Frame@d(a, b16, rest), free)
    | outside => (Drop, Frame@d(a, b16, body), free) end""")
        indexed = self.walking("""let (rest, s) = walk(body, idx(b16, Bytes), 0) in (Send(low8(s)), Frame@d(a, b16, rest), free)""")
        self.assertGreater(within.c, 0)
        self.assertEqual(indexed.c, 0)
        self.assertLess(within.c * (1 + 5) + within.d, indexed.d, "a short frame's bound is a short one")
        self.assertLess(within.c * (1 + 2048) + within.d, indexed.d + 2 * within.c, "and the longest is the same")

    def test_a_slot_walked_twice_is_paid_for_twice(self):
        """What the first within takes of the slot's amount, the second cannot."""
        once = self.walking("""match within(body, b16) with
    | inside(i) => let (rest, s) = walk(body, i, 0) in (Send(low8(s)), Frame@d(a, b16, rest), free)
    | outside => (Drop, Frame@d(a, b16, body), free) end""")
        twice = self.walking("""match within(body, b16) with
    | inside(i) => let (rest, s) = walk(body, i, 0) in
        match within(rest, b16) with
        | inside(j) => let (again, t) = walk(rest, j, s) in (Send(low8(t)), Frame@d(a, b16, again), free)
        | outside => (Drop, Frame@d(a, b16, rest), free) end
    | outside => (Drop, Frame@d(a, b16, body), free) end""")
        self.assertGreaterEqual(twice.c, 2 * once.c - 1)

    def test_the_udp_check(self):
        with open(UDP_VERIFY, "rb") as handle:
            checker = check_stream(rt_lines(handle))
        bound = certify(checker, "udp_verify", 217)
        self.assertEqual((bound.a, bound.b), (0, 0), "nothing drawn")
        self.assertEqual(checker.held_slots("udp_verify"), 1, "slots are counted apart: the item's one")
        self.assertEqual(checker.record_cells("udp_verify"), 6)
        # n is the record's six cells and the slot's bytes, so a short frame's bound is a short one.
        self.assertGreaterEqual(bound.c, 4, "a read, a widening, an add and a step down for each byte it sums")
        self.assertLess(bound.c * (6 + 5) + bound.d, bound.c * (6 + 1472) + bound.d)
        self.assertLess(bound.c * (6 + 1472) + bound.d, bound.c * 2047, "and the longest frame's is below a walk of 2047")


class InRT(unittest.TestCase):
    """rtcheck's own rules, on streams it is handed, whoever wrote them: nothing makes a slot, or loses one."""

    DECLARED = """type t16 slot Bytes 2048
fin t49152 2048
"""

    def refused(self, body: str, message: str) -> None:
        with self.assertRaisesRegex(RTError, message):
            check_stream((self.DECLARED + body).splitlines())

    def test_a_slot_is_not_made_from_a_literal(self):
        self.refused("fn f0 f() : t16\n    const r0, #0 : t16\n    ret r0\nend\n", "only a tagless type has a literal")

    def test_nor_dropped(self):
        self.refused("fn f0 f(r0 : t16) : i8\n    drop r0\n    const r1, #0 : lit\n    ret r1\nend\n",
                     "only a scalar can be dropped")

    def test_nor_copied(self):
        self.refused("fn f0 f(r0 : t16) : t16\n    copy r1, r0\n    ret r1\nend\n", "only a scalar can be copied")

    def test_nor_left_behind(self):
        self.refused("fn f0 f(r0 : t16) : i8\n    const r1, #0 : lit\n    ret r1\nend\n", "linear value still in r0")

    def test_a_byte_of_a_slot_is_read_at_its_size(self):
        self.refused("fin t49153 16\nfn f0 f(r0 : t16, r1 : t49153) : i8\n    sget r2, r0, r1\nend\n",
                     "a slot of 2048 bytes is read at a Fin 2048")

    def test_and_only_of_a_slot(self):
        self.refused("fn f0 f(r0 : i64, r1 : t49152) : i8\n    sget r2, r0, r1\nend\n", "not a slot")

    def test_only_a_fin_counts_down(self):
        self.refused("fn f0 f(r0 : i64) : i8\n    br.down r0, r1, L1, L2\nend\n", "only a Fin counts down")

    def test_what_counts_down_is_empty_where_it_is_0(self):
        self.refused("fn f0 f(r0 : t49152) : t49152\n    br.down r0, r1, L1, L2\nL1:\n    ret r1\nL2:\n    ret r1\nend\n",
                     "r1 holds nothing")

    def test_a_decreasing_set_is_cells_or_one_fin(self):
        self.refused("""type t17 tagless L
ctor c18 t17 null Nil
ctor c19 t17 - Cons
field c19 8 8 t17 tail
fn f0 f(r0 : t17, r1 : t49152) : t17
    ret r0
end decreasing 0,1
""", "parameters with cells, whose amounts are per cell, or one Fin")

    def test_a_slots_size_is_a_power_of_two(self):
        with self.assertRaisesRegex(RTError, "a power of two"):
            check_stream(["type t16 slot Bytes 1500"])


class OnX86(unittest.TestCase):
    """Lowered, and checked: src/symcheck.py reads a slot, a countdown and within for itself, and every build passes
    it, so what each means in the source is what the assembly does."""

    def built(self, source: str) -> str:
        with tempfile.TemporaryDirectory() as workdir:
            path, exe = os.path.join(workdir, "p.lfpl"), os.path.join(workdir, "p")
            with open(path, "w") as handle:
                handle.write(source)
            report = str(build(path, exe, Options(in_port=9000, out_base=9100, ctl_port=9001)))
            self.assertTrue(os.path.exists(exe))
            return report

    def test_a_slot(self):
        with open(UDP_VERIFY) as handle:
            report = self.built(handle.read())
        self.assertRegex(report, r"R\(n\) = 0 cells, W\(n\) = \d+ n \+ \d+ steps")
        with open(os.path.join(ROOT, "README.md")) as handle:
            self.assertIn(report.split(", stack")[0], handle.read())

    def test_a_countdown(self):
        self.assertRegex(self.built(counting(16)), r"R\(n\) = 0 cells, W\(n\) = \d+ steps")

    def assembly(self, source: bytes) -> str:
        out: list = []
        backend = Backend(out.append)
        for line in rt_lines(io.BytesIO(source)):
            backend.feed(line)
        return "default abs\nsection .text\n" + "\n".join(out) + "\n"

    def test_the_symbolic_check_sees_each_break(self):
        """The assembly of examples/udp_verify.lfpl, broken by hand, each way a slot read, a countdown or within
        could be got wrong: every one disagrees with what the source means."""
        with open(UDP_VERIFY, "rb") as handle:
            source = handle.read()
        asm = self.assembly(source)
        self.assertEqual(check_translation(source, asm), [])
        breaks = {"a byte past the length kept": (r"cmovae", "cmovb"),
                  "the slot one smaller": (r"shl (\w+), 11", r"shl \1, 10"),
                  "one less is two less": (r"(mov (\w+), \w+\n\s+)sub \2, 1\n", "\\1sub \\2, 2\n"),
                  "within takes the length itself": (r"(cmp \w+, r11\n\s+)jb ", r"\1jbe "),
                  "the length is the number": (r"shr r11, 32", "shr r11, 0")}
        for name, (pattern, replacement) in breaks.items():
            with self.subTest(name):
                broken, count = re.subn(pattern, replacement, asm)
                self.assertGreater(count, 0)
                self.assertNotEqual(check_translation(source, broken), [])


class OnX86Live(unittest.TestCase):
    """examples/udp_verify.lfpl on live sockets: each frame's W is the bound for its own n, worked out by admission from
    its length, and each frame runs within it, so a short frame's bound is a short one."""

    def test_every_frame_within_its_own_bound(self):
        cases = TheMachine("test_every_verdict_is_the_models").cases()
        frames = [packet for _name, packet, length in cases if length is None]
        budgets = {}
        with tempfile.TemporaryDirectory() as workdir:
            router = Forwarder(bytes(1024), workdir, build_options=("--count-steps",), source=UDP_VERIFY)
            try:
                for socket in router.outputs:
                    socket.settimeout(0.05)
                before = router.stats()
                for number, packet in enumerate(frames, 1):
                    router.raw(packet)
                    now = router.settle(number)
                    for socket in router.outputs:
                        try:
                            socket.recv(4096)
                        except TimeoutError:
                            pass
                    steps, budget = now["steps"] - before["steps"], now["budget_steps"] - before["budget_steps"]
                    self.assertLessEqual(steps, budget, packet.hex()[:24])
                    budgets[len(packet)] = budget
                    before = now
                self.assertEqual((now["pool"], now["slots"]), (now["capacity"], now["slot_capacity"]))
            finally:
                router.close()
        self.assertLess(budgets[47], budgets[1514])
        report = re.search(r"W\(n\) = (\d+) n \+ (\d+) steps", router.report)
        c, d = int(report.group(1)), int(report.group(2))
        self.assertEqual(budgets[47], c * (6 + 47 - 42) + d, "n is the record's 6 cells and the slot's 5 bytes")

    def test_a_full_queue_is_full_and_not_short(self):
        """Every slot is ingress's or a queued output's, so once the queue is full no slot is free either. The frame
        after is refused as Full, as the machine's turn refuses it, and not as Short, and when egress goes, the four
        go out and every cell and slot is back."""
        good = frame(b"query")  # sent to port 0
        with tempfile.TemporaryDirectory() as workdir:
            router = Forwarder(bytes(1024), workdir, build_options=("--k-out", "4"), source=UDP_VERIFY)
            try:
                router.stats(b"P")
                for _ in range(6):
                    router.raw(good)
                paused = router.settle(6)
                self.assertEqual((paused["served"], paused["full"], paused["short"]), (4, 2, 0))
                self.assertEqual(paused["slots"], 0, "ingress holds one, and the queue the other four")
                router.stats(b"R")
                for _ in range(4):
                    self.assertEqual(len(router.outputs[0].recv(4096)), len(good))
                drained = router.stats()
                self.assertEqual((drained["pool"], drained["slots"]), (drained["capacity"], drained["slot_capacity"]))
            finally:
                router.close()


def ones(data: bytes) -> int:
    """RFC 768's sum: 16-bit words in network order, an odd byte at the end padded with a zero, carries folded in."""
    data = data + bytes(len(data) % 2)
    total = sum(int.from_bytes(data[at:at + 2], "big") for at in range(0, len(data), 2))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return total


def udp(source: bytes, destination: bytes, data: bytes, *, ports: tuple = (40000, 53), length: int = None,
        checksum: int = None) -> bytes:
    """A UDP datagram, its checksum computed as its sender would unless told otherwise."""
    length = 8 + len(data) if length is None else length
    header = ports[0].to_bytes(2, "big") + ports[1].to_bytes(2, "big") + length.to_bytes(2, "big")
    if checksum is None:
        pseudo = source + destination + bytes((0, 17)) + length.to_bytes(2, "big")
        checksum = ~ones(pseudo + header + bytes(2) + data) & 0xFFFF or 0xFFFF  # 0 says none was computed
    return header + checksum.to_bytes(2, "big") + data


def frame(data: bytes, *, destination: bytes = bytes((10, 0, 0, 1)), ttl: int = 64, protocol: int = 17,
          offset: int = 0x4000, padding: bytes = b"", **datagram) -> bytes:
    source = bytes((192, 168, 0, 1))
    return ipv4_model.ipv4_frame(destination, source, ttl=ttl, protocol=protocol, flags_and_offset=offset,
                                 payload=udp(source, destination, data, **datagram), padding=padding)


def expected(packet: bytes, slot_length: int) -> tuple:
    """What udp_verify has to decide: Malformed, Expired, Corrupt or a port, from RFC 791 and RFC 768."""
    def be(at: int, size: int) -> int:
        return int.from_bytes(packet[at:at + size], "big")
    total, frag, ttl, protocol, usum = be(16, 2), be(20, 2), packet[22], packet[23], be(40, 2)
    header_ok = (be(12, 2) == 0x0800 and packet[14] == 0x45 and total >= 20 and total + 14 <= len(packet)
                 and ones(packet[14:34]) == 0xFFFF)
    if not header_ok:
        return ("Malformed", None)
    if protocol == 17 and frag & 0x3FFF == 0 and usum != 0:
        ulen = be(38, 2)
        if ulen < 8 or ulen != total - 20 or slot_length < ulen - 8:
            return ("Malformed", None)
        pseudo = packet[26:34] + bytes((0, 17)) + packet[38:40]
        if ones(pseudo + packet[34:34 + ulen]) != 0xFFFF:
            return ("Corrupt", None)
    if ttl <= 1:
        return ("Expired", None)
    return ("Send", packet[30] >> 6)


@SIMULATOR
class TheMachine(unittest.TestCase):
    """udp_verify's handler as a circuit, each frame's payload in a slot of the store written there directly, as the
    other tests here load cells: every verdict, and what it sends, as the model says, within the certified clocks."""

    @classmethod
    def setUpClass(cls):
        with open(UDP_VERIFY, "rb") as handle:
            cls.lines = list(rt_lines(handle))
        cls.translation = translate(cls.lines, n_max=217, slots=4)
        cls.layout = Layout(cls.lines)

    def cases(self) -> list:
        rng = random.Random(768)
        hops = [bytes((top << 6 | 10, 0, 0, 1)) for top in range(4)]
        longest = bytes(rng.randrange(256) for _ in range(1472))
        good = frame(b"query")
        wrong_byte = bytearray(frame(bytes(range(40)))); wrong_byte[-1] ^= 0x01  # noqa: E702
        wrong_odd = bytearray(frame(b"abcde")); wrong_odd[-1] ^= 0x10  # noqa: E702
        bad_ip = bytearray(good); bad_ip[24] ^= 0xFF  # noqa: E702
        return [
            ("an odd payload, its checksum right", good, None),
            ("an even payload, its checksum right", frame(bytes(range(40)), destination=hops[1]), None),
            ("the last byte changed", bytes(wrong_byte), None),
            ("an odd last byte changed", bytes(wrong_odd), None),
            ("no checksum, and the bytes anything", frame(b"query", checksum=0), None),
            ("a later fragment, whose checksum covers bytes elsewhere",
             frame(b"query", checksum=0x1234, offset=0x0001), None),
            ("the UDP length longer than the IP length says", frame(b"query", length=15), None),
            ("the UDP length shorter than its own header", frame(b"query", length=7), None),
            ("an empty payload, its checksum right", frame(b"", destination=hops[2]), None),
            ("TCP, which this does not check", ipv4_model.ipv4_frame(protocol=6, payload=bytes(28)), None),
            ("the IP header's checksum wrong", bytes(bad_ip), None),
            ("TTL 1, the checksum right", frame(b"query", ttl=1), None),
            ("the longest payload, 1472 bytes", frame(longest, destination=hops[3]), None),
            ("Ethernet's padding after a short datagram", frame(b"hey", padding=bytes(15)), None),
            ("a slot that says it holds less than the datagram", frame(b"query"), 3),
            ("a slot one byte short of the datagram, so within is asked for the length itself", frame(b"query"), 4),
        ]

    def test_every_verdict_is_the_models(self):
        cases = self.cases()
        images, wanted = [], []
        translation, layout = self.translation, self.layout
        for number, (_name, packet, slot_length) in enumerate(cases):
            slot = translation.first_slot + number % translation.slots
            loads, item, free, words = layout.image(packet, 0, slot, slot_length)
            images += image_lines(item, free, words, loads)
            wanted.append(expected(packet, len(packet) - 42 if slot_length is None else slot_length))
        results, summary = run_images(translation, images)
        self.assertEqual((summary["both_ports"], summary["same_address"]), ("0", "0"))
        verdicts = {layout.ctors[ctor][0]: next(line.split()[-1] for line in self.lines if line.startswith(f"ctor {ctor} "))
                    for ctor in layout.by_type[layout.verdict_type]}
        for (name, packet, slot_length), result, (verdict, port), number in zip(cases, results, wanted, range(99)):
            with self.subTest(name):
                got = verdicts[str(result["verdict"] & 0xFF)]
                self.assertEqual(got, verdict)
                if verdict == "Send":
                    self.assertEqual(result["verdict"] >> 8 & 0xFF, port)
                # n is the record's cells and the slot's bytes
                n = layout.shape(layout.item)[1] + (len(packet) - 42 if slot_length is None else slot_length)
                self.assertLessEqual(result["cycles"], translation.timing.bound(n))
                fields = self.fields(result["words"], result["out"])
                slot = translation.first_slot + number % translation.slots
                length = len(packet) - 42 if slot_length is None else slot_length
                self.assertEqual(fields["body"], slot | length << 32, "the slot comes back as it went")
                self.assertEqual(fields["length"], len(packet))
                sent = self.packet_from(fields)
                if verdict == "Send":
                    out = bytearray(packet[:42])
                    out[0:6], out[6:12] = ipv4_model.macs_of(port)
                    out[22] = packet[22] - 1
                    out[24:26] = ipv4_model.incremental_checksum(
                        int.from_bytes(packet[24:26], "big"), int.from_bytes(packet[22:24], "big"),
                        int.from_bytes(bytes((packet[22] - 1, packet[23])), "big")).to_bytes(2, "big")
                    self.assertEqual(sent, bytes(out))
                else:
                    self.assertEqual(sent, packet[:42], "a frame not sent is not changed")

    def fields(self, words: dict, address: int) -> dict:
        """The record's fields across its cells, by name, as integers, following its links."""
        out, ctor = {}, self.layout.item
        while True:
            word = words[address]  # in hex, its last byte first: a byte nothing ever wrote is xx
            following = None
            for offset, size, type_text, name in self.layout.ctors[ctor][1]:
                value = int(word[len(word) - 2 * (offset + size):len(word) - 2 * offset], 16)
                if name == "$next":
                    following = (value, self.layout.by_type[self.layout._resolve(type_text)][0])
                else:
                    out[name] = (value, size, type_text)
            if following is None:
                return {name: value for name, (value, _size, _type) in out.items()} | {"_raw": out}
            address, ctor = following

    def packet_from(self, fields: dict) -> bytes:
        """The header bytes the record holds, in declaration order, which is the wire's."""
        raw = fields["_raw"]
        return b"".join(value.to_bytes(size, "little") for name, (value, size, type_text) in raw.items()
                        if name not in ("length", "body"))

    def test_a_slot_read_where_its_record_holds_it(self):
        """Read before it leaves its record, so it is taken out of the cell into a register as wide as a slot is,
        and put back when the record is: its length and first byte, which no field keeps, come back in the verdict,
        with a field of the record read after them, from its own cell and not the slot's the port holds by then."""
        with tempfile.TemporaryDirectory() as workdir:
            path = os.path.join(workdir, "peek.lfpl")
            with open(path, "w") as handle:
                handle.write(SLOT + """record Frame { a: Int8, length: Int16, body: Bytes };
enum Verdict = Drop | Send(port: Int8);
handler peek(f: Frame, free: Free) : (Verdict, Frame, Free) =
  match f with Frame(d, a, length, body) =>
    let n = len(body) in
    let first = body[idx(0, Bytes)] in
    (Send(^(^(low8(n), first), a)), Frame@d(a, length, body), free)
  end;
""")
            with open(path, "rb") as handle:
                lines = list(rt_lines(handle))
        self.assertTrue(any(line.strip().startswith("take ") for line in lines), "it leaves the cell to be read")
        translation = translate(lines, n_max=8, slots=2)
        layout = Layout(lines)
        # The last slot holds bytes an earlier frame left, and says it holds none: so its first byte is 0.
        images, cases, slots = [], ((b"\x07abc", None), (bytes(range(200, 256)) * 30, None), (b"\x55\x66", 0)), []
        for number, (payload, length) in enumerate(cases):
            slots.append(translation.first_slot + number % translation.slots)
            loads, item, free, words = layout.image(b"\x01" + payload, 0, slots[-1], length)
            images += image_lines(item, free, words, loads)
        results, _summary = run_images(translation, images)
        for result, (payload, length), slot in zip(results, cases, slots):
            length = len(payload) if length is None else length
            first = payload[0] if length else 0
            self.assertEqual(result["verdict"] >> 8 & 0xFF, (length & 0xFF) ^ first ^ 1)
            self.assertEqual(self.body_of(layout, result), slot | length << 32, "and it went back whole")

    @staticmethod
    def body_of(layout: Layout, result: dict) -> int:
        word = result["words"][result["out"]]
        offset, size = next((o, s) for o, s, t, name in layout.ctors[layout.item][1] if name == "body")
        return int(word[len(word) - 2 * (offset + size):len(word) - 2 * offset], 16)

    def test_a_store_with_no_slots_is_refused(self):
        with self.assertRaisesRegex(RTError, "own slots, and this store has none"):
            translate(self.lines, n_max=217)

    def test_the_slots_are_after_the_cells_aligned_to_their_size(self):
        translation = self.translation
        per_slot = 2048 // CELL_BYTES
        self.assertGreater(translation.first_slot * per_slot, translation.cells)
        self.assertLessEqual((translation.first_slot - 1) * per_slot, translation.cells)
        self.assertIn(f"reg [127:0] cells [0:{(translation.first_slot + 4) * per_slot - 1}];", translation.verilog)

    @TOOLS
    def test_verilator_has_nothing_to_say(self):
        with tempfile.TemporaryDirectory() as workdir:
            design = os.path.join(workdir, "udp_verify.v")
            with open(design, "w") as handle:
                handle.write(self.translation.verilog)
            done = subprocess.run(["verilator", "--lint-only", "-Wall", design], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stderr.strip(), "")


if __name__ == "__main__":
    unittest.main()
