"""What a handler is written with: hex literals, record update and a verdict held in a register. LFPL-O sections 4.1,
4.10 and 4.11.

Each is a claim about what a program means and what it costs, so the programs here run, the symbolic check has to
agree with each of them without running it, and what must stay in place is counted in the stream.
"""
import io
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lfplo import Refused, build, compile_program  # noqa: E402
from src.ring import Reject  # noqa: E402
from src.rtcheck import RTError, check_stream  # noqa: E402
from src.rtlower import compiler_ring, rt_lines  # noqa: E402
from src.runtime import Options  # noqa: E402
from src.symcheck import check_translation  # noqa: E402
from tests.test_spans import FIVE, PRE, READ_THREE, THREE, answer, body_of, main_with, stores, stream_of  # noqa: E402

ONE = "record P { a: Int64, b: Int64 };\n"
READ_ONE = """fn rd(x: P, free: Free) : Res = match x with P(d, a, b) => Done@d(+(*(a, 10), b), free) end;
"""


class HexLiterals(unittest.TestCase):
    def test_a_program_in_hex_is_the_program_in_decimal(self):
        written = "fn f(a: Int16, b: Int8) : Bool = if ==(a, 0x0800) then ==(b, 0x45) else ==(a, 0xFFFF);\n"
        decimal = "fn f(a: Int16, b: Int8) : Bool = if ==(a, 2048) then ==(b, 69) else ==(a, 65535);\n"
        self.assertEqual(stream_of(written, "f"), stream_of(decimal, "f"))

    def test_digits_come_in_either_case(self):
        self.assertEqual(self.value("0xABCDEF"), 0xABCDEF)
        self.assertEqual(self.value("0xabcdef"), 0xABCDEF)
        self.assertEqual(self.value("0xaBcDeF"), 0xABCDEF)

    def test_leading_zeros_change_nothing(self):
        self.assertEqual(self.value("0x0000000000000000000000000000002A"), 42)

    def test_the_type_is_decided_by_the_value_as_it_is_for_decimal(self):
        fits = "fn f(a: Int8) : Bool = ==(a, 0x00FF);\n"
        self.assertEqual(stream_of(fits, "f"), stream_of("fn f(a: Int8) : Bool = ==(a, 255);\n", "f"))
        with self.assertRaisesRegex(Reject, "does not fit Int8"):
            stream_of("fn f(a: Int8) : Bool = ==(a, 0x100);\n", "f")

    def test_the_largest_is_64_bits_and_one_more_digit_is_refused(self):
        self.assertEqual(self.value("0xFFFFFFFFFFFFFFFF"), 2 ** 64 - 1)
        with self.assertRaisesRegex(Reject, "integer exceeds 64 bits"):
            self.value("0x10000000000000000")

    def test_what_is_not_a_hex_literal_is_refused(self):
        for text in ("0x", "0xg", "0x1g", "0x_1", "0X10", "00x10", "1x10", "0x10x"):
            with self.subTest(text), self.assertRaisesRegex(Reject, "malformed number"):
                self.value(text)

    def test_it_can_end_the_file(self):
        with self.assertRaisesRegex(Reject, "malformed number"):
            stream_of("fn f() : Int64 = 0x", "f")

    def test_a_zero_alone_and_a_decimal_with_a_leading_zero_mean_what_they_did(self):
        self.assertEqual(self.value("0"), 0)
        self.assertEqual(self.value("0x0"), 0)
        self.assertEqual(self.value("007"), 7)

    def test_a_length_and_a_width_can_be_written_in_it(self):
        written = "state s: Array[0x10] Int8;\nfn f() : Int64 = pack<0x10, 0x20>(1, 2);\n"
        decimal = "state s: Array[16] Int8;\nfn f() : Int64 = pack<16, 32>(1, 2);\n"
        self.assertEqual(stream_of(written, "f"), stream_of(decimal, "f"))

    def test_the_symbolic_check_reads_it_on_its_own_and_agrees(self):
        self.assertEqual(answer(self, PRE + main_with(1, "Done@c0(+(0xFF, 0x0100), REST)")), 511)

    @staticmethod
    def value(text: str) -> int:
        """The literal, as a program with no other source of a number sees it."""
        lines = stream_of(f"fn f() : Int64 = {text};\n", "f")
        constant = next(line for line in lines if "const" in line)
        return int(constant.split("#")[-1].split()[0]) if "#" in constant else -1


def refused(test: unittest.TestCase, source: str, message: str) -> None:
    with test.assertRaisesRegex(Reject, message):
        compile_program(source.encode())


class WhatItLowersTo(unittest.TestCase):
    """It says the same thing as the constructor with every field written out, so it has to cost the same, to the line."""

    def same(self, declarations: str, pattern: str, update: str, written: str) -> None:
        head = f"fn f(x: {pattern.split('(')[0]}, v: Int64) : {pattern.split('(')[0]} = match x with {pattern} => "
        self.assertEqual(stream_of(f"{declarations}{head}{update} end;\n", "f"),
                         stream_of(f"{declarations}{head}{written} end;\n", "f"))

    def test_a_record_of_one_cell(self):
        self.same(ONE, "P(d, a, b)", "P@d { b = +(b, v) }", "P@d(a, +(b, v))")

    def test_a_record_of_two_cells(self):
        self.same(THREE, "R(d, a, b, c)", "R@d { b = +(b, v) }", "R@d(a, +(b, v), c)")

    def test_a_field_in_the_first_cell_and_one_in_the_last(self):
        names = "a b c e g h i j k l m n o p q r s".split()
        binders = ", ".join(names)
        written = ", ".join({"a": "low32(+(pack<32>(a), v))", "s": "+(s, v)"}.get(name, name) for name in names)
        self.same(FIVE, f"F(d, {binders})", "F@d { a = low32(+(pack<32>(a), v)), s = +(s, v) }", f"F@d({written})")

    def test_every_field_named(self):
        self.same(THREE, "R(d, a, b, c)", "R@d { a = v, b = v, c = v }", "R@d(v, v, v)")

    def test_the_first_field_named_and_the_last(self):
        self.same(THREE, "R(d, a, b, c)", "R@d { a = v, c = v }", "R@d(v, b, v)")

    def test_no_field_named(self):
        self.same(ONE, "P(d, a, b)", "P@d {}", "P@d(a, b)")
        self.same(THREE, "R(d, a, b, c)", "R@d {}", "R@d(a, b, c)")

    def test_a_field_moved_is_a_field_named(self):
        self.same(THREE, "R(d, a, b, c)", "R@d { a = c, c = a }", "R@d(c, b, a)")

    def test_a_tail_call_in_a_field_is_the_same_tail_call(self):
        declarations = """data L = Nil | Cons(v: Int64, t: L);
record N { x: Int64, y: Int64, t: L };
fn f(l: L) : L = match l with Nil => Nil | Cons(d, v, t) => Cons@d(+(v, 1), f(t)) end;
"""
        update = """fn g(n: N) : N = match n with N(d, x, y, t) => N@d { y = +(y, 1), t = f(t) } end;\n"""
        written = """fn g(n: N) : N = match n with N(d, x, y, t) => N@d(x, +(y, 1), f(t)) end;\n"""
        self.assertEqual(stream_of(declarations + update, "g"), stream_of(declarations + written, "g"))

    def test_taken_apart_at_once_it_is_never_built(self):
        pattern = "match x with P(d, a, b) => match CTOR with P(e, p, q) => P@e(+(p, q), v) end end;\n"
        head = ONE + "fn f(x: P, v: Int64) : P = "
        self.assertEqual(stream_of(head + pattern.replace("CTOR", "P@d { a = 1 }"), "f"),
                         stream_of(head + pattern.replace("CTOR", "P@d(1, b)"), "f"))


class WhatItCosts(unittest.TestCase):
    def test_a_record_with_nothing_named_stores_nothing_in_any_cell(self):
        for declarations, pattern, ty in ((ONE, "P(d, a, b)", "P"), (THREE, "R(d, a, b, c)", "R")):
            source = declarations + f"fn f(x: {ty}) : {ty} = match x with {pattern} => {ty}@d {{}} end;\n"
            self.assertEqual(stores(body_of(source, "f")), [])

    def test_only_what_is_named_is_stored_across_five_cells(self):
        names = "a b c e g h i j k l m n o p q r s".split()
        source = FIVE + (f"fn f(x: F, v: Int8) : F = match x with F(d, {', '.join(names)}) => "
                         "F@d { h = v, s = 1 } end;\n")
        self.assertEqual(len(stores(body_of(source, "f"))), 2, body_of(source, "f"))

    def test_a_change_to_a_field_of_its_own_value_is_in_place(self):
        source = THREE + "fn f(x: R) : R = match x with R(d, a, b, c) => R@d { c = +(c, 1) } end;\n"
        changed = stores(body_of(source, "f"))
        self.assertEqual(len(changed), 1, changed)
        self.assertRegex(changed[0], r"^add qword \[\w+\+8\], 1$")


class WhatItMeans(unittest.TestCase):
    def test_a_field_named_and_a_field_left(self):
        source = PRE + ONE + READ_ONE + """fn u(x: P) : P = match x with P(d, a, b) => P@d { b = +(b, 7) } end;
""" + main_with(1, "rd(u(P@c0(1, 2)), REST)")
        self.assertEqual(answer(self, source), 19)

    def test_across_the_cells_of_a_record_that_spans(self):
        source = PRE + THREE + READ_THREE + """fn u(x: R) : R = match x with R(d, a, b, c) => R@d { b = +(c, 1) } end;
""" + main_with(2, "rd(u(R@(c0, c1)(1, 2, 3)), REST)")
        self.assertEqual(answer(self, source), 143)

    def test_a_name_bound_later_does_not_take_a_field_left_where_it_is(self):
        """The field stays where it is. What is called `a` by the time the record is written has nothing to say about it."""
        source = PRE + ONE + READ_ONE + """fn u(x: P) : P = match x with P(d, a, b) => let a = +(a, 8) in P@d { b = a } end;
""" + main_with(1, "rd(u(P@c0(1, 2)), REST)")
        self.assertEqual(answer(self, source), 19)

    def test_a_pattern_inside_it_does_not_either(self):
        source = PRE + ONE + READ_ONE + """record Box { a: Int64, b: Int64 };
fn u(x: P, y: Box, free: Free) : (P, Free) = match x with P(d, a, b) => match y with Box(e, a, b) =>
  (P@d { b = +(a, b) }, More@e(free)) end end;
""" + main_with(2, "let (p, f) = u(P@c0(1, 2), Box@c1(3, 4), REST) in rd(p, f)")
        self.assertEqual(answer(self, source), 17)

    def test_across_a_span_with_every_name_bound_again(self):
        source = PRE + THREE + READ_THREE + """fn u(x: R) : R = match x with R(d, a, b, c) =>
  let a = 7 in let b = 8 in let c = 9 in R@d { b = +(+(a, b), c) } end;
""" + main_with(2, "rd(u(R@(c0, c1)(1, 2, 3)), REST)")
        self.assertEqual(answer(self, source), 100 + 240 + 3)

    def test_a_field_read_again_after_the_record_is_written(self):
        source = PRE + THREE + """fn f(x: R, free: Free) : Res = match x with R(d, a, b, c) =>
  match R@d { a = 9 } with R(e, p, q, s) => let (g, rest) = peel(e) in Done@g(+(*(a, 100), +(*(p, 10), +(q, s))), More@rest(free)) end end;
""" + main_with(2, "f(R@(c0, c1)(1, 2, 3), REST)")
        self.assertEqual(answer(self, source), 100 + 90 + 5)

    def test_chosen_in_the_arms_of_an_if(self):
        source = PRE + THREE + READ_THREE + """fn f(x: R, p: Bool) : R = match x with R(d, a, b, c) =>
  if p then R@d {} else R@d { a = 5, c = 6 } end;
""" + main_with(2, "rd(f(R@(c0, c1)(1, 2, 3), COND), REST)")
        self.assertEqual(answer(self, source.replace("COND", "true")), 123)
        self.assertEqual(answer(self, source.replace("COND", "false")), 526)

    def test_a_field_in_network_order_left_and_one_named(self):
        source = PRE + """record W { a: Be16, b: Be32, c: Int64, e: Be64 };
fn u(x: W) : W = match x with W(d, a, b, c, e) => W@d { b = low32(+(pack<32>(b), 1)) } end;
fn rd(x: W, free: Free) : Res = match x with W(d, a, b, c, e) =>
  let (g, rest) = peel(d) in Done@g(+(+(pack<16>(a), pack<32>(b)), +(c, e)), More@rest(free)) end;
""" + main_with(2, "rd(u(W@(c0, c1)(1, 2, 3, 4)), REST)")
        self.assertEqual(answer(self, source), 1 + 3 + 3 + 4)

    def test_a_linear_field_left_where_it_is(self):
        source = PRE + """data List = Nil | Cons(h: Int64);
record T { l: List, a: Int64, b: Int64, m: List };
fn u(x: T) : T = match x with T(d, l, a, b, m) => T@d { b = +(a, b) } end;
fn head(l: List, free: Free) : (Int64, Free) = match l with Nil => (0, free) | Cons(d, h) => (h, More@d(free)) end;
fn rd(x: T, free: Free) : Res = match x with T(d, l, a, b, m) => let (p, f1) = head(l, free) in let (q, f2) = head(m, f1) in
  let (g, rest) = peel(d) in Done@g(+(*(p, 1000), +(*(a, 100), +(*(b, 10), q))), More@rest(f2)) end;
""" + main_with(5, "rd(u(T@(c0, c1, c2)(Cons@c3(7), 2, 3, Cons@c4(9))), REST)")
        self.assertEqual(answer(self, source), 7000 + 200 + 50 + 9)

    def test_a_linear_field_named_after_what_it_held_was_used(self):
        source = PRE + """data List = Nil | Cons(h: Int64);
record T { l: List, a: Int64 };
fn take(l: List, free: Free) : (Int64, Free) = match l with Nil => (0, free) | Cons(d, h) => (h, More@d(free)) end;
fn u(x: T, free: Free) : (T, Int64, Free) = match x with T(d, l, a) =>
  let (v, f) = take(l, free) in (T@d { l = Nil, a = +(a, 1) }, v, f) end;
fn rd(x: T, v: Int64, free: Free) : Res = match x with T(d, l, a) =>
  let (u, f) = take(l, free) in Done@d(+(*(a, 100), +(u, v)), f) end;
""" + main_with(2, "let (x, v, f) = u(T@c0(Cons@c1(5), 1), REST) in rd(x, v, f)")
        self.assertEqual(answer(self, source), 205)


class RefusedWithAReason(unittest.TestCase):
    OK = ONE + "fn f(x: P) : P = match x with P(d, a, b) => P@d { a = 1 } end;\n"

    def test_the_one_this_family_of_tests_starts_from_compiles(self):
        compile_program(self.OK.encode())

    def test_a_constructor_said_twice(self):
        refused(self, ONE + "data L = P(v: Int64) | Q;\n", "duplicate constructor name P")
        refused(self, "data L = P(v: Int64) | Q;\n" + ONE, "duplicate constructor name P")
        refused(self, "data L = P(v: Int64) | Q;\ndata M = Z | Q;\n", "duplicate constructor name Q")

    def test_a_program_can_still_spell_out_the_built_in_free(self):
        compile_program(PRE.encode() + b"fn f(x: Free) : Free = x;\n")

    def test_or_give_its_constructors_to_a_type_of_its_own(self):
        compile_program(b"data Fuel = End | More(next: Fuel);\nfn f(x: Fuel) : Fuel = x;\n")

    def test_but_not_to_a_record(self):
        refused(self, "record End { a: Int64 };\n", "duplicate constructor name End")

    def test_no_such_field(self):
        refused(self, ONE + "fn f(x: P) : P = match x with P(d, a, b) => P@d { z = 1 } end;\n", "no field z in P")

    def test_a_field_named_twice(self):
        refused(self, ONE + "fn f(x: P) : P = match x with P(d, a, b) => P@d { a = 1, a = 2 } end;\n",
                "a is named twice")

    def test_a_field_out_of_order(self):
        refused(self, ONE + "fn f(x: P) : P = match x with P(d, a, b) => P@d { b = 1, a = 2 } end;\n",
                "the order P declares them")

    def test_a_field_out_of_order_across_cells(self):
        refused(self, THREE + "fn f(x: R) : R = match x with R(d, a, b, c) => R@d { c = 1, a = 2 } end;\n",
                "the order R declares them")

    def test_what_is_not_a_record(self):
        refused(self, "data L = N | C(v: Int64);\nfn f(x: L) : L = match x with N => N | C(d, v) => C@d { v = 1 } end;\n",
                "C is not a record")

    def test_a_cell_from_the_reserve(self):
        refused(self, ONE + "fn f(free: Free) : Free = match free with End => End | More(c, r) => "
                            "match P@c { a = 1 } with P(e, x, y) => More@e(r) end end;\n",
                "c is not the cell of a P")

    def test_a_cell_of_another_record(self):
        refused(self, ONE + "record Q { a: Int64, b: Int64 };\n"
                            "fn f(x: Q) : P = match x with Q(d, a, b) => P@d { a = 1 } end;\n",
                "d is not the cell of a P")

    def test_a_cell_handed_to_a_function(self):
        refused(self, ONE + "fn f(c: Diamond) : P = P@c { a = 1 };\n", "c is not the cell of a P")

    def test_a_cell_under_another_name(self):
        refused(self, ONE + "fn f(x: P) : P = match x with P(d, a, b) => let e = d in P@e { a = 1 } end;\n",
                "e is not the cell of a P")

    def test_a_name_bound_again(self):
        refused(self, ONE + "fn f(x: P, y: Diamond) : P = match x with P(d, a, b) => let d = y in P@d { a = 1 } end;\n",
                "d is not the cell of a P")

    def test_the_cell_peel_takes_off(self):
        refused(self, ONE + THREE + "fn f(x: R) : P = match x with R(d, a, b, c) => "
                                    "let (e, rest) = peel(d) in P@e { a = 1 } end;\n", "e is not the cell of a P")

    def test_what_peel_leaves(self):
        refused(self, """record V { a: Int64, b: Int64, c: Int64, e: Int64 };
record W { a: Int64, b: Int64, c: Int64, e: Int64, g: Int64 };
fn f(x: W) : V = match x with W(d, a, b, c, e, g) => let (h, rest) = peel(d) in V@rest { a = 1 } end;
""", "rest is not the cell of a V")

    def test_cells_named_one_by_one(self):
        refused(self, THREE + "fn f(x: R, y: Diamond, z: Diamond) : R = match x with R(d, a, b, c) => "
                              "R@(y, z) { a = 1 } end;\n", "the cells of a pattern")

    def test_the_cell_used_already(self):
        refused(self, ONE + "fn g(c: Diamond, free: Free) : Free = More@c(free);\n"
                            "fn f(x: P, free: Free) : (P, Free) = match x with P(d, a, b) => "
                            "let r = g(d, free) in (P@d { a = 1 }, r) end;\n", "d used twice")

    def test_a_linear_field_gone_and_not_named(self):
        refused(self, PRE + """data List = Nil | Cons(h: Int64);
record T { l: List, a: Int64 };
fn take(l: List, free: Free) : Free = match l with Nil => free | Cons(d, h) => More@d(free) end;
fn u(x: T, free: Free) : (T, Free) = match x with T(d, l, a) => let r = take(l, free) in (T@d { a = 1 }, r) end;
""", "l used twice")

    def test_a_linear_field_replaced_and_never_used(self):
        refused(self, PRE + """data List = Nil | Cons(h: Int64);
record T { l: List, a: Int64 };
fn u(x: T) : T = match x with T(d, l, a) => T@d { l = Nil } end;
""", "l is never used")

    def test_a_value_of_the_wrong_type(self):
        refused(self, ONE + "fn f(x: P) : P = match x with P(d, a, b) => P@d { a = true } end;\n", "type mismatch")

    def test_what_is_not_a_field_and_a_value(self):
        for update in ("P@d { a }", "P@d { = 1 }", "P@d { a = 1, }", "P@d { a = 1 b = 2 }", "P@d { 1 = 1 }", "P@d { a = }"):
            with self.subTest(update), self.assertRaises(Reject):
                compile_program((ONE + f"fn f(x: P) : P = match x with P(d, a, b) => {update} end;\n").encode())


class WhatTheCompilerKeepsForIt(unittest.TestCase):
    """Every record's fields are in a table now, so that a field can be named. It is one more table in Theorem 7.2."""

    def test_the_table_is_never_bigger_than_G(self):
        fields = ", ".join(f"f{i}: Int64" for i in range(64))
        declarations = "".join(f"record R{i} {{ a: Int8 }};\n" for i in range(40)) + f"record W {{ {fields} }};\n"
        ring = compiler_ring(declarations.encode(), lambda tag, payload: None)
        ring.run()
        self.assertLessEqual(ring.stages[2].table_cells() - len(ring.stages[2].T.cells), len(ring.stages[3].G))

    def test_only_the_tables_stay_out_of_the_pool(self):
        source = PRE + THREE + READ_THREE + """fn u(x: R) : R = match x with R(d, a, b, c) => R@d { b = 7 } end;
""" + main_with(2, "rd(u(R@(c0, c1)(1, 2, 3)), REST)")
        ring = compiler_ring(source.encode(), lambda tag, payload: None, phase1=True)
        ring.run()
        passed, check = ring.stages[2], ring.stages[3]
        tables = len(check.G.cells) + len(check.TA.cells) + len(check.TF.cells) + len(check.TT.cells)
        self.assertEqual(ring.store.C - len(ring.F), tables + passed.table_cells())


VERDICT = "enum Verdict = Malformed | Expired | Send(port: Int8);\n"
FRAME = """data Bytes;
record Frame { a: Be16, ttl: Int8, length: Int16, body: Bytes };
"""
ONE_LINE = VERDICT + FRAME + """handler h(f: Frame, free: Free) : (Verdict, Frame, Free) =
  match f with Frame(d, a, ttl, length, body) =>
    if ==(a, 0) then (Malformed, Frame@d {}, free)
    else if <=(ttl, 1) then (Expired, Frame@d {}, free)
    else (Send(low8(a)), Frame@d { ttl = -(ttl, 1) }, free)
  end;
"""


def rt(source: str) -> list[str]:
    return list(rt_lines(io.BytesIO(source.encode())))


def report(test: unittest.TestCase, source: str, n_max: int = 8) -> str:
    """The build a runtime would serve: compiled, checked, validated and linked, and not run."""
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "program.lfpl")
        with open(path, "w") as handle:
            handle.write(source)
        return str(build(path, os.path.join(workdir, "program"), Options(40000, 41000, 42000, n_max)))


class AVerdictInARegister(unittest.TestCase):
    """Section 4.11: a sum whose every value fits a register is a scalar, and making one costs no cell."""

    def test_it_is_a_type_of_its_own_kind(self):
        lines = rt(VERDICT + "fn f() : Verdict = Expired;\n")
        declared = next(line for line in lines if re.fullmatch(r"type t\d+ enum Verdict", line))
        ctors = [line for line in lines if line.startswith("ctor") and line.split()[2] == declared.split()[1]]
        self.assertEqual([line.split()[3] for line in ctors], ["0", "1", "2"], ctors)
        self.assertIn("field", next(line for line in lines if line.startswith("field") and "port" in line))

    def test_a_constructor_with_no_fields_is_its_tag(self):
        lines = rt(VERDICT + "fn f() : Verdict = Expired;\n")
        body = lines[next(i for i, line in enumerate(lines) if line.startswith("fn")) + 1:]
        self.assertRegex(body[0].strip(), r"^enum r\d+, c\d+$")

    def test_its_fields_are_its_arguments(self):
        lines = rt(VERDICT + "fn f(x: Int8) : Verdict = Send(x);\n")
        self.assertTrue(any(re.fullmatch(r"enum r\d+, c\d+, r0", line.strip()) for line in lines), lines)

    def test_what_x86_makes_of_it_means_what_the_source_says(self):
        source = VERDICT + """enum Wide = A(x: Int8, y: Int16, z: Int32) | B(w: Int32) | C;
fn f(x: Int8) : Verdict = Send(x);
fn g() : Verdict = Send(7);
fn h(p: Bool) : Verdict = if p then Malformed else Expired;
fn w(x: Int8, y: Int16, z: Int32) : (Wide, Wide, Wide) = (A(x, 0xBEEF, z), B(z), C);
fn v(x: Int64) : Wide = A(low8(x), low16(>>(x, 8)), 5);
"""
        asm = compile_program(source.encode())
        self.assertEqual([str(p) for p in check_translation(source.encode(), asm)], [])

    def test_one_passed_to_a_function_is_still_there_afterwards(self):
        source = VERDICT + """fn g(v: Verdict) : Verdict = v;
fn f(x: Int8) : (Verdict, Verdict, Verdict) = let v = Send(x) in let w = g(v) in (w, v, g(v));
fn h(v: Verdict) : (Verdict, Verdict, Verdict) = let w = g(v) in (g(w), w, v);
"""
        asm = compile_program(source.encode())
        self.assertEqual([str(p) for p in check_translation(source.encode(), asm)], [])

    def test_no_cell_can_be_written_for_one_or_for_an_integer(self):
        refused(self, VERDICT + "fn g(v: Verdict) : Verdict @v = v;\n", "Verdict is no cell")
        refused(self, "fn g(v: Int64) : Int64 @v = v;\n", "Int64 is no cell")

    def test_no_cell_is_inferred_for_one(self):
        """`@` says which parameter's cell comes back, and an enum is no cell."""
        header = next(line for line in rt(VERDICT + "fn g(v: Verdict) : Verdict = v;\n") if line.startswith("fn"))
        self.assertNotIn("@", header)

    def test_a_field_comes_from_a_cell(self):
        source = VERDICT + ONE + """fn f(x: P) : (Verdict, P) = match x with P(d, a, b) =>
  (Send(low8(b)), P@d {}) end;
"""
        asm = compile_program(source.encode())
        self.assertEqual([str(p) for p in check_translation(source.encode(), asm)], [])

    def test_a_handler_that_returns_one_draws_no_cell(self):
        self.assertRegex(report(self, ONE_LINE), r"R\(n\) = 0 cells")

    def test_the_handler_of_section_3_1_with_a_verdict_and_a_record(self):
        lines = rt(ONE_LINE)
        header = next(line for line in lines if line.startswith("handler"))
        self.assertRegex(header, r"\) : t\d+$")
        check_stream(lines)


class RefusedVerdicts(unittest.TestCase):
    def test_matching_one_is_not_built_yet(self):
        refused(self, VERDICT + "fn f(v: Verdict) : Int8 = match v with Malformed => 0 | Expired => 1 "
                                "| Send(p) => p end;\n", "Verdict is an enum.*section 12")

    def test_it_cannot_be_stored_in_a_cell_yet(self):
        refused(self, VERDICT + "record R { v: Verdict, a: Int8 };\n", "an enum cannot be stored in a cell")
        refused(self, VERDICT + "data L = N | C(v: Verdict);\n", "an enum cannot be stored in a cell")

    def test_it_is_not_an_integer(self):
        refused(self, VERDICT + "fn f(v: Verdict) : Bool = ==(v, 0);\n", "integer")
        refused(self, VERDICT + "fn f(v: Verdict) : Verdict = +(v, 1);\n", "integer")

    def test_its_constructors_take_no_cell(self):
        refused(self, VERDICT + "fn f(c: Diamond) : Verdict = Send@c(1);\n", "Send is a constructor of the enum "
                                                                             "Verdict, and takes no cell")

    def test_a_constructor_of_data_takes_one(self):
        refused(self, "data L = N | C(v: Int64);\nfn f() : L = C(1);\n", "C is a constructor of L, which is "
                                                                        "built in a cell")

    def test_its_fields_have_to_fit_a_register_with_its_tag(self):
        refused(self, "enum E = A(x: Int64);\n", "does not fit a register")
        refused(self, "enum E = A(x: Int32, y: Int32);\n", "does not fit a register")
        enum = "enum E = A(x: Int8, y: Int16, z: Int32);\nfn f() : E = A(1, 2, 3);\n"
        compile_program(enum.encode())

    def test_its_fields_are_integers(self):
        refused(self, "enum E = A(x: Bool);\n", "an enum's fields are integers")
        refused(self, "enum E = A(x: Be16);\n", "an enum's fields are integers")
        refused(self, "data L = N | C(v: Int64);\nenum E = A(x: L);\n", "an enum's fields are integers")

    def test_the_wrong_number_of_arguments(self):
        refused(self, VERDICT + "fn f() : Verdict = Send(1, 2);\n", "too many arguments")
        refused(self, VERDICT + "fn f() : Verdict = Send();\n", "wrong argument count")

    def test_an_argument_of_the_wrong_width(self):
        refused(self, VERDICT + "fn f(x: Int16) : Verdict = Send(x);\n", "type mismatch")
        refused(self, VERDICT + "fn f() : Verdict = Send(256);\n", "does not fit Int8")

    def test_a_verdict_goes_with_a_record(self):
        with self.assertRaisesRegex(Refused, "a verdict goes with a record"):
            report(self, VERDICT + FRAME + "data L = N | C(v: Be16, b: Bytes);\n"
                                           "handler h(m: Frame, free: Free) : (Verdict, L, Free) = "
                                           "match m with Frame(d, a, ttl, length, body) => (Expired, C@d(a, body), free) end;\n")

    def test_a_verdict_names_a_port_or_nothing(self):
        source = ONE_LINE.replace("Send(port: Int8)", "Send(port: Int16)").replace("Send(low8(a))", "Send(a)")
        with self.assertRaisesRegex(Refused, "port: Int8"):
            report(self, source)
        source = ONE_LINE.replace("Send(port: Int8)", "Send(hop: Int8)")
        with self.assertRaisesRegex(Refused, "port: Int8"):
            report(self, source)
        source = ONE_LINE.replace("Send(port: Int8)", "Send(port: Int8, more: Int8)").replace("Send(low8(a))",
                                                                                             "Send(low8(a), 1)")
        with self.assertRaisesRegex(Refused, "port: Int8"):
            report(self, source)


class WhatRtcheckRefuses(unittest.TestCase):
    """The RT layer's own rules for an enum, on streams Lower would never write."""

    HEAD = ["type t60000 enum E", "ctor c1 t60000 0 A", "ctor c2 t60000 1 B", "field c2 1 1 i8 x"]

    def refused(self, body: list, message: str, head: list = None) -> None:
        with self.assertRaisesRegex(RTError, message):
            check_stream((head or self.HEAD) + body)

    def test_the_stream_these_start_from_types(self):
        check_stream(self.HEAD + ["fn f0 f(r0 : i8) : t60000", "    enum r1, c2, r0", "    ret r1", "end"])

    def test_a_literal_is_not_an_enum(self):
        self.refused(["fn f0 f() : t60000", "    const r0, #1 : t60000", "    ret r0", "end"], "enum")

    def test_an_operand_of_the_wrong_width(self):
        self.refused(["fn f0 f(r0 : i16) : t60000", "    enum r1, c2, r0", "    ret r1", "end"], "i16")

    def test_the_wrong_number_of_operands(self):
        self.refused(["fn f0 f(r0 : i8) : t60000", "    enum r1, c1, r0", "    ret r1", "end"], "operand")

    def test_a_constructor_of_something_else(self):
        head = self.HEAD + ["type t60001 tagless L", "ctor c3 t60001 null N", "ctor c4 t60001 - C",
                            "field c4 0 8 i64 v"]
        self.refused(["fn f0 f() : t60001", "    enum r0, c3", "    ret r0", "end"], "not an enum", head)

    def test_compared_or_computed_with(self):
        self.refused(["fn f0 f(r0 : t60000) : bool", "    cmp r1, e, r0, #0", "    ret r1", "end"], "enum")
        self.refused(["fn f0 f(r0 : t60000) : i64", "    pack r1, <8>, r0", "    ret r1", "end"], "enum")

    def test_a_field_past_the_register(self):
        self.refused([], "fit", ["type t60000 enum E", "ctor c1 t60000 0 A", "field c1 4 8 i64 x"])
        self.refused([], "fit", ["type t60000 enum E", "ctor c1 t60000 0 A", "field c1 0 1 i8 x"])

    def test_stored_in_a_cell(self):
        self.refused([], "enum", self.HEAD + ["type t60001 record R", "ctor c3 t60001 - R", "field c3 0 8 t60000 v"])


if __name__ == "__main__":
    unittest.main()
