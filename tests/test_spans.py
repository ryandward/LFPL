"""A record that spans cells, and fields in network byte order: LFPL-O sections 4.8 and 4.9.

Both are claims about what a program means and what it costs, so the programs here run, the symbolic check has to
agree with each of them without running it, and what must stay in place is counted in the assembly. Serving a record
that spans on a live socket is in tests/test_ipv4.py.
"""
import io
import os
import random
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.certify import certify  # noqa: E402
from src.lfplo import build_and_run, compile_program  # noqa: E402
from src.ring import Reject  # noqa: E402
from src.rtcheck import Checker, RTError, check_stream  # noqa: E402
from src.rtlower import Limit, compiler_ring, rt_lines  # noqa: E402
from src.spans import FIELDS_MAX, parts_of  # noqa: E402
from src import symcheck  # noqa: E402
from src.symcheck import check_translation  # noqa: E402
from src.validate import function_body, validate  # noqa: E402

PRE = """data Free = End | More(next: Free);
data Res = Fail | Done(value: Int64, rest: Free);
"""
THREE = "record R { a: Int64, b: Int64, c: Int64 };\n"
FIVE = """record F { a: Int32, b: Int16, c: Int32, e: Int16, g: Int16, h: Int8, i: Int8, j: Int16, k: Int16, l: Int16,
                     m: Int8, n: Int8, o: Int16, p: Int32, q: Int32, r: Int16, s: Int64 };
"""
READ_THREE = """fn rd(x: R, free: Free) : Res = match x with R(d, a, b, c) =>
  let (e, rest) = peel(d) in Done@e(+(*(a, 100), +(*(b, 10), c)), More@rest(free)) end;
"""


def take_cells(count: int, body: str) -> str:
    """Draw this many cells, c0 and on, from r0, leaving the rest in r<count>."""
    for i in reversed(range(count)):
        held = "End"
        for cell in range(1, i):
            held = f"More@c{cell}({held})"
        fail = "Fail" if i == 0 else f"Done@c0(0, {held})"
        body = f"match r{i} with End => {fail} | More(c{i}, r{i + 1}) => {body} end"
    return body


def main_with(count: int, body: str) -> str:
    return f"fn main(r0: Free) : Res = {take_cells(count, body.replace('REST', f'r{count}'))};\n"


def answer(test: unittest.TestCase, source: str) -> int:
    """It compiles, the symbolic check agrees, nothing redundant is in it, and this is what it prints."""
    data = source.encode()
    asm = compile_program(data)
    test.assertEqual([str(p) for p in check_translation(data, asm)], [], source)
    test.assertEqual(validate(asm), [], source)
    with tempfile.TemporaryDirectory() as workdir:
        return build_and_run(asm, workdir)


def body_of(source: str, name: str) -> list[str]:
    lines = [f"{op} {', '.join(args)}".strip() for op, args in function_body(compile_program(source.encode()), name)]
    return [line for line in lines if line not in (f"sub rsp, F_{name}", f"add rsp, F_{name}")]


def stores(lines: list[str]) -> list[str]:
    return [line for line in lines if re.match(r"(mov|add|sub|and|or|xor) (byte|word|dword|qword) \[", line)]


def stream_of(source: str, name: str) -> list[str]:
    """One function of the RT stream, with the numbers Check gave types and constructors taken out of it."""
    seen: dict[str, str] = {}

    def rename(match: re.Match) -> str:
        return seen.setdefault(match.group(0), f"#{len(seen)}")

    lines = list(rt_lines(io.BytesIO(source.encode()), phase1=True))
    start = next(i for i, line in enumerate(lines) if re.match(rf"(fn|handler) f\d+ {name}\(", line))
    stop = next(i for i in range(start, len(lines)) if lines[i].startswith("end"))
    return [re.sub(r"\b[ct]\d+\b", rename, line) for line in lines[start:stop + 1]]


class TheLayout(unittest.TestCase):
    def test_what_fits_a_cell_is_one_cell(self):
        self.assertEqual(parts_of([8, 8]), [2])
        self.assertEqual(parts_of([1] * 16), [16])
        self.assertEqual(parts_of([4, 2, 1, 1, 8]), [5])

    def test_each_cell_but_the_last_keeps_8_bytes_for_the_link(self):
        self.assertEqual(parts_of([8, 8, 8]), [1, 2])
        self.assertEqual(parts_of([1] * 17), [8, 9])
        self.assertEqual(parts_of([8, 8, 8, 8]), [1, 1, 2])

    def test_a_field_too_late_for_its_alignment_goes_to_the_next_cell(self):
        self.assertEqual(parts_of([1, 8, 8]), [1, 2])
        self.assertEqual(parts_of([2, 4, 4, 8, 8]), [2, 1, 2])

    def test_the_ipv4_headers_are_five_cells(self):
        self.assertEqual(parts_of([4, 2, 4, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 4, 4, 2, 8]), [2, 3, 5, 4, 3])

    def test_no_cell_is_past_16_bytes_and_every_field_is_placed_in_order(self):
        rng = random.Random(4816)
        for _ in range(2000):
            sizes = [rng.choice((1, 2, 4, 8)) for _ in range(rng.randrange(1, 40))]
            parts = parts_of(sizes)
            self.assertEqual(sum(parts), len(sizes))
            at = 0
            for index, count in enumerate(parts):
                self.assertGreater(count, 0)
                offset = 0
                for size in sizes[at:at + count]:
                    offset = (offset + size - 1) // size * size + size
                self.assertLessEqual(offset, 16 if index + 1 == len(parts) else 8, (sizes, parts))
                at += count

    def test_the_symbolic_check_works_it_out_the_same_way_on_its_own(self):
        rng = random.Random(1648)
        names = {1: "Int8", 2: "Int16", 4: "Int32", 8: "Int64"}
        for _ in range(2000):
            sizes = [rng.choice((1, 2, 4, 8)) for _ in range(rng.randrange(1, 40))]
            fields = [(f"f{i}", names[size]) for i, size in enumerate(sizes)]
            self.assertEqual([len(cell) for cell in symcheck._cells_of(fields)], parts_of(sizes), sizes)


class WhatAPersonWouldHaveWritten(unittest.TestCase):
    """Nothing after the rewrite knows about spans. So one record has to cost exactly what the chain costs."""

    HAND = """record R2 { b: Int64, c: Int64 };
record R { a: Int64, next: R2 };
fn f(x: R, v: Int64) : R = match x with R(d, a, n) => match n with R2(e, b, c) =>
  R@d(a, R2@e(+(b, v), c)) end end;
"""
    ONE = THREE + "fn f(x: R, v: Int64) : R = match x with R(d, a, b, c) => R@d(a, +(b, v), c) end;\n"

    def test_the_stream_is_the_one_for_the_chain_written_by_hand(self):
        self.assertEqual(stream_of(self.ONE, "f"), stream_of(self.HAND, "f"))

    def test_a_longer_one_with_changes_in_its_first_and_last_cells(self):
        hand = """record P3 { g: Int32, h: Int32, s: Int64 };
record P2 { e: Int32, f: Int32, next: P3 };
record P { a: Int32, b: Int32, next: P2 };
fn f(x: P, v: Int32) : P = match x with P(d, a, b, n) => match n with P2(d2, e, f, n2) =>
  match n2 with P3(d3, g, h, s) => P@d(+(a, v), b, P2@d2(e, f, P3@d3(g, -(h, v), s))) end end end;
"""
        one = """record P { a: Int32, b: Int32, e: Int32, f: Int32, g: Int32, h: Int32, s: Int64 };
fn f(x: P, v: Int32) : P = match x with P(d, a, b, e, f, g, h, s) => P@d(+(a, v), b, e, f, g, -(h, v), s) end;
"""
        self.assertEqual(stream_of(one, "f"), stream_of(hand, "f"))

    def test_a_record_that_fits_one_cell_is_not_touched(self):
        source = "record Q { a: Int64, b: Int64 };\nfn f(x: Q) : Q = match x with Q(d, a, b) => Q@d(b, a) end;\n"
        lines = list(rt_lines(io.BytesIO(source.encode()), phase1=True))
        self.assertFalse([line for line in lines if "$" in line], lines)


class ARecordThatSpans(unittest.TestCase):
    def test_built_and_read_back(self):
        source = PRE + THREE + READ_THREE + main_with(2, "rd(R@(c0, c1)(1, 2, 3), REST)")
        self.assertEqual(answer(self, source), 123)

    def test_a_field_in_each_cell_changes_and_the_rest_stay(self):
        source = PRE + THREE + READ_THREE + (
            "fn u(x: R) : R = match x with R(d, a, b, c) => R@d(+(a, 1), b, c) end;\n"
            "fn v(x: R) : R = match x with R(d, a, b, c) => R@d(a, +(b, 1), c) end;\n"
            "fn w(x: R) : R = match x with R(d, a, b, c) => R@d(a, b, +(c, 1)) end;\n"
        ) + main_with(2, "rd(w(v(u(u(R@(c0, c1)(1, 2, 3))))), REST)")
        self.assertEqual(answer(self, source), 334)
        for name, store in (("u", r"^add qword \[rdi\], 1$"), ("v", r"^add qword \[\w+\], 1$"),
                            ("w", r"^add qword \[\w+\+8\], 1$")):
            written = stores(body_of(source, name))
            self.assertEqual(len(written), 1, written)
            self.assertRegex(written[0], store)

    def test_rebuilt_with_nothing_changed_nothing_is_written_at_all(self):
        source = PRE + THREE + "fn same(x: R) : R = match x with R(d, a, b, c) => R@d(a, b, c) end;\n"
        self.assertEqual(stores(body_of(source, "same")), [])

    def test_a_value_moving_from_one_cell_to_another(self):
        source = PRE + THREE + READ_THREE + (
            "fn rotate(x: R) : R = match x with R(d, a, b, c) => R@d(c, a, b) end;\n"
        ) + main_with(2, "rd(rotate(R@(c0, c1)(1, 2, 3)), REST)")
        self.assertEqual(answer(self, source), 312)

    def test_every_field_of_five_cells(self):
        """Every field holds what it should, and no two fields are mixed up, at every width."""
        names = "a b c e g h i j k l m n o p q r s".split()
        values = [i + 1 for i in range(len(names))]
        binders = ", ".join(names)
        total = "0"
        for weight, name in enumerate(names):
            total = f"+(*(pack<{self._bits(name)}>({name}), {3 ** (weight % 40)}), {total})" if name != "s" else \
                f"+(*({name}, {3 ** (weight % 40)}), {total})"
        arguments = ", ".join(str(v) for v in values)
        cells = ", ".join(f"c{i}" for i in range(5))
        source = PRE + FIVE + (
            f"fn rd(x: F, free: Free) : Res = match x with F(d, {binders}) => "
            f"let (first, others) = peel(d) in Done@first({total}, More@others(free)) end;\n"
        ) + main_with(5, f"rd(F@({cells})({arguments}), REST)")
        expected = sum(v * 3 ** (weight % 40) for weight, v in enumerate(values)) & (1 << 64) - 1
        self.assertEqual(answer(self, source), expected)

    @staticmethod
    def _bits(name: str) -> int:
        return {"a": 32, "b": 16, "c": 32, "e": 16, "g": 16, "h": 8, "i": 8, "j": 16, "k": 16, "l": 16, "m": 8,
                "n": 8, "o": 16, "p": 32, "q": 32, "r": 16}[name]

    def test_its_cells_given_back_with_more(self):
        """Every cell comes back: there are 1024, one holds the answer, and counting the rest proves it."""
        source = """data Cells = None | Cell(next: Cells);
data Res = Fail | Done(value: Int64, rest: Cells);
record R { a: Int64, b: Int64, c: Int64 };
fn give(x: R, free: Cells) : Cells = match x with R(d, a, b, c) => let (e, rest) = peel(d) in Cell@e(Cell@rest(free)) end;
fn count(f: Cells, n: Int64, out: Diamond, back: Cells) : Res = match f with None => Done@out(n, back)
  | Cell(d, rest) => count(rest, +(n, 1), out, Cell@d(back)) end;
fn main(r0: Cells) : Res = match r0 with None => Fail | Cell(c0, r1) => match r1 with None => Done@c0(0, None)
  | Cell(c1, r2) => match r2 with None => Done@c0(0, Cell@c1(None))
  | Cell(out, r3) => count(give(R@(c0, c1)(1, 2, 3), r3), 0, out, None) end end end;
"""
        self.assertEqual(answer(self, source), 1023)

    def test_more_gives_back_as_many_cells_as_the_name_stands_for(self):
        source = PRE + "record S { a: Int64, b: Int64, c: Int64, e: Int64, g: Int64, h: Int64 };\n" + """
fn give(x: S, free: Free) : (Int64, Free) = match x with S(d, a, b, c, e, g, h) => (+(a, h), More@d(free)) end;
""" + main_with(5, "let (v, f) = give(S@(c0, c1, c2, c3, c4)(5, 0, 0, 0, 0, 7), REST) in "
                   "match f with End => Fail | More(e, g) => Done@e(v, g) end")
        self.assertEqual(answer(self, source), 12)

    def test_peel_takes_one_off_as_often_as_there_are_cells(self):
        source = PRE + "record R { y: Int64, z: Int64, a: Int64, b: Int64, c: Int64 };\n" + """
fn pick(x: R, free: Free) : Res = match x with R(d, y, z, a, b, c) =>
  let (p, q) = peel(d) in let (s, t) = peel(q) in Done@p(+(y, c), More@s(More@t(free))) end;
""" + main_with(4, "pick(R@(c0, c1, c2, c3)(1, 2, 3, 4, 5), REST)")
        self.assertEqual(answer(self, source), 6)

    def test_the_rest_can_have_the_name_it_came_from(self):
        source = PRE + "record R { y: Int64, z: Int64, a: Int64, b: Int64, c: Int64 };\n" + """
fn pick(x: R, free: Free) : Res = match x with R(d, y, z, a, b, c) =>
  let (p, d) = peel(d) in Done@p(+(y, c), More@d(free)) end;
""" + main_with(4, "pick(R@(c0, c1, c2, c3)(1, 2, 3, 4, 5), REST)")
        self.assertEqual(answer(self, source), 6)

    def test_linear_fields_exchanged_between_its_first_cell_and_its_last(self):
        source = PRE + """data List = Nil | Cons(h: Int64);
record T { l: List, a: Int64, b: Int64, m: List };
fn cross(x: T) : T = match x with T(d, l, a, b, m) => T@d(m, b, a, l) end;
fn head(l: List, free: Free) : (Int64, Free) = match l with Nil => (0, free) | Cons(d, h) => (h, More@d(free)) end;
fn rd(x: T, free: Free) : Res = match x with T(d, l, a, b, m) => let (u, f1) = head(l, free) in let (v, f2) = head(m, f1) in
  let (p, rest) = peel(d) in Done@p(+(*(u, 1000), +(*(a, 100), +(*(b, 10), v))), More@rest(f2)) end;
""" + main_with(5, "rd(cross(T@(c0, c1, c2)(Cons@c3(7), 2, 3, Cons@c4(9))), REST)")
        self.assertEqual(answer(self, source), 9000 + 300 + 20 + 7)

    def test_a_name_in_a_pattern_of_another_record_that_spans(self):
        source = PRE + THREE + "record T { x: Int64, y: Int64, z: Int64 };\n" + """
fn swap(p: R, q: T) : (R, T) = match p with R(d, a, b, c) => match q with T(e, x, y, z) =>
  (R@e(x, y, z), T@d(a, b, c)) end end;
fn rd2(q: T, free: Free) : Res = match q with T(d, x, y, z) =>
  let (e, rest) = peel(d) in Done@e(+(*(x, 100), +(*(y, 10), z)), More@rest(free)) end;
""" + READ_THREE + main_with(4, """let (p, q) = swap(R@(c0, c1)(1, 2, 3), T@(c2, c3)(4, 5, 6)) in
  match rd(p, REST) with Fail => rd2(q, End) | Done(d, v, f) => match rd2(q, f) with
    Fail => Done@d(v, End) | Done(e, w, g) => Done@d(+(*(v, 1000), w), More@e(g)) end end""")
        self.assertEqual(answer(self, source), 456123)

    def test_one_inside_another(self):
        source = PRE + THREE + """record W { p: Int64, r: R, q: Int64, t: Int64 };
fn rd(w: W, free: Free) : Res = match w with W(d, p, r, q, t) => match r with R(e, a, b, c) =>
  let (f, rest) = peel(d) in let (g, more) = peel(e) in
  Done@f(+(*(p, 10000), +(*(a, 1000), +(*(b, 100), +(*(c, 10), +(q, t))))), More@rest(More@g(More@more(free)))) end end;
""" + main_with(5, "rd(W@(c0, c1, c2)(1, R@(c3, c4)(2, 3, 4), 5, 6), REST)")
        self.assertEqual(answer(self, source), 12340 + 11)

    def test_chosen_in_the_arms_of_an_if(self):
        source = PRE + THREE + READ_THREE + """
fn f(x: R, p: Bool) : R = match x with R(d, a, b, c) =>
  if p then R@d(a, +(b, 1), c) else R@d(+(a, 1), b, c) end;
""" + main_with(2, "rd(f(R@(c0, c1)(1, 2, 3), COND), REST)")
        self.assertEqual(answer(self, source.replace("COND", "true")), 133)
        self.assertEqual(answer(self, source.replace("COND", "false")), 223)

    def test_a_field_read_after_its_cell_is_rebuilt(self):
        source = PRE + THREE + """
fn f(x: R, free: Free) : Res = match x with R(d, a, b, c) =>
  match R@d(9, 9, 9) with R(e, p, q, s) => let (g, rest) = peel(e) in Done@g(+(+(a, b), +(c, s)), More@rest(free)) end end;
""" + main_with(2, "f(R@(c0, c1)(1, 2, 3), REST)")
        self.assertEqual(answer(self, source), 15)

    def test_the_cell_it_returns_is_found_without_being_written(self):
        declared = THREE + "fn f(x: R) : R @x = match x with R(d, a, b, c) => R@d(c, b, a) end;\n"
        inferred = THREE + "fn f(x: R) : R = match x with R(d, a, b, c) => R@d(c, b, a) end;\n"
        self.assertEqual(stream_of(inferred, "f"), stream_of(declared, "f"))
        self.assertTrue(stream_of(inferred, "f")[0].endswith("@0"), stream_of(inferred, "f")[0])

    def test_a_list_of_them_can_be_walked(self):
        source = PRE + THREE + """data L = Nil | Cons(r: R, t: L);
fn sum(l: L, acc: Int64, free: Free) : Res = match l with
  | Nil => match free with End => Fail | More(c, rest) => Done@c(acc, rest) end
  | Cons(d, x, t) => match x with R(e, a, b, c) =>
      sum(t, +(acc, +(a, +(b, c))), More@d(More@e(free))) end end;
""" + main_with(6, "sum(Cons@c0(R@(c1, c2)(1, 2, 3), Cons@c3(R@(c4, c5)(4, 5, 6), Nil)), 0, REST)")
        self.assertEqual(answer(self, source), 21)

    def test_recursive_through_a_forward_declaration(self):
        source = PRE + """data N;
data L = Nil | Cons(n: N);
record N { a: Int64, b: Int64, c: Int64, t: L };
fn sum(l: L, acc: Int64, free: Free) : Res = match l with
  | Nil => match free with End => Fail | More(c, rest) => Done@c(acc, rest) end
  | Cons(d, n) => match n with N(e, a, b, c, t) =>
      sum(t, +(acc, +(a, +(b, c))), More@d(More@e(free))) end end;
""" + main_with(8, "sum(Cons@c0(N@(c1, c2, c3)(1, 2, 3, Cons@c4(N@(c5, c6, c7)(4, 5, 6, Nil)))), 0, REST)")
        self.assertEqual(answer(self, source), 21)

    def test_the_first_cells_fields_can_still_be_updated_with_with(self):
        source = PRE + THREE + READ_THREE + """
fn u(x: R) : R = { x with a = +(a, 8) };
""" + main_with(2, "rd(u(R@(c0, c1)(1, 2, 3)), REST)")
        self.assertEqual(answer(self, source), 923)
        self.assertEqual(stores(body_of(source, "u")), ["add qword [rdi], 8"])


class WhatTheCompilerKeeps(unittest.TestCase):
    """Theorem 7.2 and 7.1, for a program full of these: whatever it held to rewrite them, it gave back."""

    SOURCE = PRE + THREE + READ_THREE + """fn u(x: R) : R = match x with R(d, a, b, c) => R@d(c, b, a) end;
""" + main_with(4, "match rd(u(R@(c0, c1)(1, 2, 3)), REST) with Fail => rd(u(R@(c2, c3)(4, 5, 6)), End) "
                                        "| Done(d, v, f) => match rd(u(R@(c2, c3)(4, 5, 6)), f) with Fail => Done@d(v, End) "
                                        "| Done(e, w, g) => Done@d(+(v, w), More@e(g)) end end")

    def test_only_the_tables_stay_out_of_the_pool(self):
        ring = compiler_ring(self.SOURCE.encode(), lambda tag, payload: None, phase1=True)
        ring.run()
        passed, check = ring.stages[2], ring.stages[3]
        tables = len(check.G.cells) + len(check.TA.cells) + len(check.TF.cells) + len(check.TT.cells)
        self.assertEqual(ring.store.C - len(ring.F), tables + passed.table_cells())

    def test_it_holds_at_every_step(self):
        def on_step(ring) -> None:
            ring.check_conservation()

        compiler_ring(self.SOURCE.encode(), lambda tag, payload: None, on_step, phase1=True).run()

    def test_the_answer(self):
        self.assertEqual(answer(self, self.SOURCE), 321 + 654)

    def test_a_name_later_at_the_position_of_a_group_is_not_one(self):
        """The note goes with the scope, or `y` here would be refused as a group of 2."""
        source = PRE + THREE + """fn f(x: R, free: Free) : Res =
  let y = match x with R(d, a, b, c) => let (e, rest) = peel(d) in Done@e(a, More@rest(free)) end in y;
"""
        compile_program(source.encode())

    def test_a_long_name_for_the_cells_is_still_a_name(self):
        """What is hidden keeps some of it, so that it can be said when one goes unused, and two names that start
        alike are told apart by where they are."""
        long_name, other = "a_diamond_name_", "a_diamond_name2"
        source = PRE + THREE + "record T { p: Int64, q: Int64, s: Int64 };\n" + f"""
fn f(x: R, y: T, free: Free) : Free = match x with R({long_name}, a, b, c) => match y with T({other}, p, q, s) =>
  More@{long_name}(More@{other}(free)) end end;
"""
        compile_program(source.encode())
        with self.assertRaisesRegex(Reject, "a cell of a_diam"):
            compile_program(source.replace(f"More@{other}(free)", "free").encode())


class RefusedWithAReason(unittest.TestCase):
    def refuse(self, pattern: str, source: str) -> None:
        with self.assertRaisesRegex((Reject, RTError, Limit), pattern):
            compile_program(source.encode())

    MATCHED = PRE + THREE

    def test_a_pattern_with_too_few_binders(self):
        self.refuse("pattern needs 4 binders, got 3",
                    self.MATCHED + "fn f(x: R) : Int64 = match x with R(d, a, b) => 1 end;\n")
        self.refuse("pattern needs 4 binders, got 0", self.MATCHED + "fn f(x: R) : Int64 = match x with R => 1 end;\n")

    def test_a_pattern_with_too_many_binders(self):
        self.refuse("too many binders: expected 4",
                    self.MATCHED + "fn f(x: R) : Int64 = match x with R(d, a, b, c, e) => 1 end;\n")

    def test_a_constructor_with_too_few_or_too_many_arguments(self):
        self.refuse("wrong argument count: expected 3", self.MATCHED + "fn f(x: R) : R = match x with R(d, a, b, c) => R@d(a, b) end;\n")
        self.refuse("wrong argument count: expected 3", self.MATCHED + "fn f(x: R) : R = match x with R(d, a, b, c) => R@d() end;\n")
        self.refuse("too many arguments: expected 3", self.MATCHED + "fn f(x: R) : R = match x with R(d, a, b, c) => R@d(a, b, c, a) end;\n")

    def test_one_cell_cannot_hold_a_record_that_spans(self):
        self.refuse("R takes 2 cells, and this names 1", self.MATCHED + "fn f(c: Diamond) : R = R@c(1, 2, 3);\n")

    def test_the_wrong_number_of_cells_named(self):
        self.refuse("R takes 2 cells, and this names 3",
                    self.MATCHED + "fn f(a: Diamond, b: Diamond, c: Diamond) : R = R@(a, b, c)(1, 2, 3);\n")

    def test_a_group_of_the_wrong_size(self):
        self.refuse("R takes 2 cells, and d is 4", self.MATCHED + "record T { a: Int64, b: Int64, c: Int64, e: Int64, g: Int64 };\n"
                    "fn f(x: T) : R = match x with T(d, a, b, c, e, g) => R@d(a, b, c) end;\n")

    def test_a_group_in_a_list_of_cells(self):
        self.refuse("takes 2 cells, and d is 2", self.MATCHED + "fn f(x: R, c: Diamond) : R = match x with R(d, a, b, e) => R@(c, d)(a, b, e) end;\n")

    def test_a_group_is_not_one_cell(self):
        self.refuse("Done takes one cell, and d is 2: let \\(c, rest\\) = peel\\(d\\)",
                    self.MATCHED + "fn f(x: R, free: Free) : Res = match x with R(d, a, b, c) => Done@d(a, free) end;\n")

    def test_several_cells_for_a_constructor_that_takes_one(self):
        self.refuse("Done takes one cell, and this names more",
                    self.MATCHED + "fn f(a: Diamond, b: Diamond, free: Free) : Res = Done@(a, b)(1, free);\n")

    def test_a_group_is_not_a_value(self):
        for use in ("let e = d in R@e(a, b, c)", "g(d, End)", "(d, 1)"):
            self.refuse("d is 2 cells, and only a constructor or peel can take them",
                        self.MATCHED + "fn g(c: Diamond, free: Free) : Free = More@c(free);\n"
                        f"fn f(x: R) : R = match x with R(d, a, b, c) => {use} end;\n")

    def test_a_name_that_shadows_a_group_is_not_one(self):
        source = self.MATCHED + "fn f(x: R) : R = match x with R(d, a, b, c) => let d2 = d in R@d(a, b, c) end;\n"
        self.refuse("d is 2 cells", source)
        fine = self.MATCHED + """fn f(x: R) : Int64 = match x with R(d, a, b, c) => let g = R@d(a, b, c) in
  match g with R(e, p, q, r) => let d = 1 in +(d, 1) end end;
"""
        self.refuse("a cell of e is never used|linear variable e is never used", fine)

    def test_every_cell_has_to_go_somewhere(self):
        self.refuse("a cell of d is never used", self.MATCHED + "fn f(x: R) : Int64 = match x with R(d, a, b, c) => 1 end;\n")
        self.refuse("linear variable rest is never used", self.MATCHED + "fn f(x: R, free: Free) : Res = "
                    "match x with R(d, a, b, c) => let (e, rest) = peel(d) in Done@e(1, free) end;\n")

    def test_a_cell_cannot_be_used_twice(self):
        self.refuse("used twice", self.MATCHED + "fn f(x: R, free: Free) : Res = match x with R(d, a, b, c) =>"
                    " let (e, rest) = peel(d) in Done@e(1, More@rest(More@e(free))) end;\n")

    def test_peel_of_what_cannot_be_peeled(self):
        self.refuse("c is one cell", self.MATCHED + "fn f(c: Diamond, free: Free) : Res = let (a, b) = peel(c) in Done@a(1, More@b(free));\n")
        self.refuse("the name of a record's diamond", self.MATCHED + "fn f(free: Free) : Res = let (a, b) = peel(1) in Fail;\n")

    def test_peel_anywhere_but_the_one_place(self):
        for use in ("let e = peel(d) in 1", "let (e, g, h) = peel(d) in 1", "+(peel(d), 1)"):
            self.refuse("peel is written let", self.MATCHED + f"fn f(x: R) : Int64 = match x with R(d, a, b, c) => {use} end;\n")

    def test_peel_with_more_than_one_argument(self):
        self.refuse("peel takes one diamond",
                    self.MATCHED + "fn f(x: R) : Int64 = match x with R(d, a, b, c) => let (e, g) = peel(d, a) in 1 end;\n")

    def test_peel_is_a_primitive(self):
        self.refuse("peel is a primitive", "fn peel(a: Int64) : Int64 = a;\n")

    def test_with_on_a_field_past_the_first_cell_says_what_to_do(self):
        self.refuse("reaches the first cell of R, and c is not there: rebuild it with a match",
                    self.MATCHED + "fn u(x: R) : R = { x with c = 1 };\n")
        self.refuse("no field q in Q", "record Q { a: Int64 };\nfn u(x: Q) : Q = { x with q = 1 };\n")

    def test_a_tagged_constructor_still_has_to_fit_one_cell(self):
        self.refuse("does not fit in a diamond", "data T = A(a: Int64, b: Int64, c: Int64) | B;\n")

    def test_too_many_fields_for_one_record(self):
        fields = ", ".join(f"f{i}: Int8" for i in range(FIELDS_MAX + 1))
        self.refuse(f"section 8.2: record R has more than {FIELDS_MAX} fields", f"record R {{ {fields} }};\n")
        fields = ", ".join(f"f{i}: Int8" for i in range(FIELDS_MAX))
        compile_program(f"record R {{ {fields} }};\n".encode())

    def test_each_cell_after_the_first_counts_twice_against_the_nesting_limit(self):
        """Parse has no idea, and every stage after it does the nesting, so the ring is only sized for it if
        this is refused here. 24 cells is 46 frames, and enough `let`s go on top of that to pass 64."""
        fields = ", ".join(f"f{i}: Int64" for i in range(25))
        names = ", ".join(f"f{i}" for i in range(25))
        lets = "".join(f"let n{i} = 1 in " for i in range(30))
        source = f"record B {{ {fields} }};\nfn f(x: B) : Int64 = match x with B(d, {names}) => {lets}1 end;\n"
        self.refuse("nesting too deep", source)


class CountedInTheBound(unittest.TestCase):
    """Section 4.8: the cells of a record that came with an item are in its n. Those it has to draw are in R(n)."""

    ITEM = """data Bytes;
record H { a: Int64, b: Int64, c: Int64, body: Bytes };
"""

    def bound(self, source: str):
        checker = Checker()
        for line in rt_lines(io.BytesIO(source.encode())):
            checker.feed(line)
        checker.finish()
        return certify(checker, next(f.name for f in checker.functions.values() if f.handler), 216)

    def test_rebuilt_in_its_own_cells_it_draws_nothing(self):
        bound = self.bound(self.ITEM + """
handler h(m: H, free: Free) : (H, Free) = match m with H(d, a, b, c, body) => (H@d(b, a, c, body), free) end;
""")
        self.assertEqual((bound.a, bound.b), (0, 0))

    def test_a_cell_drawn_for_the_verdict_is_one(self):
        bound = self.bound(self.ITEM + """data Out = Gone(body: Bytes) | Send(port: Int8, h: H);
handler h(m: H, free: Free) : (Out, Free) = match free with
  | End => match m with H(d, a, b, c, body) => let (p, q) = peel(d) in (Gone@p(body), More@q(End)) end
  | More(c, rest) => (Send@c(1, m), rest) end;
""")
        self.assertEqual((bound.a, bound.b), (0, 1))

    def test_a_record_built_from_the_reserve_is_as_many_cells_as_it_spans(self):
        for count, record, arguments, drawn in ((2, "record Reply { x: Int64, y: Int64, body: Bytes };", "1, 2, body", ("p", "q")),
                                                (3, "record Reply { x: Int64, y: Int64, z: Int64, body: Bytes };",
                                                 "1, 2, 3, body", ("p", "q", "s"))):
            source = f"""data Bytes;
record H {{ a: Int8, body: Bytes }};
{record}
data Out = Dropped(body: Bytes) | Sent(port: Int8, r: Reply);
"""
            with self.subTest(cells=count):
                bound = self.bound(source + self._drawing(count, arguments, drawn))
                self.assertEqual((bound.a, bound.b), (0, count))

    @staticmethod
    def _drawing(count: int, arguments: str, drawn: tuple) -> str:
        """Draw that many cells for the reply. The item's own cell is the verdict, and its body is given back
        as the verdict's, so the only way to run out is the End arms, which every draw has to have."""
        body = f"(Sent@d(0, Reply@({', '.join(drawn)})({arguments})), r{count})"
        for i in reversed(range(count)):
            held = "End"
            for cell in drawn[:i]:
                held = f"More@{cell}({held})"
            refusal = f"(Dropped@d(body), {held})"
            body = f"match r{i} with End => {refusal} | More({drawn[i]}, r{i + 1}) => {body} end"
        return f"handler h(m: H, r0: Free) : (Out, Free) = match m with H(d, a, body) => {body} end;\n"


class NetworkOrder(unittest.TestCase):
    """A field of type Be16, Be32 or Be64 lies in its cell in network byte order. Nothing else knows: what a
    pattern binds is the integer, and what a constructor is given is the integer."""

    FIELDS = "record W { a: Be16, b: Be32, c: Be64 };\n"
    READ = """fn rd(w: W, free: Free) : Res = match w with W(d, a, b, c) =>
  Done@d(+(pack<16>(a), +(pack<32>(b), c)), free) end;
"""

    def test_what_goes_in_comes_out(self):
        source = PRE + self.FIELDS + self.READ + main_with(1, "rd(W@c0(258, 16909060, 72623859790382856), REST)")
        self.assertEqual(answer(self, source), 258 + 16909060 + 72623859790382856)

    def test_arithmetic_is_on_the_integer_and_not_on_the_bytes(self):
        """255 + 1 is 256: 00 ff becomes 01 00. On the bytes as they lie it would be ff 01."""
        source = PRE + "record Q { a: Be16, b: Be16 };\n" + """
fn up(q: Q) : Q = match q with Q(d, a, b) => Q@d(+(a, 1), b) end;
fn rd(q: Q, free: Free) : Res = match q with Q(d, a, b) => Done@d(pack<16>(a), free) end;
""" + main_with(1, "rd(up(Q@c0(255, 0)), REST)")
        self.assertEqual(answer(self, source), 256)
        self.assertIn("bswap", " ".join(stream_of(source, "up")))

    def test_comparison_is_on_the_integer(self):
        source = PRE + "record Q { a: Be16, b: Be16 };\n" + """
fn lt(q: Q, free: Free) : Res = match q with Q(d, a, b) => Done@d(if <(a, b) then 1 else 0, free) end;
""" + main_with(1, "lt(Q@c0(1, 256), REST)")  # 00 01 and 01 00: as they lie, the first is the larger
        self.assertEqual(answer(self, source), 1)

    def test_a_field_left_alone_is_not_swapped_twice_or_stored(self):
        source = PRE + "record Q { a: Be32, b: Be32 };\nfn u(q: Q, v: Int32) : Q = match q with Q(d, a, b) => Q@d(a, v) end;\n"
        stream = "\n".join(stream_of(source, "u"))
        self.assertEqual(stream.count("bswap"), 1, stream)
        self.assertRegex(stream, r"build r0, #\d+, none, =, r\d+")
        self.assertEqual(len(stores(body_of(source, "u"))), 1)

    def test_rebuilt_with_nothing_changed_nothing_happens(self):
        source = PRE + "record Q { a: Be32, b: Be32 };\nfn u(q: Q) : Q = match q with Q(d, a, b) => Q@d(a, b) end;\n"
        self.assertFalse([line for line in stream_of(source, "u") if "bswap" in line])
        self.assertEqual(stores(body_of(source, "u")), [])

    def test_two_fields_of_one_order_exchange_bits_with_no_swap_at_all(self):
        source = PRE + "record Q { a: Be32, b: Be32 };\nfn u(q: Q) : Q = match q with Q(d, a, b) => Q@d(b, a) end;\n"
        self.assertFalse([line for line in stream_of(source, "u") if "bswap" in line], stream_of(source, "u"))

    def test_each_way_between_the_two_orders_is_one_swap(self):
        source = PRE + "record Q { a: Be32, b: Int32 };\nfn u(q: Q) : Q = match q with Q(d, a, b) => Q@d(b, a) end;\n"
        self.assertEqual(len([line for line in stream_of(source, "u") if "bswap" in line]), 2)
        reader = """fn rd(q: Q, free: Free) : Res = match q with Q(d, a, b) => Done@d(+(*(pack<32>(a), 1000), pack<32>(b)), free) end;
"""
        self.assertEqual(answer(self, source + reader + main_with(1, "rd(u(Q@c0(3, 7)), REST)")), 7003)

    def test_a_value_through_a_join_keeps_its_word_and_not_its_order(self):
        """The arms read one field, so what the join holds is that field's value, and it is the integer. Rebuilt as a
        record in the other order, with the same layout, it has to be stored, because the bits there are the wrong way."""
        source = PRE + """record Q { a: Be32, b: Be32 };
record N { x: Int32, y: Int32 };
fn f(q: Q, p: Bool) : N = match q with Q(d, a, b) => let x = if p then a else a in N@d(x, b) end;
fn rd(n: N, free: Free) : Res = match n with N(d, x, y) => Done@d(+(*(pack<32>(x), 1000), pack<32>(y)), free) end;
""" + main_with(1, "rd(f(Q@c0(3, 7), true), REST)")
        self.assertEqual(answer(self, source), 3007)

    def test_a_literal_is_stored_already_swapped(self):
        source = PRE + "record Q { a: Be16, b: Be16 };\nfn mk(c: Diamond) : Q = Q@c(2048, 1);\n"
        self.assertTrue([line for line in stream_of(source, "mk") if re.search(r"build r0, #\d+, none, #8, #256", line)],
                        stream_of(source, "mk"))

    def test_a_literal_too_large_for_the_field(self):
        source = PRE + "record Q { a: Be16, b: Be16 };\nfn mk(c: Diamond) : Q = Q@c(65536, 1);\n"
        with self.assertRaisesRegex(Reject, "does not fit Int16"):
            compile_program(source.encode())

    def test_the_arms_of_an_if_join_as_the_integer(self):
        source = PRE + "record Q { a: Be16, b: Be16 };\n" + """
fn pick(q: Q, p: Bool) : Q = match q with Q(d, a, b) => Q@d(if p then +(a, 1) else b, 300) end;
fn rd(q: Q, free: Free) : Res = match q with Q(d, a, b) => Done@d(+(*(pack<16>(a), 1000), pack<16>(b)), free) end;
""" + main_with(1, "rd(pick(Q@c0(5, 7), COND), REST)")
        self.assertEqual(answer(self, source.replace("COND", "true")), 6300)
        self.assertEqual(answer(self, source.replace("COND", "false")), 7300)

    def test_read_after_its_cell_is_rebuilt_it_is_still_the_integer(self):
        source = PRE + "record Q { a: Be16, b: Be16 };\n" + """
fn f(q: Q, free: Free) : Res = match q with Q(d, a, b) =>
  match Q@d(1, 2) with Q(e, x, y) => Done@e(+(pack<16>(a), pack<16>(x)), free) end end;
fn g(q: Q) : Q = match q with Q(d, a, b) => let r = Q@d(7, 8) in match r with Q(e, x, y) =>
  Q@e(+(a, x), b) end end;
fn rd(q: Q, free: Free) : Res = match q with Q(d, a, b) => Done@d(+(*(pack<16>(a), 1000), pack<16>(b)), free) end;
""" + main_with(1, "rd(g(Q@c0(40, 50)), REST)")
        self.assertEqual(answer(self, source), 47050)

    def test_given_to_a_function_it_is_the_integer(self):
        source = PRE + "record Q { a: Be16, b: Be32 };\n" + """
fn twice(x: Int16) : Int16 = +(x, x);
fn f(q: Q, free: Free) : Res = match q with Q(d, a, b) => Done@d(pack<16>(twice(a)), free) end;
""" + main_with(1, "f(Q@c0(300, 0), REST)")
        self.assertEqual(answer(self, source), 600)

    def test_a_tagged_constructor_can_hold_one(self):
        source = PRE + "data T = A(x: Be16) | B(y: Be32) | C;\n" + """
fn rd(t: T, free: Free) : Res = match t with A(d, x) => Done@d(pack<16>(x), free)
  | B(d, y) => Done@d(pack<32>(y), free) | C(d) => Done@d(0, free) end;
fn bump(t: T) : T = match t with A(d, x) => A@d(+(x, 1)) | B(d, y) => B@d(+(y, 1)) | C(d) => C@d() end;
""" + main_with(1, "rd(bump(B@c0(16909060)), REST)")
        self.assertEqual(answer(self, source), 16909061)

    def test_it_is_only_a_field_type(self):
        for source in ("fn f(x: Be16) : Int64 = 0;\n", "fn f(x: Int16) : Be16 = x;\n", "fn f(x: Int16) : (Be16, Int8) = (x, 1);\n"):
            with self.subTest(source=source):
                with self.assertRaisesRegex(Reject, "says how a field lies in a cell"):
                    compile_program(source.encode())

    def test_nothing_can_be_done_to_it_as_it_lies(self):
        """A binder of a Be field has the integer's type, so this is ordinary typing, in the source's terms."""
        source = "record Q { a: Be16, b: Be32 };\nfn f(q: Q, free: Free) : Int64 = match q with Q(d, a, b) => +(a, b) end;\n"
        with self.assertRaisesRegex(Reject, "mismatch"):
            compile_program(source.encode())

    def test_a_record_that_spans_can_be_in_network_order(self):
        source = PRE + """record F { a: Be32, b: Be16, c: Be32, e: Be16, g: Be16, s: Be64 };
fn up(f: F) : F = match f with F(d, a, b, c, e, g, s) => F@d(a, +(b, 1), c, e, g, +(s, 1)) end;
fn rd(f: F, free: Free) : Res = match f with F(d, a, b, c, e, g, s) => let (p, rest) = peel(d) in
  Done@p(+(pack<32>(a), +(pack<16>(b), +(pack<32>(c), +(pack<16>(e), +(pack<16>(g), s))))), More@rest(free)) end;
""" + main_with(2, "rd(up(F@(c0, c1)(1, 2, 3, 4, 5, 6)), REST)")
        self.assertEqual(answer(self, source), 1 + 3 + 3 + 4 + 5 + 7)
        self.assertEqual(len(stores(body_of(source, "up"))), 2, body_of(source, "up"))


class WhatTheCheckerSays(unittest.TestCase):
    """rtcheck, on its own: a value in network order is not an integer until something swaps it."""

    TYPES = """type t16 record Q
ctor c0 t16 - Q
field c0 0 2 be16 a
field c0 2 2 i16 b
field c0 4 4 be32 w
"""

    def stream(self, *lines: str) -> list[str]:
        finish = [] if any(line.startswith("build") for line in lines) else ["build r0, c0, none, =, =, ="]
        return (self.TYPES + "fn f0 f(r0 : t16, r1 : i16, r2 : i32) : t16\n    open r0, c0\n"
                + "".join(f"    {line}\n" for line in (*lines, *finish))
                + "    ret r0\nend\n").splitlines()

    def accept(self, *lines: str) -> None:
        check_stream(self.stream(*lines))

    def refuse(self, pattern: str, *lines: str) -> None:
        with self.assertRaisesRegex(RTError, pattern):
            check_stream(self.stream(*lines))

    def test_read_it_is_bits_and_swapped_it_is_an_integer(self):
        self.accept("bswap r3, 16, r0[0:2]", "op r4, add/16, r3, #1", "bswap r5, 16, r4",
                    "build r0, c0, none, r5, =, =")

    def test_arithmetic_on_it_as_it_lies(self):
        self.refuse("is be16, and the operation is 16 bits", "op r3, add/16, r0[0:2], #1")
        self.refuse("is be16, and the operation is 16 bits", "read r3, r0[0:2]", "op r4, add/16, r3, #1")

    def test_a_comparison_of_it_as_it_lies(self):
        self.refuse("have to be swapped first", "cmp r3, e, r0[0:2], #8")
        self.refuse("have to be swapped first", "bswap r4, 16, r0[0:2]", "cmp r3, e, r0[0:2], r4")

    def test_packing_it_as_it_lies(self):
        self.refuse("is in network order, and has to be swapped first", "pack r3, <16>, r0[0:2]")

    def test_narrowing_it_as_it_lies(self):
        self.refuse("trunc of be16", "trunc r3, 8, r0[0:2]")

    def test_an_integer_stored_in_it_unswapped(self):
        self.refuse("Q.a: i16 against be16", "build r0, c0, none, r1, =, =")

    def test_bits_stored_in_an_integer_field(self):
        self.refuse("Q.b: be16 against i16", "read r3, r0[0:2]", "build r0, c0, none, =, r3, =")

    def test_a_register_of_the_wrong_width(self):
        self.refuse("bswap of 16 bits on r2, which is i32", "bswap r3, 16, r2")
        self.refuse("bswap of 32 bits on r0\\[0:2\\], which is be16", "bswap r3, 32, r0[0:2]")

    def test_a_byte_has_no_order(self):
        self.refuse("a byte has no order", "bswap r3, 8, r1")

    def test_a_literal_is_written_already_swapped(self):
        self.refuse("a literal is written already swapped", "bswap r3, 16, #8")
        self.accept("build r0, c0, none, #8, =, =")

    def test_a_literal_too_large_for_it(self):
        self.refuse("the literal 65536 is not a be16", "build r0, c0, none, #65536, =, =")

    def test_staying_is_for_one_order_only(self):
        other = """ctor c1 t16 - Q2
field c1 0 2 i16 a
field c1 2 2 i16 b
field c1 4 4 be32 w
""".replace("t16 - Q2", "t17 - Q2")
        stream = (self.TYPES + "type t17 record Q2\n" + other
                  + "fn f0 f(r0 : t16) : t17\n    open r0, c0\n    build r0, c1, none, =, =, =\n    ret r0\nend\n").splitlines()
        with self.assertRaisesRegex(RTError, "says STAY over a be16"):
            check_stream(stream)

    def test_a_swap_has_a_word_of_its_own(self):
        """Swapping twice is the same bits, but nothing here claims so: storing them back is a store."""
        self.accept("bswap r3, 16, r0[0:2]", "bswap r4, 16, r3", "build r0, c0, none, r4, =, =")
        self.refuse("says RESTORE", "bswap r3, 16, r0[0:2]", "bswap r4, 16, r3", "build r0, c0, none, r4!, =, =")

    def test_a_scalar_of_it_can_wait_in_a_register_and_be_dropped(self):
        self.accept("read r3, r0[0:2]", "copy r4, r3", "drop r3", "drop r4")


class WhatTheSymbolicCheckSays(unittest.TestCase):
    """src/symcheck.py, on assembly written by hand, so that it is not tied to one emitter."""

    SOURCE = b"record Q { a: Be16, b: Be32 };\n"

    def check(self, function: bytes, body: str) -> list:
        return check_translation(self.SOURCE + function, "fn_u_body:\n" + body)

    def test_a_store_that_is_swapped_means_the_field(self):
        function = b"fn u(q: Q, v: Int16) : Q @q = match q with Q(d, a, b) => Q@d(v, b) end;\n"
        good = "    mov rax, rdi\n    mov rbx, rsi\n    rol bx, 8\n    mov word [rdi], bx\n    ret\n"
        self.assertEqual(self.check(function, good), [])

    def test_a_store_that_is_not_is_found(self):
        function = b"fn u(q: Q, v: Int16) : Q @q = match q with Q(d, a, b) => Q@d(v, b) end;\n"
        bad = "    mov rax, rdi\n    mov word [rdi], si\n    ret\n"
        self.assertTrue(self.check(function, bad))

    def test_a_read_that_is_not_swapped_is_found(self):
        function = b"fn u(q: Q, r: Q) : Q @q = match q with Q(d, a, b) => Q@d(a, b) end;\n"
        self.assertEqual(self.check(function, "    mov rax, rdi\n    ret\n"), [])
        source = self.SOURCE + b"fn u(q: Q, v: Int64) : Int64 = match q with Q(d, a, b) => pack<16>(a) end;\n"
        # A diamond is never used here, so this is only about the value, which is what the check reads.
        good = "    movzx ebx, word [rdi]\n    rol bx, 8\n    mov rax, rbx\n    ret\n"
        bad = "    movzx ebx, word [rdi]\n    mov rax, rbx\n    ret\n"
        self.assertEqual(check_translation(source, "fn_u_body:\n" + good), [])
        self.assertTrue(check_translation(source, "fn_u_body:\n" + bad))

    def test_32_bits_the_same_way(self):
        function = b"fn u(q: Q, v: Int32) : Q @q = match q with Q(d, a, b) => Q@d(a, +(b, v)) end;\n"
        good = ("    mov rax, rdi\n    mov ebx, dword [rdi+4]\n    bswap ebx\n    add ebx, esi\n"
                "    bswap ebx\n    mov dword [rdi+4], ebx\n    ret\n")
        self.assertEqual(self.check(function, good), [])
        twice = good.replace("    bswap ebx\n    add ebx, esi\n    bswap ebx\n", "    add ebx, esi\n")
        self.assertTrue(self.check(function, twice))

    def test_a_rotate_that_is_not_a_swap_is_found(self):
        function = b"fn u(q: Q, v: Int16) : Q @q = match q with Q(d, a, b) => Q@d(v, b) end;\n"
        bad = "    mov rax, rdi\n    mov rbx, rsi\n    rol bx, 4\n    mov word [rdi], bx\n    ret\n"
        self.assertTrue(self.check(function, bad))

    def test_a_register_a_swap_changes_does_not_survive_the_call(self):
        """The callee's clobbers are read from its body, so a register it only swaps is one of them. The caller
        here counts on it, which only the check of the caller can see."""
        for swap in ("bswap ebx", "rol bx, 8"):
            source = (b"fn h(x: Int16) : Int16 = x;\n"
                      b"fn u(x: Int16) : Int16 = let y = h(x) in x;\n")
            asm = (f"fn_h_body:\n    mov rax, rdi\n    {swap}\n    ret\n"
                   "fn_u_body:\n    mov rbx, rdi\n    call fn_h\n    mov rax, rbx\n    ret\n")
            with self.subTest(swap=swap):
                self.assertTrue(check_translation(source, asm), asm)

    def test_a_cell_written_in_the_wrong_order_is_found(self):
        source = THREE.encode() + b"fn u(x: R, v: Int64) : R @x = match x with R(d, a, b, c) => R@d(a, b, v) end;\n"
        good = "    mov rax, rdi\n    mov rbx, qword [rdi+8]\n    mov qword [rbx+8], rsi\n    ret\n"
        wrong = "    mov rax, rdi\n    mov rbx, qword [rdi+8]\n    mov qword [rbx], rsi\n    ret\n"
        self.assertEqual(check_translation(source, "fn_u_body:\n" + good), [])
        self.assertTrue(check_translation(source, "fn_u_body:\n" + wrong))


if __name__ == "__main__":
    unittest.main()
