"""The new compiler path's memory: LFPL-O section 8.1 keeps the first specification's bound.

Nothing here may hold more than the current nesting depth asks for, however long the function
is. A long function is what catches a list that only ever grows, so the programs here are bushy:
a long text at a small depth. The cells are counted by the ring, and the Python heap is measured
from outside, in case something hid O(N) in an ordinary object.
"""
import gc
import io
import os
import sys
import tempfile
import tracemalloc
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import rtlower  # noqa: E402
from src.check import CheckStage  # noqa: E402
from src.dyck import Lexer, Parser  # noqa: E402
from src.lfplo import compile_program  # noqa: E402
from src.lower import build_and_run  # noqa: E402
from src.ring import Reject, Ring  # noqa: E402
from src.rtcheck import Checker, RTError  # noqa: E402
from src.rtlower import Limit, RTLower, required_cells, rt_lines  # noqa: E402
from src.symcheck import check_translation  # noqa: E402
from src.validate import validate  # noqa: E402
from src.x86 import Backend  # noqa: E402

PRELUDE = "data List = Nil | Cons(h: Int64, t: List);\n"


def _branching(depth: int) -> str:
    """A tree of ifs with a cell rebuilt at every leaf: each arm consumes names from outside it,
    so this is what would grow an undo log or a list of rebuilt cells."""
    if depth == 0:
        return "Cons@d(+(h, a), t)"
    return f"if <(a, {depth}) then {_branching(depth - 1)} else {_branching(depth - 1)}"


def _program(depth: int) -> bytes:
    name = "a" * 15
    body = _branching(depth).replace("(a,", f"({name},").replace(", a)", f", {name})")
    return (PRELUDE + f"fn f({name}: Int64, l: List) : List = match l with\n"
            f"  | Nil => Nil\n  | Cons(d, h, t) => {body}\n  end;\n").encode()


def _compile(source: bytes, on_step=None) -> None:
    """The whole stream, through the checker and the backend, as the driver runs it."""
    checker = Checker()
    backend = Backend(lambda line: None)
    for line in rt_lines(io.BytesIO(source), on_step):
        checker.feed(line)
        backend.feed(line)
    checker.finish()


def _heap_peak(source: bytes) -> int:
    gc.collect()
    tracemalloc.start()
    _compile(source)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


class Memory(unittest.TestCase):
    def _sources(self) -> tuple[bytes, bytes]:
        return _program(9), _program(11)

    def test_the_heap_does_not_follow_the_length_of_a_function(self):
        small, large = self._sources()
        self.assertGreater(len(large), 90_000)
        allowance = len(large) // 4
        _heap_peak(small)  # the first compile also fills the interpreter's caches
        growth = _heap_peak(large) - _heap_peak(small)
        self.assertLess(growth, allowance, (growth, allowance, len(large)))

    def test_the_cells_follow_the_depth_and_not_the_length(self):
        def peak(depth: int) -> int:
            high = [0]

            def on_step(ring: Ring) -> None:
                high[0] = max(high[0], ring.store.C - len(ring.F))

            _compile(_program(depth), on_step)
            return high[0]

        two_levels = peak(10) - peak(8)
        one_level = peak(8) - peak(7)
        self.assertLessEqual(two_levels, 2 * one_level + 8, (two_levels, one_level))

    def test_control_a_stage_that_keeps_every_line_fails_the_heap_test(self):
        def keeping_peak(source: bytes) -> int:
            kept = []
            gc.collect()
            tracemalloc.start()
            for line in rt_lines(io.BytesIO(source)):
                kept.append(line)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            return peak

        small, large = self._sources()
        growth = keeping_peak(large) - keeping_peak(small)
        self.assertGreaterEqual(growth, len(large) // 4)

    def test_nesting_to_the_limit_never_starves_the_ring(self):
        """required_cells is the claim: a program at the depth limit must fit in that many."""
        expression = "1"
        for _ in range(28):
            expression = f"let z = +(1, {expression}) in +(z, 1)"
        source = f"fn f(a: Int64) : Int64 = {expression};\n"
        stages = [Lexer(), Parser(64), CheckStage(lfplo=True), RTLower()]
        ring = Ring(required_cells(64), iter(source.encode()), stages, lambda tag, payload: None)
        ring.run()
        self.assertGreater(ring.peak_depth, 20)


# Programs that take cells: main gets a free list, and a list digests to a number, two decimal
# digits per element, so a wrong or a misplaced value changes the answer.
CELLS = """data Free = End | More(next: Free);
data Res = Fail | Done(value: Int64, rest: Free);
data List = Nil | Cons(h: Int64, t: List);
fn back(l: List, acc: Int64, free: Free) : (Int64, Free) = match l with
  | Nil => (acc, free) | Cons(d, h, t) => back(t, +(*(acc, 100), h), More@d(free)) end;
fn done(a: Int64, f: Free) : Res = match f with End => Fail | More(c, r) => Done@c(a, r) end;
"""
TAGGED = "data S = A(x: Int32) | B(y: Int32);\n"


def two_cells(definitions: str, expression: str) -> bytes:
    """A program whose answer digests the list `expression` builds from cells c0 and c1."""
    return (CELLS + definitions + """
fn main(r0: Free) : Res = match r0 with End => Fail | More(c0, r1) => match r1 with
  | End => Done@c0(0, End)
  | More(c1, r2) => let (a, f) = back(""" + expression + """, 0, r2) in done(a, f) end end;
""").encode()


def one_cell(definitions: str, body: str) -> bytes:
    """A program whose answer `body` works out, from cell c0 and the rest of the list, r1."""
    return (CELLS + definitions + """
fn main(r0: Free) : Res = match r0 with End => Fail | More(c0, r1) => """ + body + """ end;
""").encode()


class ValidPrograms(unittest.TestCase):
    """Well-typed programs that were refused, or that crashed the compiler. Each one must compile,
    pass the symbolic check and the redundancy check, and print what the source means."""

    def answer(self, source: bytes) -> int:
        asm = compile_program(source)
        self.assertEqual([str(p) for p in check_translation(source, asm)], [], asm)
        self.assertEqual(validate(asm), [], asm)
        with tempfile.TemporaryDirectory() as workdir:
            return build_and_run(asm, workdir)

    def test_a_logical_connective_in_tail_position(self):
        self.assertEqual(self.answer(b"""fn f(a: Bool) : Bool = !(a);
fn main() : Int64 = if f(false) then 7 else 9;"""), 7)
        self.assertEqual(self.answer(b"""fn f() : Bool = ||(false, true);
fn main() : Int64 = if f() then 7 else 9;"""), 7)
        self.assertEqual(self.answer(b"""fn f(a: Bool, b: Bool) : Bool = &&(a, !(b));
fn main() : Int64 = if f(true, true) then 7 else 9;"""), 9)

    def test_a_second_arm_knows_the_word_of_what_it_matched(self):
        program = two_cells("""
fn f(l: List) : List = let r = match l with
  | Nil => Nil
  | Cons(d, h, t) => match t with
      | Nil => Cons@d(1, Nil)
      | Cons(e, h2, t2) => Cons@d(5, Cons@e(h2, t2)) end end in r;""",
                            "f(Cons@c0(10, Cons@c1(20, Nil)))")
        self.assertEqual(self.answer(program), 520)

    def test_a_scalar_through_a_tuple_join_is_not_stored_back(self):
        program = one_cell("""record R { a: Int64, b: Int64 };
fn f(r: R, c: Bool) : R = match r with R(d, a, b) =>
  let (x, y) = if c then (a, b) else (a, b) in R@d(x, y) end;""",
                           "match f(R@c0(3, 4), true) with R(d, a, b) => Done@d(+(*(a, 10), b), r1) end")
        self.assertEqual(self.answer(program), 34)

    def test_a_diamond_and_its_own_field_through_a_tuple_join(self):
        program = two_cells("""
fn f(l: List, p: Bool) : List = match l with
  | Nil => Nil
  | Cons(d, h, t) => let (c, u) = if p then (d, t) else (d, t) in Cons@c(1, u) end;""",
                            "f(Cons@c0(10, Cons@c1(20, Nil)), true)")
        self.assertEqual(self.answer(program), 120)

    def test_two_diamonds_chosen_at_a_join(self):
        definitions = """
fn f(a: Diamond, b: Diamond, p: Bool) : List =
  let (x, y) = if p then (a, b) else (b, a) in Cons@x(1, Cons@y(2, Nil));"""
        self.assertEqual(self.answer(two_cells(definitions, "f(c0, c1, true)")), 102)
        self.assertEqual(self.answer(two_cells(definitions, "f(c0, c1, false)")), 102)

    def test_one_temporary_twice_in_a_tuple(self):
        self.assertEqual(self.answer(b"""
fn f(x: Int64) : Int64 = +(let (a, b) = let y = +(x, 1) in (y, y) in +(a, b), 1);
fn main() : Int64 = f(100);"""), 203)

    def test_a_known_constructor_keeps_the_tag_its_cell_has(self):
        program = one_cell("""data T = A(x: Int32) | B(y: Int32) | C;
fn f(s: T) : T = match s with
  | A(d, x) => match A@d(+(x, 1)) with A(e, x2) => A@e(x2) | B(e, y2) => B@e(y2) | C(e) => C@e() end
  | B(d, y) => B@d(y)
  | C(d) => C@d() end;""", """match f(A@c0(41)) with
    | A(d, x) => Done@d(pack<32>(x), r1) | B(d, y) => Done@d(0, r1) | C(d) => Done@d(1, r1) end""")
        self.assertEqual(self.answer(program), 42)

    def test_a_joined_diamond_still_builds(self):
        program = one_cell(TAGGED + """
fn f(s: S, p: Bool) : S = match s with
  | A(d, x) => let e = if p then d else d in A@e(1)
  | B(d, y) => B@d(y) end;""", """match f(A@c0(41), true) with
    | A(d, x) => Done@d(pack<32>(x), r1) | B(d, y) => Done@d(0, r1) end""")
        self.assertEqual(self.answer(program), 1)

    def test_a_joined_diamond_moves_a_linear_field_with_it(self):
        program = two_cells("""
fn f(l: List, p: Bool) : List = match l with
  | Nil => Nil
  | Cons(d, h, t) => let e = if p then d else d in Cons@e(7, t) end;""",
                            "f(Cons@c0(10, Cons@c1(20, Nil)), false)")
        self.assertEqual(self.answer(program), 720)

    def test_a_joined_diamond_cannot_read_a_scalar_back_yet(self):
        """Moving a diamond counts as handing it over. This is section 8.1's limit, with a reason a
        program can act on, until the right-to-left pass says the scalar is wanted later."""
        program = two_cells("""
fn f(l: List, p: Bool) : List = match l with
  | Nil => Nil
  | Cons(d, h, t) => let e = if p then d else d in Cons@e(h, t) end;""", "f(Cons@c0(1, Nil), true)")
        with self.assertRaisesRegex(Limit, "section 8.1"):
            compile_program(program)

    def test_a_rebuilt_cell_can_be_matched_again(self):
        program = one_cell(TAGGED + """
fn f(s: S) : S = match s with
  | A(d, x) => A@d(x)
  | B(d, y) => let v = A@d(5) in match v with A(e, x2) => A@e(x2) | B(e, y2) => B@e(y2) end end;""",
                           """match f(B@c0(41)) with
    | A(d, x) => Done@d(pack<32>(x), r1) | B(d, y) => Done@d(0, r1) end""")
        self.assertEqual(self.answer(program), 5)

    def test_a_diamond_returned_by_a_function(self):
        program = two_cells("""
fn idd(c: Diamond) : Diamond = c;
fn g(c: Diamond) : List = let e = idd(c) in Cons@e(1, Nil);""", "Cons@c0(4, g(c1))")
        self.assertEqual(self.answer(program), 401)

    def test_a_function_name_cannot_collide_with_a_label(self):
        self.assertEqual(self.answer(b"""fn a_body(x: Int64) : Int64 = +(x, 1);
fn a(x: Int64) : Int64 = a_body(x);
fn main() : Int64 = a(6);"""), 7)

    def test_a_known_constructor_that_is_not_the_first_arm(self):
        program = one_cell("""data T = A(x: Int8) | B(y: Int8);
fn f(d: Diamond, p0: Int8, p1: Int8, p2: Int8) : T =
  let r = match B@d(p2) with A(e, x) => A@e(x) | B(e, y) => B@e(+(y, 1)) end in r;""",
                           """match f(c0, 1, 2, 3) with
    | A(d, x) => Done@d(0, r1) | B(d, y) => Done@d(pack<8>(y), r1) end""")
        self.assertEqual(self.answer(program), 4)

    def test_cells_rotated_at_a_join_are_not_a_cycle(self):
        """Followed without regard to the path, these two cells point at each other through two
        arms that can never both happen. That once sent the symbolic check into a stack overflow."""
        rotate = """match l with
  | Nil => Nil
  | Cons(d, h, t) => match t with
      | Nil => Cons@d(h, Nil)
      | Cons(e, h2, t2) => Cons@e(h2, Cons@d(h, t2)) end end"""
        into_a_call = two_cells(f"fn g(l: List) : List = l;\nfn f(l: List) : List = g({rotate});",
                                "f(Cons@c0(10, Cons@c1(20, Nil)))")
        self.assertEqual(self.answer(into_a_call), 2010)
        into_a_tuple = two_cells(f"""fn f(l: List) : (Int8, List) = (1, {rotate});
fn second(l: List) : List = let (n, m) = f(l) in m;""", "second(Cons@c0(10, Cons@c1(20, Nil)))")
        self.assertEqual(self.answer(into_a_tuple), 2010)

    def test_a_bool_from_a_helpers_tuple_is_a_condition(self):
        self.assertEqual(self.answer(b"""fn hb(x: Int8) : (Int8, Bool) = (x, <(x, 5));
fn main() : Int64 = let (y, b) = hb(3) in if b then 7 else 9;"""), 7)

    def test_a_tagged_cell_kept_through_a_join(self):
        program = one_cell("""data X = A(x: Int8) | B(y: Int8);
fn r(s: X) : (Int8, X) = (1, match s with A(d, x) => A@d(+(x, 1)) | B(d, y) => B@d(y) end);""",
                           """let (k, v) = r(A@c0(3)) in match v with
    | A(e, x) => Done@e(+(pack<8>(x), pack<8>(k)), r1) | B(e, y) => Done@e(0, r1) end""")
        self.assertEqual(self.answer(program), 5)

    def test_a_width_that_comes_through_a_join_with_a_literal_first(self):
        program = b"""fn f(a: Int16, s: Int8) : Int16 =
  let (x, n) = if ==(s, 0) then (1, 3) else (a, 2) in >>(x, n);
fn main() : Int64 = +(pack<16>(f(65535, 1)), pack<16>(f(65535, 0)));"""
        self.assertEqual(self.answer(program), 16383)

    def test_a_cell_one_arm_gives_away_and_another_keeps(self):
        definitions = """
fn rev(l: List, acc: List) : List = match l with Nil => acc | Cons(d, h, t) => rev(t, Cons@d(h, acc)) end;
fn f(k: Bool, a: Diamond, b: Diamond) : List =
  if k then Cons@b(1, rev(Cons@a(2, Nil), Nil)) else Cons@b(3, Cons@a(4, Nil));"""
        self.assertEqual(self.answer(two_cells(definitions, "f(true, c0, c1)")), 102)
        self.assertEqual(self.answer(two_cells(definitions, "f(false, c0, c1)")), 304)

    def test_a_cell_only_another_arm_wrote_to_is_still_itself(self):
        """On the path that does not touch it, nothing has to say which constructor it holds."""
        program = one_cell(TAGGED + """
fn f(p: Bool, s: S) : (Int8, S) = (1, if p then match s with
    | A(d, x) => A@d(+(x, 1))
    | B(d, y) => B@d(y) end else s);""", """let (k, v) = f(false, A@c0(41)) in match v with
    | A(e, x) => Done@e(pack<32>(x), r1) | B(e, y) => Done@e(0, r1) end""")
        self.assertEqual(self.answer(program), 41)

    def test_a_shift_count_past_the_width_is_legal_and_gives_zero(self):
        self.assertEqual(self.answer(b"""fn f(a: Int8) : Int8 = <<(a, 300);
fn main() : Int64 = pack<8>(f(5));"""), 0)

    def test_pack_keeps_the_low_bits_of_what_it_is_given(self):
        self.assertEqual(self.answer(b"fn main() : Int64 = pack<8>(300);"), 44)

    def test_a_comment_can_hold_any_byte(self):
        self.assertEqual(self.answer(b"# caf\xe9\nfn main() : Int64 = 1;"), 1)


class TheDecreasingSet(unittest.TestCase):
    """`end` states what R3 found, because that is what carries the bound."""

    def ends(self, source: str) -> list:
        return [line for line in rt_lines(io.BytesIO((PRELUDE + source).encode()))
                if line.startswith("end")]

    def test_a_function_that_does_not_recurse_names_nothing(self):
        self.assertEqual(self.ends("fn f(x: Int64) : Int64 = +(x, 1);\n"), ["end"])

    def test_a_walk_names_the_parameter_it_walks(self):
        self.assertEqual(self.ends("""fn f(acc: List, l: List) : List = match l with
  | Nil => acc
  | Cons(d, h, t) => f(Cons@d(h, acc), t)
  end;
"""), ["end decreasing 1"])

    def test_a_walk_that_trades_between_two_names_them_both(self):
        self.assertEqual(self.ends("""fn f(x: List, y: List, acc: List) : List = match x with
  | Nil => match y with
      | Nil => acc
      | Cons(d, h, t) => f(Nil, t, Cons@d(h, acc))
      end
  | Cons(d, h, t) => f(y, t, Cons@d(h, acc))
  end;
"""), ["end decreasing 0,1"])


class RefusedWithAReason(unittest.TestCase):
    def refuse(self, why: str, source: bytes) -> None:
        with self.assertRaisesRegex((Reject, Limit, RTError), why):
            compile_program(source)

    def test_a_literal_that_does_not_fit_what_it_meets(self):
        """These gave wrong answers past every check: a register held an Int8 of 300, which was
        44 to + and not 44 to ==."""
        k = "fn k(a: Int8) : Int8 = a;\n"
        for expression in ("k(300)", "let x = 300 in k(x)", "if ==(a, 1) then 300 else a",
                           "==(300, a)", "|(a, 300)", ">>(300, a)", "+(a, 256)"):
            with self.subTest(expression=expression):
                ret = "Bool" if expression.startswith("==") else "Int8"
                self.refuse("does not fit|is not a", f"{k}fn f(a: Int8) : {ret} = {expression};".encode())

    def test_the_largest_literal_of_each_width_fits(self):
        for bits, ty, value in ((8, "Int8", 255), (16, "Int16", 65535), (32, "Int32", 4294967295),
                                (64, "Int64", 18446744073709551615)):
            with self.subTest(ty=ty):
                compile_program(f"fn f(a: {ty}) : {ty} = +(a, {value});".encode())

    def test_a_tuple_of_one(self):
        self.refuse("at least two", b"fn g(x: Int64) : Int64 = let (a) = x in a;")
        self.refuse("at least two", b"fn g(x: Int64) : Int64 = +(1, (x));")
        self.refuse("at least two", b"fn g(x: Int64) : (Int64) = x;")

    def test_a_tuple_in_a_tuple(self):
        self.refuse("cannot hold a tuple",
                    b"fn g(x: Int64) : Int64 = let (a, b) = ((x, 2), 3) in b;")

    def test_a_tuple_as_a_condition(self):
        self.refuse("must be Bool", b"fn g(x: Int64) : Int64 = if (true, false) then 1 else 2;")
        self.refuse("must be Bool", b"""fn g(x: Int8) : (Int8, Int8) = (x, x);
fn f(x: Int8) : Int8 = if g(x) then 1 else 2;""")
        self.refuse("must be Bool",
                    b"fn f(a: Array[8] Int8, i: Fin 8) : Int8 = if get(a, i) then 1 else 2;")

    def test_pack_with_nothing_to_pack(self):
        self.refuse("one expression per width", b"fn g(x: Int64) : Int64 = pack<8>();")

    def test_more_than_can_be_counted(self):
        many = ", ".join(["x"] * 257)
        self.refuse("at most 256", f"fn g(x: Int64) : Int64 = pack<8>({many});".encode())
        self.refuse("more than 6", f"fn g(x: Int64) : Int64 = let (a, b) = ({many}) in a;".encode())
        names = ", ".join(f"n{i}" for i in range(30000))
        self.refuse("more than 6", f"fn g(x: Int64) : Int64 = let ({names}) = x in x;".encode())
        types = ", ".join(["Int8"] * 30000)
        self.refuse("more than 6", f"fn g(x: Int64) : ({types}) = x;".encode())

    def test_more_registers_than_a_value_can_name(self):
        with mock.patch.object(rtlower, "MAX_REGISTERS", 8):
            self.refuse("more than 8 registers",
                        b"fn g(a: Int64) : Int64 = +(+(+(a, a), +(a, a)), +(+(a, a), +(+(a, a), +(a, a))));")

    def test_a_main_that_start_cannot_call(self):
        self.refuse("free list", b"fn main(x: Int64) : Int64 = 5;")


if __name__ == "__main__":
    unittest.main()
