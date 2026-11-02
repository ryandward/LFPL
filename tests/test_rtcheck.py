"""The RT checker against hand-written streams: docs/rt-layer.md's rules, with no compiler.

Every row of that document's "What this would have caught" table is here as a stream that
must be refused, next to the corrected stream that must pass, so a rule that refuses everything
cannot pass by accident.
"""
import ast
import os
import unittest
from unittest import mock

from fractions import Fraction

from src.certify import certify
from src.rtcheck import Budget, RTError, Unbounded, check_stream
from src.simplex import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LIST = """
type t16 tagless List
ctor c17 t16 null Nil
ctor c18 t16 - Cons
field c18 0 8 i64 head
field c18 8 8 t16 tail
"""
SHAPE = """
type t20 tagged Shape
ctor c21 t20 0 Dot
field c21 8 8 i64 x
ctor c22 t20 1 Box
field c22 1 1 i8 w
field c22 8 8 i64 y
"""
FORWARD = """
type t16 abstract Bytes
type t17 record Hdr
ctor c18 t17 - Hdr
field c18 0 1 i8 ttl
field c18 4 4 i32 dst
field c18 8 8 t16 body
type t22 tagged Verdict
ctor c23 t22 0 Drop
field c23 8 8 t16 body
ctor c25 t22 1 Fwd
field c25 1 1 i8 port
field c25 2 1 i8 ttl
field c25 4 4 i32 dst
field c25 8 8 t16 body
type t30 tagless Free
ctor c31 t30 null End
ctor c32 t30 - More
field c32 0 8 t30 next
array t32768 1024 i8
tuple t32770 t32768 t22 t30
state routes t32768
"""
HANDLER = """
handler f0 forward(r0 : t32768, r1 : t17, r2 : t30) : t32770
    open r1, c18
    br be, r1[0:1], #1, L1, L2
L1: build r1, c23, write, =
    ret r0, r1, r2
L2: idx r3, 1024, r1[4:4]
    aget r4, r0, r3, 1
    op r5, sub/8, r1[0:1], #1
    build r1, c25, write, r4, r5, =, =
    ret r0, r1, r2
end
"""
INC = """
fn f1 inc(r0 : t16) : t16 @0
    br.null r0, L2, L3
L3: tailmod r0, c18, none, add([0:8], #1), *, f1, r0[8:8]
L2: const r1, #0 : t16
    ret r1
end decreasing 0
"""


def check(*parts: str):
    return check_stream("\n".join(parts).splitlines())


class Accepted(unittest.TestCase):
    def test_the_inc_example_of_the_specification(self):
        check(LIST, INC)

    def test_the_forwarding_handler(self):
        bound = certify(check(FORWARD, HANDLER), "forward", 216)
        self.assertEqual((bound.a, bound.b, bound.c), (0, 0, 0))
        self.assertGreater(bound.d, 0)


class Refused(unittest.TestCase):
    """One test per kind of mistake. Each states the corrected stream beside the wrong one
    where the difference is small enough that passing alone would prove little."""

    def refuse(self, pattern: str, *parts: str):
        with self.assertRaisesRegex(RTError, pattern):
            check(*parts)

    def test_a_literal_result_is_the_null_of_a_tagless_type_and_nothing_else(self):
        check(LIST, "fn f1 g() : t16\n    ret #0\nend\n")
        self.refuse("only a tagless type has a literal", LIST, "fn f1 g() : t16\n    ret #1\nend\n")
        self.refuse("only a tagless type has a literal", LIST, "fn f1 g() : i64\n    ret #0\nend\n")
        self.refuse("only a tagless type has a literal", LIST, "tuple t40 t16 i64\n",
                    "fn f1 g() : t40\n    ret #0, #0\nend\n")

    def test_a_null_result_is_held_to_the_word_declared(self):
        check(LIST, INC)
        check(LIST, INC.replace("const r1, #0 : t16\n    ret r1", "ret #0"))
        self.refuse("declared @0", LIST, "fn f1 g(r0 : t16) : t16 @0\n    ret #0\nend\n")

    def test_a_join_with_two_words_cannot_claim_restore(self):
        """Phase 1's join word taken from its last arm, which dropped a store."""
        stream = """
fn f1 g(r0 : t16, r1 : i64) : t16
    br.null r0, L1, L2
L1: const r5, #0 : t16
    ret r5
L2: take r2, r0[8:8]
    br e, r1, #0, L3, L4
L3: call r3, f0, r2
    jump L5
L4: move r3, r2
    jump L5
L5: build r0, c18, none, =, {source}
    ret r0
end
"""
        callee = """
fn f0 id(r0 : t16) : t16
    ret r0
end
"""
        self.refuse("RESTORE", LIST, callee, stream.format(source="r3!"))
        check(LIST, callee, stream.format(source="r3"))

    def test_a_value_from_before_a_rebuild_is_not_what_the_cell_holds_now(self):
        """The old tail, taken out, then a new tail built in, then the cell opened again. Saying
        the old tail is still there would skip the store that puts it back."""
        stream = """
type t40 record Pair
ctor c41 t40 - Pair
field c41 0 8 t16 a
field c41 8 8 t16 b
fn f1 g(r0 : t16, r1 : cap) : t40
    br.null r0, L1, L2
L1: const r5, #0 : t16
    const r6, #0 : t16
    build r1, c41, none, r5, r6
    ret r1
L2: take r2, r0[8:8]
    build r0, c18, none, =, #0
    br.null r0, L3, L4
L3: const r6, #0 : t16
    build r1, c41, none, r2, r6
    ret r1
L4: take r3, r0[8:8]
    build r0, c18, none, =, r2{mark}
    build r1, c41, none, r0, r3
    ret r1
end
"""
        self.refuse("RESTORE", LIST, stream.format(mark="!"))
        check(LIST, stream.format(mark=""))

    def test_a_literal_is_not_the_word_of_a_location(self):
        stream = """
fn f1 g(r0 : t16) : t16
    br.null r0, L1, L2
L1: const r5, #0 : t16
    ret r5
L2: const r1, #7 : i64
    take r2, r0[8:8]
    build r0, c18, none, r1!, r2!
    ret r0
end
"""
        self.refuse("RESTORE", LIST, stream)

    def test_a_cell_cannot_be_read_after_it_is_rebuilt(self):
        stream = """
fn f1 g(r0 : t16) : i64
    br.null r0, L1, L2
L1: const r5, #0 : i64
    ret r5
L2: build r0, c18, none, #1, =
    read r1, r0[0:8]
    ret r1
end
"""
        self.refuse("capability", LIST, stream)

    def test_a_cell_cannot_be_read_after_a_callee_gets_it(self):
        stream = """
fn f0 g(r0 : cap, r1 : t16) : t16
    build r0, c18, none, #1, r1
    ret r0
end
fn f1 h(r0 : t16) : i64
    br.null r0, L1, L2
L1: const r5, #0 : i64
    ret r5
L2: take r2, r0[8:8]
    call r3, f0, r0, r2
    read r4, r0[0:8]
    ret r4
end
"""
        self.refuse("holds nothing", LIST, stream)

    def test_a_tag_fact_from_one_arm_does_not_reach_the_other(self):
        stream = """
fn f1 g(r0 : t20) : t20
    br.tag r0, 0, L1, L2
L1: build r0, c21, keep, =
    ret r0
L2: open r0, c22
    build r0, c21, {tag}, =
    ret r0
end
"""
        self.refuse("keep", SHAPE, stream.format(tag="keep"))
        check(SHAPE, stream.format(tag="write"))

    def test_a_pointer_still_owed_cannot_be_overwritten(self):
        """Phase 1's tag byte over a pointer at offset 0: nothing names the pointer."""
        stream = """
type t40 record Link
ctor c41 t40 - Link
field c41 0 8 t16 next
type t42 tagged Tagged
ctor c43 t42 0 Plain
field c43 8 8 i64 x
fn f1 g(r0 : t40) : t42
    open r0, c41
    build r0, c43, write, #5
    ret r0
end
"""
        self.refuse("would be lost", LIST, stream)

    def test_a_deferred_read_cannot_outlive_a_rebuild(self):
        stream = """
fn f1 g(r0 : t16) : t16
    br.null r0, L1, L2
L1: const r5, #0 : t16
    ret r5
L2: build r0, c18, none, #1, =
    build r0, c18, none, r0[0:8], =
    ret r0
end
"""
        self.refuse("capability", LIST, stream)

    def test_an_operation_states_its_width(self):
        stream = """
fn f1 g(r0 : i64) : i8
    op r1, add/8, r0, #1
    ret r1
end
"""
        self.refuse("64 bits|is i64", stream)

    def test_theorem_6_1_writing_a_value_back_is_ill_typed(self):
        stream = """
fn f1 g(r0 : t16) : t16
    br.null r0, L1, L2
L1: const r5, #0 : t16
    ret r5
L2: take r2, r0[8:8]
    build r0, c18, none, =, r2{mark}
    ret r0
end
"""
        self.refuse("Theorem 6.1", LIST, stream.format(mark=""))
        check(LIST, stream.format(mark="!"))

    def test_a_linear_value_cannot_be_used_twice(self):
        stream = """
fn f1 g(r0 : t16) : t16
    move r1, r0
    move r2, r0
    ret r2
end
"""
        self.refuse("holds nothing", LIST, stream)

    def test_a_linear_value_cannot_be_left_behind(self):
        stream = """
fn f1 g(r0 : t16, r1 : t16) : t16
    ret r0
end
"""
        self.refuse("still in r1", LIST, stream)

    def test_a_linear_value_cannot_exist_on_one_edge_only(self):
        stream = """
fn f1 g(r0 : t16, r1 : i64) : t16
    br e, r1, #0, L1, L2
L1: jump L3
L2: move r2, r0
    jump L3
L3: ret r0
end
"""
        self.refuse("one edge", LIST, stream)

    def test_an_index_must_carry_the_array_length(self):
        stream = """
array t32768 1024 i8
array t32769 16 i8
fn f1 g(r0 : t32768, r1 : t32769, r2 : i8) : i8
    idx r3, 16, r2
    aget r4, r1, r3, 1
    aget r5, r0, r3, 1
    ret r5
end
"""
        self.refuse("index", stream)

    def test_idx_needs_a_power_of_two(self):
        stream = """
fn f1 g(r0 : i64) : i64
    idx r1, 1000, r0
    ret r0
end
"""
        self.refuse("power of two", stream)

    def test_an_abstract_type_cannot_be_opened(self):
        stream = """
type t16 abstract Bytes
type t17 record Hdr
ctor c18 t17 - Hdr
field c18 0 8 t16 body
fn f1 g(r0 : t16) : t16
    open r0, c18
    ret r0
end
"""
        self.refuse("open", stream)

    def test_a_non_tail_recursive_call_is_refused(self):
        stream = """
fn f1 g(r0 : t16) : t16
    call r1, f1, r0
    ret r1
end
"""
        self.refuse("non-tail", LIST, stream)

    def test_tailmod_needs_the_word_the_call_returns(self):
        stream = """
fn f1 inc(r0 : t16, r1 : t16) : t16 @0
    br.null r0, L2, L3
L3: take r2, r0[8:8]
    tailmod r0, c18, none, =, *, f1, r1, r2
L2: const r3, #0 : t16
    br.null r1, L4, L5
L4: ret r3
L5: tailmod r1, c18, none, =, *, f1, r1[8:8], r3
end
"""
        self.refuse(r"hole|still", LIST, stream)

    def test_the_end_arm_of_a_draw_must_type_and_is_not_charged(self):
        """Section 5.3's DRAW, on both sides of a join: what only that arm reaches costs nothing,
        and what comes after the join is charged as the arm with a cell left it."""
        template = """
handler f1 h(r0 : t32768, r1 : t17, r2 : t30) : t32770
    open r1, c18
    build r1, c23, write, =
    br.null r2, L1, L2
L1: const r3, #0 : t30
{end_arm}    jump L3
L2: move r3, r2
    build r3, c32, none, =
    jump L3
L3: const r9, #1 : i8
    drop r9
    ret r0, r1, r3
end
"""
        costly = "".join(f"    const r{10 + k}, #1 : i8\n    drop r{10 + k}\n" for k in range(40))
        plain = certify(check(FORWARD, template.format(end_arm="")), "h", 216)
        dear = certify(check(FORWARD, template.format(end_arm=costly)), "h", 216)
        self.assertEqual(plain, dear)
        self.assertEqual(dear.b, 1)
        broken = template.format(end_arm="    drop r77\n")
        self.refuse("r77", FORWARD, broken)

    def test_a_handler_that_makes_a_free_has_no_bound(self):
        """Its End arm is reachable, and DRAW leaves End arms out. Here that would be a walk of the
        reserve whose accumulator starts from a literal. Check would not let it get this far, and
        this does not rest on that."""
        stream = """
fn f0 walk(r0 : t30, r1 : t30) : t30
    br.null r0, L1, L2
L1: ret r1
L2: take r2, r0[0:8]
    build r0, c32, none, r1
    tail f0, r2, r0
end decreasing 0
handler f1 h(r0 : t32768, r1 : t17, r2 : t30) : t32770
    const r3, #0 : t30
    call r4, f0, r2, r3
    open r1, c18
    build r1, c23, write, =
    ret r0, r1, r4
end
"""
        with self.assertRaisesRegex(Unbounded, "makes a Free"):
            certify(check(FORWARD, stream), "h", 216)

    FREE_HELPERS = """
fn f0 walk(r0 : t30, r1 : t30) : t30
    br.null r0, L1, L2
L1: ret r1
L2: take r2, r0[0:8]
    build r0, c32, none, r1
    tail f0, r2, r0
end decreasing 0
fn f1 none(r0 : i8) : t30
    const r1, #0 : t30
    ret r1
end
fn f2 wrap(r0 : cap) : t30
    build r0, c32, none, #0
    ret r0
end
"""

    @staticmethod
    def _serving(body: str) -> str:
        return f"""
handler f3 h(r0 : t32768, r1 : t17, r2 : t30) : t32770
{body}
    open r1, c18
    build r1, c23, write, =
    ret r0, r1, r9
end
"""

    def test_every_way_of_making_a_free_where_it_can_run_is_refused(self):
        """walk merges two Frees, which R3 would not let a compiled program do, so each of these
        types, and each of them reaches an End arm the bound would leave out."""
        for name, body in (
                ("a literal in a register", "    const r3, #0 : t30\n    call r9, f0, r2, r3"),
                ("a literal as an argument", "    call r9, f0, r2, #0"),
                ("a call that makes one", "    call r3, f1, #0\n    call r9, f0, r2, r3"),
                ("a literal as a field", """    br.null r2, L1, L2
L1: const r9, #0 : t30
    jump L3
L2: take r3, r2[0:8]
    call r4, f2, r2
    call r9, f0, r3, r4
    jump L3
L3:""")):
            with self.subTest(name), self.assertRaisesRegex(Unbounded, "makes a Free"):
                certify(check(FORWARD, self.FREE_HELPERS, self._serving(body)), "h", 216)

    def test_a_free_made_past_an_end_arm_is_not_a_second_one(self):
        """The reserve has gone there, and that path is out of the bound anyway."""
        kept = """L2: build r2, c32, none, =
    move r9, r2
    jump L3
L3:"""
        for name, end_arm in (("a literal", "const r9, #0 : t30"), ("a call", "call r9, f1, #0")):
            body = f"    br.null r2, L1, L2\nL1: {end_arm}\n    jump L3\n{kept}"
            with self.subTest(name):
                self.assertEqual(certify(check(FORWARD, self.FREE_HELPERS, self._serving(body)), "h", 216).b, 1)

    def test_a_draw_from_the_reserve_is_counted(self):
        stream = """
handler f1 h(r0 : t32768, r1 : t17, r2 : t30) : t32770
    open r1, c18
    build r1, c23, write, =
    br.null r2, L1, L2
L1: const r3, #0 : t30
    ret r0, r1, r3
L2: build r2, c32, none, =
    ret r0, r1, r2
end
"""
        self.assertEqual(certify(check(FORWARD, stream), "h", 216).b, 1)


ITEM = LIST + """
type t26 record Item
ctor c27 t26 - Item
field c27 0 1 i8 port
field c27 8 8 t16 body
type t30 tagless Free
ctor c31 t30 null End
ctor c32 t30 - More
field c32 0 8 t30 next
tuple t50 t26 t30
"""
REVERSE = """
fn f0 rev(r0 : t16, r1 : t16) : t16
    br.null r0, L1, L2
L1: ret r1
L2: take r2, r0[8:8]
    build r0, c18, none, =, r1
    tail f0, r2, r0
end decreasing 0
"""
ONCE = """
handler f1 h(r0 : t26, r1 : t30) : t50
    open r0, c27
    take r2, r0[8:8]
    const r3, #0 : t16
    call r4, f0, r2, r3
    build r0, c27, none, =, r4
    ret r0, r1
end
"""
TWICE = """
handler f1 h(r0 : t26, r1 : t30) : t50
    open r0, c27
    take r2, r0[8:8]
    const r3, #0 : t16
    call r4, f0, r2, r3
    const r5, #0 : t16
    call r6, f0, r4, r5
    build r0, c27, none, =, r6
    ret r0, r1
end
"""


class Potential(unittest.TestCase):
    """The bounds of LFPL-O section 5, worked out by hand beside what the checker derives. rev's
    loop is br.null, take, build and tail: four steps, so four per cell. Its null path is br.null
    and ret, so its entry is two. A handler's own steps are what its header cell's four do not cover."""

    def test_a_walk_is_paid_for_by_the_cells_it_releases(self):
        bound = certify(check(ITEM, REVERSE, ONCE), "h", 216)
        self.assertEqual((bound.a, bound.b, bound.c, bound.d), (0, 0, 4, 3))

    def test_a_walk_over_what_the_same_walk_returned_pays_for_both(self):
        """One set of numbers per function would need rev's result to carry its parameter's amount,
        which pays for the loop and has that result's amount left: it would exceed itself."""
        bound = certify(check(ITEM, REVERSE, TWICE), "h", 216)
        self.assertEqual((bound.a, bound.b, bound.c, bound.d), (0, 0, 8, 3))

    def test_recursion_is_not_given_new_numbers_at_each_call(self):
        """Otherwise this would have a linear bound: reversing the accumulator at every step."""
        quadratic = """
fn f1 quad(r0 : t16, r1 : t16) : t16
    br.null r0, L1, L2
L1: ret r1
L2: take r2, r0[8:8]
    build r0, c18, none, =, r1
    const r3, #0 : t16
    call r4, f0, r0, r3
    tail f1, r2, r4
end decreasing 0
handler f2 h(r0 : t26, r1 : t30) : t50
    open r0, c27
    take r2, r0[8:8]
    const r3, #0 : t16
    call r4, f1, r2, r3
    build r0, c27, none, =, r4
    ret r0, r1
end
"""
        with self.assertRaisesRegex(Unbounded, "no linear bound on steps"):
            certify(check(ITEM, REVERSE, quadratic), "h", 216)

    def test_recursion_that_releases_nothing_cannot_pay_for_itself(self):
        """The checker does not look at R3's argument, so this is its own proof of termination."""
        spin = """
fn f0 spin(r0 : t16) : t16
    tail f0, r0
end decreasing 0
handler f1 h(r0 : t26, r1 : t30) : t50
    open r0, c27
    take r2, r0[8:8]
    call r3, f0, r2
    build r0, c27, none, =, r3
    ret r0, r1
end
"""
        with self.assertRaisesRegex(Unbounded, "no linear bound on steps"):
            certify(check(ITEM, spin), "h", 216)

    def test_recursion_has_to_name_what_falls(self):
        with self.assertRaisesRegex(RTError, "names no decreasing set"):
            check(LIST, REVERSE.replace("end decreasing 0", "end"))

    def test_only_a_parameter_made_of_cells_can_fall(self):
        walk = """
fn f0 walk(r0 : t16, r1 : i64) : i64
    br.null r0, L1, L2
L1: ret r1
L2: take r2, r0[8:8]
    drop r0
    tail f0, r2, r1
end decreasing 1
"""
        with self.assertRaises(RTError):
            check(LIST, walk)

    def test_the_decreasing_set_shares_one_amount(self):
        """R3 only says the set's total falls, so a cell that leaves may be any member's. Neither
        parameter is ever passed as the other here, so nothing but `end` connects their amounts."""
        walk = """
fn f0 walk(r0 : t16, r1 : t16, r2 : t16) : t16
    br.null r0, L1, L2
L1: br.null r1, L3, L4
L3: ret r2
L4: take r4, r1[8:8]
    build r1, c18, none, =, r2
    const r3, #0 : t16
    tail f0, r3, r4, r1
L2: take r5, r0[8:8]
    build r0, c18, none, =, r2
    tail f0, r5, r1, r0
end decreasing 0,1
"""
        checker = check(LIST, walk)
        signature = checker.functions["f0"]
        first, second = signature.param_pots[:2]
        self.assertIn(Budget().plus(first).plus(second, -1), signature.obligations)
        self.assertIn(Budget().plus(second).plus(first, -1), signature.obligations)
        self.assertNotIn(Budget().plus(signature.param_pots[2]).plus(first, -1), signature.obligations)


class WhenTheSolverRunsOut(unittest.TestCase):
    def test_a_spent_budget_is_a_refusal_with_the_section_that_declares_it(self):
        """src.simplex.MAX_PIVOTS is what section 8.2 declares, and this holds the two together."""
        with mock.patch("src.simplex.MAX_PIVOTS", 0):
            with self.assertRaisesRegex(Unbounded, r"section 8\.2: .*more than 0 pivots"):
                certify(check(ITEM, REVERSE, TWICE), "h", 216)

    def test_the_budget_is_far_past_what_a_handler_needs(self):
        with mock.patch("src.simplex.MAX_PIVOTS", 100):
            certify(check(ITEM, REVERSE, TWICE), "h", 216)


class Certificate(unittest.TestCase):
    """Section 5.5: whatever proposed the numbers, verify evaluates every inequality with them."""

    def setUp(self):
        self.checker = check(ITEM, REVERSE, TWICE)
        self.program = self.checker.linear_program("h", 216)
        self.cells = minimize(self.program.objective(), self.program.constraints("cells"))
        self.steps = minimize(self.program.objective(), self.program.constraints("steps"))

    def verify(self, steps):
        return self.checker.verify("h", self.cells, steps)

    def test_the_solvers_numbers_pass_and_are_what_is_reported(self):
        bound = self.verify(self.steps)
        self.assertEqual((bound.c, bound.d), (8, 3))
        self.assertEqual(self.checker.bounds("h"), bound)

    def test_nothing_in_the_table_comes_from_the_solver_being_right(self):
        """A number too low anywhere in the derivation is refused, the item's or an inner one's."""
        for unknown, value in self.steps.items():
            if not value:
                continue
            for forged in (value - 1, value - Fraction(1, 216)):
                changed = {**self.steps, unknown: forged}
                with self.subTest(unknown=unknown, forged=forged):
                    if forged < 0:
                        with self.assertRaisesRegex(RTError, "negative"):
                            self.verify(changed)
                    else:
                        with self.assertRaisesRegex(RTError, "short"):
                            self.verify(changed)

    def test_an_empty_answer_and_a_negative_one_are_refused(self):
        with self.assertRaisesRegex(RTError, "short"):
            self.verify({})
        with self.assertRaisesRegex(RTError, "negative"):
            self.verify({**self.steps, self.program.entry: Fraction(-1)})

    def test_numbers_that_hold_pass_whoever_found_them(self):
        """The certificate is about the inequalities and not the optimum. Raising the handler's
        entry by one leaves every row true, and the table says what the numbers say."""
        raised = {**self.steps, self.program.entry: self.steps[self.program.entry] + 1}
        self.assertEqual(self.verify(raised).d, 4)

    def test_a_fraction_rounds_up(self):
        """Three cells' worth over two is 1.5 per cell, and the table takes integers."""
        raised = {**self.steps, self.program.entry: self.steps[self.program.entry] + Fraction(1, 2)}
        self.assertEqual(self.verify(raised).d, 4)


PAIR = """
type t24 record Pair
ctor c25 t24 - Pair
field c25 0 8 t16 l
field c25 8 8 t16 r
"""
TWO = """
type t20 tagged Two
ctor c21 t20 0 P
field c21 8 8 i64 x
ctor c22 t20 1 Q
field c22 8 8 t16 next
"""
FINS = """
array t32768 1024 i8
fin t32769 1024
type t40 record Box
ctor c41 t40 - Box
field c41 0 8 t32769 at
"""


class Ownership(unittest.TestCase):
    """Streams an earlier version of this checker accepted, each of them a real way to lose a
    pointer, forge an index, or skip a store. Some compile to a wrong answer if they get through."""

    def refuse(self, pattern: str, *parts: str):
        with self.assertRaisesRegex(RTError, pattern):
            check(*parts)

    def test_one_pointer_cannot_end_up_in_two_fields(self):
        stream = """tuple t50 t24 t16
fn f1 g(r0 : t24) : t50
    open r0, c25
    take r1, r0[0:8]
    build r0, c25, none, [8:8], =
    ret r0, r1
end
"""
        self.refuse("already takes it", LIST, PAIR, stream)

    def test_a_capability_cannot_leave_with_pointers_in_its_cell(self):
        stream = """
fn f1 g(r0 : t24) : cap
    open r0, c25
    ret r0
end
"""
        self.refuse("still at", LIST, PAIR, stream)

    def test_a_join_does_not_pick_one_edges_idea_of_a_location(self):
        erased = """
fn f0 g(r0 : t20) : t20
    br.tag r0, 0, L1, L2
L1: jump L3
L2: open r0, c22
    jump L3
L3: build r0, c21, write, #0
    ret r0
end
"""
        forged = """
fn f0 g(r0 : t20) : t16
    br.tag r0, 1, L1, L2
L1: jump L3
L2: open r0, c21
    jump L3
L3: take r1, r0[8:8]
    ret r1
end
"""
        self.refuse("on one edge and", LIST, TWO, erased)
        self.refuse("on one edge and", LIST, TWO, forged)

    def test_what_a_test_ruled_out_does_not_survive_a_rebuild(self):
        stream = """tuple t50 t16 t16 t20
fn f0 g(r0 : t20) : t50
    br.tag r0, 0, L1, L2
L1: const r3, #0 : t16
    const r4, #0 : t16
    build r0, c21, keep, =
    ret r3, r4, r0
L2: open r0, c22
    take r1, r0[8:8]
    build r0, c21, write, #12345
    open r0, c22
    take r2, r0[8:8]
    build r0, c21, write, #1
    ret r1, r2, r0
end
"""
        self.refuse("still possible", LIST, TWO, stream)

    def test_only_idx_makes_an_index(self):
        for body in ("const r1, #99999999 : t32769\n    aget r2, r0, r1, 1\n    ret r0, r2",):
            self.refuse("only way to make", FINS, f"""tuple t50 t32768 i8
fn f0 g(r0 : t32768) : t50
    {body}
end
""")
        self.refuse("only way to make", FINS, """
fn f0 g(r0 : t40) : t40
    open r0, c41
    build r0, c41, none, #99999999
    ret r0
end
""")
        self.refuse("no full integer", FINS, """
fn f0 g(r0 : t40) : t40
    open r0, c41
    build r0, c41, none, add([0:8], #99999999)
    ret r0
end
""")

    def test_the_value_a_star_stands_for_has_a_type(self):
        stream = """
type t40 tagless Tree
ctor c41 t40 null Oz
ctor c42 t40 - Oc
field c42 0 8 i64 ohead
field c42 8 8 t44 otail
type t44 tagless Third
ctor c45 t44 null Tz
ctor c46 t44 - Tc
field c46 0 8 i64 thead
field c46 8 8 t44 ttail
fn f1 g(r0 : t16) : t40 @0
    br.null r0, L2, L3
L3: tailmod r0, c42, none, =, *, f1, r0[8:8]
L2: const r1, #0 : t40
    ret r1
end decreasing 0
"""
        self.refuse("against", LIST, stream)

    def test_a_capability_cannot_be_stored_in_a_field(self):
        self.refuse("capability", "type t40 record Hold\nctor c41 t40 - Hold\nfield c41 0 8 cap h\n")

    def test_a_function_that_returns_a_tuple_cannot_be_declared_at(self):
        stream = """tuple t50 t16 t16
fn f1 g(r0 : t16, r1 : t16) : t50 @0
    ret r1, r0
end
"""
        self.refuse("returns a tuple", LIST, stream)

    def test_tailmod_can_only_link_to_itself(self):
        stream = """
fn f0 inner(r0 : t16) : t16 @0
    br.null r0, L1, L2
L1: const r1, #0 : t16
    ret r1
L2: tailmod r0, c18, none, add([0:8], #1), *, f0, r0[8:8]
end decreasing 0
fn f1 outer(r0 : t16) : t16 @0
    br.null r0, L1, L2
L1: const r1, #0 : t16
    ret r1
L2: tailmod r0, c18, none, =, *, f0, r0[8:8]
end
"""
        self.refuse("can only be itself", LIST, stream)

    def test_a_computed_source_works_in_place_and_one_instruction_can_do_it(self):
        stream = """
fn f1 g(r0 : t16) : t16
    br.null r0, L1, L2
L1: const r5, #0 : t16
    ret r5
L2: build r0, c18, none, {source}, =
    ret r0
end
"""
        self.refuse("cannot|not an operation", LIST, stream.format(source="shl([0:8], #1)"))
        self.refuse("works in place", LIST, stream.format(source="add([8:8], #1)"))
        check(LIST, stream.format(source="add([0:8], #1)"))

    def test_a_literal_must_be_a_value_of_the_type_it_meets(self):
        stream = """
fn f0 k(r0 : i8) : i8
    ret r0
end
fn f1 g(r0 : i8) : i8
{body}
end
"""
        self.refuse("not a i8", stream.format(body="    tail f0, #300"))
        self.refuse("not a i8", stream.format(body="    op r1, add/8, r0, #300\n    ret r1"))
        self.refuse("not a i8", stream.format(
            body="    br e, r0, #300, L1, L2\nL1: ret r0\nL2: ret r0"))
        self.refuse("not a i8", stream.format(body="    const r1, #300 : lit\n    tail f0, r1"))
        check(stream.format(body="    op r1, shl/8, r0, #300\n    ret r1"))  # a count is not a value

    def test_a_capability_that_comes_back_from_a_call_is_an_empty_cell(self):
        stream = """
fn f0 idd(r0 : cap) : cap
    ret r0
end
fn f1 g(r0 : cap) : t16
    call r1, f0, r0
    build r1, c18, none, #1, #0
    ret r1
end
"""
        check(LIST, stream)

    def test_two_capabilities_chosen_at_a_join_are_an_empty_cell(self):
        stream = """tuple t50 t16 t16
fn f1 g(r0 : cap, r1 : cap, r2 : bool) : t50
    br ne, r2, #0, L1, L2
L1: move r3, r0
    move r4, r1
    jump L3
L2: move r3, r1
    move r4, r0
    jump L3
L3: build r3, c18, none, #1, #0
    build r4, c18, none, #2, #0
    ret r3, r4
end
"""
        check(LIST, stream)


class Independence(unittest.TestCase):
    def test_the_checker_imports_nothing_from_the_compiler(self):
        with open(os.path.join(ROOT, "src", "rtcheck.py")) as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, f"relative import of {node.module}")


if __name__ == "__main__":
    unittest.main()
