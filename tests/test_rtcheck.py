"""The RT checker against hand-written streams: docs/rt-layer.md's rules, with no compiler.

Every row of that document's "What this would have caught" table is here as a stream that
must be refused, next to the corrected stream that must pass, so a rule that refuses everything
cannot pass by accident.
"""
import ast
import os
import unittest

from src.rtcheck import RTError, Unbounded, check_stream

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
end
"""


def check(*parts: str):
    return check_stream("\n".join(parts).splitlines())


class Accepted(unittest.TestCase):
    def test_the_inc_example_of_the_specification(self):
        check(LIST, INC)

    def test_the_forwarding_handler(self):
        checker = check(FORWARD, HANDLER)
        self.assertEqual(checker.bounds("forward")[0], 0)
        self.assertGreater(checker.bounds("forward")[1], 0)


class Refused(unittest.TestCase):
    """One test per kind of mistake. Each states the corrected stream beside the wrong one
    where the difference is small enough that passing alone would prove little."""

    def refuse(self, pattern: str, *parts: str):
        with self.assertRaisesRegex(RTError, pattern):
            check(*parts)

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

    def test_a_handler_that_reaches_recursion_has_no_bound(self):
        stream = """
fn f0 walk(r0 : t30, r1 : t30) : t30
    br.null r0, L1, L2
L1: ret r1
L2: take r2, r0[0:8]
    build r0, c32, none, r1
    tail f0, r2, r0
end
handler f1 h(r0 : t32768, r1 : t17, r2 : t30) : t32770
    const r3, #0 : t30
    call r4, f0, r2, r3
    open r1, c18
    build r1, c23, write, =
    ret r0, r1, r4
end
"""
        with self.assertRaises(Unbounded):
            check(FORWARD, stream)

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
        self.assertEqual(check(FORWARD, stream).bounds("h")[0], 1)


class Independence(unittest.TestCase):
    def test_the_checker_imports_nothing_from_the_compiler(self):
        with open(os.path.join(ROOT, "src", "rtcheck.py")) as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, f"relative import of {node.module}")


if __name__ == "__main__":
    unittest.main()
