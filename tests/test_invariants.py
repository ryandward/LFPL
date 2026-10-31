"""Executable verification of the Phase 1 invariants, on real x86-64.

Each kind of claim is marked: resource safety, or translation correctness. A
theorem test that cannot fail proves nothing, so every class holds a control
that must trip the same detector on code that really does violate the claim.
"""
import os
import re
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.check import CheckStage  # noqa: E402
from src.dyck import (CLOSE, K_ARM, K_FN, OPEN, DyckError, DyckValidator, Lexer,  # noqa: E402
                      Parser, ev_tag, parse_events, show_event)
from src.lower import (DECLARED_LIMITS, FailFast, Lower, build_and_run, compile_source,  # noqa: E402
                       compile_text, file_bytes, occupancy_bound, required_cells)
from src.ring import FStarved, Reject, Ring, Stage  # noqa: E402
from src.tokens import (CellStore, ConservationError, Flags, Hole,  # noqa: E402
                        HomeUniquenessError, LMem, LReg, PayloadOverflow, Silicon, check_homes)
from src.symcheck import check_translation  # noqa: E402
from src.validate import function_body, validate  # noqa: E402

PRE = b"""data Free = End | More(next: Free);
data Res = Fail | Done(value: Int64, rest: Free);
data List = Nil | Cons(h: Int64, t: List);
record P { x: Int32, y: Int32, z: Int64 };
record Q { a: Int64, b: Int64 };
"""


def run(source: bytes) -> int:
    with tempfile.TemporaryDirectory() as workdir:
        return build_and_run(compile_text(source, checks=True), workdir)


def body(source: bytes, name: str) -> list[str]:
    asm = compile_text(source, checks=True)
    return [f"{op} {', '.join(args)}".strip() for op, args in function_body(asm, name)]


def normalized(lines: list[str]) -> list[str]:
    """Rename labels by first appearance, so two bodies compare by structure."""
    seen: dict[str, str] = {}

    def rename(match: re.Match) -> str:
        return seen.setdefault(match.group(0), f"L#{len(seen)}")

    return [re.sub(r"\bL\d+\b", rename, line) for line in lines]


class ZeroBooleanTrampolines(unittest.TestCase):
    """Theorem-level claim of section 5.3 (translation correctness)."""

    def _cond_body(self, condition: bytes) -> list[str]:
        source = (b"fn f(a: Int64, b: Int64) : Int64 = if " + condition +
                  b" then 1 else 2;\nfn main() : Int64 = f(1, 2);")
        return body(source, "f")

    def test_if_true_false_is_exactly_the_condition(self):
        direct = self._cond_body(b"<(a, b)")
        wrapped = self._cond_body(b"if <(a, b) then true else false")
        self.assertEqual(normalized(direct), normalized(wrapped))

    def test_swapped_literals_negate_without_materializing(self):
        lines = self._cond_body(b"if <(a, b) then false else true")
        self.assertFalse([l for l in lines if l.startswith(("set", "movzx"))], lines)

    def test_logic_under_cond_never_materializes_flags(self):
        for condition in (b"&&(<(a, b), !(==(a, 0)))", b"||(<(a, b), ==(b, 7))",
                          b"!(&&(<(a, b), ||(==(a, 1), >(b, 3))))",
                          b"if &&(<(a, b), ==(a, 1)) then true else ==(b, 9)"):
            lines = self._cond_body(condition)
            with self.subTest(condition=condition):
                self.assertFalse([l for l in lines if l.startswith(("set", "movzx"))], lines)
                self.assertFalse([l for l in lines if "rsp" in l], lines)

    def test_no_label_is_only_a_jump(self):
        """A trampoline is a block that does nothing but jump onward."""
        for condition in (b"if <(a, b) then true else false",
                          b"&&(<(a, b), ==(a, 1))", b"||(<(a, b), ==(a, 1))",
                          b"if <(a, b) then ==(a, 1) else false"):
            source = (b"fn f(a: Int64, b: Int64) : Int64 = if " + condition +
                      b" then 1 else 2;\nfn main() : Int64 = f(1, 2);")
            with self.subTest(condition=condition):
                self.assertEqual(trampolines(compile_text(source)), [])

    def test_no_flag_clobber_between_cmp_and_its_branch(self):
        source = (b"fn f(a: Int64, b: Int64) : Int64 = if &&(<(a, b), !(==(a, 0))) "
                  b"then +(a, 1) else -(b, 1);\nfn main() : Int64 = f(1, 2);")
        lines = body(source, "f")
        for at, line in enumerate(lines):
            if line.startswith(("cmp", "test")):
                self.assertTrue(lines[at + 1].startswith(("j", "set")), lines)

    def test_truth_tables_on_silicon(self):
        table = [(b"&&(<(a, b), !(==(a, 0)))", lambda a, b: a < b and a != 0),
                 (b"||(<(a, b), ==(b, 1))", lambda a, b: a < b or b == 1),
                 (b"!(||(==(a, 1), ==(b, 2)))", lambda a, b: not (a == 1 or b == 2)),
                 (b"if <(a, b) then ==(a, 0) else >(b, 0)",
                  lambda a, b: a == 0 if a < b else b > 0),
                 (b"if <(a, b) then false else true", lambda a, b: not a < b)]
        for text, expected in table:
            for a, b in ((0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (3, 3)):
                source = (b"fn f(a: Int64, b: Int64) : Int64 = if " + text +
                          b" then 1 else 0;\nfn main() : Int64 = f(%d, %d);" % (a, b))
                with self.subTest(text=text, a=a, b=b):
                    self.assertEqual(run(source), int(expected(a, b)))

    def test_control_value_position_does_materialize(self):
        """The detector sees setcc when the only rule that emits it applies."""
        lines = body(b"fn f(a: Int64, b: Int64) : Bool = <(a, b);\nfn main() : Int64 = 0;", "f")
        self.assertTrue([l for l in lines if l.startswith("set")], lines)

    def test_control_trampoline_detector_fires(self):
        self.assertTrue(trampolines("L1:\n    jmp L2\nL2:\n    ret\n"))


def trampolines(asm: str) -> list[str]:
    lines = [line.strip() for line in asm.splitlines() if line.strip()]
    found = []
    for at, line in enumerate(lines[:-1]):
        if line.endswith(":") and not line.endswith("_body:"):
            following = lines[at + 1]
            if following.startswith("jmp ") and not lines[at - 1].startswith(("fn_",)):
                found.append(f"{line} {following}")
    return found


class ZeroUnchangedFieldCopies(unittest.TestCase):
    """Theorem 6.1 and Corollary 6.3 (translation correctness)."""

    FN = PRE + b"fn get(p: P, free: Free) : Res = match p with P(d, x, y, z) => " \
               b"Done@d(+(z, +(pack<32>(x), pack<32>(y))), free) end;\n"

    def _program(self, update: bytes, x=2, y=3, z=100) -> bytes:
        return self.FN + b"fn u(p: P, v: Int32) : P @p = " + update + b";\n" + (
            b"fn main(free: Free) : Res = match free with End => Fail | More(c, rest) => "
            b"get(u(P@c(%d, %d, %d), 7), rest) end;" % (x, y, z))

    def test_form_one_store_of_an_operand(self):
        source = self._program(b"{ p with y = v }")
        self.assertEqual(body(source, "u"),
                         ["mov rax, rdi", "mov dword [rdi+4], esi", "ret"])
        self.assertEqual(run(source), 2 + 7 + 100)

    def test_form_two_computation_lands_in_the_field(self):
        source = self._program(b"{ p with y = if <(v, 9) then 1 else 2 }")
        lines = body(source, "u")
        self.assertEqual([l for l in lines if "[rdi" in l and l.startswith("mov")],
                         ["mov dword [rdi+4], 1", "mov dword [rdi+4], 2"])
        self.assertEqual(run(source), 2 + 1 + 100)

    def test_form_three_single_read_modify_write(self):
        source = self._program(b"{ p with y = +(y, v) }")
        self.assertEqual(body(source, "u"),
                         ["mov rax, rdi", "add dword [rdi+4], esi", "ret"])
        self.assertEqual(run(source), 2 + (3 + 7) + 100)

    def test_no_op_update_emits_nothing(self):
        source = self._program(b"{ p with y = y }")
        self.assertEqual(body(source, "u"), ["mov rax, rdi", "ret"])

    def test_explicit_rebuild_touches_only_the_changed_field(self):
        source = self._program(b"match p with P(d, x, y, z) => P@d(x, +(y, v), z) end")
        self.assertEqual(body(source, "u"),
                         ["mov rax, rdi", "add dword [rdi+4], esi", "ret"])
        self.assertEqual(run(source), 2 + 10 + 100)

    def test_rebuild_with_a_renamed_field_is_still_restored(self):
        source = self._program(
            b"match p with P(d, x, y, z) => let w = z in P@d(x, y, w) end")
        self.assertEqual(body(source, "u"), ["mov rax, rdi", "ret"])

    def test_swap_is_the_minimum_two_loads_two_stores(self):
        source = PRE + (b"fn swap(q: Q) : Q @q = match q with Q(d, a, b) => Q@d(b, a) end;\n"
                        b"fn hi(q: Q, free: Free) : Res = match q with Q(d, a, b) => "
                        b"Done@d(a, free) end;\n"
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(c, rest) => hi(swap(Q@c(5, 6)), rest) end;")
        lines = body(source, "swap")
        self.assertEqual(len([l for l in lines if re.match(r"mov r\w+, qword \[", l)]), 2)
        self.assertEqual(len([l for l in lines if l.startswith("mov qword [")]), 2)
        self.assertEqual(run(source), 6)

    def test_section_5_6_increment_is_the_spec_sequence(self):
        source = PRE + (b"fn inc(l: List) : List @l = match l with "
                        b"| Cons(d, h, t) => Cons@d(+(h, 1), inc(t)) | Nil => Nil end;\n"
                        b"fn main() : Int64 = 0;")
        self.assertEqual(normalized(body(source, "inc")),
                         normalized(["mov rax, rdi", "test rdi, rdi", "je L2",
                                     "add qword [rdi], 1", "mov rdi, qword [rdi+8]",
                                     "jmp L1", "ret"]))

    def test_a_literal_is_never_taken_for_a_provenance_word(self):
        """An imm home keeps its value where a word would be. If the value happened
        to equal the field's word, RESTORE would drop the store. Try every small one."""
        for literal in range(0, 24):
            source = PRE + (b"fn u(q: Q) : Q @q = match q with Q(d, a, b) => "
                            b"let k = %d in Q@d(k, b) end;\n"
                            b"fn hi(q: Q, free: Free) : Res = match q with Q(d, a, b) => "
                            b"Done@d(a, free) end;\n"
                            b"fn main(free: Free) : Res = match free with End => Fail | "
                            b"More(c, rest) => hi(u(Q@c(99, 0)), rest) end;" % literal)
            with self.subTest(literal=literal):
                self.assertEqual(run(source), literal)

    def test_a_record_update_on_an_unbuilt_constructor(self):
        source = PRE + (b"fn hi(q: Q, free: Free) : Res = match q with Q(d, a, b) => "
                        b"Done@d(+(a, b), free) end;\n"
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(c, rest) => let q = Q@c(1, 2) in hi({ q with a = 40 }, rest) end;")
        self.assertEqual(run(source), 42)

    def test_read_modify_write_with_a_64_bit_immediate(self):
        """Found by the fuzzer: there is no such encoding, and NASM truncated it."""
        source = PRE + (b"fn u(q: Q) : Q @q = { q with a = +(a, 9223372036854775807) };\n"
                        b"fn hi(q: Q, free: Free) : Res = match q with Q(d, a, b) => "
                        b"Done@d(a, free) end;\n"
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(c, rest) => hi(u(Q@c(1, 0)), rest) end;")
        self.assertEqual(run(source), 1 << 63)

    def test_a_join_has_a_word_only_if_every_arm_agrees(self):
        """Found by the fuzzer. The else arm delivers the field's own word, so a
        join that trusted its last arm applied RESTORE and lost the then arm's 255."""
        source = self._program(
            b"match p with P(d, x, y, z) => P@d(x, y, if <(z, 0) then 7 else z) end")
        self.assertEqual(run(source), 2 + 3 + 100)
        lifted = self._program(
            b"match p with P(d, x, y, z) => P@d(x, y, if >(z, 0) then 7 else z) end")
        self.assertEqual(run(lifted), 2 + 3 + 7)

    def test_a_store_evacuates_a_scalar_a_frame_is_still_waiting_to_read(self):
        """Found by the fuzzer. `+(u, v)` is a pending read-modify-write whose operand
        v lives in y's cell, and the nested constructor rebuilds that cell first."""
        source = PRE + (b"data Box;\nrecord B { p: Int64, c: Box };\n"
                        b"record Box { n: Int64, m: Int64 };\n"
                        b"fn f(z: Box, y: Box) : B = match z with Box(e, u, w) => "
                        b"match y with Box(d, v, x) => B@e(+(u, v), Box@d(1, x)) end end;\n"
                        b"fn take(a: B, free: Free) : Res = match a with B(e, p, c) => "
                        b"match c with Box(d, n, m) => "
                        b"Done@e(+(p, +(n, m)), More@d(free)) end end;\n"
                        b"fn main(r0: Free) : Res = match r0 with End => Fail | More(a, r1) => "
                        b"match r1 with End => Done@a(0, End) | More(b, r2) => "
                        b"take(f(Box@a(100, 0), Box@b(10, 20)), r2) end end;")
        self.assertEqual(run(source), 110 + 1 + 20)
        self.assertEqual(validate(compile_text(source)), [])

    def test_evacuation_in_a_returning_arm_is_invisible_to_the_next_arm(self):
        """An evacuation is undone for the next arm: its field is still in the cell."""
        source = PRE + (b"data Box;\nrecord B { p: Int64, c: Box };\n"
                        b"record Box { n: Int64, m: Int64 };\n"
                        b"fn f(z: Box, y: Box, s: Bool) : B = match z with Box(e, u, w) => "
                        b"match y with Box(d, v, x) => if s "
                        b"then B@e(+(u, v), Box@d(1, x)) else B@e(+(u, 1), Box@d(v, x)) "
                        b"end end;\n"
                        b"fn take(a: B, free: Free) : Res = match a with B(e, p, c) => "
                        b"match c with Box(d, n, m) => "
                        b"Done@e(+(p, +(n, m)), More@d(free)) end end;\n"
                        b"fn main(r0: Free) : Res = match r0 with End => Fail | More(a, r1) => "
                        b"match r1 with End => Done@a(0, End) | More(b, r2) => "
                        b"take(f(Box@a(100, 0), Box@b(10, 20), false), r2) end end;")
        self.assertEqual(run(source), 101 + 10 + 20)

    def test_a_value_evacuated_in_one_arm_and_live_at_the_join_fails_fast(self):
        source = PRE + (b"data Box;\nrecord B { p: Int64, c: Box };\n"
                        b"record Box { n: Int64, m: Int64 };\n"
                        b"fn k(y: B, s: Bool) : Box = match y with B(d, p, c) => "
                        b"let r = if s then Box@d(1, 2) else Box@d(3, 4) in "
                        b"match r with Box(e, n, m) => match c with Box(f, a, b) => "
                        b"Box@e(a, n) end end end;\n")
        with self.assertRaisesRegex(FailFast, "5.3"):
            compile_text(source)

    def test_an_unread_scalar_is_erased_without_a_load(self):
        """The control for the rule above: nobody waits for b, so no load appears."""
        source = PRE + (b"fn f(q: Q) : Q @q = match q with Q(d, a, b) => Q@d(1, 2) end;\n"
                        b"fn main() : Int64 = 0;")
        self.assertEqual(body(source, "f"), ["mov rax, rdi", "mov qword [rdi], 1",
                                             "mov qword [rdi+8], 2", "ret"])

    def test_validator_is_clean_on_every_program_here(self):
        for program in (self._program(b"{ p with y = v }"),
                        self._program(b"match p with P(d, x, y, z) => P@d(y, x, z) end")):
            self.assertEqual(validate(compile_text(program)), [])

    def test_control_validator_catches_hand_written_copies(self):
        self.assertEqual(validate("    mov rax, qword [rdi+8]\n    mov qword [rdi+8], rax\n")[0].kind, "I2")
        self.assertEqual(validate("    mov rax, rax\n")[0].kind, "I1")
        self.assertEqual(validate("    mov qword [rdi], rax\n    mov rbx, qword [rdi]\n")[0].kind, "I3")

    def test_truncating_a_loaded_value_and_storing_it_is_not_a_copy(self):
        """`mov r32, r32` makes a new value. Found by the fuzzer, as a validator
        false positive: `{ p with a = pack<32>(a) }` really changes the field."""
        self.assertEqual(validate("    mov r10, qword [rdi]\n    mov r10d, r10d\n"
                                  "    mov qword [rdi], r10\n"), [])
        self.assertEqual(validate("    mov r10, qword [rdi]\n    mov r11, r10\n"
                                  "    mov qword [rdi], r11\n")[0].kind, "I2")

    def test_a_location_named_through_a_register_dies_with_that_register(self):
        """`[rdi+8]` names somewhere else after rdi changes, so a store there is not a
        write-back of what the old `[rdi+8]` held."""
        self.assertEqual(validate("    mov rax, qword [rdi+8]\n    mov rdi, rbx\n"
                                  "    mov qword [rdi+8], rax\n"), [])
        self.assertEqual(validate("    mov rax, qword [rdi+8]\n    add rdi, 16\n"
                                  "    mov qword [rdi+8], rax\n"), [])
        self.assertEqual(validate("    mov qword [rdi], rax\n    mov rdi, rcx\n"
                                  "    mov rbx, qword [rdi]\n"), [])
        self.assertEqual(validate("    mov rdi, qword [rdi+8]\n    mov qword [rdi+8], rdi\n"), [])

    def test_control_broken_lowerer_is_caught(self):
        """Disable STAY and RESTORE in the lowerer: the copies must appear, and the
        independent validator must see them. This is what the theorem rules out."""
        source = self._program(b"match p with P(d, x, y, z) => P@d(x, +(y, v), z) end")
        with mock.patch.object(Lower, "_restores", lambda self, dest, value: False), \
                mock.patch.object(Lower, "_key", lambda self, index: None):
            broken = compile_text(source)
        self.assertTrue([v for v in validate(broken) if v.kind == "I2"], broken)


class PhantomConstruction(unittest.TestCase):
    """Theorem 7.4 (translation correctness)."""

    DIRECT = PRE + (b"fn f(x: Int64, c: Diamond, y: Int64, free: Free) : Res = "
                    b"match Q@c(x, y) with Q(e, p, q) => Done@e(+(p, q), free) end;\n"
                    b"fn main() : Int64 = 0;")
    NONE = PRE + (b"fn f(x: Int64, c: Diamond, y: Int64, free: Free) : Res = "
                  b"let p = x in let q = y in Done@c(+(p, q), free);\n"
                  b"fn main() : Int64 = 0;")

    def test_direct_match_emits_nothing_for_the_constructor(self):
        self.assertEqual(normalized(body(self.DIRECT, "f")), normalized(body(self.NONE, "f")))

    def test_let_bound_match_emits_nothing_for_the_constructor(self):
        source = PRE + (b"fn f(x: Int64, c: Diamond, y: Int64, free: Free) : Res = "
                        b"let v = Q@c(x, y) in match v with Q(e, p, q) => "
                        b"Done@e(+(p, q), free) end;\nfn main() : Int64 = 0;")
        self.assertEqual(normalized(body(source, "f")), normalized(body(self.NONE, "f")))

    def test_literal_fields_write_no_bytes(self):
        source = PRE + (
            b"fn g(c: Diamond, free: Free) : Res = "
            b"match Q@c(3, 4) with Q(e, p, q) => Done@e(+(p, q), free) end;\n"
            b"fn main(free: Free) : Res = match free with End => Fail | "
            b"More(c, rest) => g(c, rest) end;")
        lines = body(source, "g")
        self.assertFalse([l for l in lines if re.match(r"mov \w+ \[\w+(\+\d+)?\], 3$", l)], lines)
        self.assertEqual(run(source), 7)

    def test_nested_phantoms_cost_nothing(self):
        source = PRE + (b"fn f(x: Int64, c: Diamond, y: Int64, free: Free) : Res = "
                        b"match Q@c(x, y) with Q(e, p, q) => "
                        b"match Q@e(q, p) with Q(g, r, s) => "
                        b"Done@g(-(r, s), free) end end;\nfn main() : Int64 = 0;")
        expected = PRE + (b"fn f(x: Int64, c: Diamond, y: Int64, free: Free) : Res = "
                          b"let r = y in let s = x in Done@c(-(r, s), free);\n"
                          b"fn main() : Int64 = 0;")
        self.assertEqual(normalized(body(source, "f")), normalized(body(expected, "f")))

    def test_other_arms_of_a_known_constructor_emit_nothing(self):
        source = PRE + (b"fn f(n: Int64, free: Free) : Res = match free with End => Fail "
                        b"| More(c, rest) => match Done@c(n, rest) with "
                        b"| Fail => Fail "
                        b"| Done(e, v, r) => Done@e(+(v, 1), r) end end;\n"
                        b"fn main() : Int64 = 0;")
        lines = body(source, "f")
        # One null result comes from `End => Fail`. The known constructor's `Fail`
        # arm would be a second if it were lowered at all.
        self.assertEqual(len([l for l in lines if l == "xor eax, eax"]), 1, lines)

    def test_phantom_result_is_correct_on_silicon(self):
        source = PRE + (b"fn g(x: Int64, c: Diamond, y: Int64, free: Free) : Res = "
                        b"match Q@c(+(x, 1), *(y, 2)) with Q(e, p, q) => Done@e(*(p, q), free) end;\n"
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(c, rest) => g(4, c, 5, rest) end;")
        self.assertEqual(run(source), 50)

    def test_control_materialization_does_write(self):
        """A constructor that reaches a real use is built, so the count is not vacuous."""
        source = PRE + (b"fn hi(q: Q, free: Free) : Res = match q with Q(d, a, b) => "
                        b"Done@d(+(a, b), free) end;\n"
                        b"fn f(x: Int64, c: Diamond, y: Int64, free: Free) : Res = "
                        b"let v = Q@c(x, y) in hi(v, free);\nfn main() : Int64 = 0;")
        lines = body(source, "f")
        self.assertEqual(len([l for l in lines if l.startswith("mov qword [")]), 2, lines)


class CellConservation(unittest.TestCase):
    """Theorem 7.1 and 7.2 (resource safety)."""

    SOURCE = PRE + (b"fn inc(l: List) : List @l = match l with "
                    b"| Cons(d, h, t) => Cons@d(+(h, 1), inc(t)) | Nil => Nil end;\n"
                    b"fn sum(l: List, acc: Int64, free: Free) : Res = match l with "
                    b"| Cons(d, h, t) => sum(t, +(acc, h), More@d(free)) "
                    b"| Nil => match free with End => Fail "
                    b"| More(d, rest) => Done@d(acc, rest) end end;\n"
                    b"fn main(free: Free) : Res = match free with End => Fail | "
                    b"More(a, r1) => match r1 with End => Done@a(0, End) | More(b, r2) => "
                    b"sum(inc(Cons@a(1, Cons@b(2, Nil))), 0, r2) end end;")

    def test_partition_holds_at_every_step_of_the_whole_pipeline(self):
        steps = []

        def on_step(ring: Ring) -> None:
            ring.check_conservation()
            steps.append(1)

        ring, _lower = compile_source(self.SOURCE, cells=512, checks=True, on_step=on_step)
        self.assertGreater(len(steps), 1000)
        ring.check_conservation()

    def test_only_the_global_tables_stay_out_of_f_at_the_end(self):
        ring, lower = compile_source(self.SOURCE, cells=512)
        check = ring.stages[2]
        self.assertEqual(ring.store.C - len(ring.F), len(check.G.cells) + len(lower.table.cells))
        self.assertEqual(len(check.K) + len(check.B) + len(lower.K) + len(lower.B), 0)

    def test_every_ring_rule_fired(self):
        ring, _ = compile_source(self.SOURCE, cells=512)
        for rule in ("ingest", "pass", "push", "drop", "expand", "egress", "frame"):
            self.assertGreater(ring.counts[rule], 0, rule)

    def test_compiled_program_keeps_every_cell(self):
        """Exactly-once linearity on silicon: walking the free list that comes back
        counts the 1023 cells that are not the result, which holds the last one."""
        source = PRE + (b"fn count(f: Free, n: Int64, out: Diamond, back: Free) : Res = "
                        b"match f with End => Done@out(n, back) "
                        b"| More(d, rest) => count(rest, +(n, 1), out, More@d(back)) end;\n"
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(out, rest) => count(rest, 0, out, End) end;")
        self.assertEqual(run(source), 1023)

    def test_register_partition_is_checked_each_step(self):
        with mock.patch.object(Silicon, "check_partition", autospec=True) as check:
            compile_source(self.SOURCE, cells=512, checks=True)
        self.assertGreater(check.call_count, 50)

    def test_home_uniqueness_control(self):
        with self.assertRaises(HomeUniquenessError):
            check_homes({"x": LReg(0), "y": LReg(0)})
        with self.assertRaises(HomeUniquenessError):
            check_homes({"x": LMem(1, 8, 8), "y": LMem(1, 8, 8)})
        check_homes({"x": LReg(0), "y": LReg(1), "z": LMem(1, 8, 8)})

    def test_register_partition_control(self):
        silicon = Silicon()
        silicon.take(0)
        with self.assertRaises(ConservationError):
            silicon.check_partition([])  # rax taken, and nobody holds it
        silicon.check_partition([0])
        with self.assertRaises(HomeUniquenessError):
            silicon.check_partition([0, 0])

    def test_control_a_leak_is_caught(self):
        class Leaky(Stage):
            name = "leaky"

            def fire(self, cell):
                self.sigma.remove(cell)  # the cell is simply forgotten

        ring = Ring(32, iter(b"fn f() : Int64 = 1;"), [Lexer(), Parser(), Leaky()],
                    lambda t, p: None)
        with self.assertRaises(ConservationError):
            ring.on_step = lambda r: r.check_conservation()
            ring.run()

    def test_control_a_duplicated_cell_is_caught(self):
        class Duplicator(Stage):
            name = "dup"

            def fire(self, cell):
                self.ring.drop(self, cell)
                self.ring.F.append(cell)  # the same location twice on the return edge

        ring = Ring(32, iter(b"fn f() : Int64 = 1;"), [Lexer(), Parser(), Duplicator()],
                    lambda t, p: None)
        with self.assertRaises(ConservationError):
            ring.on_step = lambda r: r.check_conservation()
            ring.run()

    def test_a_cell_holds_at_most_fifteen_payload_bytes(self):
        store = CellStore(1)
        store.put(0, 7, bytes(15))
        with self.assertRaises(PayloadOverflow):
            store.put(0, 7, bytes(16))


class DepthBound(unittest.TestCase):
    """Theorem 7.2 (resource safety): cells scale with depth, not source length."""

    @staticmethod
    def _balanced(depth: int, leaf: int = 0) -> tuple[str, int]:
        if depth == 0:
            return ("a", leaf + 1)
        left, leaf = DepthBound._balanced(depth - 1, leaf)
        right, leaf = DepthBound._balanced(depth - 1, leaf)
        return (f"+({left}, {right})", leaf)

    @staticmethod
    def _thin(depth: int) -> str:
        text = "a"
        for _ in range(depth):
            text = f"+({text}, 1)"
        return text

    def _peak(self, source: bytes) -> int:
        ring, _ = compile_source(source, cells=4096)
        return ring.peak_cells

    def test_source_length_does_not_move_the_peak(self):
        full, _ = self._balanced(10)
        thin = self._thin(10)
        make = "fn f(a: Int64) : Int64 = {};\nfn main() : Int64 = f(1);".format
        big, small = make(full).encode(), make(thin).encode()
        self.assertGreater(len(big), 20 * len(small))
        self.assertLess(self._peak(big), self._peak(small) + 45)

    def test_comments_and_whitespace_cost_nothing(self):
        base = b"fn main() : Int64 = 1;"
        padded = (b"# " + b"x" * 100000 + b"\n") * 5 + b" " * 100000 + base
        self.assertEqual(self._peak(padded), self._peak(base))

    @staticmethod
    def _heap_peak(text: bytes, lexer: type = Lexer) -> int:
        """Peak Python heap for one compile. The source itself was allocated before
        tracing starts, so only what the stages keep counts against them."""
        lower = Lower()
        ring = Ring(required_cells(16), iter(text), [lexer(), Parser(16), CheckStage(), lower],
                    lambda tag, payload: None)
        tracemalloc.start()
        ring.run()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak

    def _heap_sources(self) -> tuple[bytes, bytes]:
        """Two sources of one depth, a thin one and one of roughly 100 KB."""
        name = "a" * 15
        make = f"fn f({name}: Int64) : Int64 = {{}};\n".format
        full, _ = self._balanced(12)
        return (make(self._thin(12).replace("a", name)).encode(),
                make(full.replace("a", name).replace(", ", ",     ")).encode())

    def test_python_heap_does_not_follow_source_length(self):
        """The external measure. Nothing may hide O(N) in ordinary Python objects: the
        allowance is a quarter of the source, which a stage keeping it would blow."""
        small, large = self._heap_sources()
        self.assertGreater(len(large), 90_000)
        allowance = len(large) // 4
        growth = self._heap_peak(large) - self._heap_peak(small)
        self.assertLess(growth, allowance, (growth, allowance, len(large)))

    def test_control_a_stage_that_buffers_the_source_fails_the_heap_test(self):
        class Buffering(Lexer):
            def __init__(self) -> None:
                super().__init__()
                self.kept = bytearray()

            def fire(self, cell):
                if cell is not None:
                    self.kept += self.ring.store.payload(cell)[:1]
                super().fire(cell)

        small, large = self._heap_sources()
        allowance = len(large) // 4
        growth = self._heap_peak(large, Buffering) - self._heap_peak(small, Buffering)
        self.assertGreaterEqual(growth, allowance, (growth, allowance, len(large)))

    def test_peak_grows_linearly_with_depth(self):
        peaks = []
        for depth in (4, 8, 12):
            source = f"fn f(a: Int64) : Int64 = {self._thin(depth)};\nfn main() : Int64 = f(1);"
            peaks.append(self._peak(source.encode()))
        self.assertEqual(peaks[1] - peaks[0], peaks[2] - peaks[1], peaks)

    def test_globals_are_the_o_g_term_of_section_8(self):
        def source(count: int) -> bytes:
            return "\n".join(f"fn f{i}() : Int64 = {i};" for i in range(count)).encode()

        peaks = [self._peak(source(n)) for n in (10, 20, 40)]
        self.assertGreater(peaks[1], peaks[0])
        self.assertEqual((peaks[2] - peaks[1]), 2 * (peaks[1] - peaks[0]), peaks)

    def test_depth_limit_is_a_syntax_rejection_at_exactly_d_max(self):
        ok = self._thin(5)
        source = f"fn f(a: Int64) : Int64 = {ok};".encode()
        compile_source(source, cells=512, d_max=6)
        with self.assertRaisesRegex(Reject, "nesting too deep"):
            compile_source(source, cells=512, d_max=5)

    def test_rejection_depends_only_on_text_not_on_memory(self):
        source = f"fn f(a: Int64) : Int64 = {self._thin(50)};".encode()
        for cells in (2048, 4096):
            with self.assertRaisesRegex(Reject, "nesting too deep"):
                compile_source(source, cells=cells, d_max=10)

    def test_the_bound_holds_at_every_step(self):
        """Theorem 7.2 as a formula: W + 4 + (E-1) + L_max + s*D + s*G, measured."""
        full, _ = self._balanced(9)
        source = (PRE + f"fn f(a: Int64) : Int64 = {full};\n".encode()
                  + CellConservation.SOURCE[len(PRE):])

        def on_step(ring: Ring) -> None:
            check, lower = ring.stages[2], ring.stages[3]
            depth = max(stage.depth() for stage in ring.stages)
            limit = occupancy_bound(depth, len(check.G.cells) + len(lower.table.cells))
            self.assertLessEqual(ring.store.C - len(ring.F), limit)

        _ring, lower = compile_source(source, on_step=on_step)
        self.assertLessEqual(lower.max_expansion, 128)

    def test_required_cells_compiles_the_deepest_legal_program(self):
        """C = B(D_max) + 1 is enough at the deepest depth the grammar accepts."""
        depth = 12
        cells = required_cells(depth, 64)
        text = "1"
        for _ in range(depth - 1):
            text = f"let x = {text} in x"
        compile_source(f"fn main() : Int64 = {text};".encode(), cells=cells, d_max=depth)

    def test_too_few_cells_for_the_declared_depth_is_detected(self):
        """C below B(D_max) violates the theorem's hypothesis, and only that does."""
        source = f"fn f(a: Int64) : Int64 = {self._thin(12)};".encode()
        with self.assertRaises(FStarved):
            compile_source(source, cells=30)
        compile_source(source, cells=4096)


class DyckStream(unittest.TestCase):
    """Theorem 2.1 (translation correctness)."""

    SOURCE = (b"data L = N | C(h: Int64, t: L);\n"
              b"fn f(l: L, a: Int64) : L @l = match l with "
              b"| C(d, h, t) => let x = +(h, a) in if <(x, 1) then C@d(x, t) else C@d(h, t) "
              b"| N => N end;")

    def _word(self, source: bytes) -> list[tuple[int, bytes]]:
        word = []
        parse_events(source, on_event=lambda tag, payload: word.append((tag, payload)))
        return word

    def test_word_is_well_bracketed(self):
        checker = DyckValidator()
        for tag, _payload in self._word(self.SOURCE):
            checker.feed(tag)
        checker.finish()
        self.assertGreater(checker.peak, 3)

    def test_opening_precedes_children_for_operators(self):
        text = [show_event(t, p) for t, p in self._word(b"fn f(a: Int64) : Int64 = +(a, 1);")]
        at = text.index("open(BINOP)")
        self.assertLess(at, text.index("leaf(VAR,a)"))

    def test_parser_frames_equal_nesting_depth(self):
        depths = []
        word = []

        def on_event(tag, payload):
            word.append(tag)

        parse_events(self.SOURCE, on_event=on_event, on_step=lambda r: depths.append(r.stages[1].depth()))
        self.assertLessEqual(max(depths), 12)
        self.assertEqual(depths[-1], 0)

    def test_control_a_broken_word_is_rejected(self):
        checker = DyckValidator()
        checker.feed(ev_tag(OPEN, K_FN))
        with self.assertRaises(DyckError):
            checker.feed(ev_tag(CLOSE, K_ARM))
        checker = DyckValidator()
        checker.feed(ev_tag(OPEN, K_FN))
        with self.assertRaises(DyckError):
            checker.finish()

    def test_infix_is_a_syntax_error_by_design(self):
        with self.assertRaises(Reject):
            parse_events(b"fn f(a: Int64) : Int64 = a + 1;", on_event=lambda t, p: None)


class LinearityAndFailFast(unittest.TestCase):
    """Rejections carry the conflicting rule (resource safety)."""

    def _rejects(self, source: bytes, pattern: str) -> None:
        with self.assertRaisesRegex(Reject, pattern):
            compile_text(PRE + source)

    def test_linear_rules(self):
        self._rejects(b"fn f(l: List) : Int64 = 1;", "never used")
        self._rejects(b"fn f(l: List) : List = match l with Nil => Nil | "
                      b"Cons(d, h, t) => Cons@d(h, l) end;", "used twice|never used")
        self._rejects(b"fn f(l: List, m: List) : List = if true then l else m;", "different")
        self._rejects(b"fn f(l: List) : List = match l with Cons(d, h, t) => t end;",
                      "never used|non-exhaustive")

    def test_a_diagnostic_says_how_precise_its_line_is(self):
        """Check runs a few tokens behind Lex, so its line number is only near."""
        self._rejects(b"\n\nfn f(l: List) : Int64 = 1;\n", r"near line \d+: linear variable l")

    def test_no_fresh(self):
        self._rejects(b"fn f() : List = Cons@d(1, Nil);", "unbound")

    def test_a_constructor_must_fit_a_cell(self):
        self._rejects(b"record Big { a: Int64, b: Int64, c: Int8 };", "does not fit")

    def test_starvation_reports_the_exact_context(self):
        deep = "a"
        for i in range(20):
            deep = f"+(+(a, {i}), {deep})"
        with self.assertRaises(FailFast) as caught:
            compile_text(f"fn f(a: Int64) : Int64 = {deep};".encode())
        message = str(caught.exception)
        self.assertIn("register starvation", message)
        for part in ("Delta =", "H     =", "delta ="):
            self.assertIn(part, message)

    def test_a_let_rewrite_of_that_program_compiles_and_runs(self):
        text = "fn f(a: Int64) : Int64 = " + "".join(
            f"let v{i} = +(a, {i}) in " for i in range(20)) + (
            "+(v0, +(v1, +(v2, v19)));\nfn main() : Int64 = f(1);")
        self.assertEqual(run(text.encode()), 1 + 2 + 3 + 20)

    def test_an_overwritten_scalar_cannot_be_read(self):
        source = PRE + (b"fn h(q: Q) : Q @q = q;\n"
                        b"fn g(p: Q, free: Free) : Res = match p with Q(d, a, b) => "
                        b"let r = h(Q@d(1, 2)) in match r with Q(e, x, y) => Done@e(+(a, x), free) end end;\n")
        with self.assertRaisesRegex(FailFast, "erased"):
            compile_text(source)

    def test_arithmetic_width_comes_from_the_type_not_the_operands(self):
        """A literal in the last arm must not widen an Int8 operation to 64 bits."""
        source = (b"fn f(b: Bool, c: Int8) : Int64 = pack<8>(+(if b then c else 1, 1));\n"
                  b"fn main() : Int64 = f(true, 255);")
        self.assertEqual(run(source), 0)

    def test_arithmetic_on_two_literals_is_int64(self):
        with self.assertRaisesRegex(Reject, "mismatch"):
            compile_text(b"fn f(c: Int8) : Int8 = +(+(200, 100), c);")

    def test_nested_let_keeps_each_binder_name(self):
        """Found by the fuzzer: the bound expression opens another let."""
        source = b"fn main() : Int64 = let a = let b = 2 in +(b, 1) in +(a, 10);"
        self.assertEqual(run(source), 13)

    def test_non_tail_recursion_is_rejected_until_descent_exists(self):
        source = b"fn f(n: Int64) : Int64 = +(f(n), 1);"
        with self.assertRaisesRegex((FailFast, Reject), "4.4|Phase 1"):
            compile_text(source)

    def test_at_annotation_is_checked_at_every_exit(self):
        source = PRE + (b"fn bad(a: List, b: List) : List @a = match a with "
                        b"Nil => b | Cons(d, h, t) => Cons@d(h, bad(t, b)) end;")
        with self.assertRaisesRegex(Reject, "word declared"):
            compile_text(source)


class NoIdentityMorphisms(unittest.TestCase):
    """Theorem 7.5 (translation correctness), under the pressure that breaks a
    literal reading of section 5.7: found by the fuzzer's register campaign."""

    def _pressure(self, count: int, tail: str) -> bytes:
        lets = "".join(f"let t{i} = +(a, {i}) in " for i in range(count))
        return (f"fn h(x: Int64, y: Int64) : Int64 = +(*(x, 3), y);\n"
                f"fn f(a: Int64) : Int64 = {lets}{tail};\n"
                f"fn main() : Int64 = f(1);").encode()

    def test_a_spilled_value_is_not_reloaded_past_its_register(self):
        asm = compile_text(self._pressure(18, "+(h(t3, t4), +(t5, t17))"))
        self.assertEqual(validate(asm), [])

    def test_evacuation_and_arguments_are_one_parallel_move(self):
        source = self._pressure(16, "let z = h(t0, t1) in +(z, +(t0, t15))")
        self.assertEqual(validate(compile_text(source)), [])
        self.assertEqual(run(source), ((1 + 0) * 3 + (1 + 1)) + (1 + 0) + (1 + 15))

    def test_a_spilled_cell_pointer_is_reloaded_at_most_once(self):
        """Found by the fuzzer: two stores into a cell whose pointer lives in a slot."""
        lets = "".join(f"let t{i} = +(a, {i}) in " for i in range(16))
        source = PRE + (f"fn f(q: Q) : Q @q = match q with Q(d, a, b) => {lets}"
                        f"Q@d(+(t0, t15), +(t1, t14)) end;\n"
                        f"fn main() : Int64 = 0;").encode()
        self.assertEqual(validate(compile_text(source)), [])

    def test_forward_declaration_allows_mutually_recursive_types(self):
        source = (b"data Pair;\ndata T = Leaf | Node(v: Int64, kids: Pair);\n"
                  b"record Pair { l: T, r: T };\nfn main() : Int64 = 0;")
        self.assertEqual(run(source), 0)
        with self.assertRaisesRegex(Reject, "never defined"):
            compile_text(b"data Pair;\ndata T = Leaf | Node(k: Pair);")

    def test_the_root_survives_a_call_inside_a_tail_modulo_cons_loop(self):
        """Found by reading the code: the call moves the root out of rax, and the
        back-edge has to put it back, or the function returns the wrong cell."""
        source = PRE + (b"fn g(x: Int64) : Int64 = *(x, 2);\n"
                        b"fn dbl(l: List) : List @l = match l with "
                        b"| Nil => Nil | Cons(d, h, t) => Cons@d(g(h), dbl(t)) end;\n"
                        b"fn sum(l: List, acc: Int64, free: Free) : Res = match l with "
                        b"| Cons(d, h, t) => sum(t, +(acc, h), More@d(free)) "
                        b"| Nil => match free with End => Fail "
                        b"| More(d, rest) => Done@d(acc, rest) end end;\n"
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(a, r1) => match r1 with End => Done@a(0, End) | More(b, r2) => "
                        b"sum(dbl(Cons@a(5, Cons@b(7, Nil))), 0, r2) end end;")
        self.assertEqual(run(source), 24)

    def test_a_spilling_function_keeps_its_whole_name(self):
        lets = "".join(f"let t{i} = +(a, {i}) in " for i in range(18))
        source = (f"fn a_long_name_abc(a: Int64) : Int64 = {lets}+(t0, t17);\n"
                  f"fn main() : Int64 = a_long_name_abc(1);").encode()
        self.assertEqual(run(source), 1 + 18)

    def test_a_computed_operand_survives_the_spill_its_partner_needs(self):
        """Found by the fuzzer, on a wrong answer: the value to store was in r12, the
        cell pointer was in a slot, and reloading the pointer spilled r12 into itself,
        giving `mov qword [r12], r12`. An operand stays locked until it is emitted."""
        function = (b"fn s2(q: Q) : Q @q = match q with Q(d, a, b) => "
                    b"let t0 = h(b, b) in let t1 = h(b, b) in let t2 = +(a, t0) in "
                    b"let t3 = h(t1, t2) in let t4 = h(t1, t0) in let t5 = h(a, t0) in "
                    b"let t6 = h(t1, t1) in let t7 = +(t5, t5) in let t8 = h(t3, t6) in "
                    b"let t9 = +(t5, t8) in let t10 = +(t4, b) in let t11 = +(t2, t2) in "
                    b"let t12 = +(t11, t2) in let t13 = +(t6, t7) in let t14 = +(t3, t1) in "
                    b"Q@d(t0, t6) end;\n")
        source = PRE + (b"fn h(x: Int64, y: Int64) : Int64 = +(*(x, 3), y);\n"
                        b"fn rd(q: Q, free: Free) : Res = match q with Q(d, a, b) => "
                        b"Done@d(+(*(a, 5), b), free) end;\n" + function +
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(c, rest) => rd(s2(Q@c(1923856, 2544214)), rest) end;")
        self.assertEqual(run(source), 36 * 2544214)  # a' = 4b and b' = 16b

    def test_stack_frame_reaches_the_slots_through_a_thunk(self):
        asm = compile_text(self._pressure(18, "+(t0, t17)"))
        self.assertIn("sub rsp,", asm)
        self.assertRegex(asm, r"call fn_f_body")


class LinearTokens(unittest.TestCase):
    """The token forms of section 1, as Lower really holds them (resource safety)."""

    SOURCE = PRE + (b"fn u(q: Q) : Q @q = match q with Q(d, a, b) => "
                    b"let s = +(a, b) in Q@d(s, b) end;\nfn main() : Int64 = 0;")

    def _collect(self):
        seen = []

        def on_step(ring: Ring) -> None:
            seen.extend(ring.stages[3].tokens())

        compile_source(self.SOURCE, cells=512, on_step=on_step)
        return seen

    def test_a_hole_is_seen_only_after_its_location_is_written(self):
        holes = [t for t in self._collect() if isinstance(t, Hole)]
        self.assertTrue(holes)
        self.assertTrue(all(isinstance(t.a, LMem) for t in holes))

    def test_flags_are_a_linear_token_between_cmp_and_jcc(self):
        source = b"fn f(a: Int64, b: Int64) : Int64 = if <(a, b) then 1 else 2;\nfn main() : Int64 = 0;"
        states = []

        def on_step(ring: Ring) -> None:
            states.append(ring.stages[3].sil.flags)

        compile_source(source, cells=512, on_step=on_step)
        self.assertIn(Flags("b"), states)
        self.assertEqual(states[-1], Flags())


class CellsGivenToCallees(unittest.TestCase):
    """A callee owns a cell it is passed and may write it (resource safety)."""

    FILL = PRE + (b"fn fill(c: Diamond, t: List) : List = Cons@c(7, t);\n"
                  b"fn one(c: Diamond) : List = Cons@c(7, Nil);\n"
                  b"fn rev(l: List, acc: List) : List = match l with "
                  b"| Nil => acc | Cons(d, h, t) => rev(t, Cons@d(h, acc)) end;\n"
                  b"fn sum(l: List, acc: Int64, free: Free) : Res = match l with "
                  b"| Cons(d, h, t) => sum(t, +(acc, h), More@d(free)) "
                  b"| Nil => match free with End => Fail "
                  b"| More(d, rest) => Done@d(acc, rest) end end;\n")
    MAIN = (b"fn main(free: Free) : Res = match free with End => Fail | More(a, r1) => "
            b"match r1 with End => Done@a(0, End) | More(b, r2) => "
            b"sum(f(Cons@a(5, Cons@b(6, Nil))), 0, r2) end end;")

    def _program(self, body: bytes) -> bytes:
        return self.FILL + b"fn f(l: List) : List = match l with | Cons(d, h, t) => " + body + (
            b" | Nil => Nil end;\n") + self.MAIN

    def test_a_scalar_read_after_the_call_stops_instead_of_reading_the_callees_write(self):
        """The field's location now holds the callee's 7. This used to print 14."""
        source = self._program(b"match fill(d, t) with "
                               b"| Cons(e, h2, t2) => Cons@e(+(h, h2), t2) | Nil => Nil end")
        with self.assertRaisesRegex(FailFast, "erased") as caught:
            compile_text(source)
        self.assertEqual(caught.exception.limit, "section 1:")

    def test_a_copy_made_before_the_call_survives_it(self):
        source = self._program(b"let k = +(h, 0) in match fill(d, t) with "
                               b"| Cons(e, h2, t2) => Cons@e(+(k, h2), t2) | Nil => Nil end")
        self.assertEqual(run(source), 5 + 7 + 6)

    def test_a_linear_field_read_after_the_call_has_left_the_cell(self):
        """one(d) writes [c+8], where t lives, so t has to move before the call."""
        for body in (b"rev(one(d), t)", b"let m = one(d) in rev(m, t)"):
            with self.subTest(body=body):
                self.assertEqual(run(self._program(body)), 7 + 6)

    def test_a_diamond_reaching_the_call_through_a_let(self):
        source = self._program(b"let c = d in let m = one(c) in rev(m, t)")
        self.assertEqual(run(source), 7 + 6)

    def test_a_scalar_passed_to_the_same_call_needs_no_copy(self):
        source = PRE + (b"fn fill(c: Diamond, k: Int64, t: List) : List = Cons@c(k, t);\n")
        source += self.FILL[len(PRE):].split(b"\n", 1)[1] + (
            b"fn f(l: List) : List = match l with "
            b"| Cons(d, h, t) => let r = fill(d, +(h, 1), t) in r | Nil => Nil end;\n") + self.MAIN
        self.assertEqual(run(source), 5 + 1 + 6)


class TagWrites(unittest.TestCase):
    """The tag is a store like any other (translation correctness)."""

    FLIP = (b"data Free = End | More(next: Free);\n"
            b"data Res = Fail | Done(value: Int64, rest: Free);\n"
            b"data S = A(x: Int32) | B(x: Int32, y: Int16);\n"
            b"fn flip(v: S, c: Bool) : S @v = match v with "
            b"| A(d, x) => if c then B@d(x, 1) else B@d(x, 2) "
            b"| B(d, x, y) => B@d(x, y) end;\n"
            b"fn rd(s: S, free: Free) : Res = match s with "
            b"| A(d, x) => Done@d(+(1000, pack<32>(x)), free) "
            b"| B(d, x, y) => Done@d(+(2000, pack<16>(y)), free) end;\n")

    def test_each_arm_of_an_if_writes_the_tag_for_itself(self):
        """The first arm's write says nothing about the path through the second."""
        for flag, expected in ((b"false", 2002), (b"true", 2001)):
            source = self.FLIP + (b"fn main(free: Free) : Res = match free with End => Fail | "
                                  b"More(c, rest) => rd(flip(A@c(5), %s), rest) end;" % flag)
            with self.subTest(flag=flag):
                self.assertEqual(run(source), expected)

    def test_rebuilding_the_same_constructor_keeps_the_tag_without_a_store(self):
        lines = body(self.FLIP + b"fn main() : Int64 = 0;", "flip")
        self.assertEqual(len([l for l in lines if l == "mov byte [rdi], 1"]), 2, lines)

    def test_a_tag_write_does_not_corrupt_a_field_still_owed_at_offset_zero(self):
        """Found by the symbolic check, in programs whose result never showed it: the
        tag landed on the free list's next pointer before that pointer was moved out."""
        source = (b"data Free = End | More(next: Free);\n"
                  b"data Res = Fail | Done(value: Int64, rest: Free);\n"
                  b"data S = D | E(k: Int8) | G;\n"
                  b"fn count(f: Free, n: Int64, s: S, back: Free) : Res = match f with "
                  b"| More(d, rest) => count(rest, +(n, 1), s, More@d(back)) "
                  b"| End => match s with D(c) => Done@c(n, back) "
                  b"| E(c, k) => Done@c(n, back) | G(c) => Done@c(n, back) end end;\n"
                  b"fn main(free: Free) : Res = match free with End => Fail | "
                  b"More(c, rest) => count(rest, 0, D@c(), End) end;")
        self.assertEqual(run(source), 1023)


class MeaningCheck(unittest.TestCase):
    """`symcheck` against assembly written by hand, so it is not tied to one emitter."""

    SOURCE = PRE + b"fn u(p: P, v: Int32) : P @p = { p with y = if <(v, 9) then 1 else 2 };\n"

    def _check(self, source: bytes, body_text: str) -> list:
        return check_translation(source, "fn_u_body:\n" + body_text)

    def test_two_shapes_of_one_meaning_both_pass(self):
        """Stores in each arm, or one store of a register merged at the join: the merge
        rule makes these the same value, however the code reached it."""
        arms = ("    mov rax, rdi\n    cmp esi, 9\n    jae L2\n    mov dword [rdi+4], 1\n"
                "    jmp L3\nL2:\n    mov dword [rdi+4], 2\nL3:\n    ret\n")
        joined = ("    mov rax, rdi\n    cmp esi, 9\n    jae L2\n    mov r10d, 1\n    jmp L3\n"
                  "L2:\n    mov r10d, 2\nL3:\n    mov dword [rdi+4], r10d\n    ret\n")
        self.assertEqual(self._check(self.SOURCE, arms), [])
        self.assertEqual(self._check(self.SOURCE, joined), [])

    def test_a_store_missing_from_one_arm_is_found_and_named(self):
        """validate.py passes this: there is no redundant move in it, only a lost one."""
        lost = ("    mov rax, rdi\n    cmp esi, 9\n    jae L2\n    mov dword [rdi+4], 1\n"
                "    jmp L3\nL2:\nL3:\n    ret\n")
        self.assertEqual(validate(lost), [])
        problems = self._check(self.SOURCE, lost)
        self.assertEqual(len(problems), 1)
        self.assertIn("arg0", problems[0].detail)

    def test_swapped_arms_are_found(self):
        swapped = ("    mov rax, rdi\n    cmp esi, 9\n    jae L2\n    mov dword [rdi+4], 2\n"
                   "    jmp L3\nL2:\n    mov dword [rdi+4], 1\nL3:\n    ret\n")
        self.assertTrue(self._check(self.SOURCE, swapped))

    def test_a_field_written_with_the_wrong_value_in_a_straight_line(self):
        source = PRE + b"fn u(p: P, v: Int32) : P @p = { p with y = v };\n"
        self.assertEqual(self._check(source, "    mov rax, rdi\n    mov dword [rdi+4], esi\n    ret\n"), [])
        problems = self._check(source, "    mov rax, rdi\n    mov dword [rdi+4], 3\n    ret\n")
        self.assertIn("field +4 holds 3, the source says arg1", problems[0].detail)
        problems = self._check(source, "    mov rax, rdi\n    mov dword [rdi+0], esi\n    ret\n")
        self.assertTrue(problems)

    def test_a_store_into_the_wrong_cell_is_found(self):
        source = PRE + b"fn u(p: P, q: P, v: Int32) : P @p = { p with y = v };\n"
        problems = check_translation(source, "fn_u_body:\n    mov rax, rdi\n"
                                             "    mov dword [rsi+4], edx\n    ret\n")
        self.assertTrue(problems)

    def test_it_imports_nothing_from_the_compiler(self):
        import ast
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src", "symcheck.py")
        with open(path) as handle:
            tree = ast.parse(handle.read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, "a relative import reaches into the compiler")
                imported.add(node.module)
        self.assertLessEqual(imported, {"__future__", "dataclasses", "re", "sys", "typing"})

    def test_it_stays_clean_on_the_examples(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("inc.lfpl", "sum.lfpl"):
            with open(os.path.join(root, "examples", name), "rb") as handle:
                source = handle.read()
            self.assertEqual(check_translation(source, compile_text(source)), [])

    # A broken compiler, one line at a time. Each mutant is a real bug, most of them
    # ones this project actually shipped for a while. The check has to find every one
    # without running the binary.
    Q_READ = (b"fn hi(q: Q, free: Free) : Res = match q with Q(d, a, b) => "
              b"Done@d(+(a, b), free) end;\n")
    TAGS = (b"data Free = End | More(next: Free);\n"
            b"data Res = Fail | Done(value: Int64, rest: Free);\n"
            b"data S = A(x: Int32) | B(x: Int32, y: Int16);\n"
            b"fn flip(v: S, c: Bool) : S @v = match v with "
            b"| A(d, x) => if c then B@d(x, 1) else B@d(x, 2) | B(d, x, y) => B@d(x, y) end;\n")
    H = b"fn h(x: Int64, y: Int64) : Int64 = +(*(x, 3), y);\n"
    MUTANTS = [
        ("a join trusts its last arm's word",
         "return first if first == second else 0", "return second",
         PRE + b"fn u(p: P) : P @p = match p with P(d, x, y, z) => "
               b"P@d(x, y, if <(z, 0) then 7 else z) end;\n", True),
        ("a literal is taken for a provenance word",
         "return 0 if b.loc == L_IMM else b.word", "return b.word",
         PRE + b"fn u(q: Q) : Q @q = match q with Q(d, a, b) => let k = 3 in Q@d(k, b) end;\n",
         True),
        ("every store is elided",
         "return word != 0 and word == self._word_at(dest[1], dest[2], dest[3])", "return True",
         PRE + b"fn u(q: Q) : Q @q = match q with Q(d, a, b) => Q@d(b, a) end;\n", True),
        ("a deferred read is not evacuated",
         "if index not in deferred:\n                    continue", "continue",
         PRE + b"data Box;\nrecord B { p: Int64, c: Box };\n"
               b"record Box { n: Int64, m: Int64 };\n"
               b"fn f(z: Box, y: Box) : B = match z with Box(e, u, w) => "
               b"match y with Box(d, v, x) => B@e(+(u, v), Box@d(1, x)) end end;\n", False),
        ("the back-edge forgets the root",
         'if fr.b1 and self.root >= 0:\n            moves = moves + [[("r", RAX), E_BIND, 0, self.root]]',
         "if False:\n            pass",
         PRE + H.replace(b"h(", b"g(") + b"fn dbl(l: List) : List @l = match l with | Nil => Nil "
               b"| Cons(d, x, t) => Cons@d(g(x, 1), dbl(t)) end;\n", False),
        ("an operand is not locked across its partner's spill",
         "            self.locked |= 1 << b.r\n            return (O_REG, b.r, 8)",
         "            return (O_REG, b.r, 8)",
         PRE + H + b"fn s2(q: Q) : Q @q = match q with Q(d, a, b) => "
               b"let t0 = h(b, b) in let t1 = h(b, b) in let t2 = +(a, t0) in "
               b"let t3 = h(t1, t2) in let t4 = h(t1, t0) in let t5 = h(a, t0) in "
               b"let t6 = h(t1, t1) in let t7 = +(t5, t5) in let t8 = h(t3, t6) in "
               b"let t9 = +(t5, t8) in let t10 = +(t4, b) in let t11 = +(t2, t2) in "
               b"let t12 = +(t11, t2) in let t13 = +(t6, t7) in let t14 = +(t3, t1) in "
               b"Q@d(t0, t6) end;\n", False),
        ("the first arm's tag write is trusted by the second",
         "            self.i(\"mov\", (O_MEM, self._base(cell), 0, 1), (O_IMM, tag))\n",
         "            self.i(\"mov\", (O_MEM, self._base(cell), 0, 1), (O_IMM, tag))\n"
         "            b = self._b(cell)\n            b.aux = tag + 1\n            self._set_b(cell, b)\n",
         TAGS, True),
        ("a tag is never written",
         "if tagged and self._b(cell).aux != tag + 1:", "if False:", TAGS, True),
        ("the tag lands before the field under it has left",
         "        self._close_stale_window(cell, keep)\n        self._write_tag(cell, fr.b1, fr.b2)\n",
         "        self._write_tag(cell, fr.b1, fr.b2)\n        self._close_stale_window(cell, keep)\n",
         b"data Free = End | More(next: Free);\n"
         b"data Res = Fail | Done(value: Int64, rest: Free);\ndata S = D | E(k: Int8) | G;\n"
         b"fn take(s: S, f: Free) : Res = match s with D(c) => Done@c(1, f) "
         b"| E(c, k) => Done@c(2, f) | G(c) => Done@c(3, f) end;\n"
         b"fn u(f: Free) : Res = match f with End => Fail | More(c, r) => take(D@c(), r) end;\n",
         False),
        ("a cell given to a callee keeps its stale window open",
         "        for index in passed:\n            self._close_stale_window(index, passed)\n", "",
         PRE + b"fn one(c: Diamond) : List = Cons@c(7, Nil);\n"
               b"fn rev(l: List, acc: List) : List = match l with "
               b"| Nil => acc | Cons(d, h, t) => rev(t, Cons@d(h, acc)) end;\n"
               b"fn f(l: List) : List = match l with "
               b"| Cons(d, h, t) => let m = one(d) in rev(m, t) | Nil => Nil end;\n", True),
        ("registers a call clobbers are not evacuated",
         "moved = self._evacuation_moves(mask, leaving)", "moved = []",
         # The callee has to really write rdi, or leaving a value there would be correct.
         b"fn dec(x: Int64, y: Int64) : Int64 = if ==(x, 0) then y else dec(-(x, 1), y);\n"
         b"fn f(a: Int64, b: Int64) : Int64 = let t = dec(a, b) in +(t, a);\n", False),
        ("a comparison branches the wrong way",
         "self._hold_flags(cc, dest.x, dest.y)", "self._hold_flags(CC.index(NEGATED[CC[cc]]), dest.x, dest.y)",
         b"fn f(a: Int64, b: Int64) : Int64 = if <(a, b) then 1 else 2;\n", False),
        ("subtraction becomes addition",
         "        op = OP_NAMES[fr.b1]\n        second = self._second_value(fr)",
         "        op = OP_NAMES[fr.b1].replace(\"sub\", \"add\")\n        second = self._second_value(fr)",
         b"fn f(a: Int64, b: Int64) : Int64 = let c = -(a, b) in c;\n", False),
        ("a call is silently dropped",
         "        self._raw_name(\"call_name\", name)\n        self._forget_registers()\n",
         "        self._forget_registers()\n",
         H + b"fn f(a: Int64) : Int64 = let t = h(a, 1) in t;\n", False),
        ("a join leaves values where the arm put them",
         "        if fr.dk != D_COND and not self.dead:\n            self._normalize(bytes(fr.tail))\n"
         "            self._goto(fr.l3)\n",
         "        if fr.dk != D_COND and not self.dead:\n            self._goto(fr.l3)\n",
         H + b"fn f(a: Int64, b: Int64) : Int64 = let x = +(a, 1) in "
             b"let y = if <(a, b) then h(x, b) else +(x, 2) in +(+(x, y), b);\n", False),
    ]

    def _mutant_asm(self, old: str, new: str, source: bytes) -> str:
        """Compile with one line of the lowerer changed, in a separate process."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "src", "lower.py")) as handle:
            text = handle.read()
        self.assertIn(old, text, "this mutant no longer applies to the lowerer")
        with tempfile.TemporaryDirectory() as workdir:
            import shutil
            shutil.copytree(os.path.join(root, "src"), os.path.join(workdir, "src"))
            with open(os.path.join(workdir, "src", "lower.py"), "w") as handle:
                handle.write(text.replace(old, new))
            program = os.path.join(workdir, "p.lfpl")
            with open(program, "wb") as handle:
                handle.write(source)
            done = subprocess.run([sys.executable, "-m", "src.lower", program], cwd=workdir,
                                  capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout

    def test_every_mutant_is_found_without_running_it(self):
        for name, old, new, source, blind in self.MUTANTS:
            with self.subTest(mutant=name):
                self.assertEqual(check_translation(source, compile_text(source)), [],
                                 "the unbroken compiler must pass, or this proves nothing")
                asm = self._mutant_asm(old, new, source)
                self.assertTrue(check_translation(source, asm), asm)

    def test_the_redundancy_validator_is_blind_to_lost_stores(self):
        """Why this check exists: these mutants lose or misplace a store, and validate.py
        has nothing to say about any of them."""
        for name, old, new, source, blind in self.MUTANTS:
            if blind:
                with self.subTest(mutant=name):
                    self.assertEqual(validate(self._mutant_asm(old, new, source)), [])


class DeclaredLimits(unittest.TestCase):
    def test_each_limit_is_recognized_and_an_internal_state_is_not(self):
        deep = "a"
        for i in range(20):
            deep = f"+(+(a, {i}), {deep})"
        with self.assertRaises(FailFast) as caught:
            compile_text(f"fn f(a: Int64) : Int64 = {deep};".encode())
        self.assertEqual(caught.exception.limit, "register starvation")
        self.assertIsNone(FailFast("a variable has no rule for this destination", "", "", "").limit)
        self.assertTrue(all(isinstance(key, str) and key for key in DECLARED_LIMITS))


class Horner(unittest.TestCase):
    """Section 4.5's worked example, exactly (translation correctness)."""

    def test_on_a_register_destination(self):
        source = (b"fn w(k: Int8, x: Int16, y: Int32) : Int64 = "
                  b"pack<8,8,16,32>(3, k, x, y);\nfn main() : Int64 = w(7, 513, 9);")
        lines = body(source, "w")
        self.assertEqual(lines, ["mov eax, 768", "or al, dil", "shl rax, 16", "or ax, si",
                                 "shl rax, 32", "or rax, rdx", "ret"])
        self.assertEqual(run(source), 3 << 56 | 7 << 48 | 513 << 32 | 9)

    def test_memory_field_is_read_directly_exactly_once(self):
        source = PRE + (b"record S { pad: Int16, x: Int16 };\n"
                        b"fn w(k: Int8, s: S, y: Int32, free: Free) : Res = "
                        b"match s with S(d, pad, x) => "
                        b"Done@d(pack<8,8,16,32>(3, k, x, y), free) end;\n"
                        b"fn main() : Int64 = 0;")
        lines = body(source, "w")
        reads = [l for l in lines if "[rsi+2]" in l]
        self.assertEqual(len(reads), 1, lines)
        self.assertRegex(reads[0], r"^or \w+, word \[rsi\+2\]$")

    def test_a_large_literal_in_a_later_field(self):
        """imm32 sign-extends, so 0xffffffff cannot be an `or` immediate."""
        source = (b"fn w(k: Int8) : Int64 = pack<8,32>(k, 4294967295);\n"
                  b"fn main() : Int64 = w(1);")
        self.assertEqual(run(source), 1 << 32 | 0xffffffff)

    def test_leading_constants_fold_at_compile_time(self):
        lines = body(b"fn w() : Int64 = pack<8,8>(1, 2);\nfn main() : Int64 = 0;", "w")
        self.assertEqual(lines, ["mov eax, 258", "ret"])

    def test_a_32_bit_memory_field_takes_the_section_8_extra_load(self):
        source = PRE + (b"fn w(p: P, free: Free) : Res = match p with P(d, x, y, z) => "
                        b"Done@d(pack<32,32>(x, y), free) end;\n"
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(c, rest) => w(P@c(1, 2, 3), rest) end;")
        self.assertEqual(run(source), 1 << 32 | 2)


class CommandLine(unittest.TestCase):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "src.lower", *args], cwd=self.ROOT,
                              capture_output=True, text=True)

    def test_it_runs_a_program_and_prints_the_result(self):
        done = self._cli("examples/sum.lfpl", "-r")
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "9"), done.stderr)

    def test_a_rejected_program_exits_nonzero_with_the_rule(self):
        with tempfile.NamedTemporaryFile(suffix=".lfpl") as bad:
            bad.write(PRE + b"fn forget(l: List) : Int64 = 1;\n")
            bad.flush()
            done = self._cli(bad.name)
        self.assertEqual(done.returncode, 1)
        self.assertIn("linear variable l is never used", done.stderr)

    def test_the_ring_is_fed_from_the_open_file_in_small_chunks(self):
        class Counting:
            def __init__(self, data: bytes):
                self.data, self.at, self.largest = data, 0, 0

            def read(self, size: int) -> bytes:
                self.largest = max(self.largest, size)
                block = self.data[self.at:self.at + size]
                self.at += len(block)
                return block

        text = b"# padding\n" * 5000 + b"fn main() : Int64 = 1;"
        handle = Counting(text)
        ring, _ = compile_source(file_bytes(handle))
        self.assertEqual(handle.at, len(text))
        self.assertLessEqual(handle.largest, 4096)
        self.assertLess(ring.peak_cells, 64)


class Toolchain(unittest.TestCase):
    def test_no_c_compiler_on_the_path(self):
        """The binary is built by nasm and ld alone: no libc, no cc."""
        with tempfile.TemporaryDirectory() as workdir:
            asm = compile_text(b"fn main() : Int64 = 42;")
            self.assertEqual(build_and_run(asm, workdir), 42)
            listing = subprocess.run(["objdump", "-p", os.path.join(workdir, "prog")],
                                     capture_output=True, text=True).stdout
            self.assertNotIn("NEEDED", listing)
            self.assertNotIn("INTERP", listing)


if __name__ == "__main__":
    unittest.main()
