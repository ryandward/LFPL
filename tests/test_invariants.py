"""Executable verification of the first specification's invariants, on real x86-64.

Each kind of claim is marked: resource safety, or translation correctness. A
theorem test that cannot fail proves nothing, so every class holds a control
that must trip the same detector on code that really does violate the claim.
"""
import gc
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tracemalloc
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.check import CheckStage  # noqa: E402
from src.dyck import (CLOSE, K_ARM, K_FN, OPEN, DyckError, DyckValidator, Lexer,  # noqa: E402
                      Parser, ev_tag, parse_events, show_event)
from src.lfplo import build_and_run, compile_program  # noqa: E402
from src.righttoleft import RightToLeft  # noqa: E402
from src.ring import FStarved, Reject, Ring, Stage  # noqa: E402
from src.rtcheck import RTError, check_stream  # noqa: E402
from src.rtlower import (E_MAX, Limit, RTLower, compiler_ring, file_bytes, occupancy_bound,  # noqa: E402
                         required_cells, rt_lines)
from src.tokens import CellStore, ConservationError, PayloadOverflow  # noqa: E402
from src.symcheck import check_translation  # noqa: E402
from src.validate import function_body, validate  # noqa: E402
from src.x86 import Backend  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE = b"""data Free = End | More(next: Free);
data Res = Fail | Done(value: Int64, rest: Free);
data List = Nil | Cons(h: Int64, t: List);
record P { x: Int32, y: Int32, z: Int64 };
record Q { a: Int64, b: Int64 };
"""


compile_text = compile_program


def compile_source(source, *, cells=None, d_max: int = 64, on_step=None) -> Ring:
    """The whole ring, down to the RT stream, with the text thrown away. What is left to look at is the ring."""
    ring = compiler_ring(source, lambda tag, payload: None, on_step, phase1=True, cells=cells, d_max=d_max)
    ring.run()
    return ring


def run(source: bytes) -> int:
    with tempfile.TemporaryDirectory() as workdir:
        return build_and_run(compile_text(source), workdir)


def body(source: bytes, name: str) -> list[str]:
    """What the function does. The two instructions that adjust its frame are left out, because the backend
    writes them before it knows the frame's size, so a frame with nothing in it still has them: the README's gap."""
    asm = compile_text(source)
    lines = [f"{op} {', '.join(args)}".strip() for op, args in function_body(asm, name)]
    return [line for line in lines if line not in (f"sub rsp, F_{name}", f"add rsp, F_{name}")]


def normalized(lines: list[str]) -> list[str]:
    """Rename labels by first appearance, so two bodies compare by structure."""
    seen: dict[str, str] = {}

    def rename(match: re.Match) -> str:
        return seen.setdefault(match.group(0), f"L#{len(seen)}")

    return [re.sub(r"\b[LX]\d+(?:_f\d+)?\b", rename, line) for line in lines]


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
    lines = [line.strip() for line in asm.splitlines() if line.strip() and " equ " not in line]
    found = []
    for at, line in enumerate(lines[:-1]):
        if line.endswith(":") and not line.endswith("$body:"):
            following = lines[at + 1]
            if following.startswith("jmp ") and not lines[at - 1].startswith(("fn_",)):
                found.append(f"{line} {following}")
    return found


class ZeroUnchangedFieldCopies(unittest.TestCase):
    """Theorem 6.1 and Corollary 6.3 (translation correctness)."""

    FN = PRE + b"fn rd(p: P, free: Free) : Res = match p with P(d, x, y, z) => " \
               b"Done@d(+(z, +(pack<32>(x), pack<32>(y))), free) end;\n"

    def _program(self, update: bytes, x=2, y=3, z=100) -> bytes:
        return self.FN + b"fn u(p: P, v: Int32) : P @p = " + update + b";\n" + (
            b"fn main(free: Free) : Res = match free with End => Fail | More(c, rest) => "
            b"rd(u(P@c(%d, %d, %d), 7), rest) end;" % (x, y, z))

    def test_form_one_store_of_an_operand(self):
        source = self._program(b"{ p with y = v }")
        self.assertEqual(body(source, "u"),
                         ["mov rax, rdi", "mov dword [rdi+4], esi", "ret"])
        self.assertEqual(run(source), 2 + 7 + 100)

    def test_form_two_computation_reaches_the_field_in_one_store(self):
        """`build` is one atomic instruction, so a computation with arms joins in a register, and on
        every path there is one store to the field and none to any other."""
        source = self._program(b"{ p with y = if <(v, 9) then 1 else 2 }")
        lines = body(source, "u")
        stores = [l for l in lines if "[rdi" in l and l.startswith("mov")]
        self.assertEqual(len(stores), 1, lines)
        self.assertRegex(stores[0], r"^mov dword \[rdi\+4\], \w+$")
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
                                     "jmp X1", "ret"]))

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
        # What really waits on this path: a location on the value stack, with the same cell rebuilt ahead of it.
        waiting = source.replace(b"B@e(+(u, v), Box@d(1, x))", b"B@e(v, Box@d(1, x))")
        self.assertEqual(run(waiting), 10 + 1 + 20)

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

    def test_a_value_moved_out_in_both_arms_and_live_at_the_join(self):
        """The first Lower had to stop at this. A linear field gets its one register when it is bound, so
        both arms move it to the same place, and the join has nothing to reconcile."""
        source = PRE + (b"data Box;\nrecord B { p: Int64, c: Box };\n"
                        b"record Box { n: Int64, m: Int64 };\n"
                        b"fn k(y: B, s: Bool, free: Free) : Res = match y with B(d, p, c) => "
                        b"let r = if s then Box@d(1, 2) else Box@d(3, 4) in "
                        b"match r with Box(e, n, m) => match c with Box(f, a, b) => "
                        b"Done@e(+(*(n, 100), +(*(a, 10), b)), More@f(free)) end end end;\n"
                        b"fn main(r0: Free) : Res = match r0 with End => Fail | More(a, r1) => "
                        b"match r1 with End => Done@a(0, End) | More(b, r2) => "
                        b"k(B@a(9, Box@b(5, 6)), false, r2) end end;")
        self.assertEqual(run(source), 3 * 100 + 5 * 10 + 6)

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

    def test_a_narrow_load_stored_wider_is_not_a_copy(self):
        """`pack<32>` reads only the low half of a wider field. Found by the generators, as a false positive."""
        self.assertEqual(validate("    mov r10d, dword [rdi]\n    mov qword [rdi], r10\n"), [])
        self.assertEqual(validate("    movzx r10d, word [rdi]\n    mov dword [rdi], r10d\n"), [])
        self.assertEqual(validate("    mov r10, qword [rdi]\n    mov dword [rdi], r10d\n")[0].kind, "I2")
        self.assertEqual(validate("    mov r10d, dword [rdi]\n    mov dword [rdi], r10d\n")[0].kind, "I2")

    def test_taking_the_low_half_of_a_wider_field(self):
        source = PRE + (b"fn u(q: Q) : Q = match q with Q(d, a, b) => Q@d(pack<32>(a), b) end;\n"
                        b"fn hi(q: Q, free: Free) : Res = match q with Q(d, a, b) => Done@d(a, free) end;\n"
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(c, rest) => hi(u(Q@c(81985529216486895, 2)), rest) end;")
        self.assertEqual(validate(compile_text(source)), [])
        self.assertEqual(run(source), 81985529216486895 & 0xFFFFFFFF)

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

    def test_control_writing_a_value_back_is_refused_and_seen(self):
        """What the theorem rules out, written by hand, since Lower has no way left to say it: a field read
        and stored back where it came from. rtcheck refuses it as a type error, and if something got past
        that, the independent validator sees the copy in the assembly."""
        stream = ["type t16 record R", "ctor c0 t16 - R", "field c0 0 8 i64 a", "field c0 8 8 i64 b",
                  "fn f0 u(r0 : t16) : t16 @0", "    open r0, c0", "    read r1, r0[0:8]",
                  "    build r0, c0, none, r1, =", "    drop r1", "    ret r0", "end"]
        with self.assertRaisesRegex(RTError, "back where it came from"):
            check_stream(stream)
        lines = []
        backend = Backend(lines.append)
        for line in stream:
            backend.feed(line)
        self.assertTrue([v for v in validate("\n".join(lines) + "\n") if v.kind == "I2"], lines)


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

    @unittest.expectedFailure
    def test_let_bound_match_emits_nothing_for_the_constructor(self):
        """Theorem 7.4 says this costs nothing, and the first Lower made it so. The register-transfer Lower builds
        the constructor and then reads it back, which LFPL-O section 12 lists as a gap. When the gap closes, this
        passes, and unittest says so."""
        source = PRE + (b"fn f(x: Int64, c: Diamond, y: Int64, free: Free) : Res = "
                        b"let v = Q@c(x, y) in match v with Q(e, p, q) => "
                        b"Done@e(+(p, q), free) end;\nfn main() : Int64 = 0;")
        self.assertEqual(normalized(body(source, "f")), normalized(body(self.NONE, "f")))

    def test_a_constructor_bound_by_let_means_what_it_says(self):
        source = PRE + (b"fn f(x: Int64, c: Diamond, y: Int64, free: Free) : Res = "
                        b"let v = Q@c(x, y) in match v with Q(e, p, q) => "
                        b"Done@e(+(p, q), free) end;\n"
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(c, rest) => f(30, c, 12, rest) end;")
        self.assertEqual(run(source), 42)

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

        ring = compile_source(self.SOURCE, on_step=on_step)
        self.assertGreater(len(steps), 1000)
        ring.check_conservation()

    def test_only_the_global_tables_stay_out_of_f_at_the_end(self):
        ring = compile_source(self.SOURCE)
        passed, check = ring.stages[2], ring.stages[3]
        tables = (check.G, check.TA, check.TF, check.TT)
        self.assertEqual(ring.store.C - len(ring.F), sum(len(table.cells) for table in tables) + passed.table_cells())

    def test_every_ring_rule_fired(self):
        ring = compile_source(self.SOURCE)
        for rule in ("ingest", "pass", "push", "drop", "expand", "egress", "frame"):
            self.assertGreater(ring.counts[rule], 0, rule)

    def test_compiled_program_keeps_every_cell(self):
        """Exactly-once linearity on silicon: walking the list that comes back counts the
        1023 cells that are not the result, which holds the last one. The list has a type
        of its own because rule R3 does not let a function walk the reserve."""
        source = (b"data Cells = None | Cell(next: Cells);\n"
                  b"data Res = Fail | Done(value: Int64, rest: Cells);\n"
                  b"fn count(f: Cells, n: Int64, out: Diamond, back: Cells) : Res = "
                  b"match f with None => Done@out(n, back) "
                  b"| Cell(d, rest) => count(rest, +(n, 1), out, Cell@d(back)) end;\n"
                  b"fn main(free: Cells) : Res = match free with None => Fail | "
                  b"Cell(out, rest) => count(rest, 0, out, None) end;")
        self.assertEqual(run(source), 1023)

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


def homes_conflict(backend: Backend) -> list[str]:
    """Invariant H, as the x86 backend holds it: no two live values share a home, and the table and
    its inverse agree. Registers exist only down here now, so this is where the partition is."""
    found = []
    homes = list(backend.home.values())
    if len(set(homes)) != len(homes):
        found.append(f"two values share a home: {backend.home}")
    held = {home: r for r, home in backend.home.items() if isinstance(home, str)}
    if held != backend.owner:
        found.append(f"the table and its inverse differ: {backend.home} and {backend.owner}")
    return found


class RegisterHomes(unittest.TestCase):
    """Theorem 7.1's register half (resource safety), at every line of the stream."""

    def _lines_checked(self, source: bytes) -> int:
        backend = Backend(lambda line: None)
        checked = 0
        for line in rt_lines(io.BytesIO(source), phase1=True):
            backend.feed(line)
            self.assertEqual(homes_conflict(backend), [], line)
            checked += 1
        return checked

    def test_every_program_here_keeps_one_value_in_one_home(self):
        self.assertGreater(self._lines_checked(CellConservation.SOURCE), 50)

    def test_under_more_live_values_than_registers(self):
        lets = "".join(f"let t{i} = +(a, {i}) in " for i in range(30))
        total = "".join(f"+(t{i}, " for i in range(29)) + "t29" + ")" * 29
        source = (f"fn h(x: Int64, y: Int64) : Int64 = +(*(x, 3), y);\n"
                  f"fn f(a: Int64) : Int64 = {lets}+(h(t0, t1), {total});\n"
                  f"fn main() : Int64 = f(1);").encode()
        self.assertGreater(self._lines_checked(source), 50)

    def test_control_the_check_sees_two_values_in_one_home(self):
        backend = Backend(lambda line: None)
        backend.home = {0: "rax", 1: "rax"}
        backend.owner = {"rax": 1}
        self.assertEqual(len(homes_conflict(backend)), 1)
        backend.home = {0: 3, 1: 3}
        backend.owner = {}
        self.assertEqual(len(homes_conflict(backend)), 1)
        backend.home = {0: "rax"}
        backend.owner = {"rax": 1}
        self.assertEqual(len(homes_conflict(backend)), 1)
        backend.home = {0: "rax", 1: "rbx", 2: 3}
        backend.owner = {"rax": 0, "rbx": 1}
        self.assertEqual(homes_conflict(backend), [])


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
        return compile_source(source).peak_cells

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
        tracing starts, so only what the stages keep counts against them. gc.collect()
        also empties CPython's free lists, so memory that earlier tests left there
        cannot absorb a cost: without it, the result depended on test order."""
        ring = Ring(required_cells(16), iter(text),
                    [lexer(), Parser(16), RightToLeft(), CheckStage(phase1=True), RTLower()],
                    lambda tag, payload: None)
        gc.collect()
        tracemalloc.start()
        ring.run()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak

    def _heap_sources(self) -> tuple[bytes, bytes]:
        """One shape at two lengths, about 25 KB and 100 KB. A bushy tree costs a
        fixed ~340 KB more heap than a thin chain whatever its length, levelling off
        by depth 10, so comparing those two shapes measured that constant instead of
        growth, and passed only when earlier tests had warmed the process."""
        name = "a" * 15
        make = f"fn f({name}: Int64) : Int64 = {{}};\n".format

        def padded(depth: int) -> bytes:
            tree, _ = self._balanced(depth)
            return make(tree.replace("a", name).replace(", ", ",     ")).encode()

        return padded(10), padded(12)

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
        compile_source(source, d_max=6)
        with self.assertRaisesRegex(Reject, "nesting too deep"):
            compile_source(source, d_max=5)

    def test_rejection_depends_only_on_text_not_on_memory(self):
        source = f"fn f(a: Int64) : Int64 = {self._thin(50)};".encode()
        for cells in (2048, 4096):
            with self.assertRaisesRegex(Reject, "nesting too deep"):
                compile_source(source, cells=cells, d_max=10)

    def test_the_bound_holds_at_every_step(self):
        """Theorem 7.2 as a formula: W + 5 + (E-1) + L_max + s*D + s*G, measured."""
        full, _ = self._balanced(9)
        source = (PRE + f"fn f(a: Int64) : Int64 = {full};\n".encode()
                  + CellConservation.SOURCE[len(PRE):])

        def on_step(ring: Ring) -> None:
            passed, check = ring.stages[2], ring.stages[3]
            depth = max(stage.depth() for stage in ring.stages)
            tables = (check.G, check.TA, check.TF, check.TT)
            limit = occupancy_bound(depth, sum(len(table.cells) for table in tables) + passed.table_cells())
            self.assertLessEqual(ring.store.C - len(ring.F), limit)

        ring = compile_source(source, on_step=on_step)
        self.assertLessEqual(ring.peak_queue, E_MAX)

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


class LinearityAndRefusals(unittest.TestCase):
    """Refusals carry the conflicting rule (resource safety). The stream is checked as it is written, so a
    path Check has not finished with can reach rtcheck first, and either one may be what says so."""

    def _rejects(self, source: bytes, pattern: str) -> None:
        with self.assertRaisesRegex((Reject, RTError), pattern):
            compile_text(PRE + source)

    def test_linear_rules(self):
        self._rejects(b"fn f(l: List) : Int64 = 1;", "never used")
        self._rejects(b"fn f(l: List) : List = match l with Nil => Nil | "
                      b"Cons(d, h, t) => Cons@d(h, l) end;", "used twice|never used")
        self._rejects(b"fn f(l: List, m: List) : List = if true then l else m;", "different|still in")
        self._rejects(b"fn f(l: List) : List = match l with Cons(d, h, t) => t end;",
                      "never used|non-exhaustive")

    def test_a_diagnostic_says_how_precise_its_line_is(self):
        """Check runs a few tokens behind Lex, so its line number is only near."""
        self._rejects(b"\n\nfn f(l: List) : Int64 = 1;\n", r"near line \d+: linear variable l")

    def test_no_fresh(self):
        self._rejects(b"fn f() : List = Cons@d(1, Nil);", "unbound")

    def test_a_constructor_of_a_data_type_must_fit_a_cell(self):
        """A record need not: tests/test_spans.py."""
        self._rejects(b"data Big = B(a: Int64, b: Int64, c: Int8) | C;", "does not fit")

    def test_more_pending_values_than_registers_is_not_a_refusal(self):
        """The first Lower stopped here, with the state of every register. T8: nothing in this path has to."""
        deep = "a"
        for i in range(20):
            deep = f"+(+(a, {i}), {deep})"
        self.assertEqual(run(f"fn f(a: Int64) : Int64 = {deep};\nfn main() : Int64 = f(1);".encode()),
                         sum(1 + i for i in range(20)) + 1)

    def test_the_same_program_with_every_value_named(self):
        text = "fn f(a: Int64) : Int64 = " + "".join(
            f"let v{i} = +(a, {i}) in " for i in range(20)) + (
            "+(v0, +(v1, +(v2, v19)));\nfn main() : Int64 = f(1);")
        self.assertEqual(run(text.encode()), 1 + 2 + 3 + 20)

    def test_a_scalar_read_after_its_location_is_overwritten_moved_out_first(self):
        """The first Lower refused this as erased. Section 8.1's last uses say it is wanted, so it moves."""
        source = PRE + (b"fn h(q: Q) : Q @q = q;\n"
                        b"fn g(p: Q, free: Free) : Res = match p with Q(d, a, b) => "
                        b"let r = h(Q@d(1, 2)) in match r with Q(e, x, y) => Done@e(+(a, x), free) end end;\n"
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(c, rest) => g(Q@c(40, 9), rest) end;")
        self.assertEqual(run(source), 41)

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
        source = PRE + (b"fn f(l: List) : List = match l with | Nil => Nil "
                        b"| Cons(d, h, t) => let r = f(t) in Cons@d(h, r) end;")
        with self.assertRaisesRegex(Limit, "section 8.2"):
            compile_text(source)

    def test_at_annotation_is_checked_at_every_exit(self):
        source = PRE + (b"fn bad(a: List, b: List) : List @a = match a with "
                        b"Nil => b | Cons(d, h, t) => Cons@d(h, bad(t, b)) end;")
        with self.assertRaisesRegex(RTError, "declared @0"):
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

    def test_a_value_reloaded_in_one_arm_is_not_stored_back_at_the_join(self):
        """Found by the generators of records that span. t0 lives in a slot, and the other arm leaves it there, so the
        join's home for it is that slot. This arm reads it, which brings it to a register, and the slot still holds it."""
        lets = "".join(f"let t{i} = +(a, {i}) in " for i in range(18))
        total = "".join(f"+(t{i}, " for i in range(18)) + "0" + ")" * 18
        source = (f"fn f(a: Int64, b: Int64) : Int64 = {lets}let r = if <(a, b) then b else t0 in +(r, {total});\n"
                  f"fn main() : Int64 = f(5, 1);").encode()
        self.assertEqual(validate(compile_text(source)), [])
        self.assertEqual(run(source), 5 + sum(5 + i for i in range(18)))
        taken = source.replace(b"f(5, 1)", b"f(1, 5)")
        self.assertEqual(run(taken), 5 + sum(1 + i for i in range(18)))

    def test_a_frame_is_one_adjustment_with_no_thunk(self):
        """`F_f` is an `equ` at the end of the function, so its size is known at the prologue."""
        asm = compile_text(self._pressure(18, "+(t0, t17)"))
        self.assertIn("sub rsp, F_f\n", asm)
        self.assertGreater(int(re.search(r"^F_f equ (\d+)$", asm, re.M).group(1)), 0)
        self.assertNotRegex(asm, r"call fn_f\$body")


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

    def test_a_scalar_read_after_the_call_moves_out_before_the_callee_writes(self):
        """The field's location now holds the callee's 7. This once printed 14, and then the first Lower
        refused it. The right-to-left pass says the scalar is wanted, so it leaves first."""
        source = self._program(b"match fill(d, t) with "
                               b"| Cons(e, h2, t2) => Cons@e(+(h, h2), t2) | Nil => Nil end")
        self.assertEqual(run(source), 5 + 7 + 6)

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

    def test_a_tag_known_by_elimination_can_stay_for_a_constructor_of_another_type(self):
        """Found by the generators, as a false alarm that would have refused the program. The last arm of a match
        takes no test, so all a reader of the machine knows is what the tests ruled out, and that is about the
        three constructors of T. Read as an S, which has four, that leaves two, so it could not say what was there."""
        source = (b"data Free = End | More(next: Free);\n"
                  b"data Res = Fail | Done(value: Int64, rest: Free);\n"
                  b"data T = Ta(x: Int8) | Tb(y: Int8) | Tc(z: Int8);\n"
                  b"data S = Sa(x: Int32) | Sb(x: Int32, y: Int16) | Sc(z: Int8) | Sd;\n"
                  b"fn f(t: T) : S = match t with Tb(d, y) => Sb@d(1, 2) | Tc(d, z) => Sd@d() | Ta(d, x) => Sa@d(7) end;\n"
                  b"fn rd(s: S, free: Free) : Res = match s with Sa(d, x) => Done@d(pack<32>(x), free) "
                  b"| Sb(d, x, y) => Done@d(1000, free) | Sc(d, z) => Done@d(2000, free) | Sd(d) => Done@d(3000, free) end;\n"
                  b"fn main(free: Free) : Res = match free with End => Fail | More(c, rest) => rd(f(Ta@c(9)), rest) end;")
        lines = body(source, "f")
        self.assertEqual([l for l in lines if l.startswith("mov byte [rdi], ")], ["mov byte [rdi], 3"])  # Sd's, and no other
        self.assertEqual(check_translation(source, compile_text(source)), [])
        self.assertEqual(run(source), 7)

    def test_a_tag_write_does_not_corrupt_a_field_still_owed_at_offset_zero(self):
        """Found by the symbolic check, in programs whose result never showed it: the
        tag landed on the free list's next pointer before that pointer was moved out."""
        source = (b"data Cells = None | Cell(next: Cells);\n"
                  b"data Res = Fail | Done(value: Int64, rest: Cells);\n"
                  b"data S = D | E(k: Int8) | G;\n"
                  b"fn count(f: Cells, n: Int64, s: S, back: Cells) : Res = match f with "
                  b"| Cell(d, rest) => count(rest, +(n, 1), s, Cell@d(back)) "
                  b"| None => match s with D(c) => Done@c(n, back) "
                  b"| E(c, k) => Done@c(n, back) | G(c) => Done@c(n, back) end end;\n"
                  b"fn main(free: Cells) : Res = match free with None => Fail | "
                  b"Cell(c, rest) => count(rest, 0, D@c(), None) end;")
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
        path = os.path.join(ROOT, "src", "symcheck.py")
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
        for name in ("inc.lfpl", "sum.lfpl"):
            with open(os.path.join(ROOT, "examples", name), "rb") as handle:
                source = handle.read()
            self.assertEqual(check_translation(source, compile_text(source)), [])

    # A broken compiler, one line at a time. Each mutant is a real bug, most of them ones this project
    # actually shipped for a while. rtcheck would refuse a good many of Lower's before they got this far, so
    # these go around it: the point here is that the symbolic check finds every one, by itself, without
    # running the binary. Two bugs the first Lower had are gone from this list, because the RT layer has no
    # way to say them: a literal is a `#n` source with no word to mistake, and a tag is part of the one
    # atomic `build`, so all that is left to get wrong is the order of the moves it lowers to.
    Q_READ = (b"fn hi(q: Q, free: Free) : Res = match q with Q(d, a, b) => "
              b"Done@d(+(a, b), free) end;\n")
    TAGS = (b"data Free = End | More(next: Free);\n"
            b"data Res = Fail | Done(value: Int64, rest: Free);\n"
            b"data S = A(x: Int32) | B(x: Int32, y: Int16);\n"
            b"fn flip(v: S, c: Bool) : S @v = match v with "
            b"| A(d, x) => if c then B@d(x, 1) else B@d(x, 2) | B(d, x, y) => B@d(x, y) end;\n")
    H = b"fn h(x: Int64, y: Int64) : Int64 = +(*(x, 3), y);\n"
    LOWER, BACKEND = "src/rtlower.py", "src/x86.py"
    MUTANTS = [
        ("a join trusts its last arm's word", LOWER,
         "            if held.word != came.word:\n                held = replace(held, word=self.new_word())\n",
         "            held = replace(held, word=came.word)\n",
         PRE + b"fn u(p: P) : P @p = match p with P(d, x, y, z) => "
               b"P@d(x, y, if <(z, 0) then 7 else z) end;\n", True),
        ("every store is elided", LOWER,
         "same = value.off == field.off and value.size == field.size", "same = True",
         PRE + b"fn u(q: Q) : Q @q = match q with Q(d, a, b) => Q@d(b, a) end;\n", True),
        ("a waiting read is not moved out", LOWER,
         "            if value.kind == V_LOC and value.cell == cell:\n                self.V.set(index, self._hold(value))",
         "            if False:\n                self.V.set(index, self._hold(value))",
         PRE + b"data Box;\nrecord B { p: Int64, c: Box };\n"
               b"record Box { n: Int64, m: Int64 };\n"
               b"fn f(z: Box, y: Box) : B = match z with Box(e, u, w) => "
               b"match y with Box(d, v, x) => B@e(v, Box@d(1, x)) end end;\n", True),
        ("the back-edge forgets the root", BACKEND,
         'if ROOT in self.home and self.home[ROOT] != "rax":\n                moves.append(Move(Reg("rax"), _as_place(self.home[ROOT])))',
         "pass",
         PRE + H.replace(b"h(", b"g(") + b"fn dbl(l: List) : List @l = match l with | Nil => Nil "
               b"| Cons(d, x, t) => Cons@d(g(x, 1), dbl(t)) end;\n", True),
        ("an operand is not pinned across its partner's spill", BACKEND,
         "        register = self._phys(self.reg(text), pins)\n        return Reg(register), pins | {register}\n",
         "        register = self._phys(self.reg(text), pins)\n        return Reg(register), pins\n",
         PRE + H + b"fn f(a: Int64, b: Int64) : Int64 = "
               + b"".join(b"let t%d = h(a, %d) in " % (i, i) for i in range(16))
               + b"+(+(t0, t1), +(+(t2, t3), +(+(t4, t5), +(+(t6, t7), +(+(t8, t9), "
                 b"+(+(t10, t11), +(+(t12, t13), +(t14, t15))))))));\n", False),
        ("the first arm's tag write is trusted by the second", LOWER,
         'tag = "keep" if top.l2 == info.tag else "write"',
         'tag = "keep" if top.l2 == info.tag or getattr(self, "wrote", None) == top.cell else "write"\n'
         "            self.wrote = top.cell",
         TAGS, True),
        ("a tag is never written", LOWER,
         'tag = "keep" if top.l2 == info.tag else "write"', 'tag = "keep"', TAGS, True),
        ("the tag lands before the field under it has left", BACKEND,
         "moves.append(Move(Mem(base, 0, 1), Imm(int(tag_text))))",
         "self._store(Mem(base, 0, 1), Imm(int(tag_text)))",
         b"data Free = End | More(next: Free);\n"
         b"record Q { a: Int64, b: Int64 };\ndata S = X(k: Int64) | Y(j: Int8) | Z;\n"
         b"fn u(q: Q) : S = match q with Q(d, a, b) => X@d(a) end;\n", True),
        ("a store lands before a read of what it overwrites", BACKEND,
         "        return isinstance(src, Mem) and _overlap(src, dest)\n", "        return False\n",
         PRE + b"fn u(q: Q) : Q @q = match q with Q(d, a, b) => Q@d(b, a) end;\n", True),
        ("registers a call clobbers are not saved", BACKEND,
         "if isinstance(home, str) and (home in clobbered or home in wanted):", "if False:",
         # The callee has to really write rdi, or leaving a value there would be correct.
         b"fn g(x: Int64) : Int64 = +(x, 1);\n"
         b"fn dec(x: Int64, y: Int64) : Int64 = g(y);\n"
         b"fn f(a: Int64, b: Int64) : Int64 = let t = dec(a, b) in +(t, a);\n", True),
        ("a comparison branches the wrong way", LOWER,
         'f"L{top.t}, L{top.f}")\n            self.live = False\n            return\n        register = self._result(top)\n'
         '        self.i(f"cmp',
         'f"L{top.f}, L{top.t}")\n            self.live = False\n            return\n        register = self._result(top)\n'
         '        self.i(f"cmp',
         b"fn f(a: Int64, b: Int64) : Int64 = if <(a, b) then 1 else 2;\n", True),
        ("subtraction becomes addition", LOWER,
         "name = OP_NAMES[top.aux]\n", 'name = OP_NAMES[top.aux].replace("sub", "add")\n',
         b"fn f(a: Int64, b: Int64) : Int64 = let c = -(a, b) in c;\n", True),
        ("a call is silently dropped", BACKEND,
         '        self.ins("call", f"fn_{callee.name}")\n', "",
         H + b"fn f(a: Int64) : Int64 = let t = h(a, 1) in t;\n", True),
        ("an oversized literal count reaches the hardware", BACKEND,
         "if count.value >= bits:", "if False:",
         b"fn f(a: Int64) : Int64 = <<(a, 70);\n", True),
        ("a register count is not compared with the width", BACKEND,
         '        self.ins("cmp", "rcx", str(bits))\n        self.ins("jae", zero)\n', "",
         b"fn f(a: Int64, b: Int64) : Int64 = <<(a, b);\n", True),
        ("a register count is compared with 64 whatever the type", BACKEND,
         'self.ins("cmp", "rcx", str(bits))', 'self.ins("cmp", "rcx", "64")',
         b"fn f(a: Int32, b: Int32) : Int32 = >>(a, b);\n", True),
        ("a count past the width leaves the value instead of 0", BACKEND,
         '        self.place(zero)\n        self.ins("xor", self.sub(register, 4), self.sub(register, 4))\n',
         "        self.place(zero)\n",
         b"fn f(a: Int64, b: Int64) : Int64 = <<(a, b);\n", True),
        ("the count overwrites a live value in rcx", BACKEND,
         'saved = "rcx" in self.owner and count != Reg("rcx")', "saved = False",
         b"fn f(a: Int64, b: Int64, c: Int64, d: Int64) : Int64 = +(<<(a, b), d);\n", True),
        ("a join leaves values where the arm put them", BACKEND,
         "        self._pmove(moves)\n        self._table(target)\n", "        self._table(target)\n",
         H + b"fn f(a: Int64, b: Int64) : Int64 = let x = +(a, 1) in "
             b"let y = if <(a, b) then h(x, b) else +(x, 2) in +(+(x, y), b);\n", True),
    ]
    # The stream and the backend, with nothing in between to refuse what the broken Lower wrote.
    WITHOUT_RTCHECK = """
import io, sys
from src.rtlower import rt_lines
from src.x86 import Backend
out = []
backend = Backend(out.append)
with open(sys.argv[1], "rb") as handle:
    for line in rt_lines(io.BytesIO(handle.read()), phase1=True):
        backend.feed(line)
print("\\n".join(out))
"""

    def _mutant_asm(self, path: str, old: str, new: str, source: bytes) -> str:
        """Compile with one line of the compiler changed, in a separate process."""
        with open(os.path.join(ROOT, path)) as handle:
            text = handle.read()
        self.assertIn(old, text, "this mutant no longer applies")
        with tempfile.TemporaryDirectory() as workdir:
            shutil.copytree(os.path.join(ROOT, "src"), os.path.join(workdir, "src"))
            with open(os.path.join(workdir, path), "w") as handle:
                handle.write(text.replace(old, new, 1))
            program = os.path.join(workdir, "p.lfpl")
            with open(program, "wb") as handle:
                handle.write(source)
            done = subprocess.run([sys.executable, "-c", self.WITHOUT_RTCHECK, program], cwd=workdir,
                                  capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout

    def test_a_cell_given_to_a_callee_with_its_fields_still_in_it_is_a_type_error(self):
        """A bug this project shipped, which cannot reach the symbolic check any more: the stream reads the
        field through a register the call took, so rtcheck says so, and the backend could not lower it."""
        old = ("            if value.kind == V_CAP:  # a location of this cell among the arguments is read first\n"
               "                self._close_cell(value.cell, frozenset(), len(self.V))\n")
        source = PRE + (b"fn one(c: Diamond) : List = Cons@c(7, Nil);\n"
                        b"fn rev(l: List, acc: List) : List = match l with "
                        b"| Nil => acc | Cons(d, h, t) => rev(t, Cons@d(h, acc)) end;\n"
                        b"fn f(l: List) : List = match l with "
                        b"| Cons(d, h, t) => rev(one(d), t) | Nil => Nil end;\n")
        with open(os.path.join(ROOT, self.LOWER)) as handle:
            text = handle.read()
        self.assertIn(old, text, "this mutant no longer applies")
        with tempfile.TemporaryDirectory() as workdir:
            shutil.copytree(os.path.join(ROOT, "src"), os.path.join(workdir, "src"))
            with open(os.path.join(workdir, self.LOWER), "w") as handle:
                handle.write(text.replace(old, "", 1))
            program = os.path.join(workdir, "p.lfpl")
            with open(program, "wb") as handle:
                handle.write(source)
            done = subprocess.run([sys.executable, "-m", "src.lfplo", program, "--nasm"], cwd=workdir,
                                  capture_output=True, text=True)
        self.assertEqual(done.returncode, 1, done.stderr)
        self.assertIn("leaves with a linear value still at [8:8]", done.stderr)

    def test_every_mutant_is_found_without_running_it(self):
        for name, path, old, new, source, _blind in self.MUTANTS:
            with self.subTest(mutant=name):
                self.assertEqual(check_translation(source, compile_text(source)), [],
                                 "the unbroken compiler must pass, or this proves nothing")
                asm = self._mutant_asm(path, old, new, source)
                self.assertTrue(check_translation(source, asm), asm)

    def test_the_redundancy_validator_is_blind_to_lost_stores(self):
        """Why this check exists: these mutants lose or misplace a store, and validate.py
        has nothing to say about any of them."""
        for name, path, old, new, source, blind in self.MUTANTS:
            if blind:
                with self.subTest(mutant=name):
                    self.assertEqual(validate(self._mutant_asm(path, old, new, source)), [])


class DeclaredLimits(unittest.TestCase):
    """What a program can run into and fix says which section declares it. Any other failure is a bug, and
    tests/test_fuzz.py fails on one."""

    def test_each_limit_names_its_section(self):
        fields = ", ".join(f"f{i}: Int8" for i in range(16))
        record = f"record R {{ {fields} }};\n"
        opened = ""
        for level in range(6):
            names = ", ".join(f"x{level}y{i}" for i in range(16))
            opened += f"match r{level} with R(d{level}, {names}) => "
        lets = "".join(f"let n{i} = 1 in " for i in range(30))
        parameters = ", ".join(f"r{level}: R" for level in range(6))
        for source in ((record + f"fn f({parameters}) : Int64 = {opened}{lets}1{' end' * 6};").encode(),
                       PRE + b"fn f(l: List) : List = match l with | Nil => Nil "
                             b"| Cons(d, h, t) => let r = f(t) in Cons@d(h, r) end;"):
            with self.subTest(source=source[:40]):
                with self.assertRaisesRegex(Limit, r"^section 8\.2:"):
                    compile_text(source)

    def test_a_program_past_no_limit_is_not_refused_as_one(self):
        self.assertEqual(run(b"fn main() : Int64 = 0;"), 0)


class StructuralRecursion(unittest.TestCase):
    """Rule R3 as a summed measure: for some fixed set of linear argument positions, not
    counting Free, the total cell count strictly falls at every recursive call. So every
    accepted function terminates (termination)."""

    WALK = (b"data Free = End | More(next: Free);\n"
            b"data Res = Fail | Done(value: Int64, rest: Free);\n"
            b"data Pair;\ndata T = Leaf | Node(v: Int64, kids: Pair);\n"
            b"record Pair { l: T, r: T };\ndata Stack = Empty | Push(t: T, s: Stack);\n"
            b"fn finish(acc: Int64, free: Free) : Res = match free with "
            b"| End => Fail | More(c, rest) => Done@c(acc, rest) end;\n"
            b"fn walk(t: T, stack: Stack, acc: Int64, free: Free) : Res = match t with "
            b"| Node(d, v, k) => match k with Pair(e, l, r) => "
            b"walk(l, Push@e(r, stack), +(*(acc, 3), v), More@d(free)) end "
            b"| Leaf => match stack with "
            b"| Empty => finish(acc, free) "
            b"| Push(d, next, rest) => walk(next, rest, acc, More@d(free)) end end;\n")

    def _refused(self, source: bytes) -> None:
        with self.assertRaisesRegex(Reject, "R3"):
            compile_text(source)

    def _accepted(self, source: bytes) -> None:
        compile_text(source)

    # --- what must be refused: most of these really do run forever

    def test_the_specs_spin_is_refused(self):
        self._refused(b"fn spin(x: Int64) : Int64 = spin(x);")

    def test_a_scalar_counter_is_refused(self):
        """It would terminate, but no linear value shrinks, so nothing shows that it does."""
        self._refused(b"fn go(n: Int64) : Int64 = if ==(n, 0) then 0 else go(-(n, 1));")

    def test_recursion_with_no_arguments_is_refused(self):
        self._refused(b"fn f() : Int64 = f();")

    def test_a_parameter_passed_on_unchanged_is_refused(self):
        self._refused(PRE + b"fn f(l: List) : List = f(l);")

    def test_rebuilding_the_argument_is_refused(self):
        """Each call rebuilds the same list from its own parts and passes it on."""
        self._refused(PRE + b"fn f(l: List) : List = match l with "
                            b"| Cons(d, h, t) => f(Cons@d(h, t)) | Nil => Nil end;")

    def test_swapping_a_rebuilt_list_between_positions_is_refused(self):
        self._refused(PRE + b"fn f(a: List, b: List) : List = match a with "
                            b"| Cons(d, h, t) => f(b, Cons@d(h, t)) | Nil => b end;")

    def test_a_cell_moved_back_and_forth_is_refused(self):
        """One call moves a cell from a to b and the other moves it back, so each set of
        positions fails at one of the calls, and the program really runs forever."""
        self._refused(PRE + b"fn f(a: List, b: List) : List = match a with "
                            b"| Cons(d, h, t) => f(t, Cons@d(h, b)) "
                            b"| Nil => match b with "
                            b"| Cons(d, h, t) => f(Cons@d(h, Nil), t) | Nil => Nil end end;")

    def test_a_field_of_a_cell_built_here_does_not_decrease(self):
        """t is l again, through a constructor matched at once, so this loops forever."""
        self._refused(PRE + b"fn f(l: List, c: Diamond) : List = match Cons@c(1, l) with "
                            b"| Cons(e, h, t) => f(t, e) | Nil => Nil end;")
        self._refused(PRE + b"fn f(l: List, c: Diamond) : List = let x = Cons@c(1, l) in "
                            b"match x with | Cons(e, h, t) => f(t, e) | Nil => Nil end;")

    def test_a_value_returned_by_a_call_counts_for_nothing(self):
        self._refused(PRE + b"fn g(l: List) : List = l;\n"
                            b"fn f(l: List) : List = match l with "
                            b"| Cons(d, h, t) => f(Cons@d(h, g(t))) | Nil => Nil end;")
        self._refused(PRE + b"fn g(l: List) : List = l;\n"
                            b"fn f(l: List) : List = f(g(l));")

    def test_a_joined_value_counts_for_nothing(self):
        """The check is conservative here: this one would terminate."""
        self._refused(PRE + b"fn f(l: List, acc: List, c: Bool) : List = match l with "
                            b"| Nil => acc | Cons(d, h, t) => "
                            b"let x = if c then t else t in f(x, Cons@d(h, acc), c) end;")

    def test_a_grown_argument_is_refused(self):
        self._refused(PRE + b"fn f(l: List, c: Diamond, free: Free) : List = match free with "
                            b"| End => Cons@c(1, l) "
                            b"| More(d, rest) => f(Cons@c(1, l), d, rest) end;")

    def test_cells_drawn_from_free_do_not_pay_for_a_rebuild(self):
        """The new cell comes from Free, so the measured list never gets any smaller."""
        self._refused(PRE + b"fn f(l: List, free: Free) : Free = match l with "
                            b"| Nil => free "
                            b"| Cons(d, h, t) => match free with "
                            b"| End => f(Cons@d(h, t), End) "
                            b"| More(c, rest) => f(Cons@c(h, t), More@d(rest)) end end;")

    def test_free_is_not_a_measure(self):
        """Walking the reserve is refused: work has to be paid for by the item."""
        self._refused(PRE + b"fn count(f: Free, n: Int64, back: Free) : Free = match f with "
                            b"| End => back "
                            b"| More(d, rest) => count(rest, +(n, 1), More@d(back)) end;")

    # --- the claim itself, tried on every combination of a family of recursive arms

    A_ARMS = (("f(t, b, More@d(free))", lambda a, b, r: (a - 1, b, r + 1)),
              ("f(b, t, More@d(free))", lambda a, b, r: (b, a - 1, r + 1)),
              ("f(t, Cons@d(h, b), free)", lambda a, b, r: (a - 1, b + 1, r)),
              ("f(Cons@d(h, b), t, free)", lambda a, b, r: (b + 1, a - 1, r)),
              ("f(Cons@d(h, t), b, free)", lambda a, b, r: (a, b, r)),
              ("f(b, Cons@d(h, t), free)", lambda a, b, r: (b, a, r)),
              ("match free with End => f(t, b, More@d(End)) "
               "| More(c, q) => f(Cons@c(h, t), b, More@d(q)) end",
               lambda a, b, r: (a, b, r) if r else (a - 1, b, 1)))
    B_ARMS = (("f(t, Nil, More@d(free))", lambda a, b, r: (b - 1, 0, r + 1)),
              ("f(Nil, t, More@d(free))", lambda a, b, r: (0, b - 1, r + 1)),
              ("f(Cons@d(h, t), Nil, free)", lambda a, b, r: (b, 0, r)),
              ("f(Nil, Cons@d(h, t), free)", lambda a, b, r: (0, b, r)))

    @staticmethod
    def _runs_forever(a_arm, b_arm) -> bool:
        """Simulated on list lengths alone, with no part of the compiler involved."""
        for a0 in range(4):
            for b0 in range(4):
                a, b, r = a0, b0, 2
                for _ in range(400):
                    if a:
                        a, b, r = a_arm(a, b, r)
                    elif b:
                        a, b, r = b_arm(a, b, r)
                    else:
                        break
                else:
                    return True
        return False

    def test_nothing_it_accepts_runs_forever(self):
        accepted = refused_but_ends = 0
        for a_text, a_arm in self.A_ARMS:
            for b_text, b_arm in self.B_ARMS:
                source = PRE + (
                    f"fn f(a: List, b: List, free: Free) : Free = match a with "
                    f"| Cons(d, h, t) => {a_text} "
                    f"| Nil => match b with | Nil => free | Cons(d, h, t) => {b_text} end end;\n"
                    f"fn main(r0: Free) : Res = match r0 with End => Fail | More(x, r1) => "
                    f"match r1 with End => Done@x(0, End) | More(y, r2) => "
                    f"match r2 with End => Done@x(0, More@y(End)) | More(z, r3) => "
                    f"match f(Cons@x(1, Cons@y(2, Nil)), Cons@z(3, Nil), r3) with "
                    f"End => Fail | More(c, q) => Done@c(7, q) end end end end;\n").encode()
                loops = self._runs_forever(a_arm, b_arm)
                with self.subTest(a=a_text, b=b_text):
                    try:
                        asm = compile_text(source)
                    except Reject as refusal:
                        self.assertIn("R3", str(refusal))
                        refused_but_ends += not loops
                        continue
                    self.assertFalse(loops, "accepted, and it runs forever")
                    accepted += 1
                    with tempfile.TemporaryDirectory() as workdir:
                        self.assertEqual(build_and_run(asm, workdir), 7)
        self.assertGreaterEqual(accepted, 3)
        print(f"R3 family: {accepted} accepted, {refused_but_ends} refused though they end",
              file=sys.stderr)

    # --- what must be accepted

    def test_walk_is_accepted(self):
        """Neither position falls at every call, but their sum does: the first call moves
        a cell out to Free, and the second pops one."""
        self._accepted(self.WALK)

    def test_walk_runs_and_reads_the_tree_in_order(self):
        source = self.WALK + (
            b"fn main(r0: Free) : Res = match r0 with End => Fail | More(a, r1) => "
            b"match r1 with End => Done@a(0, End) | More(b, r2) => "
            b"match r2 with End => Done@a(0, More@b(End)) | More(c, r3) => "
            b"match r3 with End => Done@a(0, More@b(More@c(End))) | More(d, r4) => "
            b"walk(Node@a(2, Pair@b(Node@c(5, Pair@d(Leaf, Leaf)), Leaf)), Empty, 0, r4) "
            b"end end end end;")
        self.assertEqual(run(source), 2 * 3 + 5)

    def test_a_subset_is_enough_when_the_whole_total_holds_still(self):
        """A cell moves from l to acc, so l + acc never falls, but l alone does."""
        self._accepted(PRE + b"fn rev(l: List, acc: List) : List = match l with "
                             b"| Nil => acc | Cons(d, h, t) => rev(t, Cons@d(h, acc)) end;")

    def test_a_direct_field_is_accepted(self):
        self._accepted(PRE + b"fn inc(l: List) : List @l = match l with "
                             b"| Cons(d, h, t) => Cons@d(+(h, 1), inc(t)) | Nil => Nil end;")

    def test_a_chain_of_matches_is_accepted(self):
        self._accepted(PRE + b"fn f(l: List, acc: List) : List = match l with | Nil => acc "
                             b"| Cons(d, h, t) => match t with | Nil => Cons@d(h, acc) "
                             b"| Cons(e, x, y) => f(y, Cons@d(h, Cons@e(x, acc))) end end;")

    def test_a_chain_through_another_type_is_accepted(self):
        self._accepted(b"data Pair;\ndata T = Leaf | Node(v: Int64, kids: Pair);\n"
                       b"record Pair { l: T, r: T };\ndata S = Stop | Go(t: T, s: S);\n"
                       b"fn f(t: T, s: S) : S = match t with | Leaf => s "
                       b"| Node(d, v, k) => match k with Pair(e, l, r) => "
                       b"f(l, Go@e(r, Go@d(Leaf, s))) end end;\n")

    def test_a_rebuilt_value_counts_its_cell_and_its_arguments(self):
        """Two cells come off the list and one goes back, so the total still falls."""
        self._accepted(PRE + b"fn f(l: List, free: Free) : Free = match l with | Nil => free "
                             b"| Cons(d, h, t) => match t with | Nil => More@d(free) "
                             b"| Cons(e, x, y) => f(Cons@e(x, y), More@d(free)) end end;")

    def test_arguments_may_swap_while_the_total_falls(self):
        self._accepted(PRE + b"fn f(a: List, b: List, free: Free) : Free = match a with "
                             b"| Cons(d, h, t) => f(b, t, More@d(free)) "
                             b"| Nil => match b with | Nil => free "
                             b"| Cons(d, h, t) => f(t, Nil, More@d(free)) end end;")

    def test_let_is_a_renaming(self):
        self._accepted(PRE + b"fn f(l: List, acc: List) : List = match l with | Nil => acc "
                             b"| Cons(d, h, t) => let m = t in f(m, Cons@d(h, acc)) end;")
        self._accepted(PRE + b"fn f(l: List, acc: List) : List = let k = l in match k with "
                             b"| Nil => acc | Cons(d, h, t) => f(t, Cons@d(h, acc)) end;")

    def test_a_shadowing_binder_is_the_one_that_counts(self):
        self._accepted(PRE + b"fn f(l: List, acc: List) : List = match l with | Nil => acc "
                             b"| Cons(d, h, l) => f(l, Cons@d(h, acc)) end;")

    def test_a_tail_call_to_an_earlier_function_is_not_recursion(self):
        self._accepted(b"fn g(x: Int64) : Int64 = x;\nfn f(x: Int64) : Int64 = g(x);")

    def test_each_function_gets_its_own_set(self):
        self._accepted(PRE + b"fn p(a: List, b: List) : List = match a with | Nil => b "
                             b"| Cons(d, h, t) => p(t, Cons@d(h, b)) end;\n"
                             b"fn q(a: List, b: List) : List = match b with | Nil => a "
                             b"| Cons(d, h, t) => q(Cons@d(h, a), t) end;")

    def test_a_list_of_your_own_type_is_a_measure(self):
        """Only the type named Free is excluded."""
        self._accepted(b"data Cells = None | Cell(next: Cells);\n"
                       b"fn f(c: Cells, back: Cells) : Cells = match c with | None => back "
                       b"| Cell(d, rest) => f(rest, Cell@d(back)) end;")


class TotalShifts(unittest.TestCase):
    """Section 4.4 of LFPL-O: 0 when k >= w, never the hardware's masked count."""

    def _call(self, expr: bytes, a: int = 0, b: int = 0, ty: bytes = b"Int64") -> int:
        result = expr if ty == b"Int64" else b"pack<%d>(%s)" % (
            {b"Int8": 8, b"Int16": 16, b"Int32": 32}[ty], expr)
        source = (b"fn f(a: %s, b: %s) : Int64 = %s;\nfn main() : Int64 = f(%d, %d);"
                  % (ty, ty, result, a, b))
        return run(source)

    def test_a_literal_count_at_or_past_the_width_gives_zero(self):
        top = 1 << 63
        for expr, a, expected in ((b"<<(a, 64)", 1, 0), (b"<<(a, 70)", 1, 0), (b"<<(a, 255)", 1, 0),
                                  (b">>(a, 64)", top, 0), (b">>(a, 70)", top, 0),
                                  (b"<<(a, 4294967296)", 1, 0), (b">>(a, 18446744073709551615)", top, 0)):
            with self.subTest(expr=expr):
                self.assertEqual(self._call(expr, a), expected)

    def test_a_literal_count_below_the_width_is_the_usual_shift(self):
        top = 1 << 63
        for expr, a, expected in ((b"<<(a, 0)", 5, 5), (b">>(a, 0)", 5, 5), (b"<<(a, 1)", 5, 10),
                                  (b"<<(a, 63)", 1, top), (b">>(a, 63)", top, 1), (b"<<(a, 63)", 3, top)):
            with self.subTest(expr=expr):
                self.assertEqual(self._call(expr, a), expected)

    def test_the_width_is_the_types_not_the_registers(self):
        """x86 masks a 32-bit shift to 5 bits and has no 8-bit width here at all."""
        for ty, bits in ((b"Int8", 8), (b"Int16", 16), (b"Int32", 32)):
            high = 1 << bits - 1
            for expr, a, expected in ((b">>(a, %d)" % (bits - 1), high, 1), (b">>(a, %d)" % bits, high, 0),
                                      (b">>(a, %d)" % (bits + 8), high, 0), (b">>(a, 40)", high, 0),
                                      (b"<<(a, %d)" % (bits - 1), 1, high), (b"<<(a, %d)" % bits, 1, 0),
                                      (b"<<(a, 40)", 1, 0), (b"<<(a, 1)", high, 0)):
                with self.subTest(ty=ty, expr=expr):
                    self.assertEqual(self._call(expr, a, 0, ty), expected)

    def test_two_literals(self):
        self.assertEqual(run(b"fn main() : Int64 = <<(1, 70);"), 0)
        self.assertEqual(run(b"fn main() : Int64 = >>(1, 70);"), 0)
        self.assertEqual(run(b"fn main() : Int64 = <<(1, 3);"), 8)

    def test_a_let_bound_literal_count(self):
        self.assertEqual(self._call(b"let k = 70 in <<(a, k)", 1), 0)
        self.assertEqual(self._call(b"let k = 3 in <<(a, k)", 1), 8)

    def test_an_oversized_literal_emits_no_shift_at_all(self):
        lines = body(b"fn f(a: Int64) : Int64 = <<(a, 70);\nfn main() : Int64 = 0;", "f")
        self.assertFalse([l for l in lines if l.startswith(("shl", "shr"))], lines)
        self.assertIn("xor ebx, ebx", lines)

    def test_a_register_count_is_compared_with_the_width_first(self):
        top = 1 << 63
        for count in (0, 1, 62, 63, 64, 65, 70, 128, 256, 1 << 32, (1 << 64) - 1):
            with self.subTest(count=count):
                self.assertEqual(self._call(b"<<(a, b)", 1, count),
                                 1 << count if count < 64 else 0)
                self.assertEqual(self._call(b">>(a, b)", top, count),
                                 top >> count if count < 64 else 0)

    def test_a_register_count_at_a_narrow_width(self):
        for ty, bits in ((b"Int8", 8), (b"Int16", 16), (b"Int32", 32)):
            high = 1 << bits - 1
            for count in (0, 1, bits - 1, bits, bits + 1, 31, 32, 33, 63, 64, 70, 255):
                if count >= 1 << bits:
                    continue
                with self.subTest(ty=ty, count=count):
                    self.assertEqual(self._call(b">>(a, b)", high, count, ty),
                                     high >> count if count < bits else 0)
                    self.assertEqual(self._call(b"<<(a, b)", high, count, ty),
                                     (high << count) & ((1 << bits) - 1) if count < bits else 0)

    # rcx is the only register x86 takes a count from, so every way of already
    # holding it has to come out the other side intact.
    FOUR = b"fn f(a: Int64, b: Int64, c: Int64, d: Int64) : Int64 = %s;\nfn main() : Int64 = f(%d, %d, %d, %d);"

    def test_a_live_value_in_rcx_survives_a_shift(self):
        self.assertEqual(run(self.FOUR % (b"+(<<(a, b), d)", 1, 3, 0, 100)), 8 + 100)
        self.assertEqual(run(self.FOUR % (b"+(<<(a, b), d)", 1, 70, 0, 100)), 0 + 100)

    def test_the_shifted_value_can_start_in_rcx(self):
        self.assertEqual(run(self.FOUR % (b"+(<<(d, a), d)", 3, 0, 0, 5)), 40 + 5)

    def test_the_count_can_start_in_rcx(self):
        self.assertEqual(run(self.FOUR % (b"+(<<(a, d), d)", 5, 0, 0, 3)), 40 + 3)
        self.assertEqual(run(self.FOUR % (b"+(<<(a, d), d)", 5, 0, 0, 64)), 0 + 64)

    def test_the_result_can_be_wanted_in_rcx(self):
        source = (b"fn g(p: Int64, q: Int64, r: Int64, s: Int64) : Int64 = +(p, +(q, +(r, s)));\n"
                  b"fn f(a: Int64, b: Int64) : Int64 = g(1, 2, 3, <<(a, b));\n"
                  b"fn main() : Int64 = f(5, %d);")
        self.assertEqual(run(source % 3), 1 + 2 + 3 + 40)
        self.assertEqual(run(source % 70), 1 + 2 + 3)

    def test_computed_operands(self):
        self.assertEqual(self._call(b"<<(+(a, 1), +(b, 1))", 2, 2), 3 << 3)
        self.assertEqual(self._call(b"<<(+(a, 1), +(b, 1))", 2, 63), 0)
        self.assertEqual(self._call(b">>(a, if <(b, 5) then 1 else 99)", 8, 1), 4)
        self.assertEqual(self._call(b">>(a, if <(b, 5) then 1 else 99)", 8, 9), 0)

    def test_a_count_read_from_a_field(self):
        source = PRE + (b"fn g(q: Q, free: Free) : Res = match q with Q(d, x, k) => "
                        b"Done@d(<<(x, k), free) end;\n"
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(c, rest) => g(Q@c(3, %d), rest) end;")
        self.assertEqual(run(source % 2), 12)
        self.assertEqual(run(source % 70), 0)

    def test_a_shift_inside_a_condition(self):
        self.assertEqual(self._call(b"if <(<<(a, b), 10) then 1 else 2", 1, 2), 1)
        self.assertEqual(self._call(b"if <(<<(a, b), 10) then 1 else 2", 1, 5), 2)
        self.assertEqual(self._call(b"if <(<<(a, b), 10) then 1 else 2", 1, 70), 1)

    def test_a_shift_into_a_field(self):
        source = PRE + (b"fn g(q: Q, n: Int64, free: Free) : Res = match q with Q(d, x, k) => "
                        b"Done@d(+(0, >>(x, n)), free) end;\n"
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(c, rest) => g(Q@c(16, 0), %d, rest) end;")
        self.assertEqual(run(source % 2), 4)
        self.assertEqual(run(source % 64), 0)

    def test_the_symbolic_check_agrees_with_every_lowering_here(self):
        for text in (b"<<(a, 70)", b">>(a, 64)", b"<<(a, 3)", b"<<(a, b)", b">>(a, b)",
                     b"<<(+(a, 1), +(b, 1))", b"if <(<<(a, b), 10) then 1 else 2"):
            source = b"fn f(a: Int64, b: Int64) : Int64 = %s;\n" % text
            with self.subTest(text=text):
                self.assertEqual(check_translation(source, compile_text(source)), [])

    def test_a_constant_count_the_guard_sends_around_the_shift_is_no_problem(self):
        """Found by the fuzzer, as a false alarm: on the path where the count is that
        constant, the comparison has already jumped, so the shift is never reached."""
        source = (b"fn f(a: Int64, b: Int64) : Int64 = "
                  b">>(a, if ==(a, b) then 9223372036854775807 else b);\n")
        self.assertEqual(check_translation(source, compile_text(source)), [])

    def test_the_symbolic_check_sees_the_hardware_mask(self):
        """`shl rax, 70` assembles without a warning and shifts by 6."""
        source = b"fn f(a: Int64) : Int64 = <<(a, 70);\n"
        wrong = "fn_f_body:\n    mov rax, rdi\n    shl rax, 70\n    ret\n"
        self.assertTrue(check_translation(source, wrong))
        right = "fn_f_body:\n    xor eax, eax\n    ret\n"
        self.assertEqual(check_translation(source, right), [])

    def test_the_symbolic_check_sees_an_unguarded_register_count(self):
        source = b"fn f(a: Int64, b: Int64) : Int64 = <<(a, b);\n"
        wrong = ("fn_f_body:\n    mov rax, rdi\n    mov rcx, rsi\n    shl rax, cl\n    ret\n")
        self.assertTrue(check_translation(source, wrong))
        guarded = ("fn_f_body:\n    mov rax, rdi\n    mov rcx, rsi\n    cmp rcx, 64\n    jae L1\n"
                   "    shl rax, cl\n    jmp L2\nL1:\n    xor eax, eax\nL2:\n    ret\n")
        self.assertEqual(check_translation(source, guarded), [])
        loose = guarded.replace("cmp rcx, 64", "cmp rcx, 65")
        self.assertTrue(check_translation(source, loose))


class Horner(unittest.TestCase):
    """Section 4.5's worked example (translation correctness)."""

    def test_on_a_register_destination(self):
        """The example, with two more moves. The value is computed in a register of its own and moved to
        where `ret` wants it, and the 32-bit field is narrowed through scratch, because RT does not tell
        the backend the operand is already that narrow. A 32-bit `or` would clear the upper half."""
        source = (b"fn w(k: Int8, x: Int16, y: Int32) : Int64 = "
                  b"pack<8,8,16,32>(3, k, x, y);\nfn main() : Int64 = w(7, 513, 9);")
        lines = body(source, "w")
        self.assertEqual(lines, ["mov ebx, 768", "or bl, dil", "shl rbx, 16", "or bx, si",
                                 "shl rbx, 32", "mov r11d, edx", "or rbx, r11", "mov rax, rbx", "ret"])
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
        self.assertEqual(lines, ["mov ebx, 258", "mov rax, rbx", "ret"])

    def test_a_literal_after_a_field(self):
        """No constant ahead of it to fold into, a zero to skip, and one too wide for an `or` immediate."""
        for widths, literal, expected in ((b"8,8", 3, 7 << 8 | 3), (b"8,8", 0, 7 << 8),
                                          (b"8,32", 0x80000000, 7 << 32 | 0x80000000)):
            source = (b"fn w(k: Int8) : Int64 = pack<%s>(k, %d);\nfn main() : Int64 = w(7);" % (widths, literal))
            with self.subTest(widths=widths, literal=literal):
                self.assertEqual(run(source), expected)

    def test_a_wider_operand_keeps_only_the_fields_width(self):
        """The first field, a later field in a subregister, and a later field through scratch."""
        for widths, expected in ((b"8,8", 0xBB << 8 | 0xBB), (b"16,16", 0xABBB << 16 | 0xABBB),
                                 (b"32,32", 0xAAAAABBB << 32 | 0xAAAAABBB)):
            source = (b"fn w(k: Int64) : Int64 = pack<%s>(k, k);\nfn main() : Int64 = w(%d);"
                      % (widths, 0xAAAAAAAAAAAAABBB))
            with self.subTest(widths=widths):
                self.assertEqual(run(source), expected)

    def test_a_field_read_at_a_width_other_than_its_own(self):
        source = PRE + (b"record S { a: Int8, b: Int16, c: Int32, e: Int64 };\n"
                        b"fn w(s: S, free: Free) : Res = match s with S(d, a, b, c, e) => "
                        b"Done@d(pack<16,8,16,32>(a, b, e, e), free) end;\n"
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(c, rest) => w(S@c(1, 515, 3, 72057594037927941), rest) end;")
        self.assertEqual(run(source), 1 << 56 | 3 << 48 | 5 << 32 | 5)

    def test_a_32_bit_memory_field_takes_the_section_8_extra_load(self):
        source = PRE + (b"fn w(p: P, free: Free) : Res = match p with P(d, x, y, z) => "
                        b"Done@d(pack<32,32>(x, y), free) end;\n"
                        b"fn main(free: Free) : Res = match free with End => Fail | "
                        b"More(c, rest) => w(P@c(1, 2, 3), rest) end;")
        self.assertEqual(run(source), 1 << 32 | 2)


class CommandLine(unittest.TestCase):
    def _cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "src.lfplo", *args], cwd=ROOT,
                              capture_output=True, text=True)

    def test_it_runs_a_program_and_prints_the_result(self):
        done = self._cli("examples/sum.lfpl", "--run")
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "9"), done.stderr)

    def test_it_prints_the_assembly(self):
        done = self._cli("examples/inc.lfpl", "--nasm")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("fn_inc$body:", done.stdout)

    def test_a_rejected_program_exits_nonzero_with_the_rule(self):
        with tempfile.NamedTemporaryFile(suffix=".lfpl") as bad:
            bad.write(PRE + b"fn forget(l: List) : Int64 = 1;\n")
            bad.flush()
            done = self._cli(bad.name, "--nasm")
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
        ring = compile_source(file_bytes(handle))
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
