"""Break the bound machinery one line at a time, in a copy of the tree, and see whether the tests notice.

Each mutant is something that would make a certified bound wrong, or the runtime's use of it wrong. A
mutant that survives is a claim in the README that nothing is testing, so this exits with a failure
if any does. It takes a long time, so it is not part of the test suite: run it after changing
src/rtcheck.py, src/simplex.py or the admission code in src/runtime.py.

    python3 tools/mutate_bounds.py [name of a mutant...]
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATIENCE = 300  # seconds: a mutant with nothing to catch it takes about 90
RUNTIME_TESTS = ["tests.test_forward.WhatTheRuntimeDoesWithIt", "tests.test_forward.WhatTheKernelSaid"]
VERILOG_TESTS = ["tests.test_rtverilog.TheEmittersOwnArithmetic", "tests.test_rtverilog.TheChecksum",
                 "tests.test_rtverilog.RandomWalks", "tests.test_rtverilog.OtherHandlers",
                 "tests.test_rtverilog.Synthesis"]
# Running code first, so that a verdict says whether a mutant is caught by what a binary does, and
# not only by the numbers in a unit test.
TESTS = ["tests.test_walks.RandomWalks", "tests.test_walks.TheDearestWayThrough", "tests.test_walks.TheEndArm",
         "tests.test_walks.OneFunctionAtSeveralAmounts", "tests.test_walks.TooLargeForAnyMachine",
         "tests.test_checksum.TheBound", "tests.test_checksum.TheChecksum",
         "tests.test_checksum.WhatTheChainTypeOpensUp", "tests.test_rtcheck", "tests.test_simplex"]

# The pass's own tests first, then what runs the generators with their annotations taken out.
PASS_TESTS = ["tests.test_rtlower.TheRightToLeftPass", "tests.test_rtlower.ValidPrograms",
              "tests.test_generated.GeneratedPrograms", "tests.test_fuzz.Differential.test_diamonds_passed_to_calls",
              "tests.test_fuzz.Differential.test_lists", "tests.test_fuzz.Differential.test_tagged_cells"]

MUTANTS = {
    # --- the rules that derive the inequalities
    "cells released twice": ("src/rtcheck.py",
        "self.S.budget = self.S.budget.plus(token.pot)\n",
        "self.S.budget = self.S.budget.plus(token.pot, 2)\n"),
    "build not charged a step": ("src/rtcheck.py",
        'NO_WORK = {"drop", "open", "move"}',
        'NO_WORK = {"drop", "open", "move", "build"}'),
    "take not charged a step": ("src/rtcheck.py",
        'NO_WORK = {"drop", "open", "move"}',
        'NO_WORK = {"drop", "open", "move", "take"}'),
    "tail not charged a step": ("src/rtcheck.py",
        'NO_WORK = {"drop", "open", "move"}',
        'NO_WORK = {"drop", "open", "move", "tail"}'),
    "the new cell's share not paid": ("src/rtcheck.py",
        "self.S.budget = self.S.budget.plus(carried, -1)  # the new cell's share, paid now\n",
        "pass\n"),
    "a draw not charged a cell": ("src/rtcheck.py",
        "        if reserve:\n            self.S.budget = self.S.budget.spend(cells=1)\n",
        ""),
    "every null arm unreachable": ("src/rtcheck.py",
        "null_edge.unreachable = base.unreachable or reserve\n",
        "null_edge.unreachable = True\n"),
    "a Free can be made anywhere": ("src/rtcheck.py",
        "        if handler.makes_a_free:\n",
        "        if False:\n"),
    "a callee's making a Free does not count": ("src/rtcheck.py",
        "            fn.makes_a_free = fn.makes_a_free or callee.makes_a_free\n",
        "            pass\n"),
    "a Free made past an End arm counts": ("src/rtcheck.py",
        "if self.is_reserve(ty) and not self.S.unreachable:",
        "if self.is_reserve(ty):"),
    "a callee's entry not charged": ("src/rtcheck.py",
        "self.S.budget = self.S.budget.plus(signature.entry, -1)\n",
        "pass\n"),
    "a callee's rows not copied": ("src/rtcheck.py",
        "        for row in callee.obligations:\n            self.obligate(Budget(tuple(sorted((copy(u), c) for u, c in row.terms)), row.cells, row.steps))\n",
        ""),
    "arguments need not carry what parameters want": ("src/rtcheck.py",
        "            self.at_least(token.pot, signature.param_pots[position])\n",
        ""),
    "a result need not carry what the function promises": ("src/rtcheck.py",
        "            self.at_least(token.pot, fn.result_pots[position])\n",
        ""),
    "a tail call's result not tied to the caller's": ("src/rtcheck.py",
        "                self.at_least(theirs, ours)\n",
        "                pass\n"),
    "a field need not carry what its new cell does (=)": ("src/rtcheck.py",
        '                self.at_least(held.pot if held is not None else None, None if is_scalar(ty) else carried)\n',
        ""),
    "a field need not carry what its new cell does (store)": ("src/rtcheck.py",
        "                self.at_least(token.pot, carried)\n        if star is not None and stars != 1:",
        "        if star is not None and stars != 1:"),
    "a join takes the larger budget": ("src/rtcheck.py",
        "            return Budget(a.budget.terms, min(a.budget.cells, b.budget.cells),\n                          min(a.budget.steps, b.budget.steps))\n",
        "            return Budget(a.budget.terms, max(a.budget.cells, b.budget.cells),\n                          max(a.budget.steps, b.budget.steps))\n"),
    "a join's edges owe nothing": ("src/rtcheck.py",
        "        self.obligate(a.budget.plus(joined, -1))\n        self.obligate(b.budget.plus(joined, -1))\n",
        ""),
    "a join's value keeps the first edge's amount": ("src/rtcheck.py",
        "        joined = self.unknown()\n        self.at_least(x, joined)\n        self.at_least(y, joined)\n        return joined\n",
        "        return x\n"),
    "the exit owes nothing": ("src/rtcheck.py",
        "        self.obligate(self.S.budget)\n        self.S = None\n\n    def i_tail(",
        "        self.S = None\n\n    def i_tail("),
    "the tail call owes nothing": ("src/rtcheck.py",
        "        assert self.S is not None\n        self.obligate(self.S.budget)\n        self.S = None\n\n    def i_tailmod(",
        "        self.S = None\n\n    def i_tailmod("),
    "the reserve is given a share": ("src/rtcheck.py",
        "            rows[Budget(((reserve, -1),))] = None\n",
        "            pass\n"),
    "a recursive call gets its own numbers": ("src/rtcheck.py",
        "            fn.recursive = True\n            signature = fn\n",
        "            fn.recursive = True\n            signature = replace(fn, entry=self.unknown(), param_pots=[None if p is None else self.unknown() for p in fn.param_pots], result_pots=[None if p is None else self.unknown() for p in fn.result_pots])\n"),
    "the decreasing set is not tied": ("src/rtcheck.py",
        "            self.at_least(first, second)\n            self.at_least(second, first)\n",
        "            pass\n"),
    "a table entry rounds down": ("src/rtcheck.py",
        "return 0 if unknown is None else math.ceil(Fraction(values.get(unknown, 0)))",
        "return 0 if unknown is None else math.floor(Fraction(values.get(unknown, 0)))"),
    "verify does not look at the rows": ("src/rtcheck.py",
        "                if total < 0:\n",
        "                if False:\n"),
    "verify lets a negative through": ("src/rtcheck.py",
        "            if any(value < 0 for value in values.values()):\n",
        "            if False:\n"),
    "the item's amount is read from the wrong parameter": ("src/rtcheck.py",
        "        item = handler.param_pots[-2]\n",
        "        item = handler.param_pots[-1]\n"),
    # --- the backend's counter, which the tests time things with
    "the counter misses build": ("src/x86.py",
        "if self.count_steps and head not in NO_WORK:",
        'if self.count_steps and head not in NO_WORK and head != "build":'),
    # --- the solver
    "the solver stops a pivot early": ("src/simplex.py",
        "    while True:\n        entering = min(",
        "    for _ in range(1):\n        entering = min("),
    "the solver ignores an unbounded objective": ("src/simplex.py",
        '            raise UnboundedProblem("the objective has no minimum")\n',
        "            return\n"),
    # --- what the runtime does with the table
    "admission's n is one short": ("src/runtime.py",
        'add("    lea r14, [rax+1]")', 'add("    lea r14, [rax+0]")'),
    "the reserve is half what it should be": ("src/runtime.py",
        'add("    mov rax, [admission]")\n    add("    imul rax, r14")\n',
        'add("    mov rax, [admission]")\n    add("    imul rax, r14")\n    add("    shr rax, 1")\n'),
    "stale bytes are not cleared": ("src/runtime.py",
        'add("    and [r10], rax")', 'pass'),
    "egress does not clamp a forged count": ("src/runtime.py",
        'add(f"    jbe .counted_bytes_{position}")', 'add(f"    jmp .counted_bytes_{position}")'),
    "egress does not bound the chain": ("src/runtime.py",
        'add(f"    jae .too_long_{position}")', 'pass'),
    "a whole cell's used count is left stale": ("src/runtime.py",
        'add(f"    mov byte [rdi], {BODY_PAYLOAD}")', 'pass'),
    # --- the Verilog backend: what it builds, and what it says about what it built
    "a field is read before its cell is": ("src/rtverilog.py",
        "        if self.loaded == register:\n            return\n        pointer = self._pointer(register)\n",
        "        self.loaded = register\n        return\n        pointer = self._pointer(register)\n"),
    "a word at the port survives a call that reads": ("src/rtverilog.py",
        "        if callee.reads:  # it leaves its own word at the port\n            self.loaded = None\n",
        ""),
    "a word at the port survives another read": ("src/rtverilog.py",
        "        self._edge(a[1], self.loaded)\n        self._edge(a[2], p)\n",
        "        self._edge(a[1], self.loaded)\n        self._edge(a[2], self.loaded)\n"),
    "a word at the port survives a join it does not reach": ("src/rtverilog.py",
        "self.loaded = incoming[0] if incoming and all(i == incoming[0] for i in incoming) else None",
        "self.loaded = incoming[0] if incoming else None"),
    "a word at the port outlives a call's result in its register": ("src/rtverilog.py",
        "            site.results.append((fn.flop(register), fn.widths[register]))\n            if self.loaded == register:\n                self.loaded = None\n",
        "            site.results.append((fn.flop(register), fn.widths[register]))\n"),
    "a word at the port outlives a move into its register": ("src/rtverilog.py",
        "        elif self.loaded == target:\n            self.loaded = None  # it held that cell once, and it does not now\n",
        ""),
    "a callee's reads do not count for what calls it": ("src/rtverilog.py",
        "fn.reads = any(state.reads for state in fn.states) or any(callee.reads for callee in fn.calls)",
        "fn.reads = any(state.reads for state in fn.states)"),
    "a write covers the whole cell": ("src/rtverilog.py",
        "            enables |= ((1 << size) - 1) << offset\n            low, high, width",
        "            enables = 0xffff\n            low, high, width"),
    "a tag is not written": ("src/rtverilog.py",
        "        if tag == \"write\":\n            enables |= 1\n",
        "        if False:\n            enables |= 1\n"),
    "a RESTORE writes": ("src/rtverilog.py",
        "            if source in (\"=\", \"*\") or source.endswith(\"!\"):\n                continue\n            enables",
        "            if source in (\"=\", \"*\"):\n                continue\n            enables"),
    "the emitter forgets the states in front when it works out k": ("src/rtverilog.py",
        "        cost = 1 + self.attached\n",
        "        cost = 1\n"),
    "the emitter forgets the states in front when it works out c": ("src/rtverilog.py",
        "            own = 1 if state.hardware else 0\n",
        "            own = 0\n"),
    "c does not look inside what a handler calls": ("src/rtverilog.py",
        "                own += self._hardware_states(state.next.callee, walking, known)\n",
        "                pass\n"),
    "k does not look at what a walk calls": ("src/rtverilog.py",
        "                waiting.extend(fn.calls)\n",
        "                pass\n"),
    "a tail call returns to nobody's caller": ("src/rtverilog.py",
        "                    if inherited not in callee.sites:\n                        callee.sites.append(inherited)\n",
        "                    pass\n"),
    "an @ function returns where it got to": ("src/rtverilog.py",
        "        if fn.at is not None:  # whatever this iteration holds, the caller gets the cell it passed\n            values",
        "        if False:\n            values"),
    "a waiting move is forgotten": ("src/rtverilog.py",
        "        for target, source in self.pending.items():  # every waiting move happens in this clock\n",
        "        for target, source in ():\n"),
    "a latched operand is not what is used": ("src/rtverilog.py",
        "        if text in self.latched:\n            name, width = self.latched[text]\n            return _whole(name, width, want)\n",
        ""),
    "pack puts its operands the other way round": ("src/rtverilog.py",
        "        parts = [self._value(operand, bits) for bits, operand in zip(widths, a[2:])]\n",
        "        parts = [self._value(operand, bits) for bits, operand in zip(widths, a[2:])][::-1]\n"),
    "a comparison is one wider than it should be": ("src/rtverilog.py",
        "COMPARISONS = {\"b\": \"<\", \"be\": \"<=\", ",
        "COMPARISONS = {\"b\": \"<=\", \"be\": \"<=\", "),
    "the host can read while it writes": ("src/rtverilog.py",
        "a_en = host_en && host_we == {CELL_BYTES}'d0;",
        "a_en = host_en;"),
    "a write reads as well": ("src/rtverilog.py",
        "        return [f\"b_we = {CELL_BYTES}'h{enables:04x};\", ",
        "        return [\"a_en = 1'b1;\", f\"b_we = {CELL_BYTES}'h{enables:04x};\", "),
    "port A's output does not hold": ("src/rtverilog.py",
        "        add(\"        if (a_en) a_dout <= cells[a_addr];\")\n",
        "        add(\"        a_dout <= cells[a_addr];\")\n"),
    # --- what the kernel says about ingress
    "the drop count is taken whole": ("src/runtime.py",
        'add("    sub edx, [last_drops]")', 'pass', RUNTIME_TESTS),
    "the last count is not kept": ("src/runtime.py",
        'add("    mov [last_drops], r8d")', 'pass', RUNTIME_TESTS),
    "the difference is taken in 64 bits": ("src/runtime.py",
        'add("    sub edx, [last_drops]")', 'add("    mov r9d, [last_drops]"); add("    sub rdx, r9")', RUNTIME_TESTS),
    "the socket is not asked for the count": ("src/runtime.py",
        "for option in (SO_RXQ_OVFL, SO_TIMESTAMPNS) if o.measure_age else (SO_RXQ_OVFL,):",
        "for option in (SO_TIMESTAMPNS,) if o.measure_age else ():", RUNTIME_TESTS),
    "the socket is not asked for the time": ("src/runtime.py",
        "for option in (SO_RXQ_OVFL, SO_TIMESTAMPNS) if o.measure_age else (SO_RXQ_OVFL,):",
        "for option in (SO_RXQ_OVFL,):", RUNTIME_TESTS),
    "the room for control messages is not reset": ("src/runtime.py",
        'add(f"    mov qword [in_msg+40], {CONTROL_BYTES}")', 'pass', RUNTIME_TESTS),
    "a message is not aligned to the next": ("src/runtime.py",
        'add("    add rcx, 7")', 'pass', RUNTIME_TESTS),
    "a negative age is not zero": ("src/runtime.py",
        'add("    jns .known")', 'add("    jmp .known")', RUNTIME_TESTS),
    "the age is the last and not the largest": ("src/runtime.py",
        'add("    jbe .done")\n        add("    mov [cnt_max_age_ns], rax")', 'add("    mov [cnt_max_age_ns], rax")', RUNTIME_TESTS),
    "an item with no timestamp has an age": ("src/runtime.py",
        'add("    test rdx, rdx")\n        add("    jz .done")\n        add("    mov [item_arrival_sec], rdx")',
        'add("    mov [item_arrival_sec], rdx")', RUNTIME_TESTS),
    "a refused item is aged": ("src/runtime.py",
        'add("    call read_control")', 'add("    call read_control"); add("    push rax"); add("    call note_age"); add("    pop rax")',
        RUNTIME_TESTS),
    # --- back to the runtime
    "the image is sized without the reserve": ("src/runtime.py",
        "return (K_IN * n_max + self.reserve_at_most + n_max\n",
        "return (K_IN * n_max + n_max\n"),
    "a queued output is taken to be one item's size": ("src/runtime.py",
        "return self.options.n_max + self.reserve_at_most\n",
        "return self.options.n_max\n"),
    "one message gathers only one item's cells": ("src/runtime.py",
        "out_iov = min(IOV_MAX, HEADER_BYTES_MAX + max(1, image.largest_output))",
        "out_iov = HEADER_BYTES_MAX + max(1, image.options.n_max)"),
    "an image of any size is built": ("src/runtime.py",
        "if image.bytes > MAX_IMAGE_BYTES:",
        "if False:"),
    "W(N_max) of any size is built": ("src/runtime.py",
        "if image.bound.c * o.n_max + image.bound.d >= WORD_LIMIT:",
        "if False:"),
    "a linear program of any size is solved": ("src/rtcheck.py",
        "if len(fn.obligations) > MAX_OBLIGATIONS:",
        "if False:"),
    # --- the right-to-left pass: last uses
    "pass: a use does not make a name wanted": ("src/righttoleft.py",
        "                flags |= F_LAST\n            self.live |= bit\n", "                flags |= F_LAST\n", PASS_TESTS),
    "pass: every use is the last": ("src/righttoleft.py",
        "if kind == K_VAR and not self.live & bit:", "if kind == K_VAR:", PASS_TESTS),
    "pass: no use is the last": ("src/righttoleft.py",
        "                flags |= F_LAST\n", "                pass\n", PASS_TESTS),
    "pass: a binder in a pattern stays wanted": ("src/righttoleft.py",
        "        elif kind in (K_BINDER, K_FIELD):\n            self.live &= ~bit\n", "", PASS_TESTS),
    "pass: a let's name stays wanted": ("src/righttoleft.py",
        "            for position in range(stamp, min(stamp + count, MASK_BITS)):\n                self.live &= ~(1 << position)\n",
        "", PASS_TESTS),
    "pass: an arm starts from what the last arm wanted": ("src/righttoleft.py",
        "            elif kind == K_ARM:\n                self.live = self._branch()[0]\n", "", PASS_TESTS),
    "pass: a match wants only what its first arm does": ("src/righttoleft.py",
        "            self._set_branch(after, before | self.live)\n", "            self._set_branch(after, self.live)\n", PASS_TESTS),
    "pass: an else starts from what the then wanted": ("src/righttoleft.py",
        "                self._set_branch(after, self.live)\n                self.live = after\n",
        "                self._set_branch(after, self.live)\n", PASS_TESTS),
    "pass: an if wants only what its then does": ("src/righttoleft.py",
        "                self.live |= before\n", "                pass\n", PASS_TESTS),
    "pass: an arm is not told what is wanted": ("src/righttoleft.py",
        "elif kind == K_ARM or (kind == K_WITH and index == 2)", "elif (kind == K_WITH and index == 2)", PASS_TESTS),
    "pass: an if is not told what is wanted": ("src/righttoleft.py",
        "        elif kind == K_IF:\n            self._annotate(self.cursor, record, F_LIVE, self.live)\n",
        "        elif kind == K_IF:\n", PASS_TESTS),
    "pass: a let is not told what is wanted": ("src/righttoleft.py",
        "        elif kind == K_LET:\n            self._annotate(self.cursor, record, F_LIVE, self.live)\n",
        "        elif kind == K_LET:\n", PASS_TESTS),
    "pass: a name's position is off by one": ("src/righttoleft.py",
        "                return position, tag - 2\n", "                return position + 1, tag - 2\n", PASS_TESTS),
    "pass: an error says where the function ended": ("src/righttoleft.py",
        "        self.ring.replay_line = line\n", "", PASS_TESTS),
    "lower: a scalar is never moved": ("src/rtlower.py",
        "            elif self._wanted_later(index):\n", "            elif False:\n", PASS_TESTS),
    "lower: every scalar is moved": ("src/rtlower.py",
        "            elif self._wanted_later(index):\n", "            else:\n", PASS_TESTS),
    "lower: a last use does not end a name": ("src/rtlower.py",
        "            self.live_now &= ~(1 << src)\n", "            pass\n", PASS_TESTS),
    "lower: a scalar moved in the last arm of an if is left in its register": ("src/rtlower.py",
        "        self._finish_moves_of_arm(top)\n        self._note_arm(top)\n", "        self._note_arm(top)\n", PASS_TESTS),
    "lower: a scalar moved in the last arm of a match is left in its register": ("src/rtlower.py",
        "        self._finish_moves_of_arm(match)\n", "", PASS_TESTS),
    "lower: each arm moves a scalar somewhere else": ("src/rtlower.py",
        "self.B.set(undo.index, replace(undo.binder, refuge=self.B.at(undo.index).refuge))",
        "self.B.set(undo.index, undo.binder)", PASS_TESTS),
    "lower: names past the annotation are taken to be unused": ("src/rtlower.py",
        "return index >= MASK_BITS or bool(self.live_now >> index & 1)", "return bool(self.live_now >> index & 1)", PASS_TESTS),
    # --- the right-to-left pass: the cell every exit returns
    "pass: a let's name has no word": ("src/righttoleft.py",
        "self._bind(name, self.word if count == 1 else NOTHING_KNOWN)", "self._bind(name, NOTHING_KNOWN)", PASS_TESTS),
    "pass: a diamond has no word": ("src/righttoleft.py",
        "position = self._bind(payload, frame.word if frame.count == 0 else NOTHING_KNOWN)",
        "position = self._bind(payload, NOTHING_KNOWN)", PASS_TESTS),
    "pass: a field has its cell's word": ("src/righttoleft.py",
        "position = self._bind(payload, frame.word if frame.count == 0 else NOTHING_KNOWN)",
        "position = self._bind(payload, frame.word)", PASS_TESTS),
    "pass: an if keeps a word its arms disagree on": ("src/righttoleft.py",
        "            self.word = self.word if self.word == frame.join else NOTHING_KNOWN\n", "", PASS_TESTS),
    "pass: a match keeps a word its arms disagree on": ("src/righttoleft.py",
        "match.join = self.word if match.join in (NOT_YET, self.word) else NOTHING_KNOWN", "match.join = self.word", PASS_TESTS),
    "pass: any exit allows anything": ("src/righttoleft.py",
        "        else:\n            self.allowed = 0\n\n    def _finished", "\n    def _finished", PASS_TESTS),
    "pass: a null arm says nothing": ("src/righttoleft.py",
        "                self.nulls |= 1 << frame.word\n", "                pass\n", PASS_TESTS),
    # There is no mutant for a null staying known after its arm. A function returning one value has to return every
    # cell it was given, so an exit of null means every parameter of that type is null on that path, and a stale
    # fact about one cannot change what is allowed, on a program that gets past Check.
    "pass: a recursive call allows anything": ("src/righttoleft.py",
        "for k in range(min(frame.count, MAX_PARAMETERS)) if frame.args[k] == k)",
        "for k in range(min(frame.count, MAX_PARAMETERS)))", PASS_TESTS),
    "pass: a parameter of any type can come back": ("src/righttoleft.py",
        "            if self.P.at(position)[1] == self.result_type:\n                self.allowed |= 1 << position\n",
        "            self.allowed |= 1 << position\n", PASS_TESTS),
    "pass: a scalar is a cell": ("src/righttoleft.py",
        'if self.result_type.rstrip(b"\\0") in _NOT_A_CELL or not self.result_type:', "if not self.result_type:", PASS_TESTS),
    "pass: a callee's cell is not looked up": ("src/righttoleft.py",
        "            frame.join = self._cell_of(payload)\n", "", PASS_TESTS),
    "pass: a function is not told what it returns": ("src/righttoleft.py",
        "            self.after.append((ev_tag(LEAF, K_RESWORD), self.write_at))\n", "            pass\n", PASS_TESTS),
    "symcheck: it does not work out the cell": ("src/symcheck.py",
        "        program.infer_cells()\n", "", PASS_TESTS),
    "symcheck: a word is updated only once": ("src/symcheck.py",
        "lambda x: T.select(cond, result, x) if self._pointer_of(x) == pointer else x, term)",
        "lambda x: T.select(cond, result, x) if x == pointer else x, term)", PASS_TESTS),
}


def run(work: str, path: str, old: str, new: str, tests: Optional[list] = None) -> tuple:
    tests = tests or (VERILOG_TESTS if path == "src/rtverilog.py" else TESTS)
    """What became of one mutant: its verdict, and whether the tests failed to notice it."""
    tree = os.path.join(work, "tree")
    shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(ROOT, tree, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    target = os.path.join(tree, path)
    with open(target) as handle:
        text = handle.read()
    if old not in text:
        return f"NOT APPLIED: the text is gone: {old[:60]!r}", True
    with open(target, "w") as handle:
        handle.write(text.replace(old, new, 1))
    start = time.time()
    try:
        done = subprocess.run([sys.executable, "-W", "ignore::ResourceWarning", "-m", "unittest", "-f", *tests],
                              cwd=tree, env=dict(os.environ, TMPDIR=work, LFPL_FUZZ="30"),
                              capture_output=True, text=True, timeout=PATIENCE)
    except subprocess.TimeoutExpired:  # what removing a size limit looks like
        return f"caught: the tests were still running after {PATIENCE}s", False
    took = time.time() - start
    if done.returncode == 0:
        return f"SURVIVED ({took:.0f}s)", True
    lines = done.stderr.splitlines()
    first = next((line for line in lines if line.startswith(("FAIL:", "ERROR:"))), None)
    if first is None:  # nothing ran: this says the mutation is broken, and not that a test noticed
        return f"NOT RUN ({took:.0f}s): {lines[-1][:80] if lines else '?'}", True
    why = next((line for line in lines if "Error" in line.split(":")[0]), "?")
    return f"caught ({took:.0f}s) by {first.split('(')[0].strip()}: {why[:90]}", False


def main() -> int:
    survivors = 0
    wanted = sys.argv[1:] or list(MUTANTS)
    with tempfile.TemporaryDirectory() as work:
        for name in wanted:
            verdict, survived = run(work, *MUTANTS[name])
            survivors += survived
            print(f"{name}: {verdict}", flush=True)
    print(f"{len(wanted) - survivors} of {len(wanted)} caught")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
