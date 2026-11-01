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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Running code first, so that a verdict says whether a mutant is caught by what a binary does, and
# not only by the numbers in a unit test.
TESTS = ["tests.test_walks.RandomWalks", "tests.test_walks.WhatAFieldCarries", "tests.test_walks.TheEndArm",
         "tests.test_walks.OneFunctionAtSeveralAmounts", "tests.test_walks.TooLargeForAnyMachine",
         "tests.test_checksum.TheBound", "tests.test_checksum.TheChecksum",
         "tests.test_checksum.WhatTheChainTypeOpensUp", "tests.test_rtcheck", "tests.test_simplex"]

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
    "the image is sized without the reserve": ("src/runtime.py",
        "return (K_IN * n_max + self.reserve_at_most + n_max\n",
        "return (K_IN * n_max + n_max\n"),
    "a queued output is taken to be one item's size": ("src/runtime.py",
        "return self.options.n_max + self.reserve_at_most\n",
        "return self.options.n_max\n"),
    "one message gathers only one item's cells": ("src/runtime.py",
        "out_iov = min(IOV_MAX, HEADER_BYTES_MAX + max(1, image.largest_output))",
        "out_iov = HEADER_BYTES_MAX + max(1, image.options.n_max)"),
    "a message may gather past what sendmsg takes": ("src/runtime.py",
        "out_iov = min(IOV_MAX, HEADER_BYTES_MAX + max(1, image.largest_output))",
        "out_iov = HEADER_BYTES_MAX + max(1, image.largest_output)"),
    "an image of any size is built": ("src/runtime.py",
        "if image.bytes > MAX_IMAGE_BYTES:",
        "if False:"),
    "W(N_max) of any size is built": ("src/runtime.py",
        "if image.bound.c * o.n_max + image.bound.d >= WORD_LIMIT:",
        "if False:"),
    "a linear program of any size is solved": ("src/rtcheck.py",
        "if len(fn.obligations) > MAX_OBLIGATIONS:",
        "if False:"),
}


def run(work: str, path: str, old: str, new: str) -> tuple:
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
    done = subprocess.run([sys.executable, "-W", "ignore::ResourceWarning", "-m", "unittest", "-f", *TESTS],
                          cwd=tree, env=dict(os.environ, TMPDIR=work, LFPL_FUZZ="30"),
                          capture_output=True, text=True, timeout=900)
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
