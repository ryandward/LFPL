"""Break the bound machinery one line at a time, in a copy of the tree, and see whether the tests notice.

Each mutant is something that would make a certified bound wrong, or the runtime's use of it wrong. A
mutant that survives is a claim in the README that nothing is testing, so this exits with a failure
if any does. It takes a long time, so it is not part of the test suite: run it after changing
src/rtcheck.py, src/simplex.py or the admission code in src/runtime.py.

The forwarder's mutants are caught by tests that put an XDP program on a virtual pair, so they need root,
and without it they are reported as not run, and not as caught:

    python3 tools/mutate_bounds.py [name of a mutant...]
    sudo env PATH="$PATH" PYTHONDONTWRITEBYTECODE=1 python3 tools/mutate_bounds.py --forwarder
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
FORWARDER_TESTS = ["tests.test_ipv4.TheTwoAgree", "tests.test_ipv4.TheRings", "tests.test_ipv4.OtherHandlers",
                   "tests.test_ipv4.WhatItCosts", "tests.test_ipv4.RefusalsAtStartup", "tests.test_ipv4.TheLoader",
                   "tests.test_ipv4.TheTable", "tests.test_ipv4.TheBaselinesRoutes",
                   "tests.test_ipv4.WithoutRoot", "tests.test_ipv4.Binding",
                   "tests.test_ipv4.TheProgramsBytes", "tests.test_ipv4.TheHarness"]
# Running code first, so that a verdict says whether a mutant is caught by what a binary does, and
# not only by the numbers in a unit test.
TESTS = ["tests.test_walks.RandomWalks", "tests.test_walks.TheDearestWayThrough", "tests.test_walks.TheEndArm",
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
    # --- the AF_XDP runtime
    "xdp: a frame too short to read is admitted": ("src/runtime_xdp.py",
        'add(f"    cmp ecx, {item.header_bytes}")\n    add("    jb .malformed")', 'pass', FORWARDER_TESTS),
    "xdp: a full ring is not a refusal": ("src/runtime_xdp.py",
        'add("    jae .full")', 'pass', FORWARDER_TESTS),
    "xdp: a refusal loses its frame": ("src/runtime_xdp.py",
        'add("    mov [rsi+rcx*8], rax")\n    add("    inc edx")', 'pass', FORWARDER_TESTS),
    "xdp: what was sent does not go back on fill": ("src/runtime_xdp.py",
        'add("    mov [r9+rdi*8], rsi")\n    add("    inc ecx")\n    add("    inc r10d")',
        'add("    inc ecx")', FORWARDER_TESTS),
    "xdp: the kernel is not told it has rx entries back": ("src/runtime_xdp.py",
        'add("    mov rax, [rx_cons_at]")\n    add("    mov edx, [rx_cons]")\n    add("    mov [rax], edx")',
        'pass', FORWARDER_TESTS),
    "xdp: fill is not published after a batch": ("src/runtime_xdp.py",
        'add("    mov rax, [fill_prod_at]")\n    add("    mov edx, [fill_prod]")\n    add("    mov [rax], edx")\n    add("    call flush_tx")',
        'add("    call flush_tx")', FORWARDER_TESTS),
    "xdp: tx is not published": ("src/runtime_xdp.py",
        'add("    mov rax, [tx_prod_at]")\n    add("    mov edx, [tx_prod]")\n    add("    mov [rax], edx")\n    add("    mov rax, [tx_flags_at]")',
        'add("    mov edx, [tx_prod]")\n    add("    mov rax, [tx_flags_at]")', FORWARDER_TESTS),
    "xdp: the kernel is never woken": ("src/runtime_xdp.py",
        f'add(f"    test dword [rax], {{XDP_RING_NEED_WAKEUP}}")\n    add("    jz .done")', 'add("    jmp .done")', FORWARDER_TESTS),
    "xdp: egress leaves its cells out of the pool": ("src/runtime_xdp.py",
        'add(f"    add qword [pool_count], {output.cells}")', 'pass', FORWARDER_TESTS),
    "xdp: egress loses a cell": ("src/runtime_xdp.py",
        'add("    mov [pool], rax")\n    add(f"    add qword [pool_count]',
        'add("    mov rax, [rax]")\n    add("    mov [pool], rax")\n    add(f"    add qword [pool_count]', FORWARDER_TESTS),
    "xdp: the output's fields are not written back": ("src/runtime_xdp.py",
        "_free_cells(add, ctor, True)", "_free_cells(add, ctor, False)", FORWARDER_TESTS),
    "xdp: a field goes back one byte along": ("src/runtime_xdp.py",
        "writes.append(Scalar(name, cell, off, size, source.wire))",
        "writes.append(Scalar(name, cell, off, size, source.wire + 1))", FORWARDER_TESTS),
    "xdp: a frame leaves with the wrong length": ("src/runtime_xdp.py",
        'add("    shr rbx, 32")', 'add("    shr rbx, 33")', FORWARDER_TESTS),
    "xdp: the frame's length is not given to the program": ("src/runtime_xdp.py",
        'add(f"    mov [rax+{item.length[1]}], cx")', 'pass', FORWARDER_TESTS),
    "xdp: a gap in the cell is copied across": ("src/runtime_xdp.py",
        'if runs and runs[-1][0] + runs[-1][2] == scalar.wire and runs[-1][1] + runs[-1][2] == scalar.offset:',
        'if runs and runs[-1][0] + runs[-1][2] == scalar.wire:', FORWARDER_TESTS),
    "xdp: length counts as a wire field": ("src/runtime_xdp.py",
        'length = (cell, off)\n            continue', 'length = (cell, off)', FORWARDER_TESTS),
    "xdp: P does not pause": ("src/runtime_xdp.py",
        'add("    mov qword [egress_paused], 1")', 'pass', FORWARDER_TESTS),
    "xdp: R does not resume": ("src/runtime_xdp.py",
        'add("    cmp al, 82")\n    add("    jne .answer")\n    add("    mov qword [egress_paused], 0")',
        'pass', FORWARDER_TESTS),
    "xdp: Q leaves what is queued": ("src/runtime_xdp.py",
        'add("    mov qword [egress_paused], 0")\n    add(f"    mov qword [patience], {QUIT_PATIENCE}")',
        'add("    jmp .leave")\n    add(f"    mov qword [patience], {QUIT_PATIENCE}")', FORWARDER_TESTS),
    "xdp: a bad port is sent anyway": ("src/runtime_xdp.py",
        'add(f"    jne .bad_port_{position}")', 'pass', FORWARDER_TESTS),
    "xdp: the pool is not counted when asked": ("src/runtime_xdp.py",
        'add("    mov [cnt_pool], rcx")', 'pass', FORWARDER_TESTS),
    "xdp: what the kernel dropped is not asked for": ("src/runtime_xdp.py",
        'add("    mov [cnt_kernel_dropped], rax")', 'pass', FORWARDER_TESTS),
    "xdp: the reserve is not handed to the program": ("src/runtime_xdp.py",
        'add(f"    mov {PARAMS[k + 1]}, r13")', 'add(f"    xor {PARAMS[k + 1]}, {PARAMS[k + 1]}")', FORWARDER_TESTS),
    "xdp: the image is sized without the reserve": ("src/runtime_xdp.py",
        "return self.item.cells + self.reserve\n", "return self.item.cells\n", FORWARDER_TESTS),
    "xdp: what is left of the reserve does not come back": ("src/runtime_xdp.py",
        'add("    test r13, r13")\n    add("    jz .given_back")', 'add("    jmp .given_back")', FORWARDER_TESTS),
    "xdp: the state is not handed to the program": ("src/runtime_xdp.py",
        'add(f"    mov {PARAMS[position]}, [state_{position}]")', 'pass', FORWARDER_TESTS),
    "xdp: an image of any size is built": ("src/runtime_xdp.py",
        "if image.bytes > MAX_IMAGE_BYTES:", "if False:", FORWARDER_TESTS),
    "xdp: a constructor without a frame is bound": ("src/runtime_xdp.py",
        'if frame is None:\n        raise BindingError', 'if False:\n        raise BindingError', FORWARDER_TESTS),
    "xdp: an output may write what the item lacks": ("src/runtime_xdp.py",
        'if source is None:', 'if source is None:\n            continue\n        if False:', FORWARDER_TESTS),
    "xdp: a missing map is not a refusal": ("src/runtime_xdp.py",
        'syscall(321, "js .refuse_map", str(BPF_OBJ_GET), "obj_attr", "24")\n'
        '    add("    mov [update_attr], eax")\n'
        '    syscall(321, "jnz .refuse_map", str(BPF_MAP_UPDATE_ELEM), "update_attr", "32")',
        'syscall(321, "", str(BPF_OBJ_GET), "obj_attr", "24")\n'
        '    add("    mov [update_attr], eax")\n'
        '    syscall(321, "", str(BPF_MAP_UPDATE_ELEM), "update_attr", "32")', FORWARDER_TESTS),
    "xdp: the socket is not put in the map": ("src/runtime_xdp.py",
        'syscall(321, "jnz .refuse_map", str(BPF_MAP_UPDATE_ELEM), "update_attr", "32")', 'pass', FORWARDER_TESTS),
    "xdp: the packet memory is outside what startup touches": ("src/runtime_xdp.py",
        'add(f"umem: resb {frames * CHUNK}")\n    add(f"bss_end_page: resb {PAGE}")\n    add("bss_end:")',
        'add(f"bss_end_page: resb {PAGE}")\n    add("bss_end:")\n    add(f"umem: resb {frames * CHUNK}")', FORWARDER_TESTS),
    "xdp: a frame the kernel dropped is not counted": ("src/runtime_xdp.py",
        'for number in (EAGAIN, ENOBUFS):', 'for number in (EAGAIN, ENOBUFS, 16):', FORWARDER_TESTS),
    # --- the harness: only what can be shown without a card
    "speed: a window where the forwarder heard nothing is a result": ("tools/speed.py",
        'if after["frames"] == before["frames"]:', 'if False:', FORWARDER_TESTS),
    "speed: a virtual port can give a rate": ("tools/speed.py",
        'if not args.plumbing:\n        for port in (args.generator, args.forwarder):\n            if not has_a_device(port):',
        'if False:\n        for port in (args.generator, args.forwarder):\n            if not has_a_device(port):',
        FORWARDER_TESTS),
    "speed: a port with no carrier is not waited for": ("tools/speed.py",
        'while "LOWER_UP" not in link(port, namespace)["flags"]:', 'while False:', FORWARDER_TESTS),
    "speed: a signal does not put things back": ("tools/speed.py",
        "signal.signal(number, _leave)", "pass", FORWARDER_TESTS),
    "speed: a program is left on the generator's port": ("tools/speed.py",
        "xdp_load.detach_program(args.generator, self.mode)", "pass", FORWARDER_TESTS),
    # --- the loader
    "xdp: a program already there is replaced": ("tools/xdp_load.py",
        "set_xdp(interface, program_fd, _mode_flag(mode) | XDP_FLAGS_UPDATE_IF_NOEXIST)\n    except (LoadError, OSError):",
        "set_xdp(interface, program_fd, _mode_flag(mode))\n    except (LoadError, OSError):", FORWARDER_TESTS),
    "xdp: a failed attach leaves its map for a runtime to find": ("tools/xdp_load.py",
        "if os.path.exists(path):\n            os.unlink(path)\n        raise", "raise", FORWARDER_TESTS),
    "xdp: returning frames are passed on": ("tools/xdp_load.py",
        "_instruction(0xb7, dst=0, immediate=XDP_DROP)", "_instruction(0xb7, dst=0, immediate=XDP_PASS)", FORWARDER_TESTS),
    "xdp: what no socket wants is dropped": ("tools/xdp_load.py",
        "XDP_DROP, XDP_PASS = 1, 2", "XDP_DROP, XDP_PASS = 1, 1", FORWARDER_TESTS),
    "xdp: the program reads the wrong field for the queue": ("tools/xdp_load.py",
        "RX_QUEUE_INDEX = 16", "RX_QUEUE_INDEX = 12", FORWARDER_TESTS),
    # --- the handler
    "ipv4: a TTL of 1 is forwarded": ("examples/ipv4.lfpl", "if <=(ttl, 1) then", "if <(ttl, 1) then", FORWARDER_TESTS),
    "ipv4: a TTL of 2 is refused": ("examples/ipv4.lfpl", "if <=(ttl, 1) then", "if <=(ttl, 2) then", FORWARDER_TESTS),
    "ipv4: the ethertype is not checked": ("examples/ipv4.lfpl",
        "if !=(ethertype, 8) then false            # 08 00 on the wire\n  else if", "if", FORWARDER_TESTS),
    "ipv4: the ethertype is read the other way round": ("examples/ipv4.lfpl",
        "if !=(ethertype, 8) then", "if !=(ethertype, 2048) then", FORWARDER_TESTS),
    "ipv4: only the version is checked": ("examples/ipv4.lfpl",
        "else if !=(version, 69) then false", "else if !=(&(version, 240), 64) then false", FORWARDER_TESTS),
    "ipv4: only the header length is checked": ("examples/ipv4.lfpl",
        "else if !=(version, 69) then false", "else if !=(&(version, 15), 5) then false", FORWARDER_TESTS),
    "ipv4: a total of 19 passes": ("examples/ipv4.lfpl", "else if <(total, 20) then false", "else if <(total, 19) then false", FORWARDER_TESTS),
    "ipv4: a total of 20 is refused": ("examples/ipv4.lfpl", "else if <(total, 20) then false", "else if <=(total, 20) then false", FORWARDER_TESTS),
    "ipv4: a total past the frame passes": ("examples/ipv4.lfpl",
        "else if >(+(total, 14), pack<16>(length)) then false", "else if >(+(total, 13), pack<16>(length)) then false", FORWARDER_TESTS),
    "ipv4: a total of exactly the frame is refused": ("examples/ipv4.lfpl",
        "else if >(+(total, 14), pack<16>(length)) then false", "else if >=(+(total, 14), pack<16>(length)) then false", FORWARDER_TESTS),
    "ipv4: the total is read with its bytes swapped": ("examples/ipv4.lfpl",
        "pack<8,8>(low8(total), low8(>>(total, 8)))", "pack<8,8>(low8(>>(total, 8)), low8(total))", FORWARDER_TESTS),
    "ipv4: the checksum is not checked": ("examples/ipv4.lfpl",
        "else ==(fold(sum), 65535);", "else true;", FORWARDER_TESTS),
    "ipv4: the check does not fold": ("examples/ipv4.lfpl",
        "else ==(fold(sum), 65535);", "else ==(sum, 65535);", FORWARDER_TESTS),
    "ipv4: a word is left out of the sum": ("examples/ipv4.lfpl",
        "+(pack<16>(total), +(pack<16>(id),", "+(pack<16>(total), +(0,", FORWARDER_TESTS),
    "ipv4: the TTL goes down by 2": ("examples/ipv4.lfpl", "-(ttl, 1), 2,", "-(ttl, 2), 2,", FORWARDER_TESTS),
    "ipv4: the checksum is not patched": ("examples/ipv4.lfpl",
        "Tx2@d3(fresh_sum, frame)", "Tx2@d3(sum, frame)", FORWARDER_TESTS),
    "ipv4: the update leaves out the new word": ("examples/ipv4.lfpl",
        "+(^(m, 65535), -(m, 1))", "+(^(m, 65535), m)", FORWARDER_TESTS),
    "ipv4: the update does not fold": ("examples/ipv4.lfpl",
        "low16(^(fold(+(^(pack<16>(sum), 65535), +(^(m, 65535), -(m, 1)))), 65535))",
        "low16(^(+(^(pack<16>(sum), 65535), +(^(m, 65535), -(m, 1))), 65535))", FORWARDER_TESTS),
    "ipv4: the update is RFC 1141's": ("examples/ipv4.lfpl",
        "low16(^(fold(+(^(pack<16>(sum), 65535), +(^(m, 65535), -(m, 1)))), 65535))",
        "low16(fold(+(pack<16>(sum), 1)))", FORWARDER_TESTS),
    "ipv4: the boundary at 64 is off by one": ("examples/ipv4.lfpl", "if <(first, 64) then 0", "if <=(first, 64) then 0", FORWARDER_TESTS),
    "ipv4: the boundary at 128 is off by one": ("examples/ipv4.lfpl", "if <(first, 128) then 1", "if <=(first, 128) then 1", FORWARDER_TESTS),
    "ipv4: the boundary at 192 is off by one": ("examples/ipv4.lfpl", "if <(first, 192) then 2", "if <=(first, 192) then 2", FORWARDER_TESTS),
    "ipv4: the hop comes from the wrong byte": ("examples/ipv4.lfpl",
        "next_hop(low8(dst_hi))", "next_hop(low8(>>(dst_hi, 8)))", FORWARDER_TESTS),
    "ipv4: the destination MAC is one off": ("examples/ipv4.lfpl", "low16(<<(mac_end, 8))", "low16(<<(+(mac_end, 1), 8))", FORWARDER_TESTS),
    "ipv4: the source MAC loses its 1": ("examples/ipv4.lfpl", "+(<<(mac_end, 24), 65536)", "<<(mac_end, 24)", FORWARDER_TESTS),
    "ipv4: an expired frame is called malformed": ("examples/ipv4.lfpl",
        "(Expired@d1(frame), More@d2", "(Malformed@d1(frame), More@d2", FORWARDER_TESTS),
    # --- the baseline, which is an oracle only if it is right. There is no mutant for its check that a
    # frame has 34 bytes: a shorter one fails the check of its total length anyway, whatever is in it.
    "c: a TTL of 1 is forwarded": ("baseline/ipv4_forward.c", "if (ttl <= 1)", "if (ttl < 1)", FORWARDER_TESTS),
    "c: the ethertype is not checked": ("baseline/ipv4_forward.c",
        "    if (read16(frame + 12) != ETHERTYPE_IPV4_LE)\n        return MALFORMED;\n", "", FORWARDER_TESTS),
    "c: the version byte is not checked": ("baseline/ipv4_forward.c",
        "    if (ip[0] != VERSION_AND_LENGTH)\n        return MALFORMED;\n", "", FORWARDER_TESTS),
    "c: a total past the frame passes": ("baseline/ipv4_forward.c", "total > length - 14", "total > length - 13", FORWARDER_TESTS),
    # (void) keeps each of these three compiling, because -Werror refusing it would say nothing.
    "c: the checksum is not checked": ("baseline/ipv4_forward.c",
        "    if (fold(sum) != 0xFFFF)\n        return MALFORMED;\n", "    (void)sum;\n", FORWARDER_TESTS),
    "c: the checksum is not patched": ("baseline/ipv4_forward.c",
        "    memcpy(frame + 24, &checksum, 2);\n", "    (void)checksum;\n", FORWARDER_TESTS),
    "c: the MACs are not rewritten": ("baseline/ipv4_forward.c", "    memcpy(frame, hop_macs[hop], 12);\n", "", FORWARDER_TESTS),
    "c: routes are not sorted": ("baseline/ipv4_forward.c",
        "    qsort(routes, used, sizeof routes[0], compare_length);\n", "    (void)compare_length;\n", FORWARDER_TESTS),
    "c: a new second level starts empty": ("baseline/ipv4_forward.c",
        "        for (uint32_t i = 0; i < TBL8_GROUP; i++)\n            tbl8[group * TBL8_GROUP + i] = tbl24[top];\n", "", FORWARDER_TESTS),
    "c: the second level is not looked at": ("baseline/ipv4_forward.c",
        "    if (entry & TBL8_FLAG)\n        entry = tbl8[(entry & ~TBL8_FLAG) * TBL8_GROUP + destination[3]];\n",
        "    if (entry & TBL8_FLAG)\n        entry = 0;\n", FORWARDER_TESTS),
    "c: a /24 spills over": ("baseline/ipv4_forward.c",
        "uint32_t cells = 1u << (24 - length);", "uint32_t cells = (1u << (24 - length)) + 1;", FORWARDER_TESTS),
    "c: no route is not refused": ("baseline/ipv4_forward.c",
        "    if (hop == NO_ROUTE)\n        return MALFORMED;\n", "    if (hop == NO_ROUTE)\n        hop = 0;\n", FORWARDER_TESTS),
    "c: match is off at 128": ("baseline/ipv4_forward.c", "first < 128 ? 1", "first <= 128 ? 1", FORWARDER_TESTS),
    "c: a line of the routes that does not parse is skipped": ("baseline/ipv4_forward.c",
        'fail("a line of the routes file does not parse");', 'continue;', FORWARDER_TESTS),
    "c: octets are not checked": ("baseline/ipv4_forward.c",
        "a > 255 || b > 255 || c > 255 || d > 255 || ", "", FORWARDER_TESTS),
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


def needs_root(name: str) -> bool:
    return len(MUTANTS[name]) > 3 and MUTANTS[name][3] is FORWARDER_TESTS


def main() -> int:
    survivors = 0
    arguments = sys.argv[1:]
    if arguments == ["--forwarder"]:
        wanted = [name for name in MUTANTS if needs_root(name)]
    else:
        wanted = arguments or list(MUTANTS)
    not_run = []
    if os.geteuid() != 0:
        not_run = [name for name in wanted if needs_root(name)]
        wanted = [name for name in wanted if name not in not_run]
    with tempfile.TemporaryDirectory() as work:
        for name in wanted:
            verdict, survived = run(work, *MUTANTS[name])
            survivors += survived
            print(f"{name}: {verdict}", flush=True)
    print(f"{len(wanted) - survivors} of {len(wanted)} caught")
    if not_run:
        print(f"{len(not_run)} not run: they need root")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
