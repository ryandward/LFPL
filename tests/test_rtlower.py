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
import tracemalloc
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.check import CheckStage  # noqa: E402
from src.dyck import Lexer, Parser  # noqa: E402
from src.ring import Ring  # noqa: E402
from src.rtcheck import Checker  # noqa: E402
from src.rtlower import RTLower, required_cells, rt_lines  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
