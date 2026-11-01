"""The exact solver, against an oracle that shares no method with it: enumerating every vertex.

A linear program's minimum, when one exists, is at a vertex of the feasible region, and a vertex is
where enough constraints hold with equality. For a small problem that can simply be listed, so
random problems can be solved both ways and compared exactly.
"""
import itertools
import os
import random
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simplex import UnboundedProblem, minimize  # noqa: E402


def _solve_square(matrix: list, target: list):
    """Gaussian elimination, or None if the system has no single solution."""
    n = len(matrix)
    rows = [list(row) + [value] for row, value in zip(matrix, target)]
    for column in range(n):
        pivot = next((r for r in range(column, n) if rows[r][column]), None)
        if pivot is None:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [v / scale for v in rows[column]]
        for r in range(n):
            if r != column and rows[r][column]:
                factor = rows[r][column]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[column])]
    return [row[n] for row in rows]


def vertex_minimum(cost: list, rows: list):
    """The best vertex, by listing them all. None if there is no feasible point at a vertex."""
    n = len(cost)
    boundaries = [(row, k) for row, k in rows]
    boundaries += [([Fraction(i == j) for j in range(n)], Fraction(0)) for i in range(n)]
    best = None
    for chosen in itertools.combinations(boundaries, n):
        point = _solve_square([list(row) for row, _k in chosen], [-k for _row, k in chosen])
        if point is None or any(v < 0 for v in point):
            continue
        if any(sum(a * v for a, v in zip(row, point)) + k < 0 for row, k in rows):
            continue
        value = sum(c * v for c, v in zip(cost, point))
        if best is None or value < best:
            best = value
    return best


class Solver(unittest.TestCase):
    def test_a_bound_from_an_iteration_cost(self):
        """c >= 5 + r, r >= 3, minimizing 100 c + d, with d >= 2: the shape a walk produces."""
        solved = minimize({"c": 100, "d": 1},
                          [({"c": 1, "r": -1}, -5), ({"r": 1}, -3), ({"d": 1}, -2)])
        self.assertEqual((solved["c"], solved["d"], solved["r"]), (8, 2, 3))

    def test_an_exact_fraction_comes_back_exact(self):
        solved = minimize({"x": 1}, [({"x": 3}, -1)])
        self.assertEqual(solved["x"], Fraction(1, 3))

    def test_no_constraints_leave_everything_at_zero(self):
        self.assertEqual(minimize({"x": 1, "y": 2}, []), {"x": 0, "y": 0})

    def test_a_cost_that_cannot_be_paid_is_infeasible(self):
        """What recursion with nothing shrinking gives: q >= q + 1."""
        self.assertIsNone(minimize({"q": 1}, [({}, -1)]))
        self.assertIsNone(minimize({"x": 1}, [({"x": 1}, -1), ({"x": -1}, 0)]))

    def test_the_cycle_of_a_superlinear_walk_is_infeasible(self):
        """a >= w, w >= b, b >= 3 + r, r >= a: each step needs more than the last left."""
        self.assertIsNone(minimize({"a": 1}, [({"a": 1, "w": -1}, 0), ({"w": 1, "b": -1}, 0),
                                              ({"b": 1, "r": -1}, -3), ({"r": 1, "a": -1}, 0)]))

    def test_an_objective_with_no_minimum_says_so(self):
        with self.assertRaises(UnboundedProblem):
            minimize({"x": -1}, [])

    def test_beales_problem_cycles_without_blands_rule(self):
        """The classic degenerate example, on which the usual pivoting rule never terminates."""
        solved = minimize(
            {"a": Fraction(-3, 4), "b": 150, "c": Fraction(-1, 50), "d": 6},
            [({"a": Fraction(-1, 4), "b": 60, "c": Fraction(1, 25), "d": -9}, 0),
             ({"a": Fraction(-1, 2), "b": 90, "c": Fraction(1, 50), "d": -3}, 0),
             ({"c": -1}, 1)])
        value = (Fraction(-3, 4) * solved["a"] + 150 * solved["b"]
                 + Fraction(-1, 50) * solved["c"] + 6 * solved["d"])
        self.assertEqual(value, Fraction(-1, 20))

    def test_a_redundant_row_does_not_disturb_the_answer(self):
        solved = minimize({"x": 1}, [({"x": 1}, -2), ({"x": 1}, -2), ({"x": 2}, -4)])
        self.assertEqual(solved["x"], 2)

    def test_random_problems_agree_with_the_list_of_vertices(self):
        rng = random.Random(7)
        names = ["a", "b", "c", "d"]
        feasible = infeasible = 0
        for _ in range(400):
            n = rng.randrange(1, 5)
            used = names[:n]
            cost = [Fraction(rng.randrange(0, 7)) for _ in used]
            rows = []
            for _ in range(rng.randrange(0, 6)):
                row = [Fraction(rng.randrange(-4, 5), rng.randrange(1, 4)) for _ in used]
                rows.append((row, Fraction(rng.randrange(-6, 7))))
            solved = minimize(dict(zip(used, cost)),
                              [(dict(zip(used, row)), k) for row, k in rows])
            expected = vertex_minimum(cost, rows)
            if solved is None:
                self.assertIsNone(expected, (cost, rows))
                infeasible += 1
                continue
            feasible += 1
            values = [solved.get(v, Fraction(0)) for v in used]
            self.assertTrue(all(v >= 0 for v in values))
            for row, k in rows:
                self.assertGreaterEqual(sum(a * v for a, v in zip(row, values)) + k, 0)
            self.assertEqual(sum(c * v for c, v in zip(cost, values)), expected, (cost, rows))
        self.assertGreater(feasible, 100)
        self.assertGreater(infeasible, 20)


if __name__ == "__main__":
    unittest.main()
