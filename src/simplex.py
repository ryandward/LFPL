"""An exact linear program solver: two-phase simplex over fractions, with Bland's rule.

LFPL-O section 5.5 asks for one linear program per handler, and says the solver need not be
trusted. It is not: src/rtcheck.py derives the constraints from the code, this proposes numbers,
and rtcheck evaluates every inequality with those numbers before believing any of them. So a wrong
answer here can only cost a refusal. It still has to be right about the arithmetic, and floating
point could not be: a bound that is off by a rounding error is not a bound, so everything is an
exact Fraction.

    minimize   sum(objective[v] * v)
    subject to sum(row[v] * v) + constant >= 0   for each (row, constant)
               v >= 0

Bland's rule picks the lowest-numbered column and row whenever there is a choice, which is what
guarantees termination on a degenerate problem. It is slower than the usual rules on large
problems, and these are small: one row for each join, build and exit of a handler.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Hashable, Iterable, Mapping, Optional, TypeVar

V = TypeVar("V", bound=Hashable)
Number = int | Fraction


class UnboundedProblem(Exception):
    """The objective can fall forever, so no minimum exists."""


def minimize(objective: Mapping[V, Number],
             constraints: Iterable[tuple[Mapping[V, Number], Number]]) -> Optional[dict[V, Fraction]]:
    """The values of a minimum, for the variables anything mentions. None if nothing is feasible."""
    rows = [({v: Fraction(c) for v, c in row.items() if c}, Fraction(k)) for row, k in constraints]
    names: list[V] = []
    position: dict[V, int] = {}
    for mapping in [objective, *(row for row, _k in rows)]:
        for v in mapping:
            if v not in position:
                position[v] = len(names)
                names.append(v)
    solved = _solve([Fraction(objective.get(v, 0)) for v in names],
                    [([row.get(v, Fraction(0)) for v in names], k) for row, k in rows])
    if solved is None:
        return None
    return dict(zip(names, solved))


def _solve(cost: list[Fraction], rows: list[tuple[list[Fraction], Fraction]]) -> Optional[list[Fraction]]:
    """Every row becomes an equation with a slack, -a.x + s = k. A row with k >= 0 starts with its
    slack basic. A row with k < 0 is negated, and needs an artificial variable to start from."""
    count = len(cost)
    slacks = len(rows)
    needing = [i for i, (_a, k) in enumerate(rows) if k < 0]
    artificial_of = {row: count + slacks + n for n, row in enumerate(needing)}
    width = count + slacks + len(needing)
    tableau: list[list[Fraction]] = []
    basis: list[int] = []
    for i, (a, k) in enumerate(rows):
        line = [Fraction(0)] * (width + 1)
        if k < 0:
            for j, value in enumerate(a):
                line[j] = value
            line[count + i] = Fraction(-1)
            line[artificial_of[i]] = Fraction(1)
            line[width] = -k
            basis.append(artificial_of[i])
        else:
            for j, value in enumerate(a):
                line[j] = -value
            line[count + i] = Fraction(1)
            line[width] = k
            basis.append(count + i)
        tableau.append(line)
    if needing:
        phase_one = [Fraction(0)] * width
        for column in artificial_of.values():
            phase_one[column] = Fraction(1)
        _optimize(tableau, basis, phase_one, width, width)
        infeasibility = sum((tableau[i][width] for i in range(len(rows)) if basis[i] >= count + slacks),
                            Fraction(0))
        if infeasibility != 0:
            return None
        _drive_out_artificials(tableau, basis, count + slacks, width)
    full_cost = list(cost) + [Fraction(0)] * (width - count)
    _optimize(tableau, basis, full_cost, count + slacks, width)
    values = [Fraction(0)] * count
    for i, column in enumerate(basis):
        if column < count:
            values[column] = tableau[i][width]
    return values


def _optimize(tableau: list[list[Fraction]], basis: list[int], cost: list[Fraction], columns: int,
              width: int) -> None:
    """Pivot until no column can lower the cost. Only the first `columns` columns may enter."""
    while True:
        entering = _entering(tableau, basis, cost, columns)
        if entering is None:
            return
        leaving = None
        best = None
        for i, line in enumerate(tableau):
            if line[entering] > 0:
                ratio = line[width] / line[entering]
                if best is None or ratio < best or (ratio == best and basis[i] < basis[leaving]):
                    best, leaving = ratio, i
        if leaving is None:
            raise UnboundedProblem("the objective has no minimum")
        _pivot(tableau, basis, leaving, entering)


def _entering(tableau: list[list[Fraction]], basis: list[int], cost: list[Fraction],
              columns: int) -> Optional[int]:
    for j in range(columns):
        if j in basis:
            continue
        reduced = cost[j]
        for i, line in enumerate(tableau):
            if line[j]:
                reduced -= cost[basis[i]] * line[j]
        if reduced < 0:
            return j
    return None


def _pivot(tableau: list[list[Fraction]], basis: list[int], row: int, column: int) -> None:
    line = tableau[row]
    scale = line[column]
    tableau[row] = line = [value / scale for value in line]
    for i, other in enumerate(tableau):
        if i != row and other[column]:
            factor = other[column]
            tableau[i] = [a - factor * b for a, b in zip(other, line)]
    basis[row] = column


def _drive_out_artificials(tableau: list[list[Fraction]], basis: list[int], first_artificial: int,
                           width: int) -> None:
    """Phase one ended at zero, but an artificial can still be basic at zero. Swap it for a real
    column, or if the row has none, the row was redundant and stays at zero for good."""
    for i in range(len(tableau)):
        if basis[i] < first_artificial:
            continue
        for j in range(first_artificial):
            if tableau[i][j]:
                _pivot(tableau, basis, i, j)
                break
        else:
            basis[i] = width + i  # a redundant row: nothing can enter it, and nothing reads it
            tableau[i][width] = Fraction(0)
