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
problems, and these are small: one row for each join, build and exit of a handler and of what it
calls. Those rows name two or three unknowns each, so the tableau is kept sparse: a dense one
spends nearly all of its time subtracting zeros.

Bland's rule terminates, but its worst case is exponential, and a compiler that may not finish has
no bound of its own. So a solve has a budget of pivots, declared in LFPL-O section 8.2, and spending
it is a refusal and not a long wait. The linear programs handlers really make take about one pivot
per row, so at the largest size section 8.2 allows, this is about ten times what one has needed.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Hashable, Iterable, Mapping, Optional, TypeVar

V = TypeVar("V", bound=Hashable)
Number = int | Fraction
Line = dict[int, Fraction]  # column -> coefficient, with no zeros, and the right side at RHS
RHS = -1
MAX_PIVOTS = 10240  # for one solve: LFPL-O section 8.2


class UnboundedProblem(Exception):
    """The objective can fall forever, so no minimum exists."""


class PivotBudgetSpent(Exception):
    """The solve used all the pivots it may. Nothing is known about the problem either way."""

    def __init__(self, budget: int) -> None:
        super().__init__(f"more than {budget} pivots")
        self.budget = budget


class _Pivots:
    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.left = budget

    def spend(self) -> None:
        if not self.left:
            raise PivotBudgetSpent(self.budget)
        self.left -= 1


def minimize(objective: Mapping[V, Number], constraints: Iterable[tuple[Mapping[V, Number], Number]],
             *, pivots: Optional[int] = None) -> Optional[dict[V, Fraction]]:
    """The values of a minimum, for the variables anything mentions. None if nothing is feasible.
    PivotBudgetSpent if finding out would take more than `pivots`, by default MAX_PIVOTS."""
    rows = [({v: Fraction(c) for v, c in row.items() if c}, Fraction(k)) for row, k in constraints]
    names: list[V] = []
    position: dict[V, int] = {}
    for mapping in [objective, *(row for row, _k in rows)]:
        for v in mapping:
            if v not in position:
                position[v] = len(names)
                names.append(v)
    solved = _solve([Fraction(objective.get(v, 0)) for v in names],
                    [([row.get(v, Fraction(0)) for v in names], k) for row, k in rows],
                    _Pivots(MAX_PIVOTS if pivots is None else pivots))
    if solved is None:
        return None
    return dict(zip(names, solved))


def _solve(cost: list[Fraction], rows: list[tuple[list[Fraction], Fraction]],
           budget: _Pivots) -> Optional[list[Fraction]]:
    """Every row becomes an equation with a slack, -a.x + s = k. A row with k >= 0 starts with its
    slack basic. A row with k < 0 is negated, and needs an artificial variable to start from."""
    count = len(cost)
    slacks = len(rows)
    first_artificial = next_artificial = count + slacks
    tableau: list[Line] = []
    basis: list[int] = []
    for i, (a, k) in enumerate(rows):
        sign = Fraction(1) if k < 0 else Fraction(-1)
        line: Line = {j: sign * value for j, value in enumerate(a) if value}
        line[count + i] = -sign
        if k:
            line[RHS] = abs(k)
        if k < 0:
            line[next_artificial] = Fraction(1)
            basis.append(next_artificial)
            next_artificial += 1
        else:
            basis.append(count + i)
        tableau.append(line)
    if next_artificial > first_artificial:
        artificial = {b: Fraction(1) for b in basis if b >= first_artificial}
        _optimize(tableau, basis, _priced(artificial, tableau, basis), None, budget)
        if any(tableau[i].get(RHS) for i, b in enumerate(basis) if b >= first_artificial):
            return None
        _drive_out_artificials(tableau, basis, first_artificial, budget)
    wanted = {j: value for j, value in enumerate(cost) if value}
    _optimize(tableau, basis, _priced(wanted, tableau, basis), first_artificial, budget)
    values = [Fraction(0)] * count
    for i, column in enumerate(basis):
        if column < count:
            values[column] = tableau[i].get(RHS, Fraction(0))
    return values


def _priced(cost: Line, tableau: list[Line], basis: list[int]) -> Line:
    """The reduced costs of the basis as it stands: the cost row, with every basic column cancelled."""
    reduced = dict(cost)
    for i, column in enumerate(basis):
        if column in cost:
            _subtract(reduced, tableau[i], cost[column])
    return reduced


def _optimize(tableau: list[Line], basis: list[int], reduced: Line, limit: Optional[int],
              budget: _Pivots) -> None:
    """Pivot until no column can lower the cost. Columns at `limit` and past it may not enter.
    Bland's rule: the lowest column that helps, and of the tightest rows, the lowest basic column."""
    while True:
        entering = min((j for j, value in reduced.items()
                        if value < 0 and j != RHS and (limit is None or j < limit)), default=None)
        if entering is None:
            return
        leaving = None
        best = None
        for i, line in enumerate(tableau):
            coefficient = line.get(entering)
            if coefficient is not None and coefficient > 0:
                ratio = line.get(RHS, Fraction(0)) / coefficient
                if best is None or ratio < best or (ratio == best and basis[i] < basis[leaving]):
                    best, leaving = ratio, i
        if leaving is None:
            raise UnboundedProblem("the objective has no minimum")
        budget.spend()
        _pivot(tableau, basis, leaving, entering)
        _subtract(reduced, tableau[leaving], reduced[entering])


def _subtract(line: Line, other: Line, factor: Fraction) -> None:
    """line -= factor * other, keeping no entry that came to exactly zero."""
    for j, value in other.items():
        result = line.get(j, Fraction(0)) - factor * value
        if result:
            line[j] = result
        else:
            line.pop(j, None)


def _pivot(tableau: list[Line], basis: list[int], row: int, column: int) -> None:
    scale = tableau[row][column]
    line = tableau[row] = {j: value / scale for j, value in tableau[row].items()}
    for i, other in enumerate(tableau):
        if i != row and column in other:
            _subtract(other, line, other[column])
    basis[row] = column


def _drive_out_artificials(tableau: list[Line], basis: list[int], first_artificial: int,
                           budget: _Pivots) -> None:
    """Phase one ended at zero, but an artificial can still be basic at zero. Swap it for a real
    column. A row with no real column was redundant: nothing can enter it, and it stays at zero."""
    for i in range(len(tableau)):
        if basis[i] >= first_artificial:
            column = min((j for j in tableau[i] if 0 <= j < first_artificial), default=None)
            if column is not None:
                budget.spend()
                _pivot(tableau, basis, i, column)
