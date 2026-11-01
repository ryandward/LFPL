"""A handler's bound, for any backend: LFPL-O section 5.5.

src/rtcheck.py writes the inequalities as it types the code, src/simplex.py proposes numbers, and the
checker evaluates its own inequalities with them, so the solver is not trusted. A wrong answer from it
can only cost a refusal, and so can its running out of pivots.
"""
from __future__ import annotations

from .rtcheck import Bound, Checker, Unbounded
from .simplex import PivotBudgetSpent, minimize


def certify(checker: Checker, name: str, n_max: int) -> Bound:
    program = checker.linear_program(name, n_max)
    proposed = {}
    for resource in ("cells", "steps"):
        try:
            solved = minimize(program.objective(), program.constraints(resource))
        except PivotBudgetSpent as spent:
            raise Unbounded(f"section 8.2: the linear program for {resource} of handler {name} took "
                            f"more than {spent.budget} pivots") from spent
        if solved is None:
            raise Unbounded(f"section 8.2: handler {name} has no linear bound on {resource}: no "
                            f"amount per cell of its item pays for what it does")
        proposed[resource] = solved
    return checker.verify(name, proposed["cells"], proposed["steps"])
