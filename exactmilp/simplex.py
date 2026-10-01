"""Two phase primal simplex over exact rational arithmetic.

The solver works on a small, explicit problem description:

    minimise    c . x + c0
    subject to  rows[i] . x  (<=, >=, =)  rhs[i]
                lb[j] <= x[j] <= ub[j]       (None means infinite)

Every number is a ``fractions.Fraction`` so pivots are exact: there is no
epsilon anywhere in this file, a value is either zero or it is not.

Internally the problem is rewritten into standard form (A x = b, x >= 0,
b >= 0) by shifting, mirroring or splitting variables, turning finite upper
bounds into extra rows and adding slack, surplus and artificial columns. A
dense tableau is then driven through phase 1 (minimise the sum of
artificials) and phase 2 (minimise the real objective).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence

LE, GE, EQ = "<=", ">=", "="

OPTIMAL = "optimal"
INFEASIBLE = "infeasible"
UNBOUNDED = "unbounded"


def to_fraction(v) -> Fraction:
    """Convert ints, floats, strings and Fractions without binary noise.

    ``Fraction(0.1)`` is 3602879701896397/36028797018963968, which is
    almost never what a modeller meant, so floats go through ``repr``.
    """
    if isinstance(v, Fraction):
        return v
    if isinstance(v, float):
        return Fraction(repr(v))
    return Fraction(v)


@dataclass
class LPProblem:
    """A linear program in the solver's native, user facing form."""

    c: List[Fraction]
    rows: List[List[Fraction]] = field(default_factory=list)
    senses: List[str] = field(default_factory=list)
    rhs: List[Fraction] = field(default_factory=list)
    lb: List[Optional[Fraction]] = field(default_factory=list)
    ub: List[Optional[Fraction]] = field(default_factory=list)
    c0: Fraction = Fraction(0)

    def __post_init__(self):
        n = len(self.c)
        self.c = [to_fraction(v) for v in self.c]
        self.rows = [[to_fraction(v) for v in r] for r in self.rows]
        self.rhs = [to_fraction(v) for v in self.rhs]
        if not self.lb:
            self.lb = [Fraction(0)] * n
        if not self.ub:
            self.ub = [None] * n
        self.lb = [None if v is None else to_fraction(v) for v in self.lb]
        self.ub = [None if v is None else to_fraction(v) for v in self.ub]
        self.c0 = to_fraction(self.c0)
        if not (len(self.rows) == len(self.senses) == len(self.rhs)):
            raise ValueError("rows, senses and rhs must have equal length")
        for r in self.rows:
            if len(r) != n:
                raise ValueError("every row needs one coefficient per variable")
        for s in self.senses:
            if s not in (LE, GE, EQ):
                raise ValueError(f"unknown constraint sense {s!r}")
        if len(self.lb) != n or len(self.ub) != n:
            raise ValueError("lb and ub need one entry per variable")

    @property
    def num_vars(self) -> int:
        return len(self.c)

    def with_bounds(self, lb, ub) -> "LPProblem":
        """Cheap copy sharing rows but with new variable bounds (used by B&B)."""
        p = object.__new__(LPProblem)
        p.c, p.rows, p.senses, p.rhs, p.c0 = self.c, self.rows, self.senses, self.rhs, self.c0
        p.lb, p.ub = list(lb), list(ub)
        return p


@dataclass
class LPResult:
    status: str
    objective: Optional[Fraction] = None
    x: Optional[List[Fraction]] = None
    duals: Optional[List[Fraction]] = None
    iterations: int = 0


class _Tableau:
    """Dense simplex tableau. Row ``m`` (the last one) is the reduced cost row."""

    def __init__(self, T, basis, allowed, rule):
        self.T = T
        self.basis = basis
        self.allowed = allowed  # columns that may enter the basis
        self.rule = rule
        self.iterations = 0

    def pivot(self, r: int, col: int) -> None:
        T = self.T
        row = T[r]
        piv = row[col]
        if piv != 1:
            row = [v / piv for v in row]
            T[r] = row
        nz = [(j, v) for j, v in enumerate(row) if v]
        for i in range(len(T)):
            if i == r:
                continue
            f = T[i][col]
            if f:
                Ti = T[i]
                for j, v in nz:
                    Ti[j] -= f * v
        self.basis[r] = col
        self.iterations += 1

    def _entering(self, degenerate_streak: int) -> Optional[int]:
        d = self.T[-1]
        if self.rule == "dantzig" and degenerate_streak < 50:
            best, best_val = None, Fraction(0)
            for j in self.allowed:
                if d[j] < best_val:
                    best, best_val = j, d[j]
            return best
        # Bland: first improving column. Guaranteed not to cycle.
        for j in self.allowed:
            if d[j] < 0:
                return j
        return None

    def _leaving(self, col: int) -> Optional[int]:
        T, basis = self.T, self.basis
        best_r, best_ratio = None, None
        for i in range(len(T) - 1):
            a = T[i][col]
            if a > 0:
                ratio = T[i][-1] / a
                if (best_r is None or ratio < best_ratio
                        or (ratio == best_ratio and basis[i] < basis[best_r])):
                    best_r, best_ratio = i, ratio
        return best_r

    def run(self, max_iter: int) -> str:
        streak = 0
        while True:
            col = self._entering(streak)
            if col is None:
                return OPTIMAL
            r = self._leaving(col)
            if r is None:
                return UNBOUNDED
            streak = streak + 1 if self.T[r][-1] == 0 else 0
            self.pivot(r, col)
            if self.iterations > max_iter:
                raise RuntimeError("simplex iteration limit exceeded")


def solve_lp(problem: LPProblem, rule: str = "bland", max_iter: int = 100_000) -> LPResult:
    """Solve ``problem`` exactly. ``rule`` is ``"bland"`` or ``"dantzig"``.

    Dantzig's largest coefficient rule usually needs fewer pivots but can
    cycle on degenerate problems, so after a run of degenerate pivots it
    falls back to Bland's rule, which provably terminates.
    """
    if rule not in ("bland", "dantzig"):
        raise ValueError("rule must be 'bland' or 'dantzig'")
    p = problem
    n = p.num_vars
    zero = Fraction(0)

    for j in range(n):
        if p.lb[j] is not None and p.ub[j] is not None and p.lb[j] > p.ub[j]:
            return LPResult(INFEASIBLE)

    # 1. Map each user variable onto non negative internal columns.
    #    x = offset + sum(sign * y_col)
    maps = []
    ncols = 0
    extra_rows = []  # (col, bound) meaning y_col <= bound
    for j in range(n):
        lo, hi = p.lb[j], p.ub[j]
        if lo is not None:
            maps.append((lo, [(ncols, 1)]))
            if hi is not None:
                extra_rows.append((ncols, hi - lo))
            ncols += 1
        elif hi is not None:
            maps.append((hi, [(ncols, -1)]))
            ncols += 1
        else:
            maps.append((zero, [(ncols, 1), (ncols + 1, -1)]))
            ncols += 2

    cost = [zero] * ncols
    const = p.c0
    for j in range(n):
        off, parts = maps[j]
        const += p.c[j] * off
        for col, s in parts:
            cost[col] += s * p.c[j]

    # 2. Build internal rows with b >= 0.
    A, senses, b, flipped = [], [], [], []
    for row, sense, rhs in zip(p.rows, p.senses, p.rhs):
        a = [zero] * ncols
        r = rhs
        for j, coef in enumerate(row):
            if coef:
                off, parts = maps[j]
                r -= coef * off
                for col, s in parts:
                    a[col] += s * coef
        A.append(a), senses.append(sense), b.append(r), flipped.append(False)
    for col, bound in extra_rows:
        a = [zero] * ncols
        a[col] = Fraction(1)
        A.append(a), senses.append(LE), b.append(bound), flipped.append(False)
    m = len(A)
    for i in range(m):
        if b[i] < 0:
            A[i] = [-v for v in A[i]]
            b[i] = -b[i]
            senses[i] = {LE: GE, GE: LE, EQ: EQ}[senses[i]]
            flipped[i] = True

    # 3. Slack / surplus / artificial columns. id_col[i] is the column that
    #    starts as the identity column for row i; its final reduced cost gives
    #    the dual value of that row.
    n_slack = sum(1 for s in senses if s != EQ)
    n_art = sum(1 for s in senses if s != LE)
    width = ncols + n_slack + n_art
    T = []
    basis = [0] * m
    id_col = [0] * m
    artificial = []
    sc, ac = ncols, ncols + n_slack
    for i in range(m):
        row = A[i] + [zero] * (n_slack + n_art) + [b[i]]
        if senses[i] == LE:
            row[sc] = Fraction(1)
            basis[i] = id_col[i] = sc
            sc += 1
        else:
            if senses[i] == GE:
                row[sc] = Fraction(-1)
                sc += 1
            row[ac] = Fraction(1)
            basis[i] = id_col[i] = ac
            artificial.append(ac)
            ac += 1
        T.append(row)

    art_set = set(artificial)
    structural = [j for j in range(width) if j not in art_set]
    iterations = 0

    # 4. Phase 1.
    if artificial:
        obj = [zero] * (width + 1)
        for i in range(m):
            if basis[i] in art_set:
                for j in range(width + 1):
                    obj[j] -= T[i][j]
        for j in artificial:
            obj[j] = zero
        T.append(obj)
        tab = _Tableau(T, basis, structural, rule)
        tab.run(max_iter)
        iterations += tab.iterations
        if T[-1][-1] != 0:  # -(sum of artificials) at optimum
            return LPResult(INFEASIBLE, iterations=iterations)
        # Drive zero valued artificials out of the basis where possible. A
        # row whose structural entries are all zero is redundant: its
        # artificial stays basic at level 0 and never blocks a pivot.
        for i in range(m):
            if basis[i] in art_set:
                for j in structural:
                    if T[i][j] != 0:
                        tab.pivot(i, j)
                        break
        T.pop()

    # 5. Phase 2 reduced costs: d_j = c_j - c_B . column_j for every column.
    full_cost = cost + [zero] * (n_slack + n_art)
    obj = list(full_cost) + [zero]
    for i in range(m):
        cb = full_cost[basis[i]]
        if cb:
            Ti = T[i]
            for j in range(width + 1):
                obj[j] -= cb * Ti[j]
    T.append(obj)
    tab = _Tableau(T, basis, structural, rule)
    status = tab.run(max_iter)
    iterations += tab.iterations
    if status == UNBOUNDED:
        return LPResult(UNBOUNDED, iterations=iterations)

    y = [zero] * ncols
    for i in range(m):
        if basis[i] < ncols:
            y[basis[i]] = T[i][-1]
    x = []
    for j in range(n):
        off, parts = maps[j]
        x.append(off + sum((s * y[col] for col, s in parts), zero))
    objective = sum((p.c[j] * x[j] for j in range(n)), zero) + p.c0

    duals = []
    for i in range(len(p.rows)):
        d = -T[-1][id_col[i]]
        duals.append(-d if flipped[i] else d)
    return LPResult(OPTIMAL, objective, x, duals, iterations)
