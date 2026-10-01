"""A small algebraic modelling layer on top of the solvers.

    m = Model()
    x = m.var("x", ub=4)
    y = m.int_var("y")
    m.add(x + 2 * y <= 14, name="capacity")
    m.maximize(3 * x + 4 * y)
    sol = m.solve()
    sol["x"], sol.objective, sol.duals
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional

from .branch_bound import solve_milp
from .simplex import EQ, GE, LE, OPTIMAL, LPProblem, solve_lp, to_fraction

_NUMBER = (int, float, Fraction)


class LinExpr:
    """sum(coef * var) + constant, with var indices as keys."""

    __slots__ = ("model", "terms", "const")

    def __init__(self, model=None, terms=None, const=Fraction(0)):
        self.model = model
        self.terms: Dict[int, Fraction] = dict(terms or {})
        self.const = to_fraction(const)

    @staticmethod
    def of(value) -> "LinExpr":
        if isinstance(value, LinExpr):
            return value
        if isinstance(value, Var):
            return LinExpr(value.model, {value.index: Fraction(1)})
        if isinstance(value, _NUMBER):
            return LinExpr(None, {}, value)
        raise TypeError(f"cannot use {type(value).__name__} in a linear expression")

    def _combine(self, other, sign):
        o = LinExpr.of(other)
        if self.model and o.model and self.model is not o.model:
            raise ValueError("cannot mix variables from different models")
        terms = dict(self.terms)
        for k, v in o.terms.items():
            terms[k] = terms.get(k, Fraction(0)) + sign * v
            if terms[k] == 0:
                del terms[k]
        return LinExpr(self.model or o.model, terms, self.const + sign * o.const)

    def __add__(self, other):
        return self._combine(other, 1)

    __radd__ = __add__

    def __sub__(self, other):
        return self._combine(other, -1)

    def __rsub__(self, other):
        return LinExpr.of(other)._combine(self, -1)

    def __neg__(self):
        return self * -1

    def __pos__(self):
        return self

    def __mul__(self, k):
        if not isinstance(k, _NUMBER):
            raise TypeError("products of variables are not linear")
        k = to_fraction(k)
        if k == 0:
            return LinExpr(self.model)
        return LinExpr(self.model, {i: v * k for i, v in self.terms.items()}, self.const * k)

    __rmul__ = __mul__

    def __truediv__(self, k):
        if not isinstance(k, _NUMBER):
            raise TypeError("can only divide by a number")
        return self * (1 / to_fraction(k))

    def _rel(self, other, sense):
        diff = self - other
        return Constraint(diff.model, diff.terms, sense, -diff.const)

    def __le__(self, other):
        return self._rel(other, LE)

    def __ge__(self, other):
        return self._rel(other, GE)

    def __eq__(self, other):  # type: ignore[override]
        return self._rel(other, EQ)

    __hash__ = None  # expressions are not hashable values

    def value(self, sol: "Solution") -> Fraction:
        return self.const + sum((c * sol.x[i] for i, c in self.terms.items()), Fraction(0))

    def __repr__(self):
        names = self.model.names if self.model else {}
        parts = [f"{v} {names[i] if names else 'x' + str(i)}" for i, v in sorted(self.terms.items())]
        if self.const or not parts:
            parts.append(str(self.const))
        return " + ".join(parts)


class Var:
    __slots__ = ("model", "index", "name")

    def __init__(self, model, index, name):
        self.model, self.index, self.name = model, index, name

    def _e(self):
        return LinExpr.of(self)

    def __add__(self, o): return self._e() + o
    def __radd__(self, o): return self._e() + o
    def __sub__(self, o): return self._e() - o
    def __rsub__(self, o): return LinExpr.of(o) - self._e()
    def __mul__(self, k): return self._e() * k
    def __rmul__(self, k): return self._e() * k
    def __truediv__(self, k): return self._e() / k
    def __neg__(self): return self._e() * -1
    def __le__(self, o): return self._e() <= o
    def __ge__(self, o): return self._e() >= o
    def __eq__(self, o): return self._e() == o  # type: ignore[override]

    def __hash__(self):
        return hash((id(self.model), self.index))

    def __repr__(self):
        return self.name


@dataclass
class Constraint:
    model: object
    terms: Dict[int, Fraction]
    sense: str
    rhs: Fraction
    name: Optional[str] = None

    def __bool__(self):
        raise TypeError("a Constraint has no truth value; pass it to Model.add()")


@dataclass
class Solution:
    status: str
    objective: Optional[Fraction]
    x: Optional[List[Fraction]]
    names: Dict[int, str]
    duals: Optional[Dict[str, Fraction]] = None
    nodes: int = 0
    iterations: int = 0
    log: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        if isinstance(key, Var):
            key = key.index
        elif isinstance(key, str):
            key = next(i for i, n in self.names.items() if n == key)
        return self.x[key]

    @property
    def values(self) -> Dict[str, Fraction]:
        return {self.names[i]: v for i, v in enumerate(self.x or [])}


class Model:
    def __init__(self, name: str = "model"):
        self.name = name
        self.names: Dict[int, str] = {}
        self.lb: List[Optional[Fraction]] = []
        self.ub: List[Optional[Fraction]] = []
        self.integer: List[int] = []
        self.constraints: List[Constraint] = []
        self.objective = LinExpr(self)
        self.sense = "min"
        self._by_name: Dict[str, Var] = {}

    # variables -----------------------------------------------------------
    def var(self, name=None, lb=0, ub=None, integer=False) -> Var:
        idx = len(self.lb)
        name = name or f"x{idx}"
        if name in self._by_name:
            raise ValueError(f"duplicate variable name {name!r}")
        v = Var(self, idx, name)
        self.names[idx] = name
        self.lb.append(None if lb is None else to_fraction(lb))
        self.ub.append(None if ub is None else to_fraction(ub))
        if integer:
            self.integer.append(idx)
        self._by_name[name] = v
        return v

    def int_var(self, name=None, lb=0, ub=None) -> Var:
        return self.var(name, lb, ub, integer=True)

    def bin_var(self, name=None) -> Var:
        return self.var(name, 0, 1, integer=True)

    def get_var(self, name) -> Var:
        return self._by_name[name]

    @property
    def variables(self) -> List[Var]:
        return list(self._by_name.values())

    # constraints / objective --------------------------------------------
    def add(self, constraint: Constraint, name: Optional[str] = None) -> Constraint:
        if not isinstance(constraint, Constraint):
            raise TypeError("Model.add expects a constraint such as x + y <= 3")
        if constraint.model not in (None, self):
            raise ValueError("constraint belongs to another model")
        constraint.name = name or constraint.name or f"c{len(self.constraints)}"
        if any(c.name == constraint.name for c in self.constraints):
            raise ValueError(f"duplicate constraint name {constraint.name!r}")
        self.constraints.append(constraint)
        return constraint

    def minimize(self, expr) -> None:
        self.objective, self.sense = LinExpr.of(expr), "min"

    def maximize(self, expr) -> None:
        self.objective, self.sense = LinExpr.of(expr), "max"

    # solving -------------------------------------------------------------
    def to_problem(self) -> LPProblem:
        n = len(self.lb)
        sign = 1 if self.sense == "min" else -1
        c = [Fraction(0)] * n
        for i, v in self.objective.terms.items():
            c[i] = sign * v
        rows, senses, rhs = [], [], []
        for con in self.constraints:
            row = [Fraction(0)] * n
            for i, v in con.terms.items():
                row[i] = v
            rows.append(row), senses.append(con.sense), rhs.append(con.rhs)
        return LPProblem(c, rows, senses, rhs, list(self.lb), list(self.ub),
                         sign * self.objective.const)

    def solve(self, *, rule="bland", strategy="best", branching="most_fractional",
              node_limit=100_000, verbose=False) -> Solution:
        prob = self.to_problem()
        sign = 1 if self.sense == "min" else -1
        if not self.integer:
            r = solve_lp(prob, rule=rule)
            duals = None
            if r.status == OPTIMAL:
                duals = {c.name: sign * d for c, d in zip(self.constraints, r.duals)}
            obj = None if r.objective is None else sign * r.objective
            return Solution(r.status, obj, r.x, dict(self.names), duals,
                            nodes=1, iterations=r.iterations)
        r = solve_milp(prob, self.integer, strategy=strategy, branching=branching,
                       rule=rule, node_limit=node_limit, verbose=verbose,
                       report_sign=sign)
        obj = None if r.objective is None else sign * r.objective
        return Solution(r.status, obj, r.x, dict(self.names), None,
                        nodes=r.nodes, iterations=r.lp_iterations, log=r.log)
