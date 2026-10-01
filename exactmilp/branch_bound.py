"""LP based branch and bound for mixed integer programs.

Each node is the root LP with tightened variable bounds. Because the LP
solver is exact, "is this value an integer" is ``x.denominator == 1`` and
pruning compares Fractions directly, so the search never accepts a
solution that is only integral up to a tolerance.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence

from .simplex import INFEASIBLE, OPTIMAL, UNBOUNDED, LPProblem, solve_lp

NODE_LIMIT = "node_limit"


@dataclass
class MILPResult:
    status: str
    objective: Optional[Fraction] = None
    x: Optional[List[Fraction]] = None
    best_bound: Optional[Fraction] = None
    nodes: int = 0
    lp_iterations: int = 0
    log: List[str] = field(default_factory=list)

    @property
    def gap(self) -> Optional[Fraction]:
        if self.objective is None or self.best_bound is None:
            return None
        return abs(self.objective - self.best_bound)


def _objective_is_integral(problem: LPProblem, integer: Sequence[int]) -> bool:
    """True when every feasible integer solution has an integer objective.

    Then a node whose LP bound is 7.2 cannot hold anything better than 8
    (when minimising), which lets us prune far more aggressively.
    """
    ints = set(integer)
    if problem.c0.denominator != 1:
        return False
    for j, cj in enumerate(problem.c):
        if cj and (j not in ints or cj.denominator != 1):
            return False
    return True


def _ceil(q: Fraction) -> Fraction:
    return Fraction(math.ceil(q))


def solve_milp(problem: LPProblem, integer: Sequence[int], *, strategy: str = "best",
               branching: str = "most_fractional", rule: str = "bland",
               node_limit: int = 100_000, verbose: bool = False,
               report_sign: int = 1) -> MILPResult:
    """Minimise ``problem`` with variables in ``integer`` restricted to Z.

    strategy:  "best"  explores the open node with the smallest bound first
               (fewest nodes to prove optimality), "depth" dives depth first
               (finds incumbents fast, small memory).
    branching: "most_fractional" or "first" fractional variable.
    report_sign: multiplier for objective values in log lines (-1 when the
               caller turned a maximisation into this minimisation).
    """
    if strategy not in ("best", "depth"):
        raise ValueError("strategy must be 'best' or 'depth'")
    if branching not in ("most_fractional", "first"):
        raise ValueError("branching must be 'most_fractional' or 'first'")

    integer = sorted(set(integer))
    integral_obj = _objective_is_integral(problem, integer)
    # Integer variables can have their bounds rounded inward up front.
    lb0, ub0 = list(problem.lb), list(problem.ub)
    for j in integer:
        if lb0[j] is not None:
            lb0[j] = _ceil(lb0[j])
        if ub0[j] is not None:
            ub0[j] = Fraction(math.floor(ub0[j]))

    res = MILPResult(INFEASIBLE)
    incumbent: Optional[Fraction] = None
    best_x = None
    seq = 0

    def node_bound(obj: Fraction) -> Fraction:
        return _ceil(obj) if integral_obj else obj

    def solve_node(lb, ub):
        lp = solve_lp(problem.with_bounds(lb, ub), rule=rule)
        res.lp_iterations += lp.iterations
        res.nodes += 1
        return lp

    root = solve_node(lb0, ub0)
    if root.status == INFEASIBLE:
        return res
    if root.status == UNBOUNDED:
        # With rational data an unbounded relaxation of a feasible MILP means
        # the MILP is unbounded too (Meyer, 1974); we report it as such.
        res.status = UNBOUNDED
        return res

    open_nodes = []  # (bound, -depth, seq, lb, ub, lp)

    def push(bound, depth, lb, ub, lp):
        nonlocal seq
        seq += 1
        node = (bound, -depth, seq, lb, ub, lp)
        if strategy == "best":
            heapq.heappush(open_nodes, node)
        else:
            open_nodes.append(node)

    push(node_bound(root.objective), 0, lb0, ub0, root)

    while open_nodes:
        if res.nodes >= node_limit:
            res.status = NODE_LIMIT
            break
        item = heapq.heappop(open_nodes) if strategy == "best" else open_nodes.pop()
        bound, negdepth, _, lb, ub, lp = item
        if incumbent is not None and bound >= incumbent:
            continue  # pruned by bound

        frac = [j for j in integer if lp.x[j].denominator != 1]
        if not frac:
            if incumbent is None or lp.objective < incumbent:
                incumbent, best_x = lp.objective, lp.x
                msg = f"node {res.nodes}: new incumbent {report_sign * incumbent}"
                res.log.append(msg)
                if verbose:
                    print(msg)
            continue

        if branching == "first":
            j = frac[0]
        else:  # distance from nearest integer, largest first
            j = max(frac, key=lambda k: min(lp.x[k] - math.floor(lp.x[k]),
                                            math.ceil(lp.x[k]) - lp.x[k]))
        v = lp.x[j]
        down_ub = list(ub); down_ub[j] = Fraction(math.floor(v))
        up_lb = list(lb); up_lb[j] = Fraction(math.ceil(v))
        children = [(lb, down_ub), (up_lb, ub)]
        if strategy == "depth":
            children.reverse()  # explore the "down" branch first after pop
        for clb, cub in children:
            child = solve_node(clb, cub)
            if child.status != OPTIMAL:
                continue  # infeasible child; unbounded cannot appear below a bounded parent
            cb = node_bound(child.objective)
            if incumbent is None or cb < incumbent:
                push(cb, -negdepth + 1, clb, cub, child)

    if incumbent is not None:
        res.objective, res.x = incumbent, best_x
        if res.status != NODE_LIMIT:
            res.status = OPTIMAL
            res.best_bound = incumbent
        else:
            res.best_bound = min([n[0] for n in open_nodes] + [incumbent])
    elif res.status != NODE_LIMIT:
        res.status = INFEASIBLE
    return res
