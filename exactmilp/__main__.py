"""Command line entry point: python -m exactmilp model.lp"""

from __future__ import annotations

import argparse
import sys
import time

from .lp_format import LPFormatError, read_lp


def fmt(q, as_float):
    if as_float:
        return f"{float(q):.6g}"
    return str(q)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="exactmilp", description="Solve an LP/MILP in CPLEX LP format exactly.")
    ap.add_argument("file", help="path to a .lp file")
    ap.add_argument("--rule", choices=["bland", "dantzig"], default="bland", help="simplex pivoting rule")
    ap.add_argument("--strategy", choices=["best", "depth"], default="best", help="branch and bound node selection")
    ap.add_argument("--branching", choices=["most_fractional", "first"], default="most_fractional")
    ap.add_argument("--node-limit", type=int, default=100_000)
    ap.add_argument("--float", action="store_true", help="print decimals instead of exact fractions")
    ap.add_argument("-v", "--verbose", action="store_true", help="log new incumbents")
    args = ap.parse_args(argv)

    try:
        model = read_lp(args.file)
    except (OSError, LPFormatError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    sol = model.solve(rule=args.rule, strategy=args.strategy, branching=args.branching,
                      node_limit=args.node_limit, verbose=args.verbose)
    dt = time.perf_counter() - t0

    kind = "MILP" if model.integer else "LP"
    print(f"{kind} with {len(model.lb)} variables, {len(model.constraints)} constraints")
    print(f"status:     {sol.status}")
    if sol.objective is not None:
        print(f"objective:  {fmt(sol.objective, args.float)}")
        width = max(len(n) for n in sol.values)
        print("values:")
        for name, v in sol.values.items():
            print(f"  {name:<{width}} = {fmt(v, args.float)}")
    if sol.duals:
        width = max(len(n) for n in sol.duals)
        print("duals (shadow prices):")
        for name, v in sol.duals.items():
            print(f"  {name:<{width}} = {fmt(v, args.float)}")
    print(f"nodes: {sol.nodes}  simplex pivots: {sol.iterations}  time: {dt * 1000:.1f} ms")
    return 0 if sol.status == "optimal" else 1


if __name__ == "__main__":
    sys.exit(main())
