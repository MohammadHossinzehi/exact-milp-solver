# exact-milp-solver

A linear and mixed integer programming solver in pure Python where every number is an exact fraction. Two phase simplex, LP based branch and bound, a small algebraic modelling API, and a reader for the CPLEX LP file format. No NumPy, no dependencies, no epsilons.

```
$ python -m exactmilp examples/diet.lp
LP with 6 variables, 4 constraints
status:     optimal
objective:  35171/4275
values:
  oats    = 4
  milk    = 188/171
  beans   = 556/57
  ...
duals (shadow prices):
  calories = 73/17100
  calcium  = 77/68400
  ...
```

## Why exact arithmetic?

Industrial solvers work in floating point and lean on a forest of tolerances: a pivot below 1e-9 is "zero", a variable within 1e-6 of an integer is "integral", a constraint violated by 1e-7 is "satisfied". That is the right trade for speed, but it means the answer you get is a numerical approximation of the optimum and occasionally just wrong (accepting a 0.9999999 as 1, or declaring a feasible problem infeasible).

This project takes the opposite trade. Every coefficient is a `fractions.Fraction`, so:

* a pivot element is either zero or it is not, so there is no tolerance in the ratio test;
* integrality in branch and bound is `x.denominator == 1`;
* the optimal objective and the dual values come out as exact rationals you can check by hand;
* the tests can assert *equality* against brute force enumeration rather than "close enough".

That makes it useful as a teaching tool, as a reference oracle when testing a faster solver, and for small problems where a certified answer matters more than milliseconds (scheduling a handful of shifts, verifying a textbook exercise, checking a paper's counterexample).

## What is inside

| Module | What it does |
| --- | --- |
| `exactmilp/simplex.py` | Two phase dense tableau simplex. Converts any bounds (free, shifted, upper only, boxed) and any row sense into standard form, runs phase 1 on artificials, drives zero level artificials out of the basis, then phase 2. Returns primal values, dual values (shadow prices) and the pivot count. |
| `exactmilp/branch_bound.py` | Branch and bound over the exact LP. Best bound or depth first node selection, most fractional or first fractional branching, bound rounding when the objective is provably integral, node limits. |
| `exactmilp/model.py` | `Model`, `Var`, `LinExpr`: write `m.add(3*x + 2*y <= 12)` instead of building matrices. Maximisation, named constraints, named duals. |
| `exactmilp/lp_format.py` | Reader for a practical subset of CPLEX LP: objective, constraints, ranged rows (`1 <= x + y <= 3`), bounds (`free`, `-inf`, two sided), `General` and `Binary` sections, decimals and `3/4` style fractions, line numbered errors. |
| `exactmilp/__main__.py` | Command line front end. |

## Running it

Python 3.8 or newer, nothing to install:

```bash
git clone https://github.com/MohammadHossinzehi/exact-milp-solver
cd exact-milp-solver

python -m exactmilp examples/knapsack.lp -v       # 0/1 knapsack, logs each new incumbent
python -m exactmilp examples/facility.lp          # capacitated facility location (mixed binary)
python -m exactmilp examples/diet.lp --float      # decimals instead of fractions
python -m exactmilp examples/diet.lp --rule dantzig
python demo.py                                    # Python API: production plan + 6 queens

python -m unittest discover -s tests -v           # full test suite (about 2 seconds)
```

CLI flags: `--rule {bland,dantzig}`, `--strategy {best,depth}`, `--branching {most_fractional,first}`, `--node-limit N`, `--float`, `-v`.

### Python API

```python
from exactmilp import Model

m = Model()
x = m.int_var("x", ub=10)
y = m.var("y")                      # continuous, y >= 0
m.add(2*x + 2*y <= 7, name="budget")
m.add(x <= 5/2)
m.maximize(3*x + y)

sol = m.solve()                     # strategy="best", rule="bland", ...
sol.status       # 'optimal'
sol.objective    # Fraction(15, 2)
sol["x"], sol[y] # Fraction(2, 1), Fraction(3, 2)
```

For a pure LP (no integer variables) `sol.duals` maps each constraint name to its shadow price: how much the optimal objective moves per unit increase of that constraint's right hand side.

You can also skip the modelling layer and call `solve_lp(LPProblem(c, rows, senses, rhs, lb, ub))` or `solve_milp(problem, integer_indices)` directly.

## Design notes

**Standard form conversion.** A variable with a finite lower bound is shifted (`x = l + x'`), one with only an upper bound is mirrored (`x = u - x'`), and a free one is split (`x = x⁺ - x⁻`). Finite upper bounds become explicit rows. Rows with a negative right hand side are negated and their sense flipped, and that flip is remembered so the dual sign can be restored afterwards.

**Duals for free.** Each row starts with an identity column (its slack, or its artificial for `>=` and `=` rows). Those columns are kept in the tableau through phase 2 even though they may not re-enter, so the dual of row *i* is simply minus the final reduced cost of its identity column. The tests check strong duality (`c·x == b·y`), complementary slackness, dual sign conventions and dual feasibility on hundreds of random LPs.

**Redundant rows.** If phase 1 ends with an artificial in the basis at level 0 and its row has no non zero structural entry, the row is linearly dependent on the others. Rather than deleting it (which would scramble the basis inverse used for the duals), the artificial is left basic at zero. Its row is all zeros in the eligible columns, so it can never be chosen by the ratio test.

**Anti cycling.** Bland's rule (smallest index entering, smallest basic index on ratio ties) is the default and provably terminates. `rule="dantzig"` picks the most negative reduced cost, which usually needs fewer pivots, and switches to Bland after 50 consecutive degenerate pivots. Beale's classic cycling example is in the tests for both rules, and Klee Minty cubes check the solver survives exponential paths.

**Branch and bound.** Each node re-solves the LP from scratch with tightened bounds. That is wasteful compared to a dual simplex warm start, but it keeps the node logic tiny and obviously correct, and with exact arithmetic correctness was the point. Integer bounds are rounded inward before the search. When every objective coefficient sits on an integer variable and is itself an integer, node bounds are rounded up, which often lets the first integral solution close the whole tree.

**Limits.** The tableau is dense and fractions grow, so this is for problems with tens to low hundreds of variables, not thousands. An unbounded relaxation is reported as `unbounded` for the MILP (for rational data this is correct whenever the MILP is feasible).

## Testing

`tests/` holds 34 tests and every assertion about an objective is an exact equality:

* **test_simplex.py**: textbook cases with known primal and dual answers, infeasible, unbounded, free and shifted bounds, redundant equalities, Beale's cycling LP, Klee Minty cubes up to n = 7, 300 random LPs checked against full vertex enumeration under both pivot rules, and 200 random LPs checked for strong duality, complementary slackness and dual feasibility.
* **test_milp.py**: 40 random knapsacks under all four strategy and branching combinations versus brute force, 150 random bounded integer programs (including infeasible ones) versus exhaustive enumeration, LP feasible but integer infeasible problems, mixed integer models, unbounded MILPs, the integral bound rounding and node limits.
* **test_model_and_format.py**: expression algebra, constant handling, rejection of nonlinear products, named shadow prices, LP file parsing (ranges, free, binary, reversed bounds, fractions), line numbered parse errors, and an end to end run of the CLI on every file in `examples/`.

## License

MIT
