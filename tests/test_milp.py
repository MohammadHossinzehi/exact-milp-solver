import itertools
import random
import unittest
from fractions import Fraction as F

from exactmilp import EQ, GE, INFEASIBLE, LE, OPTIMAL, UNBOUNDED, LPProblem, Model, solve_milp


def brute_force_ip(c, rows, senses, rhs, box):
    best, arg = None, None
    for x in itertools.product(*(range(b + 1) for b in box)):
        ok = True
        for r, s, b in zip(rows, senses, rhs):
            lhs = sum(a * v for a, v in zip(r, x))
            if (s == LE and lhs > b) or (s == GE and lhs < b) or (s == EQ and lhs != b):
                ok = False
                break
        if ok:
            val = sum(a * v for a, v in zip(c, x))
            if best is None or val < best:
                best, arg = val, x
    return best, arg


class KnapsackTests(unittest.TestCase):
    def test_random_knapsacks(self):
        rng = random.Random(3)
        for _ in range(40):
            n = rng.randint(3, 9)
            w = [rng.randint(1, 20) for _ in range(n)]
            v = [rng.randint(1, 30) for _ in range(n)]
            cap = rng.randint(10, sum(w))
            m = Model()
            xs = [m.bin_var(f"x{i}") for i in range(n)]
            m.add(sum(wi * x for wi, x in zip(w, xs)) <= cap)
            m.maximize(sum(vi * x for vi, x in zip(v, xs)))
            expected, _ = brute_force_ip([-a for a in v], [w], [LE], [cap], [1] * n)
            for strategy in ("best", "depth"):
                for branching in ("most_fractional", "first"):
                    sol = m.solve(strategy=strategy, branching=branching)
                    self.assertEqual(sol.status, OPTIMAL)
                    self.assertEqual(sol.objective, -expected)
                    self.assertTrue(all(sol[x] in (0, 1) for x in xs))
                    self.assertLessEqual(sum(wi * sol[x] for wi, x in zip(w, xs)), cap)


class GeneralIntegerTests(unittest.TestCase):
    def test_against_enumeration(self):
        rng = random.Random(5)
        seen_infeasible = seen_optimal = 0
        for _ in range(150):
            n, m = rng.randint(2, 3), rng.randint(1, 3)
            box = [rng.randint(1, 5) for _ in range(n)]
            c = [rng.randint(-7, 7) for _ in range(n)]
            rows = [[rng.randint(-4, 6) for _ in range(n)] for _ in range(m)]
            senses = [rng.choice([LE, GE, EQ]) for _ in range(m)]
            rhs = [rng.randint(-3, 15) for _ in range(m)]
            expected, _ = brute_force_ip(c, rows, senses, rhs, box)
            p = LPProblem(c, rows, senses, rhs, lb=[0] * n, ub=box)
            r = solve_milp(p, range(n))
            if expected is None:
                self.assertEqual(r.status, INFEASIBLE)
                seen_infeasible += 1
            else:
                self.assertEqual(r.status, OPTIMAL)
                self.assertEqual(r.objective, expected)
                seen_optimal += 1
        self.assertGreater(seen_infeasible, 5)
        self.assertGreater(seen_optimal, 50)

    def test_lp_feasible_but_integer_infeasible(self):
        # 2x = 1 has x = 1/2 but no integer solution
        p = LPProblem([1], [[2]], [EQ], [1], ub=[10])
        self.assertEqual(solve_milp(p, [0]).status, INFEASIBLE)

    def test_mixed_integer(self):
        # max x + y, x integer, y continuous: 2x + 2y <= 7, x - y <= 1/2... y keeps the fractional part
        m = Model()
        x = m.int_var("x")
        y = m.var("y")
        m.add(2 * x + 2 * y <= 7)
        m.add(x <= F(5, 2))
        m.maximize(3 * x + y)
        sol = m.solve()
        self.assertEqual(sol.status, OPTIMAL)
        self.assertEqual(sol["x"], 2)
        self.assertEqual(sol["y"], F(3, 2))
        self.assertEqual(sol.objective, F(15, 2))

    def test_unbounded_milp(self):
        m = Model()
        x = m.int_var("x")
        m.add(x >= 1)
        m.maximize(x)
        self.assertEqual(m.solve().status, UNBOUNDED)

    def test_integral_objective_bound_prunes(self):
        # x + y <= 5/2 has a whole ridge of LP optima at -5/2. Rounding the
        # bound up to -2 lets the first integral solution close the search;
        # without it every fractional point on the ridge would be branched on.
        p = LPProblem([-1, -1], [[2, 2]], [LE], [5], ub=[3, 3])
        r = solve_milp(p, [0, 1])
        self.assertEqual(r.objective, -2)
        self.assertLessEqual(r.nodes, 5)
        # Same problem with a fractional objective weight: no rounding allowed.
        p2 = LPProblem([-1, F(-1001, 1000)], [[2, 2]], [LE], [5], ub=[3, 3])
        r2 = solve_milp(p2, [0, 1])
        self.assertEqual(r2.objective, F(-2002, 1000))
        self.assertEqual(r2.x, [0, 2])

    def test_node_limit(self):
        rng = random.Random(1)
        n = 14
        m = Model()
        xs = [m.bin_var() for _ in range(n)]
        w = [rng.randint(30, 60) * 2 for _ in range(n)]
        m.add(sum(wi * x for wi, x in zip(w, xs)) <= sum(w) // 2 + 1)
        m.maximize(sum(wi * x for wi, x in zip(w, xs)))
        sol = m.solve(node_limit=3)
        self.assertIn(sol.status, ("node_limit", "optimal"))


if __name__ == "__main__":
    unittest.main()
