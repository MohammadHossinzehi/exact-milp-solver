import itertools
import random
import unittest
from fractions import Fraction as F

from exactmilp import EQ, GE, INFEASIBLE, LE, OPTIMAL, UNBOUNDED, LPProblem, solve_lp


def solve_linear(M, rhs):
    """Exact Gaussian elimination; returns None if singular."""
    n = len(M)
    A = [list(r) + [b] for r, b in zip(M, rhs)]
    for c in range(n):
        p = next((r for r in range(c, n) if A[r][c] != 0), None)
        if p is None:
            return None
        A[c], A[p] = A[p], A[c]
        for r in range(n):
            if r != c and A[r][c]:
                f = A[r][c] / A[c][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [A[i][-1] / A[i][i] for i in range(n)]


def brute_force_lp(c, rows, senses, rhs):
    """Enumerate every vertex of {x >= 0, rows} and return the best (min)."""
    n = len(c)
    hyper = [(r, b) for r, b in zip(rows, rhs)]
    hyper += [([F(int(i == j)) for i in range(n)], F(0)) for j in range(n)]
    best = None
    for combo in itertools.combinations(hyper, n):
        x = solve_linear([h[0] for h in combo], [h[1] for h in combo])
        if x is None or any(v < 0 for v in x):
            continue
        ok = True
        for r, s, b in zip(rows, senses, rhs):
            lhs = sum(a * v for a, v in zip(r, x))
            if (s == LE and lhs > b) or (s == GE and lhs < b) or (s == EQ and lhs != b):
                ok = False
                break
        if ok:
            val = sum(a * v for a, v in zip(c, x))
            if best is None or val < best:
                best = val
    return best


class TextbookTests(unittest.TestCase):
    def test_wyndor_glass(self):
        # max 3x + 5y st x <= 4, 2y <= 12, 3x + 2y <= 18  (Hillier & Lieberman)
        p = LPProblem([-3, -5], [[1, 0], [0, 2], [3, 2]], [LE] * 3, [4, 12, 18])
        r = solve_lp(p)
        self.assertEqual(r.status, OPTIMAL)
        self.assertEqual(r.objective, -36)
        self.assertEqual(r.x, [2, 6])
        # duals of the min problem; shadow prices of the max are their negation
        self.assertEqual(r.duals, [0, F(-3, 2), -1])

    def test_ge_and_eq_rows_need_phase_one(self):
        # min x + y st x + 2y >= 4, x - y = 1
        p = LPProblem([1, 1], [[1, 2], [1, -1]], [GE, EQ], [4, 1])
        r = solve_lp(p)
        self.assertEqual(r.status, OPTIMAL)
        self.assertEqual(r.x, [2, 1])
        self.assertEqual(r.objective, 3)

    def test_infeasible(self):
        p = LPProblem([1, 1], [[1, 1], [1, 1]], [LE, GE], [1, 2])
        self.assertEqual(solve_lp(p).status, INFEASIBLE)

    def test_unbounded(self):
        p = LPProblem([-1, 0], [[1, -1]], [LE], [3])
        self.assertEqual(solve_lp(p).status, UNBOUNDED)

    def test_negative_rhs_is_flipped(self):
        # min x st -x <= -5   ->  x >= 5
        r = solve_lp(LPProblem([1], [[-1]], [LE], [-5]))
        self.assertEqual(r.x, [5])
        self.assertEqual(r.duals, [-1])

    def test_free_and_shifted_bounds(self):
        # min x + y with x free, y in [-3, 7], x - y >= -10
        p = LPProblem([1, 1], [[1, -1]], [GE], [-10], lb=[None, -3], ub=[None, 7])
        r = solve_lp(p)
        self.assertEqual(r.x, [-13, -3])
        self.assertEqual(r.objective, -16)

    def test_upper_bound_only(self):
        r = solve_lp(LPProblem([-1], [], [], [], lb=[None], ub=[F(5, 2)]))
        self.assertEqual(r.x, [F(5, 2)])

    def test_crossed_bounds_infeasible(self):
        r = solve_lp(LPProblem([1], [], [], [], lb=[3], ub=[2]))
        self.assertEqual(r.status, INFEASIBLE)

    def test_redundant_equalities(self):
        # second row is twice the first; phase 1 leaves an artificial at 0
        p = LPProblem([1, 2], [[1, 1], [2, 2], [1, 0]], [EQ, EQ, LE], [4, 8, 3])
        r = solve_lp(p)
        self.assertEqual(r.status, OPTIMAL)
        self.assertEqual(r.x, [3, 1])

    def test_objective_constant(self):
        r = solve_lp(LPProblem([1], [[1]], [GE], [2], c0=10))
        self.assertEqual(r.objective, 12)

    def test_float_input_is_read_decimally(self):
        r = solve_lp(LPProblem([1], [[1]], [GE], [0.1]))
        self.assertEqual(r.x, [F(1, 10)])


class DegeneracyTests(unittest.TestCase):
    def beale(self):
        # Beale (1955): Dantzig's rule with naive tie breaking cycles forever.
        c = [F(-3, 4), 150, F(-1, 50), 6]
        rows = [[F(1, 4), -60, F(-1, 25), 9], [F(1, 2), -90, F(-1, 50), 3], [0, 0, 1, 0]]
        return LPProblem(c, rows, [LE] * 3, [0, 0, 1])

    def test_beale_bland(self):
        r = solve_lp(self.beale(), rule="bland")
        self.assertEqual(r.objective, F(-1, 20))

    def test_beale_dantzig_with_fallback(self):
        r = solve_lp(self.beale(), rule="dantzig")
        self.assertEqual(r.objective, F(-1, 20))

    def test_klee_minty(self):
        for n in (3, 5, 7):
            c = [-(2 ** (n - 1 - j)) for j in range(n)]
            rows, rhs = [], []
            for i in range(n):
                rows.append([2 ** (i - j + 1) if j < i else (1 if j == i else 0) for j in range(n)])
                rhs.append(5 ** (i + 1))
            for rule in ("bland", "dantzig"):
                r = solve_lp(LPProblem(c, rows, [LE] * n, rhs), rule=rule)
                self.assertEqual(r.objective, -(5 ** n), (n, rule))


class RandomisedTests(unittest.TestCase):
    def random_problem(self, rng, n, m):
        c = [F(rng.randint(-9, 9)) for _ in range(n)]
        rows = [[F(rng.randint(-5, 9)) for _ in range(n)] for _ in range(m)]
        senses = [rng.choice([LE, LE, GE, EQ]) for _ in range(m)]
        rhs = [F(rng.randint(-5, 30)) for _ in range(m)]
        # box every variable so the optimum is finite whenever feasible
        for j in range(n):
            rows.append([F(int(i == j)) for i in range(n)])
            senses.append(LE)
            rhs.append(F(rng.randint(1, 12)))
        return c, rows, senses, rhs

    def test_matches_vertex_enumeration(self):
        rng = random.Random(7)
        checked = 0
        for _ in range(300):
            n, m = rng.randint(1, 3), rng.randint(1, 3)
            c, rows, senses, rhs = self.random_problem(rng, n, m)
            expected = brute_force_lp(c, rows, senses, rhs)
            for rule in ("bland", "dantzig"):
                r = solve_lp(LPProblem(c, rows, senses, rhs), rule=rule)
                if expected is None:
                    self.assertEqual(r.status, INFEASIBLE)
                else:
                    self.assertEqual(r.status, OPTIMAL)
                    self.assertEqual(r.objective, expected)
            checked += expected is not None
        self.assertGreater(checked, 100)

    def test_strong_duality_and_complementary_slackness(self):
        rng = random.Random(11)
        for _ in range(200):
            n, m = rng.randint(2, 5), rng.randint(2, 5)
            c, rows, senses, rhs = self.random_problem(rng, n, m)
            r = solve_lp(LPProblem(c, rows, senses, rhs))
            if r.status != OPTIMAL:
                continue
            y = r.duals
            # strong duality: c.x == b.y (all lb are 0, no explicit ub)
            self.assertEqual(r.objective, sum(b * yi for b, yi in zip(rhs, y)))
            for i, (row, s, b) in enumerate(zip(rows, senses, rhs)):
                slack = sum(a * v for a, v in zip(row, r.x)) - b
                if slack != 0:
                    self.assertEqual(y[i], 0)
                if s == LE:
                    self.assertLessEqual(y[i], 0)
                if s == GE:
                    self.assertGreaterEqual(y[i], 0)
            # dual feasibility: reduced costs c - A^T y >= 0
            for j in range(n):
                red = c[j] - sum(rows[i][j] * y[i] for i in range(len(rows)))
                self.assertGreaterEqual(red, 0)
                if r.x[j] > 0:
                    self.assertEqual(red, 0)


if __name__ == "__main__":
    unittest.main()
