import os
import subprocess
import sys
import unittest
from fractions import Fraction as F

from exactmilp import GE, LE, OPTIMAL, LPFormatError, Model, parse_lp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelTests(unittest.TestCase):
    def test_expression_algebra(self):
        m = Model()
        x, y = m.var("x"), m.var("y")
        e = 3 * x + 2 * (y - x) + 4 - y / 2
        self.assertEqual(e.terms, {0: 1, 1: F(3, 2)})
        self.assertEqual(e.const, 4)
        self.assertEqual((x - x).terms, {})

    def test_constraint_moves_constants_right(self):
        m = Model()
        x = m.var("x")
        con = 2 * x + 3 >= x + 10
        self.assertEqual(con.sense, GE)
        self.assertEqual(con.terms, {0: 1})
        self.assertEqual(con.rhs, 7)
        con2 = 5 <= x
        self.assertEqual((con2.sense, con2.rhs), (GE, 5))

    def test_nonlinear_rejected(self):
        m = Model()
        x, y = m.var("x"), m.var("y")
        with self.assertRaises(TypeError):
            x * y
        with self.assertRaises(TypeError):
            m.add(x)

    def test_constraint_has_no_truth_value(self):
        m = Model()
        x = m.var("x")
        with self.assertRaises(TypeError):
            if x <= 3:
                pass

    def test_maximize_and_shadow_prices(self):
        m = Model()
        x, y = m.var("x"), m.var("y")
        m.add(x <= 4, "plant1")
        m.add(2 * y <= 12, "plant2")
        m.add(3 * x + 2 * y <= 18, "plant3")
        m.maximize(3 * x + 5 * y)
        sol = m.solve()
        self.assertEqual(sol.objective, 36)
        self.assertEqual(sol.values, {"x": 2, "y": 6})
        self.assertEqual(sol.duals, {"plant1": 0, "plant2": F(3, 2), "plant3": 1})

    def test_duplicate_names(self):
        m = Model()
        m.var("x")
        with self.assertRaises(ValueError):
            m.var("x")


SAMPLE = r"""
\ production planning toy
Maximize
 profit: 5 x + 4 y + 3/2 z
Subject To
 labour:   2 x + 3 y + z <= 5
 material: 4 x + y + 2 z <= 11
 mix:      1 <= x + y <= 3
 c4:       -x + 0.5 y >= -2
Bounds
 0 <= x <= 2
 z <= 4
General
 x
End
"""


class FormatTests(unittest.TestCase):
    def test_parse_structure(self):
        m = parse_lp(SAMPLE)
        self.assertEqual([v.name for v in m.variables], ["x", "y", "z"])
        self.assertEqual([c.name for c in m.constraints], ["labour", "material", "mix_lo", "mix_hi", "c4"])
        self.assertEqual(m.sense, "max")
        self.assertEqual(m.objective.terms[2], F(3, 2))
        self.assertEqual(m.ub, [2, None, 4])
        self.assertEqual(m.integer, [0])
        self.assertEqual(m.constraints[4].terms[1], F(1, 2))

    def test_parse_and_solve(self):
        sol = parse_lp(SAMPLE).solve()
        self.assertEqual(sol.status, OPTIMAL)
        self.assertEqual(sol["x"], 2)
        self.assertEqual(sol.objective, 5 * 2 + 4 * sol["y"] + F(3, 2) * sol["z"])
        self.assertEqual(sol.objective, F(23, 2))

    def test_free_binary_and_reversed_bounds(self):
        m = parse_lp("min\n x - y + b\nst\n x + y >= -3\n y <= 2\nbounds\n x free\n -5 <= x\n y >= -1\nbinary\n b\nend")
        self.assertEqual(m.lb, [-5, -1, 0])
        self.assertEqual(m.ub, [None, None, 1])
        sol = m.solve()
        self.assertEqual(sol.values, {"x": -5, "y": 2, "b": 0})

    def test_errors_report_line(self):
        with self.assertRaises(LPFormatError) as ctx:
            parse_lp("max\n x\nst\n x <= 3 <= \nend")
        self.assertIn("line 4", str(ctx.exception))
        with self.assertRaises(LPFormatError):
            parse_lp("st\n x <= 1\n")

    def test_cli_on_examples(self):
        for name in sorted(os.listdir(os.path.join(ROOT, "examples"))):
            path = os.path.join(ROOT, "examples", name)
            out = subprocess.run([sys.executable, "-m", "exactmilp", path], cwd=ROOT,
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr + out.stdout)
            self.assertIn("status:     optimal", out.stdout)


if __name__ == "__main__":
    unittest.main()
