"""Two short models built with the Python API.

    python demo.py
"""

from exactmilp import Model


def production_plan():
    """Continuous LP: how many units of each product, and what is an extra
    hour of each machine worth (the shadow price)?"""
    m = Model("factory")
    chairs, tables, desks = m.var("chairs"), m.var("tables"), m.var("desks")
    m.add(1 * chairs + 3 * tables + 2 * desks <= 120, "saw_hours")
    m.add(2 * chairs + 1 * tables + 2 * desks <= 100, "lathe_hours")
    m.add(chairs + tables + desks <= 70, "storage")
    m.add(desks >= 5, "desk_contract")
    m.maximize(30 * chairs + 40 * tables + 45 * desks)
    sol = m.solve()
    print("== production plan:", sol.status, "profit", sol.objective)
    for name, v in sol.values.items():
        print(f"   {name:<7} {v}")
    for name, y in sol.duals.items():
        print(f"   shadow price of {name:<13} {y}")


def n_queens(n=6):
    """Pure feasibility MILP: place n non attacking queens."""
    m = Model(f"{n}-queens")
    q = [[m.bin_var(f"q{r}{c}") for c in range(n)] for r in range(n)]
    for i in range(n):
        m.add(sum(q[i]) == 1, f"row{i}")
        m.add(sum(q[r][i] for r in range(n)) == 1, f"col{i}")
    for d in range(-(n - 2), n - 1):
        cells = [q[r][r - d] for r in range(n) if 0 <= r - d < n]
        m.add(sum(cells) <= 1, f"diag{d}")
    for s in range(1, 2 * n - 2):
        cells = [q[r][s - r] for r in range(n) if 0 <= s - r < n]
        m.add(sum(cells) <= 1, f"anti{s}")
    m.minimize(0)
    sol = m.solve(strategy="depth")
    print(f"== {n} queens:", sol.status, f"({sol.nodes} nodes)")
    for r in range(n):
        print("   " + " ".join("Q" if sol[q[r][c]] == 1 else "." for c in range(n)))


if __name__ == "__main__":
    production_plan()
    n_queens()
