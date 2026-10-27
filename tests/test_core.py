import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import RLS, aci_widen, quantile_regression, run_direct_rls  # noqa: E402
from scheduling import (aggregate_envelope, cvar_schedule, disaggregate,  # noqa: E402
                        exact_fleet_schedule, greedy_schedule, make_fleet)


class TestRLS(unittest.TestCase):
    def test_matches_weighted_least_squares(self):
        rng = np.random.default_rng(0)
        n, d, lam, delta = 400, 4, 0.98, 1e-3
        X = rng.normal(size=(n, d))
        y = X @ rng.normal(size=d) + 0.1 * rng.normal(size=n)
        m = RLS(d, lam, delta)
        for i in range(n):
            m.update(X[i], y[i])
        w = lam ** np.arange(n)[::-1]
        A = X.T @ (w[:, None] * X) + lam ** n * delta * np.eye(d)
        theta = np.linalg.solve(A, X.T @ (w * y))
        np.testing.assert_allclose(m.theta, theta, rtol=1e-6, atol=1e-8)

    def test_no_look_ahead(self):
        """Forecasts must not change when targets after the issue time are altered."""
        rng = np.random.default_rng(1)
        n, h = 300, 5
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = rng.normal(size=n)
        org = np.arange(n)
        p1 = run_direct_rls(X, y, org, h, 0.99)
        y2 = y.copy()
        y2[200 - h + 1:] += 100.0          # targets of origins >= 200-h+1 resolve after origin 200
        p2 = run_direct_rls(X, y2, org, h, 0.99)
        np.testing.assert_allclose(p1[:201], p2[:201])


class TestQuantiles(unittest.TestCase):
    def test_intercept_only_is_sample_quantile(self):
        rng = np.random.default_rng(2)
        y = rng.exponential(size=501)
        for tau in (0.1, 0.5, 0.9):
            b = quantile_regression(np.ones((501, 1)), y, tau)
            loss = lambda q: np.sum(np.maximum(tau * (y - q), (tau - 1) * (y - q)))
            self.assertLessEqual(loss(b[0]), loss(np.quantile(y, tau)) + 1e-8)

    def test_aci_uses_only_resolved_misses(self):
        rng = np.random.default_rng(3)
        n, h = 1000, 3
        y = rng.normal(size=n)
        lo, hi = -np.ones(n), np.ones(n)
        org = np.arange(n)
        a1 = aci_widen(lo, hi, y, org, h, 0.1, min_scores=50)
        y2 = y.copy()
        y2[600:] = 50.0
        a2 = aci_widen(lo, hi, y2, org, h, 0.1, min_scores=50)
        np.testing.assert_allclose(a1[0][:600 - h + 1], a2[0][:600 - h + 1])


class TestScheduling(unittest.TestCase):
    def test_greedy_is_lp_optimal(self):
        rng = np.random.default_rng(4)
        for _ in range(20):
            c = rng.uniform(0, 100, 24)
            p = greedy_schedule(c, 6.5, 1.0)
            r = linprog(c, A_eq=np.ones((1, 24)), b_eq=[6.5], bounds=[(0, 1)] * 24, method="highs")
            self.assertAlmostEqual(c @ p, r.fun, places=6)

    def test_cvar_with_identical_scenarios_is_greedy(self):
        c = np.linspace(1, 24, 24)[::-1]
        S = np.tile(c, (50, 1))
        p = cvar_schedule(S, 5.0, 1.0, beta=1.0, alpha=0.9)
        self.assertAlmostEqual(c @ p, c @ greedy_schedule(c, 5.0, 1.0), places=6)

    def test_exact_profile_inside_envelope_and_disaggregable(self):
        rng = np.random.default_rng(5)
        f = make_fleet(15, rng)
        c = rng.uniform(10, 200, 24)
        prof, X = exact_fleet_schedule(c, f)
        pmax, lo, hi = aggregate_envelope(f)
        cum = np.cumsum(prof)
        self.assertTrue(np.all(prof <= pmax + 1e-7))
        self.assertTrue(np.all(cum >= lo - 1e-6) and np.all(cum <= hi + 1e-6))
        _, s = disaggregate(prof, f)
        self.assertLess(s.sum(), 1e-6)


if __name__ == "__main__":
    unittest.main()
