"""Point and probabilistic forecasters for hourly CO2 intensity.

Information discipline: a forecast issued at origin t (after hour t is observed)
for target t+h may use y up to t and day-ahead inputs (wind/solar forecasts,
day-ahead price) published before t.  Every adaptive model is updated with the
pair (x_t, y_{t+h}) only once y_{t+h} has been observed, i.e. at time t+h.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm


# --------------------------------------------------------------------------
# Recursive least squares with exponential forgetting
# --------------------------------------------------------------------------
class RLS:
    """theta_t = argmin sum_s lam^(t-s) (y_s - x_s' theta)^2 + lam^t delta |theta|^2."""

    def __init__(self, dim: int, lam: float = 0.995, delta: float = 1.0):
        self.lam = lam
        self.theta = np.zeros(dim)
        self.P = np.eye(dim) / delta

    def predict(self, x: np.ndarray) -> float:
        return float(x @ self.theta)

    def update(self, x: np.ndarray, y: float) -> None:
        Px = self.P @ x
        k = Px / (self.lam + x @ Px)
        self.theta = self.theta + k * (y - x @ self.theta)
        self.P = (self.P - np.outer(k, Px)) / self.lam
        # keep P symmetric (numerical hygiene)
        self.P = 0.5 * (self.P + self.P.T)


def run_direct_rls(X: np.ndarray, y: np.ndarray, origins: np.ndarray, h: int,
                   lam: float, delta: float = 1.0) -> np.ndarray:
    """Direct h-step RLS forecasts.

    X[i] are features built at origin origins[i] for target origins[i]+h.
    Returns the forecast for every origin, using only pairs whose target time
    origins[j]+h <= origins[i] (observed before the forecast is issued).
    """
    n, d = X.shape
    model = RLS(d, lam, delta)
    pred = np.full(n, np.nan)
    tgt = origins + h
    j = 0  # next pair to absorb (pairs are ordered by target time)
    for i in range(n):
        while j < n and tgt[j] <= origins[i]:
            if np.isfinite(y[j]) and np.all(np.isfinite(X[j])):
                model.update(X[j], y[j])
            j += 1
        if np.all(np.isfinite(X[i])):
            pred[i] = model.predict(X[i])
    return pred


# --------------------------------------------------------------------------
# Quantile regression (weighted pinball loss) by linear programming
# --------------------------------------------------------------------------
def quantile_regression(X: np.ndarray, y: np.ndarray, tau: float,
                        w: np.ndarray | None = None) -> np.ndarray:
    """min_b sum_i w_i rho_tau(y_i - x_i' b), solved exactly with HiGHS."""
    n, d = X.shape
    w = np.ones(n) if w is None else w
    # variables: b+ (d), b- (d), u+ (n), u- (n)
    c = np.concatenate([np.zeros(2 * d), tau * w, (1 - tau) * w])
    from scipy.sparse import csr_matrix, hstack, identity
    Xs = csr_matrix(X)
    I = identity(n, format="csr")
    A = hstack([Xs, -Xs, I, -I], format="csr")
    res = linprog(c, A_eq=A, b_eq=y, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    z = res.x
    return z[:d] - z[d:2 * d]


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    u = y - q
    return np.maximum(tau * u, (tau - 1) * u)


# --------------------------------------------------------------------------
# Adaptive conformal inference (Gibbs & Candes, 2021) for a central interval
# --------------------------------------------------------------------------
def aci_widen(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, origins: np.ndarray, h: int,
              alpha: float, gamma: float = 0.005, scale_window: int = 24 * 60,
              min_scores: int = 200) -> tuple:
    """Symmetric additive widening of [lo, hi] with an online-updated level.

    The miss indicator for origin j is only known at time origins[j]+h, so the
    level used at origin i is updated with misses of earlier targets only.
    The widening is the (1-alpha_t) empirical quantile of the conformity score
    max(lo - y, y - hi) over the last `scale_window` resolved forecasts.
    """
    n = len(y)
    a_t = alpha
    out_lo, out_hi = lo.copy(), hi.copy()
    scores: list = []
    tgt = origins + h
    j = 0
    for i in range(n):
        while j < n and tgt[j] <= origins[i]:
            if np.isfinite(y[j]) and np.isfinite(lo[j]) and np.isfinite(out_lo[j]):
                err = float(y[j] < out_lo[j] or y[j] > out_hi[j])
                a_t = a_t + gamma * (alpha - err)
                scores.append(max(lo[j] - y[j], y[j] - hi[j]))
                if len(scores) > scale_window:
                    scores.pop(0)
            j += 1
        if len(scores) >= min_scores and np.isfinite(lo[i]):
            lev = min(max(1 - a_t, 0.0), 1.0)
            qhat = np.quantile(scores, lev) if lev < 1 else max(scores)
            out_lo[i] = lo[i] - qhat
            out_hi[i] = hi[i] + qhat
    return out_lo, out_hi


# --------------------------------------------------------------------------
# Scenario generation: Gaussian copula on probability-integral transforms
# --------------------------------------------------------------------------
def pit_from_quantiles(y: float, taus: np.ndarray, q: np.ndarray) -> float:
    """Piecewise-linear predictive CDF evaluated at y (clipped to (0,1))."""
    if y <= q[0]:
        return taus[0] / 2
    if y >= q[-1]:
        return 1 - (1 - taus[-1]) / 2
    return float(np.interp(y, q, taus))


def sample_from_quantiles(u: np.ndarray, taus: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Inverse of the piecewise-linear CDF with linear tails to taus[0], taus[-1]."""
    lo_slope = (q[1] - q[0]) / (taus[1] - taus[0])
    hi_slope = (q[-1] - q[-2]) / (taus[-1] - taus[-2])
    x = np.interp(u, taus, q)
    x = np.where(u < taus[0], q[0] - lo_slope * (taus[0] - u), x)
    x = np.where(u > taus[-1], q[-1] + hi_slope * (u - taus[-1]), x)
    return x


def gaussian_copula_scenarios(Q: np.ndarray, taus: np.ndarray, R: np.ndarray,
                              n_scen: int, rng: np.random.Generator) -> np.ndarray:
    """Q: (H, K) quantiles for H target hours; R: (H, H) correlation of normal scores."""
    H = Q.shape[0]
    L = np.linalg.cholesky(R + 1e-8 * np.eye(H))
    Z = rng.standard_normal((n_scen, H)) @ L.T
    U = norm.cdf(Z)
    S = np.empty_like(U)
    for k in range(H):
        S[:, k] = sample_from_quantiles(U[:, k], taus, Q[k])
    return np.maximum(S, 0.0)
