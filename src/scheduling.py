"""Day-ahead scheduling of flexible data-centre load with carbon forecasts.

Part B  one site: deferrable energy E within day D, power in [0, P].
        Strategies differ only in the forecast they use; realised emissions are
        always evaluated with the realised average intensity (location-based
        accounting, not marginal emission reductions).
Part C  a fleet of heterogeneous sites with individual time windows.  The
        aggregator plans with a compact aggregate model (sum of power limits and
        of cumulative-energy bounds, a 'virtual battery') and we test whether the
        aggregate profile can be split into site schedules that respect every
        site's own window, power and energy constraints.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, identity, kron, eye


# ------------------------------------------------------------- Part B
def greedy_schedule(c: np.ndarray, E: float, P: float) -> np.ndarray:
    """Exact minimiser of c'p s.t. sum p = E, 0 <= p <= P (fill cheapest hours)."""
    p = np.zeros_like(c, dtype=float)
    rem = E
    for k in np.argsort(c, kind="stable"):
        take = min(P, rem)
        p[k] = take
        rem -= take
        if rem <= 1e-12:
            break
    return p


def cvar_schedule(S: np.ndarray, E: float, P: float, beta: float, alpha: float) -> np.ndarray:
    """min (1-beta) mean_s c_s'p + beta CVaR_alpha(c_s'p), sum p = E, 0<=p<=P.

    Rockafellar-Uryasev LP.  S: (n_scen, H) intensity scenarios.
    Variables: p (H), eta (1), z (n_scen) >= 0 with z_s >= S_s p - eta.
    """
    n, H = S.shape
    mean_c = S.mean(axis=0)
    c = np.concatenate([(1 - beta) * mean_c, [beta], beta / ((1 - alpha) * n) * np.ones(n)])
    # S p - eta - z <= 0
    A_ub = hstack([csr_matrix(S), -np.ones((n, 1)), -identity(n)], format="csr")
    b_ub = np.zeros(n)
    A_eq = np.concatenate([np.ones(H), [0.0], np.zeros(n)])[None, :]
    bounds = [(0, P)] * H + [(None, None)] + [(0, None)] * n
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[E], bounds=bounds, method="highs")
    if not r.success:
        raise RuntimeError(r.message)
    return r.x[:H]


# ------------------------------------------------------------- Part C
def make_fleet(n_sites: int, rng: np.random.Generator, H: int = 24) -> dict:
    """Heterogeneous deferrable workloads: window [a, b), energy E, power cap P."""
    a = rng.integers(0, 14, n_sites)
    length = rng.integers(6, 15, n_sites)
    b = np.minimum(a + length, H)
    P = rng.uniform(0.2, 1.0, n_sites)
    fill = rng.uniform(0.3, 0.8, n_sites)            # share of the window's max energy
    E = fill * P * (b - a)
    return {"a": a, "b": b, "P": P, "E": E}


def site_bounds(f: dict, H: int = 24) -> np.ndarray:
    """U[i, h] = power cap of site i in hour h (0 outside its window)."""
    n = len(f["a"])
    U = np.zeros((n, H))
    for i in range(n):
        U[i, f["a"][i]:f["b"][i]] = f["P"][i]
    return U


def exact_fleet_schedule(c: np.ndarray, f: dict) -> tuple:
    """Joint LP over all site schedules (reference with full site information)."""
    U = site_bounds(f, len(c))
    n, H = U.shape
    cost = np.tile(c, n)
    A_eq = kron(eye(n), np.ones((1, H)), format="csr")
    bounds = [(0, U[i, h]) for i in range(n) for h in range(H)]
    r = linprog(cost, A_eq=A_eq, b_eq=f["E"], bounds=bounds, method="highs")
    if not r.success:
        raise RuntimeError(r.message)
    X = r.x.reshape(n, H)
    return X.sum(axis=0), X


def aggregate_envelope(f: dict, H: int = 24) -> tuple:
    """Virtual-battery model: power bound and cumulative energy bounds summed over sites."""
    U = site_bounds(f, H)
    pmax = U.sum(axis=0)
    cumU = np.cumsum(U, axis=1)                         # max energy deliverable by end of hour h
    rest = U.sum(axis=1)[:, None] - cumU                 # max energy deliverable after hour h
    lo = np.maximum(0.0, f["E"][:, None] - rest)         # must have been delivered by end of h
    hi = np.minimum(f["E"][:, None], cumU)
    return pmax, lo.sum(axis=0), hi.sum(axis=0)


def aggregate_schedule(c: np.ndarray, f: dict) -> np.ndarray:
    H = len(c)
    pmax, lo, hi = aggregate_envelope(f, H)
    L = np.tril(np.ones((H, H)))                         # cumulative sums
    A_ub = np.vstack([L, -L])
    b_ub = np.concatenate([hi, -lo])
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, H)), b_eq=[f["E"].sum()],
                bounds=list(zip(np.zeros(H), pmax)), method="highs")
    if not r.success:
        raise RuntimeError(r.message)
    return r.x


def disaggregate(profile: np.ndarray, f: dict) -> tuple:
    """Split an aggregate profile into site schedules; minimise unserved site energy.

    Site energy may fall short (slack s_i >= 0) but the aggregate profile must be
    followed exactly or reduced, so the shortfall measures how much of the
    aggregate commitment cannot be delivered by the sites."""
    U = site_bounds(f, len(profile))
    n, H = U.shape
    nv = n * H
    # variables x (n*H), s (n)
    cost = np.concatenate([np.zeros(nv), np.ones(n)])
    A_site = hstack([kron(eye(n), np.ones((1, H))), identity(n)], format="csr")     # sum_h x + s = E
    A_hour = hstack([kron(np.ones((1, n)), eye(H)), csr_matrix((H, n))], format="csr")  # sum_i x <= profile
    bounds = [(0, U[i, h]) for i in range(n) for h in range(H)] + [(0, None)] * n
    r = linprog(cost, A_ub=A_hour, b_ub=profile + 1e-9, A_eq=A_site, b_eq=f["E"],
                bounds=bounds, method="highs")
    if not r.success:
        raise RuntimeError(r.message)
    X = r.x[:nv].reshape(n, H)
    return X, r.x[nv:]


def group_sites(f: dict, k: int) -> np.ndarray:
    """Cluster sites into k groups of similar windows (sort by window midpoint, then start)."""
    n = len(f["a"])
    order = np.lexsort((f["a"], (f["a"] + f["b"]) / 2.0))
    labels = np.empty(n, dtype=int)
    for g, chunk in enumerate(np.array_split(order, k)):
        labels[chunk] = g
    return labels


def grouped_aggregate_schedule(c: np.ndarray, f: dict, labels: np.ndarray) -> list:
    """One virtual battery per group, scheduled jointly; returns per-group profiles.

    k = 1 is the single fleet envelope; k = number of sites is the exact model."""
    H = len(c)
    groups = np.unique(labels)
    L = np.tril(np.ones((H, H)))
    blocks_ub, b_ub, blocks_eq, b_eq, bounds = [], [], [], [], []
    for g in groups:
        sub = {key: val[labels == g] for key, val in f.items()}
        pmax, lo, hi = aggregate_envelope(sub, H)
        blocks_ub.append(np.vstack([L, -L]))
        b_ub.append(np.concatenate([hi, -lo]))
        blocks_eq.append(np.ones((1, H)))
        b_eq.append(sub["E"].sum())
        bounds += list(zip(np.zeros(H), pmax))
    from scipy.linalg import block_diag
    A_ub = block_diag(*blocks_ub)
    A_eq = block_diag(*blocks_eq)
    r = linprog(np.tile(c, len(groups)), A_ub=A_ub, b_ub=np.concatenate(b_ub), A_eq=A_eq,
                b_eq=np.array(b_eq), bounds=bounds, method="highs")
    if not r.success:
        raise RuntimeError(r.message)
    return [r.x[j * H:(j + 1) * H] for j in range(len(groups))]
