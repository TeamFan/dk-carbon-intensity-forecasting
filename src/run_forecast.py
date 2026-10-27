"""Rolling-origin forecasting experiment for DK1 and DK2.

Products
  ST: issued every hour, horizons 1-12 h.
  DA: issued once a day after hour 15:00-16:00 UTC of D-1, all 24 hours of day D
      (horizons 9-32 h); this is the product used for next-day scheduling.

Point models: persistence (ST) / diurnal persistence (DA), adaptive ARX by RLS
with exponential forgetting (lambda chosen on 2023H2), gradient-boosted trees
refitted monthly on the trailing year (non-linear benchmark).
Probabilistic models on top of the RLS forecast: Gaussian with recursive
(EWMA) variance; linear quantile regression refitted monthly on the trailing
year with exponential down-weighting; the latter with adaptive conformal
widening of the central 80% and 90% intervals.

Evaluation period: 2024-01-01 to 2026-09-30 (UTC).
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

from features import (TEST_END, TRAIN_END, VALID_END, design_day_ahead,
                      design_short_term, load)
from models import aci_widen, quantile_regression, run_direct_rls

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "forecasts"
TAUS = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
LAM_GRID = {"ST": [0.99, 0.995, 0.998, 0.999], "DA": [0.97, 0.98, 0.99, 0.995]}
ST_H = list(range(1, 13))
ST_PROB_H = [1, 2, 3, 6, 9, 12]
QR_DECAY_PER_DAY = 0.995
MONTHS = pd.date_range(TRAIN_END, TEST_END, freq="MS")


# ---------------------------------------------------------------- helpers
def resolved_mask(idx: pd.DatetimeIndex, h: int, cutoff: pd.Timestamp, span_days: int = 365):
    tgt = idx + pd.Timedelta(hours=h)
    return (tgt < cutoff) & (idx >= cutoff - pd.Timedelta(days=span_days))


def gbm_monthly(idx, X, y, h, seed=0):
    pred = np.full(len(idx), np.nan)
    for m0, m1 in zip(MONTHS[:-1], MONTHS[1:]):
        tr = resolved_mask(idx, h, m0) & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        te = (idx >= m0) & (idx < m1) & np.all(np.isfinite(X), axis=1)
        if tr.sum() < 100 or te.sum() == 0:
            continue
        g = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=15, min_samples_leaf=20, random_state=seed)
        g.fit(X[tr][:, 1:], y[tr])
        pred[te] = g.predict(X[te][:, 1:])
    return pred


def gaussian_ewma(pred, y, origins, h, decay):
    """Predictive sd from an EWMA of squared errors of already-resolved forecasts."""
    n = len(y)
    sd = np.full(n, np.nan)
    v, have = None, 0
    tgt = origins + h
    j = 0
    for i in range(n):
        while j < n and tgt[j] <= origins[i]:
            if np.isfinite(y[j]) and np.isfinite(pred[j]):
                e2 = (y[j] - pred[j]) ** 2
                v = e2 if v is None else decay * v + (1 - decay) * e2
                have += 1
            j += 1
        if have >= 30:
            sd[i] = np.sqrt(v)
    return pred[:, None] + sd[:, None] * norm.ppf(TAUS)[None, :]


def _qr_job(args):
    Xtr, ytr, w, Xte, tau = args
    b = quantile_regression(Xtr, ytr, tau, w)
    return Xte @ b


def qr_monthly(idx, Z, y, h, pool, subsample=1):
    """Monthly refitted, exponentially weighted linear quantile regression."""
    Q = np.full((len(idx), len(TAUS)), np.nan)
    jobs, slots = [], []
    for m0, m1 in zip(MONTHS[:-1], MONTHS[1:]):
        tr = resolved_mask(idx, h, m0) & np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
        te = (idx >= m0) & (idx < m1) & np.all(np.isfinite(Z), axis=1)
        tri = np.where(tr)[0][::subsample]
        if len(tri) < 60 or te.sum() == 0:
            continue
        age = (m0 - idx[tri]).total_seconds().to_numpy() / 86400.0
        w = QR_DECAY_PER_DAY ** age
        for k, tau in enumerate(TAUS):
            jobs.append((Z[tri], y[tri], w, Z[te], tau))
            slots.append((te, k))
    for (te, k), q in zip(slots, pool.map(_qr_job, jobs, chunksize=4)):
        Q[te, k] = q
    return np.sort(Q, axis=1)  # remove quantile crossing


def choose_lambda(builder, keys, product):
    best, scores = None, {}
    for lam in LAM_GRID[product]:
        errs = []
        for k in keys:
            idx, X, y, org, h = builder(k)
            p = run_direct_rls(X, y, org, h, lam, 1e-2)
            val = (idx >= TRAIN_END) & (idx < VALID_END)
            errs.append(np.nanmean(np.abs(y - p)[val]))
        scores[lam] = float(np.mean(errs))
    best = min(scores, key=scores.get)
    return best, scores


# ---------------------------------------------------------------- main
def run_area(area: str, pool, quick: bool = False):
    d = load(area)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"area": area, "flagged_hours": int(d["flag"].sum())}

    # ---------------- short-term product
    def st_builder(h):
        idx, X, yT, yt, _ = design_short_term(d, h, area)
        return idx, X, yT, np.arange(len(idx)), h

    lam_st, sc = choose_lambda(st_builder, [1, 3, 6, 12], "ST")
    meta["lambda_ST"], meta["lambda_ST_scores"] = lam_st, sc
    rows = []
    for h in ST_H:
        idx, X, yT, yt, _ = design_short_term(d, h, area)
        org = np.arange(len(idx))
        rls = run_direct_rls(X, yT, org, h, lam_st, 1e-2)
        gbm = np.full(len(idx), np.nan) if quick else gbm_monthly(idx, X, yT, h)
        df = pd.DataFrame({"origin": idx, "h": h, "y": yT, "persistence": yt, "rls": rls, "gbm": gbm})
        if h in ST_PROB_H:
            G = gaussian_ewma(rls, yT, org, h, decay=0.995)
            Z = np.column_stack([np.ones(len(idx)), rls, X[:, 4], X[:, 5], X[:, 6]])
            Q = qr_monthly(idx, Z, yT, h, pool, subsample=3)
            for k, tau in enumerate(TAUS):
                df[f"gauss_q{tau:.2f}"] = G[:, k]
                df[f"qr_q{tau:.2f}"] = Q[:, k]
            for lo_t, hi_t, a in [(0.1, 0.9, 0.2), (0.05, 0.95, 0.1)]:
                lo, hi = aci_widen(df[f"qr_q{lo_t:.2f}"].to_numpy(), df[f"qr_q{hi_t:.2f}"].to_numpy(),
                                   yT, org, h, alpha=a, gamma=0.005)
                df[f"aci_lo{int(100 * (1 - a))}"], df[f"aci_hi{int(100 * (1 - a))}"] = lo, hi
        rows.append(df[df.origin >= TRAIN_END])
        print(area, "ST h", h, "done", flush=True)
    pd.concat(rows).to_csv(OUT / f"st_{area}.csv.gz", index=False, float_format="%.3f", compression="gzip")

    # ---------------- day-ahead product
    def da_builder(th):
        idx, o, h, X, yT, _ = design_day_ahead(d, th, area)
        return idx, X, yT, o, h

    lam_da, sc = choose_lambda(da_builder, list(range(0, 24, 3)), "DA")
    meta["lambda_DA"], meta["lambda_DA_scores"] = lam_da, sc
    rows = []
    for th in range(24):
        idx, o, h, X, yT, _ = design_day_ahead(d, th, area)
        rls = run_direct_rls(X, yT, o, h, lam_da, 1e-2)
        gbm = np.full(len(idx), np.nan) if quick else gbm_monthly(idx, X, yT, h)
        diur = d["co2"].to_numpy()[o + h - 24 * int(np.ceil(h / 24))]
        prog = d["co2_prog"].to_numpy()[o + h]
        target_time = d.index[o + h]
        df = pd.DataFrame({"origin": idx, "target": target_time, "target_hour": th, "h": h, "y": yT,
                           "diurnal": diur, "rls": rls, "gbm": gbm, "energinet_prog_archived": prog})
        G = gaussian_ewma(rls, yT, o, h, decay=0.95)
        Z = np.column_stack([np.ones(len(idx)), rls, X[:, 4], X[:, 5], X[:, 6]])
        Q = qr_monthly(idx, Z, yT, h, pool, subsample=1)
        for k, tau in enumerate(TAUS):
            df[f"gauss_q{tau:.2f}"] = G[:, k]
            df[f"qr_q{tau:.2f}"] = Q[:, k]
        for lo_t, hi_t, a in [(0.1, 0.9, 0.2), (0.05, 0.95, 0.1)]:
            lo, hi = aci_widen(df[f"qr_q{lo_t:.2f}"].to_numpy(), df[f"qr_q{hi_t:.2f}"].to_numpy(),
                               yT, o, h, alpha=a, gamma=0.01, scale_window=365, min_scores=60)
            df[f"aci_lo{int(100 * (1 - a))}"], df[f"aci_hi{int(100 * (1 - a))}"] = lo, hi
        rows.append(df[df.origin >= TRAIN_END])
        print(area, "DA hour", th, "done", flush=True)
    pd.concat(rows).sort_values(["origin", "target_hour"]).to_csv(
        OUT / f"da_{area}.csv.gz", index=False, float_format="%.3f", compression="gzip")
    (OUT / f"meta_{area}.json").write_text(json.dumps(meta, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", nargs="+", default=["DK1", "DK2"])
    ap.add_argument("--quick", action="store_true", help="skip the GBM benchmark")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        for area in a.areas:
            run_area(area, pool, a.quick)


if __name__ == "__main__":
    main()
