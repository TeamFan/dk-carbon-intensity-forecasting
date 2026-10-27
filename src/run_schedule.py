"""Value of day-ahead carbon forecasts for flexible data-centre load (Parts B and C)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from evaluate import COL, load_fc, quantile_cols
from models import gaussian_copula_scenarios, pit_from_quantiles
from run_forecast import TAUS
from scheduling import (cvar_schedule, disaggregate, exact_fleet_schedule, greedy_schedule,
                        group_sites, grouped_aggregate_schedule, make_fleet)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
E, P = 6.0, 1.0           # deferrable energy per day (MWh) and power headroom (MW)
N_SCEN, COP_DAYS, SHRINK = 500, 90, 0.1
ALPHA = 0.9


def day_matrix(da: pd.DataFrame) -> dict:
    """Arrange the day-ahead table as (days, 24) arrays; keep complete days only."""
    da = da.assign(day=da.target.dt.normalize())
    full = da.groupby("day").target_hour.transform("count") == 24
    da = da[full].sort_values(["day", "target_hour"])
    days = da.day.unique()
    out = {"days": pd.DatetimeIndex(days)}
    for col in ["y", "diurnal", "rls", "gbm"] + quantile_cols("qr"):
        out[col] = da[col].to_numpy().reshape(len(days), 24)
    out["Q"] = np.stack([out[c] for c in quantile_cols("qr")], axis=2)  # (days, 24, K)
    return out


def part_b(area: str, rng: np.random.Generator) -> pd.DataFrame:
    da = load_fc("da", area).dropna(subset=["diurnal", "rls", "gbm"] + quantile_cols("qr"))
    M = day_matrix(da)
    y, Q = M["y"], M["Q"]
    nd = len(M["days"])
    # normal scores of PIT values, for the copula correlation
    Z = np.full((nd, 24), np.nan)
    for i in range(nd):
        for k in range(24):
            u = pit_from_quantiles(y[i, k], TAUS, np.sort(Q[i, k]))
            Z[i, k] = norm.ppf(np.clip(u, 1e-3, 1 - 1e-3))
    rows, corr_log = [], []
    for i in range(nd):
        if not (np.all(np.isfinite(y[i])) and np.all(np.isfinite(Q[i]))):
            continue
        # copula correlation from the previous COP_DAYS fully resolved days (day i-1 resolves after
        # the day-i forecast is issued at 16:00 UTC on day i-1, so use days <= i-2)
        dayi = M["days"][i]
        sel = (M["days"] <= dayi - pd.Timedelta(days=2)) & (M["days"] > dayi - pd.Timedelta(days=2 + COP_DAYS))
        past = Z[sel]
        past = past[np.all(np.isfinite(past), axis=1)]
        if len(past) < 30:
            continue
        R = np.corrcoef(past.T)
        corr_log.append(float(R[np.triu_indices(24, 1)].mean()))
        R = (1 - SHRINK) * R + SHRINK * np.eye(24)
        S = gaussian_copula_scenarios(np.sort(Q[i], axis=1), TAUS, R, N_SCEN, rng)
        sched = {
            "uniform": np.full(24, E / 24),
            "diurnal": greedy_schedule(M["diurnal"][i], E, P),
            "gbm": greedy_schedule(M["gbm"][i], E, P),
            "rls": greedy_schedule(M["rls"][i], E, P),
            "qr_median": greedy_schedule(Q[i, :, list(TAUS).index(0.5)], E, P),
            "scen_mean": greedy_schedule(S.mean(axis=0), E, P),
            "cvar_b0.5": cvar_schedule(S, E, P, beta=0.5, alpha=ALPHA),
            "cvar_b1.0": cvar_schedule(S, E, P, beta=1.0, alpha=ALPHA),
            "oracle": greedy_schedule(y[i], E, P),
        }
        row = {"day": M["days"][i]}
        for k, p in sched.items():
            row[k] = float(p @ y[i]) / 1000.0      # t CO2 (MWh * g/kWh / 1000)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["mean_copula_corr"] = float(np.mean(corr_log))
    return out


def part_c(area: str, n_sites: int = 30, ks=(1, 2, 5, 10, 30), seed: int = 7) -> pd.DataFrame:
    da = load_fc("da", area).dropna(subset=["rls"])
    M = day_matrix(da)
    f = make_fleet(n_sites, np.random.default_rng(seed))
    labels = {k: group_sites(f, k) for k in ks}
    rows = []
    for i in range(len(M["days"])):
        c, y = M["rls"][i], M["y"][i]
        if not (np.all(np.isfinite(c)) and np.all(np.isfinite(y))):
            continue
        prof_exact, X = exact_fleet_schedule(c, f)
        row = {"day": M["days"][i], "planned_exact": float(c @ prof_exact),
               "realised_exact": float(np.sum(X * y[None, :]))}
        for k in ks:
            if k == n_sites:
                continue
            profs = grouped_aggregate_schedule(c, f, labels[k])
            short = 0.0
            for g, pr in enumerate(profs):
                sub = {key: val[labels[k] == g] for key, val in f.items()}
                _, s = disaggregate(pr, sub)
                short += float(s.sum())
            row[f"planned_k{k}"] = float(c @ sum(profs))
            row[f"shortfall_k{k}"] = short / float(f["E"].sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["fleet"] = {k: v.tolist() for k, v in f.items()}
    return df


def main(areas=("DK1", "DK2")):
    rng = np.random.default_rng(2026)
    summary = {}
    figB, axB = plt.subplots(1, len(areas), figsize=(4.75 * len(areas), 2.8), squeeze=False)
    axB = axB[0]
    for j, area in enumerate(areas):
        b = part_b(area, rng)
        b.to_csv(RES / f"schedule_{area}.csv", index=False, float_format="%.5f")
        base = b["uniform"].sum()
        orac = base - b["oracle"].sum()
        sb = {"days": len(b), "uniform_tCO2": float(base), "mean_copula_corr": b.attrs["mean_copula_corr"],
              "first_day": str(b["day"].min().date()), "last_day": str(b["day"].max().date())}
        for k in [c for c in b.columns if c not in ("day",)]:
            red = base - b[k].sum()
            regret = b[k] - b["oracle"]
            sb[k] = {"reduction_vs_uniform_pct": 100 * red / base,
                     "share_of_oracle_reduction_pct": 100 * red / orac if orac > 0 else np.nan,
                     "regret_mean_kg": 1000 * regret.mean(),
                     "regret_p95_kg": 1000 * regret.quantile(0.95),
                     "regret_cvar90_kg": 1000 * regret[regret >= regret.quantile(0.9)].mean()}
        summary[area] = {"part_b": sb}
        names = ["diurnal", "gbm", "rls", "qr_median", "cvar_b0.5", "oracle"]
        labs = ["diurnal\npersist.", "boosted\ntrees", "adaptive\nARX", "QR\nmedian", "CVaR\n(beta=0.5)", "perfect\ninfo"]
        vals = [sb[n]["reduction_vs_uniform_pct"] for n in names]
        axB[j].bar(range(len(names)), vals, color=["#8a8a8a", COL["gbm"], COL["rls"], COL["qr"], COL["aci"], "#2f2f2f"])
        for x, v in enumerate(vals):
            axB[j].text(x, v + 0.3, f"{v:.1f}", ha="center", fontsize=7)
        axB[j].set_xticks(range(len(names)), labs, fontsize=7)
        axB[j].set(ylabel="CO$_2$ reduction vs uniform (%)", title=f"{area}: flexible load, {len(b)} days")

        c = part_c(area)
        c.to_csv(RES / f"fleet_{area}.csv", index=False, float_format="%.6f")
        sc = {"days": len(c)}
        for k in (1, 2, 5, 10):
            sc[f"k{k}"] = {"planned_optimism_pct": float(100 * (1 - c[f"planned_k{k}"].sum() / c["planned_exact"].sum())),
                           "shortfall_mean_pct": float(100 * c[f"shortfall_k{k}"].mean()),
                           "shortfall_max_pct": float(100 * c[f"shortfall_k{k}"].max()),
                           "days_with_shortfall_pct": float(100 * (c[f"shortfall_k{k}"] > 1e-6).mean())}
        summary[area]["part_c"] = sc
        summary[area]["fleet"] = c.attrs["fleet"]
        print(area, json.dumps({k: v for k, v in summary[area].items() if k != "fleet"}, indent=1), flush=True)
    figB.tight_layout()
    figB.savefig(FIG / "scheduling_value.pdf")
    figB.savefig(FIG / "scheduling_value.png")
    plt.close(figB)

    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    for area, mk in [(a, m) for a, m in [("DK1", "o"), ("DK2", "s")] if a in areas]:
        ks = [1, 2, 5, 10, 30]
        sh = [summary[area]["part_c"][f"k{k}"]["shortfall_mean_pct"] for k in ks[:-1]] + [0.0]
        op = [summary[area]["part_c"][f"k{k}"]["planned_optimism_pct"] for k in ks[:-1]] + [0.0]
        ax.plot(ks, sh, marker=mk, color=COL["aci"], label=f"{area}: undeliverable energy")
        ax.plot(ks, op, marker=mk, color=COL["rls"], ls="--", label=f"{area}: planned-CO$_2$ optimism")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 30], ["1", "2", "5", "10", "30\n(exact)"])
    ax.set(xlabel="number of aggregate groups (30 sites)", ylabel="% (mean over days)",
           title="Compact aggregate models of a heterogeneous fleet")
    ax.legend(frameon=False, fontsize=6.5)
    fig.savefig(FIG / "aggregation.pdf")
    fig.savefig(FIG / "aggregation.png")
    plt.close(fig)
    (RES / "scheduling_summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or ("DK1", "DK2"))
