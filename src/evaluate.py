"""Scores, tests and figures for the forecasting experiment (test period 2024-01 to 2026-09)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from features import TEST_END, VALID_END
from run_forecast import TAUS

ROOT = Path(__file__).resolve().parents[1]
FC = ROOT / "results" / "forecasts"
RES = ROOT / "results"
FIG = RES / "figures"
AREAS = ["DK1", "DK2"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
COL = {"persistence": "#8a8a8a", "diurnal": "#8a8a8a", "rls": "#1b6a8a", "gbm": "#d08c2a",
       "gauss": "#8a8a8a", "qr": "#1b6a8a", "aci": "#b03a48"}


def load_fc(kind: str, area: str) -> pd.DataFrame:
    df = pd.read_csv(FC / f"{kind}_{area}.csv.gz", parse_dates=["origin"] + (["target"] if kind == "da" else []))
    tgt = df["target"] if kind == "da" else df["origin"] + pd.to_timedelta(df["h"], unit="h")
    return df[(df["origin"] >= VALID_END) & (tgt < TEST_END) & df["y"].notna()].copy()


def pinball(y, q, tau):
    u = y - q
    return np.maximum(tau * u, (tau - 1) * u)


def dm_test(loss_a, loss_b, lag):
    """Diebold-Mariano statistic with Newey-West variance (positive: b is better)."""
    d = np.asarray(loss_a - loss_b)
    d = d[np.isfinite(d)]
    n = len(d)
    dc = d - d.mean()
    v = dc @ dc / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(v / n)
    return float(stat), float(2 * (1 - norm.cdf(abs(stat))))


def quantile_cols(prefix):
    return [f"{prefix}_q{t:.2f}" for t in TAUS]


def prob_scores(df, prefix):
    Q = df[quantile_cols(prefix)].to_numpy()
    y = df["y"].to_numpy()[:, None]
    qs = np.mean(pinball(y, Q, TAUS[None, :]), axis=1)
    out = {"QS": float(np.mean(qs))}
    for lo_t, hi_t, lev in [(0.10, 0.90, 80), (0.05, 0.95, 90)]:
        lo, hi = df[f"{prefix}_q{lo_t:.2f}"], df[f"{prefix}_q{hi_t:.2f}"]
        out[f"cov{lev}"] = float(np.mean((df.y >= lo) & (df.y <= hi)))
        out[f"width{lev}"] = float(np.mean(hi - lo))
    out["reliability"] = [float(np.mean(df["y"].to_numpy() <= Q[:, k])) for k in range(len(TAUS))]
    return out


def aci_scores(df):
    out = {}
    for lev in (80, 90):
        lo, hi = df[f"aci_lo{lev}"], df[f"aci_hi{lev}"]
        out[f"cov{lev}"] = float(np.mean((df.y >= lo) & (df.y <= hi)))
        out[f"width{lev}"] = float(np.mean(hi - lo))
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    summary = {}
    st_tab, da_tab = [], []
    for area in AREAS:
        s = {}
        # ---------------------------------------------- short-term point scores
        st = load_fc("st", area)
        st = st.dropna(subset=["persistence", "rls", "gbm"])
        for h, g in st.groupby("h"):
            row = {"area": area, "h": h, "n": len(g)}
            for m in ["persistence", "rls", "gbm"]:
                e = g["y"] - g[m]
                row[f"MAE_{m}"] = e.abs().mean()
                row[f"RMSE_{m}"] = np.sqrt((e ** 2).mean())
            row["DM_rls_vs_pers"], row["p_rls_vs_pers"] = dm_test((g.y - g.persistence).abs(), (g.y - g.rls).abs(), lag=h)
            st_tab.append(row)
        # ---------------------------------------------- short-term probabilistic
        sp = {}
        for h in sorted(st.h.unique()):
            g = st[(st.h == h)].dropna(subset=quantile_cols("qr") + quantile_cols("gauss") + ["aci_lo90"])
            if len(g) == 0:
                continue
            sp[int(h)] = {"gauss": prob_scores(g, "gauss"), "qr": prob_scores(g, "qr"), "aci": aci_scores(g), "n": len(g)}
        s["st_prob"] = sp
        # ---------------------------------------------- day-ahead
        da = load_fc("da", area).dropna(subset=["diurnal", "rls", "gbm"])
        for th, g in da.groupby("target_hour"):
            row = {"area": area, "target_hour": th, "h": int(g.h.iloc[0]), "n": len(g)}
            for m in ["diurnal", "rls", "gbm", "energinet_prog_archived"]:
                e = g["y"] - g[m]
                row[f"MAE_{m}"] = e.abs().mean()
                row[f"RMSE_{m}"] = np.sqrt((e ** 2).mean())
            da_tab.append(row)
        tot = {}
        for m in ["diurnal", "rls", "gbm", "energinet_prog_archived"]:
            e = da["y"] - da[m]
            tot[m] = {"MAE": float(e.abs().mean()), "RMSE": float(np.sqrt((e ** 2).mean()))}
        daily = da.assign(day=da.target.dt.normalize())
        la = daily.groupby("day").apply(lambda g: (g.y - g.diurnal).abs().mean(), include_groups=False)
        lb = daily.groupby("day").apply(lambda g: (g.y - g.rls).abs().mean(), include_groups=False)
        lc = daily.groupby("day").apply(lambda g: (g.y - g.gbm).abs().mean(), include_groups=False)
        tot["DM_rls_vs_diurnal"] = dm_test(la, lb, lag=7)
        tot["DM_rls_vs_gbm"] = dm_test(lc, lb, lag=7)
        tot["mean_y"] = float(da.y.mean())
        s["da_point"] = tot
        gp = da.dropna(subset=quantile_cols("qr") + quantile_cols("gauss") + ["aci_lo90"])
        s["da_prob"] = {"gauss": prob_scores(gp, "gauss"), "qr": prob_scores(gp, "qr"), "aci": aci_scores(gp), "n": len(gp)}
        # coverage by year and in high-intensity hours (top quintile of the median forecast)
        byyear = {}
        for yr, g in gp.groupby(gp.target.dt.year):
            byyear[int(yr)] = {"qr90": float(np.mean((g.y >= g["qr_q0.05"]) & (g.y <= g["qr_q0.95"]))),
                               "aci90": float(np.mean((g.y >= g.aci_lo90) & (g.y <= g.aci_hi90))),
                               "gauss90": float(np.mean((g.y >= g["gauss_q0.05"]) & (g.y <= g["gauss_q0.95"])))}
        s["da_cov_by_year"] = byyear
        hi_mask = gp["qr_q0.50"] >= gp["qr_q0.50"].quantile(0.8)
        g = gp[hi_mask]
        s["da_cov_top_quintile"] = {
            "qr90": float(np.mean((g.y >= g["qr_q0.05"]) & (g.y <= g["qr_q0.95"]))),
            "aci90": float(np.mean((g.y >= g.aci_lo90) & (g.y <= g.aci_hi90))),
            "gauss90": float(np.mean((g.y >= g["gauss_q0.05"]) & (g.y <= g["gauss_q0.95"])))}
        summary[area] = s
        plot_area(area, st, da, gp, sp)

    pd.DataFrame(st_tab).round(3).to_csv(RES / "st_point_scores.csv", index=False)
    pd.DataFrame(da_tab).round(3).to_csv(RES / "da_point_scores_by_hour.csv", index=False)
    (RES / "forecast_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({a: {"da_point": summary[a]["da_point"], "da_prob": {k: {kk: vv for kk, vv in v.items() if kk != "reliability"} if isinstance(v, dict) else v for k, v in summary[a]["da_prob"].items()}} for a in AREAS}, indent=1))
    print(pd.DataFrame(st_tab)[["area", "h", "MAE_persistence", "MAE_rls", "MAE_gbm", "RMSE_persistence", "RMSE_rls", "p_rls_vs_pers"]].round(2).to_string())


def plot_area(area, st, da, gp, sp):
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9))
    g = st.groupby("h")[["y", "persistence", "rls", "gbm"]].apply(
        lambda g: pd.Series({m: (g.y - g[m]).abs().mean() for m in ["persistence", "rls", "gbm"]}))
    for m, lab in [("persistence", "persistence"), ("rls", "adaptive ARX (RLS)"), ("gbm", "boosted trees")]:
        ax[0].plot(g.index, g[m], marker="o", ms=3, color=COL[m], label=lab)
    ax[0].set(xlabel="horizon (h)", ylabel="MAE (g/kWh)", title=f"{area}: hourly-issued, 1-12 h")
    ax[0].legend(frameon=False, fontsize=7)
    g = da.groupby("target_hour")[["y", "diurnal", "rls", "gbm"]].apply(
        lambda g: pd.Series({m: (g.y - g[m]).abs().mean() for m in ["diurnal", "rls", "gbm"]}))
    for m, lab in [("diurnal", "diurnal persistence"), ("rls", "adaptive ARX (RLS)"), ("gbm", "boosted trees")]:
        ax[1].plot(g.index, g[m], marker="o", ms=3, color=COL[m], label=lab)
    ax[1].set(xlabel="target hour of day D (UTC)", ylabel="MAE (g/kWh)", title=f"{area}: day-ahead, issued 16:00 UTC")
    ax[1].legend(frameon=False, fontsize=7)
    Q = gp[quantile_cols("qr")].to_numpy()
    G = gp[quantile_cols("gauss")].to_numpy()
    y = gp.y.to_numpy()[:, None]
    ax[2].plot([0, 1], [0, 1], color="k", lw=0.6)
    ax[2].plot(TAUS, (y <= G).mean(0), marker="s", ms=3, color=COL["gauss"], label="Gaussian, EWMA variance")
    ax[2].plot(TAUS, (y <= Q).mean(0), marker="o", ms=3, color=COL["qr"], label="quantile regression")
    ax[2].set(xlabel="nominal level", ylabel="observed frequency", title=f"{area}: day-ahead reliability")
    ax[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"forecast_skill_{area}.pdf")
    fig.savefig(FIG / f"forecast_skill_{area}.png")
    plt.close(fig)

    # fan chart, one week
    wk = gp[(gp.target >= "2026-03-02") & (gp.target < "2026-03-09")].sort_values("target")
    if len(wk):
        fig, ax = plt.subplots(figsize=(10.5, 2.6))
        ax.fill_between(wk.target, wk["qr_q0.05"], wk["qr_q0.95"], color=COL["qr"], alpha=0.18, lw=0, label="90% interval")
        ax.fill_between(wk.target, wk["qr_q0.20"], wk["qr_q0.80"], color=COL["qr"], alpha=0.30, lw=0, label="60% interval")
        ax.plot(wk.target, wk["qr_q0.50"], color=COL["qr"], lw=1, label="median")
        ax.plot(wk.target, wk.y, color="k", lw=0.9, label="realised")
        ax.set(ylabel="g CO$_2$/kWh", title=f"{area}: day-ahead quantile forecasts, 2-8 March 2026")
        ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper left")
        fig.savefig(FIG / f"fan_{area}.pdf")
        fig.savefig(FIG / f"fan_{area}.png")
        plt.close(fig)

    # monthly coverage of the 90% interval
    m = gp.assign(month=gp.target.dt.to_period("M").dt.to_timestamp()).groupby("month").apply(
        lambda g: pd.Series({"qr": np.mean((g.y >= g["qr_q0.05"]) & (g.y <= g["qr_q0.95"])),
                             "aci": np.mean((g.y >= g.aci_lo90) & (g.y <= g.aci_hi90)),
                             "gauss": np.mean((g.y >= g["gauss_q0.05"]) & (g.y <= g["gauss_q0.95"]))}),
        include_groups=False)
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    for k, lab in [("gauss", "Gaussian, EWMA variance"), ("qr", "quantile regression"), ("aci", "QR + adaptive conformal")]:
        ax.plot(m.index, m[k], marker="o", ms=2.5, color=COL[k], label=lab)
    ax.axhline(0.9, color="k", lw=0.6, ls="--")
    import matplotlib.dates as mdates
    loc = mdates.MonthLocator(bymonth=(1, 7))
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set(ylabel="monthly coverage", title=f"{area}: day-ahead 90% interval")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIG / f"coverage_{area}.pdf")
    fig.savefig(FIG / f"coverage_{area}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
