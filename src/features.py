"""Cleaning and information-set-consistent design matrices."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

TRAIN_END = pd.Timestamp("2023-07-01")   # burn-in for the adaptive models
VALID_END = pd.Timestamp("2024-01-01")   # hyper-parameters chosen on 2023H2
TEST_END = pd.Timestamp("2026-10-01")
DA_ISSUE_HOUR = 15                       # forecast issued after hour 15:00-16:00 UTC of D-1


def load(area: str) -> pd.DataFrame:
    d = pd.read_csv(PROC / f"hourly_{area}.csv.gz", index_col=0, parse_dates=True)
    d = d.asfreq("h")
    # Energinet notes that realised values contain uncorrected measurement errors.
    # Flag physically impossible values and isolated spikes that the operator's own
    # schedule-based estimate does not share (typical at DST transitions).
    med = d["co2"].rolling(5, center=True, min_periods=3).median()
    spike = ((d["co2"] - med).abs() > 100) & ((d["co2"] - d["co2_prog"]).abs() > 100)
    bad = (d["co2"] < 0) | spike
    d["flag"] = bad.astype(int)
    d.loc[bad, "co2"] = np.nan
    return d


NEIGHBOURS = {"DK1": ("DE", "NO2", "SE3"), "DK2": ("DE", "SE4")}


def spread_features(d: pd.DataFrame, area: str) -> np.ndarray:
    """Day-ahead price spreads to interconnected zones (known after the auction).

    A near-zero spread means the zones are coupled in the auction, i.e. the
    marginal supply is shared with that neighbour."""
    cols = []
    for z in NEIGHBOURS[area]:
        sp = (d["price_da"] - d[f"price_{z}"]).to_numpy()
        cols += [np.tanh(sp / 10.0), (np.abs(sp) < 0.5).astype(float)]
    return np.column_stack(cols)


def _scales(d: pd.DataFrame) -> dict:
    tr = d.loc[:TRAIN_END]
    return {c: (tr[c].mean(), tr[c].std()) for c in ["co2", "wind_da", "solar_da", "price_da"]}


def _z(x, s):
    return (x - s[0]) / s[1]


def design_short_term(d: pd.DataFrame, h: int, area: str) -> tuple:
    """Origins t (hourly), target t+h. Returns origin times, X, y_target, y_origin."""
    s = _scales(d)
    y = d["co2"].to_numpy()
    n = len(d)
    t = np.arange(n - h)
    T = t + h
    hod = d.index.hour.to_numpy()[T]
    diurnal = T - 24 * int(np.ceil(h / 24))       # most recent same-hour value observed at t
    wz = _z(d["wind_da"].to_numpy(), s["wind_da"])
    sz = _z(d["solar_da"].to_numpy(), s["solar_da"])
    pz = _z(d["price_da"].to_numpy(), s["price_da"])
    yz = _z(y, s["co2"])
    lagd = np.where(diurnal >= 0, yz[np.clip(diurnal, 0, None)], np.nan)
    X = np.column_stack([
        np.ones(len(t)),
        yz[t], yz[np.clip(t - 1, 0, None)], lagd,
        wz[T], sz[T], pz[T],
        wz[T] - wz[t], pz[T] - pz[t],
        np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
        np.sin(4 * np.pi * hod / 24), np.cos(4 * np.pi * hod / 24),
        spread_features(d, area)[T],
    ])
    X[: max(1, 24 - h + 1)] = np.nan
    return d.index[t], X, y[T], y[t], (s["co2"])


def day_ahead_origins(d: pd.DataFrame) -> np.ndarray:
    """Positions of the issue hour (15:00 UTC) of every day D-1 with a full day D after it."""
    pos = np.where(d.index.hour == DA_ISSUE_HOUR)[0]
    return pos[pos + 9 + 23 < len(d)]


def design_day_ahead(d: pd.DataFrame, target_hour: int, area: str) -> tuple:
    """Origin = issue hour on D-1; target = hour `target_hour` (UTC) of day D."""
    s = _scales(d)
    y = d["co2"].to_numpy()
    o = day_ahead_origins(d)
    h = 24 - DA_ISSUE_HOUR + target_hour        # 9 ... 32
    T = o + h
    k = int(np.ceil(h / 24))
    diurnal = T - 24 * k
    yz = _z(y, s["co2"])
    wz = _z(d["wind_da"].to_numpy(), s["wind_da"])
    sz = _z(d["solar_da"].to_numpy(), s["solar_da"])
    pz = _z(d["price_da"].to_numpy(), s["price_da"])
    # mean price of the target day (published with the day-ahead auction, before issue)
    pday = pd.Series(pz, index=d.index).groupby(d.index.normalize()).transform("mean").to_numpy()
    wday = pd.Series(wz, index=d.index).groupby(d.index.normalize()).transform("mean").to_numpy()
    last24 = pd.Series(yz).rolling(24, min_periods=18).mean().to_numpy()
    X = np.column_stack([
        np.ones(len(o)),
        yz[o], last24[o], np.where(diurnal >= 0, yz[np.clip(diurnal, 0, None)], np.nan),
        wz[T], sz[T], pz[T], pday[T], wday[T],
        pz[T] - pday[T],
        spread_features(d, area)[T],
    ])
    X[:2] = np.nan
    return d.index[o], o, h, X, y[T], (s["co2"])
