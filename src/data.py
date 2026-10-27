"""Download Energinet (Energi Data Service) data and build the hourly modelling table.

Sources (all public, no key):
  CO2Emis        5-min realised CO2 intensity of consumed electricity [g/kWh]
  CO2EmisProg    5-min schedule-based CO2 prognosis (released ~15:00 local for the
                 next day, then updated intraday; the archive keeps the last update)
  Forecasts_Hour hourly wind/solar forecasts; only the day-ahead column is used
  Elspotprices   hourly day-ahead price (until 2025-09-30)
  DayAheadPrices 15-min day-ahead price (from 2025-10-01), averaged to hours

All timestamps are UTC interval starts.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://api.energidataservice.dk/dataset/"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
AREAS = ("DK1", "DK2")
NEIGHBOURS = ("DE", "NO2", "SE3", "SE4")   # interconnected bidding zones


def fetch(dataset: str, start: str, end: str, areas=AREAS, columns: str | None = None) -> pd.DataFrame:
    params = {"start": start, "end": end, "limit": 0,
              "filter": json.dumps({"PriceArea": list(areas)})}
    if columns:
        params["columns"] = columns
    url = API + dataset + "?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                recs = json.load(r)["records"]
            return pd.DataFrame.from_records(recs)
        except urllib.error.HTTPError as e:  # rate limit: back off
            if e.code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"failed: {url}")


def month_starts(start: str, end: str):
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    edges = list(pd.date_range(s, e, freq="MS"))
    if edges[0] != s:
        edges = [s] + edges
    if edges[-1] != e:
        edges.append(e)
    return [(a.strftime("%Y-%m-%dT%H:%M"), b.strftime("%Y-%m-%dT%H:%M")) for a, b in zip(edges[:-1], edges[1:])]


def year_windows(start: str, end: str):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    edges = sorted({s, e, *[t for t in pd.date_range(s, e, freq="YS") if s < t < e]})
    return [(a.strftime("%Y-%m-%dT%H:%M"), b.strftime("%Y-%m-%dT%H:%M")) for a, b in zip(edges[:-1], edges[1:])]


def download(start: str, end: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    specs = {
        "CO2Emis": "Minutes5UTC,PriceArea,CO2Emission",
        "CO2EmisProg": "Minutes5UTC,PriceArea,CO2Emission",
        "Forecasts_Hour": "HourUTC,PriceArea,ForecastType,ForecastDayAhead",
        "Elspotprices": "HourUTC,PriceArea,SpotPriceEUR",
        "DayAheadPrices": "TimeUTC,PriceArea,DayAheadPriceEUR",
        "Elspotprices_nb": "HourUTC,PriceArea,SpotPriceEUR",
        "DayAheadPrices_nb": "TimeUTC,PriceArea,DayAheadPriceEUR",
    }
    for name, cols in specs.items():
        ds, areas = (name[:-3], NEIGHBOURS) if name.endswith("_nb") else (name, AREAS)
        out = RAW / f"{name}.csv.gz"
        if out.exists():
            print("exists", out.name)
            continue
        parts = []
        windows = month_starts(start, end)
        if ds in ("Elspotprices", "DayAheadPrices"):  # small datasets: one call per year
            windows = year_windows(start, end)
        for a, b in windows:
            if ds == "Elspotprices":
                b = min(b, "2025-10-01T00:00")
            if ds == "DayAheadPrices":
                a = max(a, "2025-10-01T00:00")
            if a >= b:
                continue
            df = fetch(ds, a, b, areas=areas, columns=cols)
            print(ds, a, len(df), flush=True)
            parts.append(df)
            time.sleep(1.0)
        pd.concat(parts, ignore_index=True).to_csv(out, index=False, compression="gzip")


def _hourly_mean(df: pd.DataFrame, tcol: str, vcol: str) -> pd.DataFrame:
    d = df.copy()
    d["hour"] = pd.to_datetime(d[tcol]).dt.floor("h")
    g = d.groupby(["PriceArea", "hour"])[vcol]
    out = g.mean().to_frame(vcol)
    out["n"] = g.count()
    return out


def build(area: str, start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="h", inclusive="left", name="hour")
    co2 = pd.read_csv(RAW / "CO2Emis.csv.gz")
    prog = pd.read_csv(RAW / "CO2EmisProg.csv.gz")
    fh = pd.read_csv(RAW / "Forecasts_Hour.csv.gz")
    sp1 = pd.read_csv(RAW / "Elspotprices.csv.gz")
    sp2 = pd.read_csv(RAW / "DayAheadPrices.csv.gz")

    c = _hourly_mean(co2, "Minutes5UTC", "CO2Emission").loc[area]
    # an hour needs at least 9 of its 12 five-minute values
    y = c["CO2Emission"].where(c["n"] >= 9).reindex(idx)
    p = _hourly_mean(prog, "Minutes5UTC", "CO2Emission").loc[area]["CO2Emission"].reindex(idx)

    fh = fh[fh.PriceArea == area].copy()
    fh["hour"] = pd.to_datetime(fh["HourUTC"])
    w = fh.pivot_table(index="hour", columns="ForecastType", values="ForecastDayAhead", aggfunc="mean")
    w = w.reindex(idx)
    wind = w[["Offshore Wind", "Onshore Wind"]].sum(axis=1, min_count=2)
    solar = w["Solar"]

    s1 = sp1[sp1.PriceArea == area].assign(hour=lambda d: pd.to_datetime(d["HourUTC"])).set_index("hour")["SpotPriceEUR"]
    s2 = _hourly_mean(sp2, "TimeUTC", "DayAheadPriceEUR").loc[area]["DayAheadPriceEUR"]
    price = pd.concat([s1, s2]).groupby(level=0).mean().reindex(idx)

    n1 = pd.read_csv(RAW / "Elspotprices_nb.csv.gz")
    n2 = pd.read_csv(RAW / "DayAheadPrices_nb.csv.gz")
    nb = {}
    for z in NEIGHBOURS:
        a1 = n1[n1.PriceArea == z].assign(hour=lambda d: pd.to_datetime(d["HourUTC"])).set_index("hour")["SpotPriceEUR"]
        a2 = _hourly_mean(n2, "TimeUTC", "DayAheadPriceEUR").loc[z]["DayAheadPriceEUR"]
        nb[f"price_{z}"] = pd.concat([a1, a2]).groupby(level=0).mean().reindex(idx)

    out = pd.DataFrame({"co2": y, "co2_prog": p, "wind_da": wind, "solar_da": solar, "price_da": price, **nb})
    out.index.name = "hour_utc"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01T00:00")
    ap.add_argument("--end", default="2026-10-01T00:00")
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    if not a.skip_download:
        download(a.start, a.end)
    PROC.mkdir(parents=True, exist_ok=True)
    for area in AREAS:
        df = build(area, a.start, a.end)
        df.round(3).to_csv(PROC / f"hourly_{area}.csv.gz", compression="gzip")
        print(area, df.shape, df.isna().sum().to_dict())


if __name__ == "__main__":
    main()
