from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


COMP_DIR = Path(__file__).resolve().parent
ROOT = COMP_DIR.parent

UPSTREAM_CACHE = (
    ROOT
    / "competition_treasury_yieldcurve_inversion20d_regime_shift"
    / "_cache"
    / "par-yield-curve-rates-1990-2023.csv"
)

N_BINS = 12
EPS = 1e-12

ID_COLUMN = "row_id"
SEGMENT_COLUMN = "segment_id"
ERA_COLUMN = "event_era"
MONTH_COLUMN = "month"
DOW_COLUMN = "dow"
MISSING_COLUMN = "missing_count"

TARGET_COLUMN = "target_bear_steepen_shock_10d"
PRED_COLUMN = "pred_bear_steepen_shock_10d"
SLICE_COLUMN = "slice_high_rate_vol"

MIN_TRAIN_ROWS = 10_000


def _hash_bucket(text: str, mod: int) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % mod


def _event_era(year: int) -> int:
    if year <= 1999:
        return 0
    if year <= 2007:
        return 1
    if year <= 2015:
        return 2
    return 3


def _quantile_edges(train_values: np.ndarray, n_bins: int) -> np.ndarray:
    train_values = train_values[np.isfinite(train_values)]
    if train_values.size == 0:
        return np.array([], dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(train_values, qs).astype(float)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + EPS
    return edges


def _bin_with_edges(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], -1, dtype=int)
    mask = np.isfinite(values)
    if edges.size == 0:
        out[mask] = 0
        return out
    out[mask] = np.digitize(values[mask], edges, right=False).astype(int)
    return out


def _make_bins(df_all: pd.DataFrame, df_train: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    edges_map: Dict[str, List[float]] = {}
    for c in cols:
        edges = _quantile_edges(df_train[c].to_numpy(dtype=float), N_BINS)
        edges_map[c] = edges.tolist()
        df_all[f"{c}_bin"] = _bin_with_edges(df_all[c].to_numpy(dtype=float), edges)
    return df_all, edges_map


def main() -> None:
    if not UPSTREAM_CACHE.exists():
        raise FileNotFoundError(f"Missing upstream cache file: {UPSTREAM_CACHE}")

    raw = pd.read_csv(UPSTREAM_CACHE, low_memory=False)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    if "date" not in raw.columns:
        raise RuntimeError("Expected a 'date' column in upstream yield curve CSV.")

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Pick a stable subset of maturities.
    # Upstream uses names like '3_mo', '2_yr', '10_yr', '30_yr' after normalization.
    def _col(name: str) -> str:
        return name

    needed = ["3_mo", "2_yr", "10_yr", "30_yr", "1_yr", "5_yr"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build a panel over multiple curve segments to ensure enough rows and avoid
    # a "single-series is too small" rejection risk.
    segments = [
        {"name": "10y2y", "short": "2_yr", "long": "10_yr"},
        {"name": "10y3m", "short": "3_mo", "long": "10_yr"},
        {"name": "30y10y", "short": "10_yr", "long": "30_yr"},
        {"name": "5y2y", "short": "2_yr", "long": "5_yr"},
    ]

    for seg in segments:
        s = seg["short"]
        l = seg["long"]
        spread_col = f"s_{seg['name']}"
        df[spread_col] = df[l] - df[s]
        for c in [s, l, spread_col]:
            df[f"{c}_chg1"] = df[c].diff(1)
            df[f"{c}_chg5"] = df[c].diff(5)
            df[f"{c}_std20"] = df[c].rolling(20, min_periods=10).std()

        df[f"delta_{spread_col}_10d"] = df[spread_col].shift(-10) - df[spread_col]
        df[f"delta_{s}_10d"] = df[s].shift(-10) - df[s]

    df["year"] = df["date"].dt.year.astype(int)
    df[MONTH_COLUMN] = df["date"].dt.month.astype(int)
    df[DOW_COLUMN] = df["date"].dt.dayofweek.astype(int)  # 0=Mon .. 6=Sun (data is trading days)
    df[ERA_COLUMN] = df["year"].map(_event_era).astype(int)

    # Define test: newer era emphasized, deterministic bridge mix.
    bridge_bucket = df.apply(lambda r: _hash_bucket(f"{r['year']}_{int(r[MONTH_COLUMN])}_{int(r[DOW_COLUMN])}_{r.name}", 10), axis=1)
    is_test = (df["year"] >= 2018) | ((df["year"] >= 2016) & (df["year"] <= 2017) & (bridge_bucket == 0))
    df["is_test"] = is_test.astype(int)

    panel_rows = []
    taus: Dict[str, Dict[str, float]] = {}

    # Feature schema for panel rows (segment-specific but column-stable).
    panel_features = [
        "short_yield",
        "long_yield",
        "spread",
        "short_chg1",
        "short_chg5",
        "long_chg1",
        "long_chg5",
        "spread_chg1",
        "spread_chg5",
        "short_std20",
        "spread_std20",
    ]

    for seg in segments:
        seg_name = seg["name"]
        s = seg["short"]
        l = seg["long"]
        spread_col = f"s_{seg_name}"

        seg_df = df.copy()
        seg_df["short_yield"] = seg_df[s]
        seg_df["long_yield"] = seg_df[l]
        seg_df["spread"] = seg_df[spread_col]
        seg_df["short_chg1"] = seg_df[f"{s}_chg1"]
        seg_df["short_chg5"] = seg_df[f"{s}_chg5"]
        seg_df["long_chg1"] = seg_df[f"{l}_chg1"]
        seg_df["long_chg5"] = seg_df[f"{l}_chg5"]
        seg_df["spread_chg1"] = seg_df[f"{spread_col}_chg1"]
        seg_df["spread_chg5"] = seg_df[f"{spread_col}_chg5"]
        seg_df["short_std20"] = seg_df[f"{s}_std20"]
        seg_df["spread_std20"] = seg_df[f"{spread_col}_std20"]

        seg_df["delta_spread_10d"] = seg_df[f"delta_{spread_col}_10d"]
        seg_df["delta_short_10d"] = seg_df[f"delta_{s}_10d"]

        seg_df = seg_df[seg_df["delta_spread_10d"].notna() & seg_df["delta_short_10d"].notna()].copy()
        seg_df[SEGMENT_COLUMN] = int(hashlib.md5(f"segment::{seg_name}".encode("utf-8")).hexdigest()[:15], 16)

        seg_train = seg_df[seg_df["is_test"] == 0].copy()
        if seg_train.empty:
            continue

        tau_spread = float(np.nanquantile(seg_train["delta_spread_10d"].to_numpy(dtype=float), 0.85))
        tau_short = float(np.nanquantile(seg_train["delta_short_10d"].to_numpy(dtype=float), 0.75))
        taus[seg_name] = {"tau_spread_p85_train": tau_spread, "tau_short_p75_train": tau_short}

        seg_df[TARGET_COLUMN] = ((seg_df["delta_spread_10d"] >= tau_spread) & (seg_df["delta_short_10d"] >= tau_short)).astype(int)

        seg_df[MISSING_COLUMN] = seg_df[panel_features].isna().sum(axis=1).astype(int)
        panel_rows.append(seg_df[[SEGMENT_COLUMN, ERA_COLUMN, MONTH_COLUMN, DOW_COLUMN, MISSING_COLUMN, *panel_features, TARGET_COLUMN, "is_test"]])

    if not panel_rows:
        raise RuntimeError("No segment rows produced; cannot build dataset.")

    panel = pd.concat(panel_rows, axis=0, ignore_index=True)
    train_rows = panel[panel["is_test"] == 0].copy()
    if train_rows.empty:
        raise RuntimeError("No training rows after split; cannot build dataset.")

    # Bin all panel features using training rows only.
    panel_all, edges_map = _make_bins(panel, train_rows, panel_features)

    # Slice: high short-rate volatility (derived from visible bin).
    panel_all[SLICE_COLUMN] = (panel_all["short_std20_bin"] >= 9).astype(int)

    keep = [SEGMENT_COLUMN, ERA_COLUMN, MONTH_COLUMN, DOW_COLUMN, MISSING_COLUMN] + [f"{c}_bin" for c in panel_features] + [SLICE_COLUMN]
    out = panel_all[keep + [TARGET_COLUMN, "is_test"]].copy()
    out[ID_COLUMN] = np.arange(out.shape[0], dtype=np.int64)

    train = out[out["is_test"] == 0].drop(columns=["is_test"]).copy()
    test = out[out["is_test"] == 1].drop(columns=["is_test", TARGET_COLUMN]).copy()
    solution = out[out["is_test"] == 1][[ID_COLUMN, TARGET_COLUMN, SLICE_COLUMN]].copy()

    train = train.sort_values(ID_COLUMN).reset_index(drop=True)
    test = test.sort_values(ID_COLUMN).reset_index(drop=True)
    solution = solution.sort_values(ID_COLUMN).reset_index(drop=True)

    if len(train) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Train set too small ({len(train)} rows). Target is >= {MIN_TRAIN_ROWS} rows to avoid rejection risk.")

    sample = test[[ID_COLUMN]].copy()
    sample[PRED_COLUMN] = 0.5

    perfect = solution[[ID_COLUMN, TARGET_COLUMN]].copy()
    perfect[PRED_COLUMN] = perfect[TARGET_COLUMN].astype(float)
    perfect = perfect[[ID_COLUMN, PRED_COLUMN]]

    train.to_csv(COMP_DIR / "train.csv", index=False)
    test.to_csv(COMP_DIR / "test.csv", index=False)
    solution.to_csv(COMP_DIR / "solution.csv", index=False)
    sample.to_csv(COMP_DIR / "sample_submission.csv", index=False)
    perfect.to_csv(COMP_DIR / "perfect_submission.csv", index=False)

    meta = {
        "upstream_cache": str(UPSTREAM_CACHE),
        "n_bins": N_BINS,
        "horizon_trading_days": 10,
        "segment_thresholds_train": taus,
        "bin_edges": edges_map,
        "slice_definition": {
            "slice_column": SLICE_COLUMN,
            "rule": "slice_high_rate_vol = 1 if short_std20_bin >= 9 (n_bins=12).",
        },
        "row_counts": {"train": int(len(train)), "test": int(len(test))},
        "positive_rate": {
            "train": float(train[TARGET_COLUMN].mean()) if len(train) else None,
            "test": float(solution[TARGET_COLUMN].mean()) if len(solution) else None,
        },
    }
    (COMP_DIR / "build_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

    print("Wrote competition files to", COMP_DIR)
    print("train rows:", len(train), "test rows:", len(test))
    print("train positive rate:", float(train[TARGET_COLUMN].mean()) if len(train) else float("nan"))
    print("test positive rate:", float(solution[TARGET_COLUMN].mean()) if len(solution) else float("nan"))


if __name__ == "__main__":
    main()

