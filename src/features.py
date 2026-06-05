"""
features.py — leak-free feature pipeline for the Traffic Demand Prediction challenge.

Design contract
---------------
* "Base" features depend only on the row's own inputs (time, geohash decode, categoricals,
  missing flags). They are computed once on any dataframe and never touch the target.
* "Target-derived" features (target-encodings, day-48 lookups) are ALWAYS fit on a designated
  `fit_df` and applied to a `transform_df`. For the model's own training rows we instead use
  out-of-fold (OOF) encodings so a row never sees its own target. This is the single most
  important guard against the metric-bloat / overfitting the brief warns about.

Key data facts this module encodes (all independently verified on train.csv):
* target `demand` in (0, 1], right-skewed, mean ~0.094.
* `geohash` 6-char; prefix-5 (56 regions) / prefix-4 (6 regions) are robust fallbacks.
* time is 15-min slots; tmin in [0, 1425]; clear time-of-day cycle -> sin/cos encoding.
* RoadType is the strongest robust signal (Highway 0.61 >> Street 0.27 >> Residential 0.057).
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SEED = 42
TARGET = "demand"

# Project root = parent of this file's directory (src/). Lets scripts run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "dataset")

# Ordinal maps (RoadType ordered by mean demand: Residential < Street < Highway)
ROADTYPE_ORD = {"Residential": 0, "Street": 1, "Highway": 2}
LARGEVEH_ORD = {"Not Allowed": 0, "Allowed": 1}
LANDMARK_ORD = {"No": 0, "Yes": 1}
WEATHER_CATS = ["Sunny", "Rainy", "Foggy", "Snowy", "Missing"]

# Geohash base-32 alphabet -> used to decode lat/lon without external deps as a fallback.
_GEO_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"


# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------
def load_raw(data_dir: str = DATA_DIR):
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    return train, test


# --------------------------------------------------------------------------------------
# geohash decode (pure-python, matches pygeohash; avoids a hard dependency)
# --------------------------------------------------------------------------------------
def _decode_geohash(gh: str):
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    even = True
    for ch in gh:
        idx = _GEO_B32.index(ch)
        for bit in (16, 8, 4, 2, 1):
            if even:
                mid = (lon_lo + lon_hi) / 2
                if idx & bit:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if idx & bit:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2


def _decode_geohash_frame(geohashes: pd.Series):
    uniq = geohashes.unique()
    dec = {g: _decode_geohash(g) for g in uniq}
    lat = geohashes.map(lambda g: dec[g][0])
    lon = geohashes.map(lambda g: dec[g][1])
    return lat.to_numpy(), lon.to_numpy()


# --------------------------------------------------------------------------------------
# Base (target-independent) features
# --------------------------------------------------------------------------------------
def parse_tmin(ts: str) -> int:
    h, m = ts.split(":")
    return int(h) * 60 + int(m)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # --- time ---
    df["tmin"] = df["timestamp"].map(parse_tmin)
    df["hour"] = df["tmin"] // 60
    df["sin_t"] = np.sin(2 * np.pi * df["tmin"] / 1440.0)
    df["cos_t"] = np.cos(2 * np.pi * df["tmin"] / 1440.0)
    df["abs_t"] = df["day"] * 1440 + df["tmin"]

    # --- spatial ---
    df["p5"] = df["geohash"].str[:5]
    df["p4"] = df["geohash"].str[:4]
    lat, lon = _decode_geohash_frame(df["geohash"])
    df["lat"] = lat
    df["lon"] = lon

    # --- missing flags (before imputation) ---
    df["RoadType_missing"] = df["RoadType"].isna().astype(int)
    df["Temperature_missing"] = df["Temperature"].isna().astype(int)
    df["Weather_missing"] = df["Weather"].isna().astype(int)

    # --- categorical ordinals ---
    df["RoadType_ord"] = df["RoadType"].map(ROADTYPE_ORD)
    df["LargeVehicles_ord"] = df["LargeVehicles"].map(LARGEVEH_ORD).fillna(0).astype(int)
    df["Landmarks_ord"] = df["Landmarks"].map(LANDMARK_ORD).fillna(0).astype(int)

    # --- weather one-hot (incl. Missing) ---
    w = df["Weather"].fillna("Missing")
    for c in WEATHER_CATS:
        df[f"Weather_{c}"] = (w == c).astype(int)

    # --- lanes numeric; Temperature kept (NaN allowed for LightGBM, median elsewhere) ---
    df["NumberofLanes"] = pd.to_numeric(df["NumberofLanes"], errors="coerce")
    return df


def impute_static(fit_df: pd.DataFrame, *frames: pd.DataFrame):
    """Compute imputation stats on fit_df, apply to every frame. Returns transformed frames.

    RoadType_ord  <- per-geohash mode then global mode
    Temperature   <- global median (a *_imp copy; raw column also kept with NaN for LGBM)
    """
    # global stats from fit_df
    global_road = fit_df["RoadType_ord"].mode(dropna=True)
    global_road = int(global_road.iloc[0]) if len(global_road) else 0
    temp_median = fit_df["Temperature"].median()
    # per-geohash road mode
    gh_road = (
        fit_df.dropna(subset=["RoadType_ord"])
        .groupby("geohash")["RoadType_ord"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
    )

    out = []
    for f in frames:
        f = f.copy()
        filled = f["RoadType_ord"].fillna(f["geohash"].map(gh_road))
        f["RoadType_ord"] = filled.fillna(global_road).astype(int)
        f["Temperature_imp"] = f["Temperature"].fillna(temp_median)
        out.append(f)
    return out if len(out) > 1 else out[0]


# --------------------------------------------------------------------------------------
# Smoothed target encoding (leak-free)
# --------------------------------------------------------------------------------------
def _smooth_map(fit_df: pd.DataFrame, col: str, m: float, target: str = TARGET):
    gm = fit_df[target].mean()
    agg = fit_df.groupby(col)[target].agg(["mean", "count"])
    enc = (agg["mean"] * agg["count"] + gm * m) / (agg["count"] + m)
    return enc, gm


def fit_target_encoders(fit_df: pd.DataFrame, cols_m: dict, target: str = TARGET):
    """cols_m: {col: m}. Returns ({col: enc_series}, global_mean)."""
    encoders = {}
    gm = fit_df[target].mean()
    for col, m in cols_m.items():
        enc, _ = _smooth_map(fit_df, col, m, target)
        encoders[col] = enc
    return encoders, gm


def apply_target_encoders(df: pd.DataFrame, encoders: dict, gm: float,
                          fallbacks: dict | None = None) -> pd.DataFrame:
    """Add `<col>_te`. Unseen keys fall back to `fallbacks[col]` map (e.g. coarser region)
    then to the global mean."""
    df = df.copy()
    fallbacks = fallbacks or {}
    for col, enc in encoders.items():
        te = df[col].map(enc)
        if col in fallbacks:
            te = te.fillna(df[fallbacks[col][0]].map(fallbacks[col][1]))
        df[f"{col}_te"] = te.fillna(gm)
    return df


def oof_target_encode(train_df: pd.DataFrame, col: str, m: float,
                      n_splits: int = 5, seed: int = SEED, target: str = TARGET) -> np.ndarray:
    """Out-of-fold smoothed TE for the training rows themselves (no row sees its own target)."""
    oof = np.full(len(train_df), np.nan)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx = np.arange(len(train_df))
    for tr_i, va_i in kf.split(idx):
        enc, gm = _smooth_map(train_df.iloc[tr_i], col, m, target)
        oof[va_i] = train_df.iloc[va_i][col].map(enc).fillna(gm).to_numpy()
    return oof


# --------------------------------------------------------------------------------------
# Day-48 history lookups (the legitimate "memorization" boosters)
# --------------------------------------------------------------------------------------
def build_history_lookups(history_df: pd.DataFrame, target: str = TARGET):
    """history_df is the reference day (typically day 48). Returns a dict of lookup maps."""
    gm = history_df[target].mean()
    return {
        "gm": gm,
        "slot": history_df.groupby(["geohash", "tmin"])[target].mean(),
        "geohr": history_df.groupby(["geohash", "hour"])[target].mean(),
        "geo_mean": history_df.groupby("geohash")[target].mean(),
        "geo_median": history_df.groupby("geohash")[target].median(),
        "geo_std": history_df.groupby("geohash")[target].std(),
        "geo_count": history_df.groupby("geohash")[target].count(),
    }


def apply_history_lookups(df: pd.DataFrame, maps: dict) -> pd.DataFrame:
    df = df.copy()
    gm = maps["gm"]
    geo_mean = df["geohash"].map(maps["geo_mean"])
    # exact-slot lookup (geohash, tmin) on history day; fallback geohash mean -> global
    slot = pd.Series(df.set_index(["geohash", "tmin"]).index.map(maps["slot"]), index=df.index)
    df["d48_slot"] = slot.fillna(geo_mean).fillna(gm)
    # affine-corrected day-to-day shift (coeffs measured on train: 0.032 + 1.255*d48)
    df["d48_shift_adj"] = 0.032 + 1.255 * df["d48_slot"]
    # geohash x hour
    geohr = pd.Series(df.set_index(["geohash", "hour"]).index.map(maps["geohr"]), index=df.index)
    df["d48_geohr"] = geohr.fillna(geo_mean).fillna(gm)
    # geohash-level day-48 stats
    df["d48_geo_mean"] = geo_mean.fillna(gm)
    df["d48_geo_median"] = df["geohash"].map(maps["geo_median"]).fillna(gm)
    df["d48_geo_std"] = df["geohash"].map(maps["geo_std"]).fillna(0.0)
    df["d48_geo_count"] = df["geohash"].map(maps["geo_count"]).fillna(0.0)
    return df


# --------------------------------------------------------------------------------------
# Feature column groups
# --------------------------------------------------------------------------------------
BASE_NUMERIC = [
    "tmin", "hour", "sin_t", "cos_t", "lat", "lon",
    "NumberofLanes", "Temperature_imp",
    "RoadType_ord", "LargeVehicles_ord", "Landmarks_ord",
    "RoadType_missing", "Temperature_missing", "Weather_missing",
] + [f"Weather_{c}" for c in WEATHER_CATS]

# target-encoding columns and their smoothing strength (geohash light -> memorization kept)
TE_COLS_M = {"geohash": 8.0, "p5": 20.0, "p4": 30.0, "RoadType": 50.0}

TE_FEATURES = [f"{c}_te" for c in TE_COLS_M]

HISTORY_FEATURES = [
    "d48_slot", "d48_shift_adj", "d48_geohr",
    "d48_geo_mean", "d48_geo_median", "d48_geo_std", "d48_geo_count",
]

# Robust-core (generalizes to new locations) vs memorization boosters (seen locations)
ROBUST_FEATURES = BASE_NUMERIC + ["p5_te", "p4_te", "RoadType_te"]
MEMORIZATION_FEATURES = ["geohash_te", "lat", "lon"] + HISTORY_FEATURES


def all_features(use_history: bool = True) -> list:
    feats = BASE_NUMERIC + TE_FEATURES
    if use_history:
        feats = feats + HISTORY_FEATURES
    # de-dup preserving order
    seen, out = set(), []
    for f in feats:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out
