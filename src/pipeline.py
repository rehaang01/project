"""
pipeline.py — assembles leak-free feature matrices from base + target-derived features.

Two entry points:
  build_oof(train_df, seed, n_splits)       -> X for the model's OWN training rows (OOF, no self-leak)
  build_fit_transform(fit_df, txf_df)       -> X for valid/test rows (encoders fit on fit_df)

Target-derived features are smoothed group-means of `demand` at several granularities, each with a
hierarchical fallback to coarser groups then the global mean. Finer granularities (slot, geohash x
hour) memorize seen-location signal; coarser ones (p5, p4, RoadType) regularize and cover unseen keys.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import features as F

SEED = F.SEED
TARGET = F.TARGET

# Each entry: (feature_name, [groupby_cols], smoothing_m, [fallback_feature_names...])
# Fallbacks are resolved in order; the first non-null wins, else global mean.
TE_SPECS = [
    ("te_slot",      ["geohash", "tmin"], 10.0, ["te_geohr", "te_geo", "te_p5", "te_p4"]),
    ("te_geohr",     ["geohash", "hour"], 12.0, ["te_geo", "te_p5", "te_p4"]),
    ("te_geo",       ["geohash"],          8.0, ["te_p5", "te_p4"]),
    ("te_p5",        ["p5"],              20.0, ["te_p4"]),
    ("te_p4",        ["p4"],              30.0, []),
    ("te_road",      ["RoadType_ord"],    50.0, []),
    ("te_road_hour", ["RoadType_ord", "hour"], 30.0, ["te_road"]),
    # robust regional/structural time patterns (enough samples/cell to be stable, not memorization)
    ("te_p5_hour",   ["p5", "hour"],      25.0, ["te_road_hour", "te_p5"]),
    ("te_p4_hour",   ["p4", "hour"],      30.0, ["te_road_hour", "te_p4"]),
    ("te_road_lanes", ["RoadType_ord", "NumberofLanes"], 30.0, ["te_road"]),
]
TE_NAMES = [s[0] for s in TE_SPECS]

# Derived features (order MUST match the order they are added in _transform).
DERIVED_NAMES = ["d48_shift_adj", "geo_count", "mult_geo_p5hr", "mult_geo_roadhr"]
MULT_NAMES = ["mult_geo_p5hr", "mult_geo_roadhr"]

FEATURE_NAMES = F.BASE_NUMERIC + TE_NAMES + DERIVED_NAMES

# ---- Canonical feature sets (decided by H1/H2/H3 ablation) ----
# ROBUST_SET maximizes the faithful forward metric (H1) AND the spatial guardrail (H3); finer
# memorization features only help the optimistic same-day H2 and hurt forward generalization.
ROBUST_SET = F.BASE_NUMERIC + ["te_p5", "te_p4", "te_road", "te_road_hour"]
# MEMO_SET = honest daytime-memorization features (board ~88.3-88.4).
MEMO_SET = ROBUST_SET + ["te_geo", "te_geohr", "te_slot", "d48_shift_adj", "geo_count"]
# MEMO_PLUS adds the multiplicative E[demand|geohash,time] estimates (push toward the ~0.90 ceiling).
MEMO_PLUS = MEMO_SET + MULT_NAMES


# Global smoothing multiplier. <1 => memorize harder (lighter shrinkage toward the mean).
# The board showed daytime demand repeats strongly from day 48, so lighter smoothing helps here.
SMOOTH_SCALE = 1.0


def _smoothed_map(df: pd.DataFrame, cols: list, m: float, gm: float):
    m = m * SMOOTH_SCALE
    g = df.groupby(cols)[TARGET].agg(["mean", "count"])
    return (g["mean"] * g["count"] + gm * m) / (g["count"] + m)


def _map_series(txf: pd.DataFrame, enc: pd.Series, cols: list) -> pd.Series:
    if len(cols) == 1:
        return txf[cols[0]].map(enc)
    idx = txf.set_index(cols).index
    return pd.Series(idx.map(enc), index=txf.index)


def _fit_encoders(fit_df: pd.DataFrame):
    gm = fit_df[TARGET].mean()
    encs = {name: _smoothed_map(fit_df, cols, m, gm) for name, cols, m, _ in TE_SPECS}
    geo_count = fit_df.groupby("geohash")[TARGET].count()
    return {"gm": gm, "encs": encs, "geo_count": geo_count}


def _transform(txf: pd.DataFrame, state: dict) -> pd.DataFrame:
    gm = state["gm"]
    encs = state["encs"]
    spec_by_name = {s[0]: s for s in TE_SPECS}
    raw = {}  # raw mapped (with NaN) before fallback
    for name, cols, m, _ in TE_SPECS:
        raw[name] = _map_series(txf, encs[name], cols)
    out = pd.DataFrame(index=txf.index)
    for name, cols, m, fbacks in TE_SPECS:
        s = raw[name].copy()
        for fb in fbacks:
            s = s.fillna(raw[fb])
        out[name] = s.fillna(gm)
    # derived
    out["d48_shift_adj"] = 0.032 + 1.255 * out["te_slot"]
    out["geo_count"] = txf["geohash"].map(state["geo_count"]).fillna(0.0)
    # multiplicative estimates of E[demand | geohash, time]: geohash level x time-shape.
    # Borrows strength (geo level from ~55 obs, shape from a whole region/road) -> low-variance
    # estimate even for sparse cells; targets the ~0.90 honest ceiling.
    eps = 1e-4
    out["mult_geo_p5hr"] = out["te_geo"] * out["te_p5_hour"] / (out["te_p5"] + eps)
    out["mult_geo_roadhr"] = out["te_geo"] * out["te_road_hour"] / (out["te_road"] + eps)
    return out


def _assemble(txf: pd.DataFrame, te_df: pd.DataFrame) -> pd.DataFrame:
    base = txf[F.BASE_NUMERIC].reset_index(drop=True)
    te = te_df.reset_index(drop=True)
    X = pd.concat([base, te], axis=1)
    return X[FEATURE_NAMES]


def build_fit_transform(fit_df: pd.DataFrame, txf_df: pd.DataFrame) -> pd.DataFrame:
    """Encoders fit on fit_df, applied to txf_df. Use for validation / test."""
    state = _fit_encoders(fit_df)
    te_df = _transform(txf_df, state)
    return _assemble(txf_df, te_df)


def build_oof(train_df: pd.DataFrame, seed: int = SEED, n_splits: int = 5) -> pd.DataFrame:
    """OOF target features for the training rows themselves (no row sees its own target)."""
    te_oof = pd.DataFrame(index=np.arange(len(train_df)), columns=TE_NAMES + DERIVED_NAMES, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pos = np.arange(len(train_df))
    for tr_i, va_i in kf.split(pos):
        state = _fit_encoders(train_df.iloc[tr_i])
        te_df = _transform(train_df.iloc[va_i], state)
        te_oof.iloc[va_i] = te_df.to_numpy()
    te_oof = te_oof.astype(float)
    return _assemble(train_df, te_oof)


def prepare(df: pd.DataFrame, fit_df_for_impute: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add base features + static imputation. If fit_df_for_impute is None, impute from df itself."""
    df = F.add_base_features(df)
    src = F.add_base_features(fit_df_for_impute) if fit_df_for_impute is not None else df
    df = F.impute_static(src, df)
    return df
