"""
cv.py — validation harness.

Headline metrics (the anti-overfit core):
  H1  temporal forward   : train day 48 -> validate day 49 (captures day-to-day drift)   [PRIMARY]
  H2  daytime forward    : hold out a day-48 daytime block (test hours), train on the rest [PRIMARY]
  H3  spatial GroupKFold : predict unseen geohashes                          [GUARDRAIL >= ~0.70]

All target-derived features are OOF on the model's own training rows and fit-on-fit-set for the
held-out rows, via pipeline.build_oof / build_fit_transform.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import features as F
import pipeline as P

SEED = F.SEED
TARGET = F.TARGET

# Test window on day 49 is 02:15-13:45 -> minutes [135, 825]. We mirror it inside day 48 for H2.
TEST_TMIN_LO, TEST_TMIN_HI = 135, 825


def r2(actual, pred) -> float:
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    return float(1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum())


# --------------------------------------------------------------------------------------
# model factory (regularized LightGBM by default)
# --------------------------------------------------------------------------------------
def lgbm_factory(params: dict | None = None):
    import lightgbm as lgb
    base = dict(
        objective="regression", n_estimators=1200, learning_rate=0.03,
        num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=1.0,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    if params:
        base.update(params)
    return lgb.LGBMRegressor(**base)


def xgb_factory(params: dict | None = None):
    import xgboost as xgb
    base = dict(
        objective="reg:squarederror", n_estimators=600, learning_rate=0.03,
        max_depth=4, min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
        reg_lambda=2.0, reg_alpha=1.0, random_state=SEED, n_jobs=-1, verbosity=0,
    )
    if params:
        base.update(params)
    return xgb.XGBRegressor(**base)


def cat_factory(params: dict | None = None):
    from catboost import CatBoostRegressor
    base = dict(
        loss_function="RMSE", iterations=800, learning_rate=0.03, depth=5,
        l2_leaf_reg=6.0, subsample=0.8, random_seed=SEED, verbose=False,
        allow_writing_files=False,
    )
    if params:
        base.update(params)
    return CatBoostRegressor(**base)


def ridge_factory(params: dict | None = None):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    alpha = (params or {}).get("alpha", 1.0)
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=SEED))


def fit_predict(model_factory, fit_df: pd.DataFrame, valid_df: pd.DataFrame,
                seed: int = SEED, n_splits: int = 5, feat_cols: list | None = None):
    """Train on fit_df (OOF features), predict valid_df (encoders fit on full fit_df)."""
    Xtr = P.build_oof(fit_df, seed=seed, n_splits=n_splits)
    ytr = fit_df[TARGET].to_numpy()
    Xva = P.build_fit_transform(fit_df, valid_df)
    if feat_cols is not None:
        Xtr, Xva = Xtr[feat_cols], Xva[feat_cols]
    model = model_factory()
    model.fit(Xtr, ytr)
    pred = np.clip(model.predict(Xva), 0.0, 1.0)
    return pred, model


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------
def split_H1(train_df: pd.DataFrame):
    return train_df[train_df.day == 48].copy(), train_df[train_df.day == 49].copy()


def split_H2(train_df: pd.DataFrame):
    d48 = train_df[train_df.day == 48]
    block = (d48.tmin >= TEST_TMIN_LO) & (d48.tmin <= TEST_TMIN_HI)
    valid = d48[block].copy()
    fit = pd.concat([d48[~block], train_df[train_df.day == 49]]).copy()
    return fit, valid


def eval_H1(model_factory, train_df, **kw):
    fit, valid = split_H1(train_df)
    pred, _ = fit_predict(model_factory, fit, valid, **kw)
    return r2(valid[TARGET], pred)


def eval_H2(model_factory, train_df, **kw):
    fit, valid = split_H2(train_df)
    pred, _ = fit_predict(model_factory, fit, valid, **kw)
    return r2(valid[TARGET], pred)


def eval_H3(model_factory, train_df, n_groups: int = 5, **kw):
    gkf = GroupKFold(n_splits=n_groups)
    oof = np.full(len(train_df), np.nan)
    groups = train_df["geohash"].to_numpy()
    for tr_i, va_i in gkf.split(train_df, groups=groups):
        fit = train_df.iloc[tr_i]
        valid = train_df.iloc[va_i]
        pred, _ = fit_predict(model_factory, fit, valid, **kw)
        oof[va_i] = pred
    return r2(train_df[TARGET], oof)


# --------------------------------------------------------------------------------------
# Baseline ladder (group-mean predictors, no model) — sanity-checks the pipeline
# --------------------------------------------------------------------------------------
def baseline_ladder(train_df: pd.DataFrame):
    fit, valid = split_H1(train_df)
    gm = fit[TARGET].mean()
    rows = []

    def grp(cols):
        m = fit.groupby(cols)[TARGET].mean()
        if len(cols) == 1:
            p = valid[cols[0]].map(m)
        else:
            p = pd.Series(valid.set_index(cols).index.map(m), index=valid.index)
        return p.fillna(gm)

    rows.append(("global_mean", r2(valid[TARGET], np.full(len(valid), gm))))
    rows.append(("RoadType_mean", r2(valid[TARGET], grp(["RoadType_ord"]))))
    rows.append(("per_geohash_mean", r2(valid[TARGET], grp(["geohash"]))))
    # exact-slot affine lookup
    slot = fit.groupby(["geohash", "tmin"])[TARGET].mean()
    look = pd.Series(valid.set_index(["geohash", "tmin"]).index.map(slot), index=valid.index)
    look_geo = look.fillna(valid["geohash"].map(fit.groupby("geohash")[TARGET].mean())).fillna(gm)
    rows.append(("d48_slot_raw", r2(valid[TARGET], look_geo)))
    rows.append(("d48_slot_affine", r2(valid[TARGET], (0.032 + 1.255 * look_geo).clip(0, 1))))
    return rows


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def main():
    train, _ = F.load_raw()
    train = P.prepare(train)  # base features + static imputation (fit on full train)

    print("=" * 64)
    print("BASELINE LADDER (H1 forward: day48 -> day49)")
    print("=" * 64)
    for name, val in baseline_ladder(train):
        print(f"  {name:22s} R2 = {val:.4f}")

    print("\n" + "=" * 64)
    print("LightGBM robust+memorization core")
    print("=" * 64)
    h1 = eval_H1(lgbm_factory, train)
    h2 = eval_H2(lgbm_factory, train)
    h3 = eval_H3(lgbm_factory, train)
    print(f"  H1 temporal-forward  R2 = {h1:.4f}   [PRIMARY]")
    print(f"  H2 daytime-block     R2 = {h2:.4f}   [PRIMARY]")
    print(f"  H3 spatial guardrail R2 = {h3:.4f}   [floor ~0.70]")
    print(f"  headline min(H1,H2)  = {min(h1, h2):.4f}")


if __name__ == "__main__":
    main()
