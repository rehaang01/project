"""
train.py — tune + train the robust ensemble, evaluate on H1/H2/H3, write test predictions.

Outputs (in ../models and ../submissions):
  models/oof_h1.npz                 per-model H1 validation predictions + weights
  models/best_lgbm_params.json      tuned LightGBM params
  submissions/submission_robust.csv PRIMARY (robust ensemble)  [data-backed best on forward task]
  submissions/submission_memo.csv   diagnostic A/B (memorization model; optimistic on same-day only)
"""
from __future__ import annotations
import json
import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import features as F
import pipeline as P
import cv

warnings.filterwarnings("ignore")
SEED = F.SEED
TARGET = F.TARGET
ROOT = F.ROOT
N_TRIALS = int(os.environ.get("N_TRIALS", "40"))


def optuna_tune_lgbm(Xtr, ytr, Xva, yva, cols, n_trials=N_TRIALS):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(t):
        params = dict(
            n_estimators=t.suggest_int("n_estimators", 150, 700),
            num_leaves=t.suggest_int("num_leaves", 7, 31),
            min_child_samples=t.suggest_int("min_child_samples", 100, 600),
            learning_rate=t.suggest_float("learning_rate", 0.02, 0.06, log=True),
            subsample=t.suggest_float("subsample", 0.6, 0.95),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 0.95),
            reg_lambda=t.suggest_float("reg_lambda", 0.0, 8.0),
            reg_alpha=t.suggest_float("reg_alpha", 0.0, 5.0),
        )
        m = cv.lgbm_factory(params)
        m.fit(Xtr[cols], ytr)
        return cv.r2(yva, np.clip(m.predict(Xva[cols]), 0, 1))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def ensemble_weights(preds: dict, y):
    """Non-negative weights summing to 1 that maximize R² on y."""
    names = list(preds)
    M = np.column_stack([preds[n] for n in names])

    def neg_r2(w):
        return -cv.r2(y, M @ w)

    w0 = np.full(len(names), 1 / len(names))
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bnds = [(0, 1)] * len(names)
    res = minimize(neg_r2, w0, method="SLSQP", bounds=bnds, constraints=cons)
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    return dict(zip(names, w)), cv.r2(y, M @ w)


def model_suite(best_lgb):
    return {
        "lgbm": lambda: cv.lgbm_factory(best_lgb),
        "xgb": cv.xgb_factory,
        "cat": cv.cat_factory,
        "ridge": cv.ridge_factory,
    }


def fit_all(suite, Xtr, ytr, Xva, cols):
    """Fit every model on Xtr[cols], return dict of clipped predictions on Xva[cols]."""
    out = {}
    for name, fac in suite.items():
        m = fac()
        m.fit(Xtr[cols], ytr)
        out[name] = np.clip(m.predict(Xva[cols]), 0, 1)
    return out


def main():
    print("Loading + preparing data ...")
    train, test = F.load_raw()
    train = P.prepare(train)
    test = P.prepare(test, fit_df_for_impute=pd.read_csv(F.DATA_DIR + "/train.csv"))

    cols = P.ROBUST_SET

    # ---- H1 matrices (for tuning + weight selection) ----
    fit1, val1 = cv.split_H1(train)
    X1tr = P.build_oof(fit1); y1tr = fit1[TARGET].to_numpy()
    X1va = P.build_fit_transform(fit1, val1); y1va = val1[TARGET].to_numpy()

    print(f"Tuning LightGBM on H1 ({N_TRIALS} trials) ...")
    best_lgb, best_h1 = optuna_tune_lgbm(X1tr, y1tr, X1va, y1va, cols)
    print(f"  best single-LGBM H1 = {best_h1:.4f}")
    print(f"  params = {best_lgb}")

    suite = model_suite(best_lgb)

    # ---- per-model H1 preds + ensemble weights ----
    print("Fitting model suite on H1 split ...")
    preds_h1 = fit_all(suite, X1tr, y1tr, X1va, cols)
    for n, p in preds_h1.items():
        print(f"  {n:6s} H1 = {cv.r2(y1va, p):.4f}")
    weights, ens_h1 = ensemble_weights(preds_h1, y1va)
    print(f"  ENSEMBLE H1 = {ens_h1:.4f}  weights={ {k: round(v,3) for k,v in weights.items()} }")

    # ---- H2 + H3 for the chosen ensemble (sanity / guardrail) ----
    fit2, val2 = cv.split_H2(train)
    X2tr = P.build_oof(fit2); y2tr = fit2[TARGET].to_numpy()
    X2va = P.build_fit_transform(fit2, val2); y2va = val2[TARGET].to_numpy()
    preds_h2 = fit_all(suite, X2tr, y2tr, X2va, cols)
    ens2 = sum(weights[n] * preds_h2[n] for n in weights)
    print(f"  ENSEMBLE H2 = {cv.r2(y2va, ens2):.4f}")

    h3 = cv.eval_H3(lambda: cv.lgbm_factory(best_lgb), train, feat_cols=cols)
    print(f"  LGBM   H3 (guardrail) = {h3:.4f}")

    # ---- Final: train on FULL train, predict test ----
    print("Training on full train, predicting test ...")
    Xtr_full = P.build_oof(train); ytr_full = train[TARGET].to_numpy()
    Xte = P.build_fit_transform(train, test)
    preds_te = fit_all(suite, Xtr_full, ytr_full, Xte, cols)
    robust_pred = np.clip(sum(weights[n] * preds_te[n] for n in weights), 0, 1)

    # ---- Diagnostic memorization model (full features, single tuned LGBM) ----
    memo_cols = P.MEMO_SET
    m_memo = cv.lgbm_factory(best_lgb)
    m_memo.fit(Xtr_full[memo_cols], ytr_full)
    memo_pred = np.clip(m_memo.predict(Xte[memo_cols]), 0, 1)

    # ---- Save ----
    os.makedirs(f"{ROOT}/models", exist_ok=True)
    os.makedirs(f"{ROOT}/submissions", exist_ok=True)
    np.savez(f"{ROOT}/models/oof_h1.npz",
             **{f"h1_{n}": p for n, p in preds_h1.items()}, y1va=y1va,
             weights=np.array([weights[n] for n in suite]), names=np.array(list(suite)))
    with open(f"{ROOT}/models/best_lgbm_params.json", "w") as f:
        json.dump({"params": best_lgb, "weights": weights,
                   "H1": ens_h1, "H2": cv.r2(y2va, ens2), "H3": h3}, f, indent=2)

    write_submission(test, robust_pred, f"{ROOT}/submissions/submission_robust.csv")
    write_submission(test, memo_pred, f"{ROOT}/submissions/submission_memo.csv")

    print("\n=== SUMMARY ===")
    print(f"  Ensemble H1 (forward, PRIMARY) = {ens_h1:.4f}")
    print(f"  Ensemble H2 (daytime same-day) = {cv.r2(y2va, ens2):.4f}")
    print(f"  LGBM     H3 (spatial guardrail)= {h3:.4f}")
    print(f"  robust submission -> submissions/submission_robust.csv")
    print(f"  memo   submission -> submissions/submission_memo.csv")


def write_submission(test_df, preds, path):
    sub = pd.DataFrame({"Index": test_df["Index"].to_numpy(), "demand": np.clip(preds, 0, 1)})
    assert sub.shape == (41778, 2), sub.shape
    assert list(sub.columns) == ["Index", "demand"]
    assert sub["demand"].notna().all() and sub["demand"].between(0, 1).all()
    sub.to_csv(path, index=False)


if __name__ == "__main__":
    main()
