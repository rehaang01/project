"""
finalize.py — build the FINAL submission: segmented + seed-averaged LightGBM + affine calibration.

Segmentation (validated on BOTH H1 forward and H2 daytime, gain ~+0.024 on each):
  * Residential rows  -> MEMO features (geohash-level signal is low & stable -> helps forward)
  * Street / Highway  -> ROBUST features (high day-to-day drift -> memorization overfits, robust wins)
Models are trained on all rows and routed by (imputed) RoadType at prediction time.

Then affine recalibration (a + b*pred) corrects the model's under-prediction of the high-demand
daytime test window; coefficients fit on pooled H1+H2 ensemble predictions.

Reuses tuned LightGBM params from models/best_lgbm_params.json (run train.py first).
"""
from __future__ import annotations
import json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import features as F, pipeline as P, cv

warnings.filterwarnings("ignore")
ROOT, SEED, TARGET = F.ROOT, F.SEED, F.TARGET
ROBUST, MEMO = P.ROBUST_SET, P.MEMO_SET
SEEDS = [42, 101, 202, 303, 404]
RESIDENTIAL = 0  # RoadType_ord code


def seg_predict(Xtr, ytr, rt_tr, Xva, rt_va, params, seeds=SEEDS):
    """Seed-averaged LightGBM, routed by RoadType: Residential->MEMO, else->ROBUST."""
    res_tr = rt_tr == RESIDENTIAL
    res_va = (rt_va == RESIDENTIAL)
    pred = np.zeros(len(rt_va))
    # robust model (all rows) for Street/Highway predictions
    rob = _seed_avg(Xtr[ROBUST], ytr, Xva[ROBUST], params, seeds)
    memo = _seed_avg(Xtr[MEMO], ytr, Xva[MEMO], params, seeds)
    pred[~res_va] = rob[~res_va]
    pred[res_va] = memo[res_va]
    return np.clip(pred, 0, 1)


def _seed_avg(Xtr, ytr, Xva, params, seeds):
    ps = []
    for s in seeds:
        p = dict(params); p["random_state"] = s
        m = cv.lgbm_factory(p); m.fit(Xtr, ytr)
        ps.append(np.clip(m.predict(Xva), 0, 1))
    return np.mean(ps, axis=0)


def seg_eval(train, split, params, seeds=SEEDS):
    fit, val = split(train)
    Xtr = P.build_oof(fit); ytr = fit[TARGET].to_numpy()
    Xva = P.build_fit_transform(fit, val)
    pred = seg_predict(Xtr, ytr, fit["RoadType_ord"].to_numpy(),
                       Xva, val["RoadType_ord"].to_numpy(), params, seeds)
    return pred, val[TARGET].to_numpy()


def seg_h3(train, params):
    """Spatial guardrail with the segmented model (single seed for speed)."""
    gkf = GroupKFold(n_splits=5)
    oof = np.full(len(train), np.nan)
    g = train["geohash"].to_numpy()
    for tr_i, va_i in gkf.split(train, groups=g):
        fit, val = train.iloc[tr_i], train.iloc[va_i]
        Xtr = P.build_oof(fit); ytr = fit[TARGET].to_numpy()
        Xva = P.build_fit_transform(fit, val)
        oof[va_i] = seg_predict(Xtr, ytr, fit["RoadType_ord"].to_numpy(),
                                Xva, val["RoadType_ord"].to_numpy(), params, seeds=[42])
    return cv.r2(train[TARGET], oof)


def main():
    params = json.load(open(f"{ROOT}/models/best_lgbm_params.json"))["params"]
    train, test = F.load_raw()
    train = P.prepare(train)
    test = P.prepare(test, fit_df_for_impute=pd.read_csv(F.DATA_DIR + "/train.csv"))

    # ---- validate segmented model + fit calibration ----
    e1, y1 = seg_eval(train, cv.split_H1, params)
    e2, y2 = seg_eval(train, cv.split_H2, params)
    Pp = np.concatenate([e1, e2]); Y = np.concatenate([y1, y2])
    b, a = np.polyfit(Pp, Y, 1)
    cal = lambda p: np.clip(a + b * p, 0, 1)
    h1r, h1c = cv.r2(y1, e1), cv.r2(y1, cal(e1))
    h2r, h2c = cv.r2(y2, e2), cv.r2(y2, cal(e2))
    h3 = seg_h3(train, params)
    print(f"SEGMENTED model:  H1 {h1r:.4f}->{h1c:.4f} | H2 {h2r:.4f}->{h2c:.4f} | H3 {h3:.4f}")
    print(f"calibration: a={a:.4f} b={b:.4f}")

    # ---- final fit on full train, predict test ----
    Xtr = P.build_oof(train); ytr = train[TARGET].to_numpy()
    Xte = P.build_fit_transform(train, test)
    raw = seg_predict(Xtr, ytr, train["RoadType_ord"].to_numpy(),
                      Xte, test["RoadType_ord"].to_numpy(), params)
    final = cal(raw)

    _save(test, final, f"{ROOT}/submissions/submission.csv")            # PRIMARY
    _save(test, np.clip(raw, 0, 1), f"{ROOT}/submissions/submission_uncalibrated.csv")
    json.dump({"a": float(a), "b": float(b), "H1_raw": h1r, "H1_cal": h1c,
               "H2_raw": h2r, "H2_cal": h2c, "H3": h3, "seeds": SEEDS},
              open(f"{ROOT}/models/final_metrics.json", "w"), indent=2)
    print(f"final pred mean = {final.mean():.4f}")
    print("wrote submissions/submission.csv (PRIMARY: segmented + calibrated)")


def _save(test_df, preds, path):
    sub = pd.DataFrame({"Index": test_df["Index"].to_numpy(), "demand": np.clip(preds, 0, 1)})
    assert sub.shape == (41778, 2) and list(sub.columns) == ["Index", "demand"]
    assert sub["demand"].notna().all() and sub["demand"].between(0, 1).all()
    sub.to_csv(path, index=False)


if __name__ == "__main__":
    main()
