"""
final_model.py — the FINAL high-score model, reproducible from scratch.

A diversity ensemble of three calibrated, seed-averaged memorization models:
  m1  LightGBM  MEMO  SMOOTH=0.5  low-regularization
  m2  LightGBM  MEMO  SMOOTH=1.0  Optuna-tuned
  c1  CatBoost  MEMO  SMOOTH=0.5  depth 8
blended  ENS_W = (0.45, 0.25, 0.30)  -> board ~88.8 (honest; provided data only).

Each model is affine-recalibrated (a + b*pred) on pooled H1+H2, correcting the daytime
under-prediction. Writes submissions/submission_highscore.csv.
"""
from __future__ import annotations
import json, warnings, numpy as np, pandas as pd
import features as F, pipeline as P, cv
warnings.filterwarnings("ignore")
ROOT, TARGET = F.ROOT, F.TARGET
MEMO = P.MEMO_SET
SEEDS = [42, 101, 202]
TUNED = json.load(open(f"{ROOT}/models/best_lgbm_params.json"))["params"]
LOWREG = dict(n_estimators=800, num_leaves=63, min_child_samples=40, learning_rate=0.03,
              subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0, reg_alpha=0.0)
ENS_W = (0.45, 0.25, 0.30)  # (m1 lgb-lowreg, m2 lgb-tuned, c1 catboost)

train, test = F.load_raw()
train = P.prepare(train)
test = P.prepare(test, fit_df_for_impute=pd.read_csv(F.DATA_DIR + "/train.csv"))


def calibrated_pred(make, scale, seeds=SEEDS):
    """make(seed)->estimator. Seed-averaged, affine-calibrated on pooled H1+H2 -> test preds."""
    def seed_avg(Xtr, ytr, Xte):
        return np.mean([np.clip(make(s).fit(Xtr[MEMO], ytr).predict(Xte[MEMO]), 0, 1) for s in seeds], axis=0)
    pr, ys = [], []
    for split in (cv.split_H1, cv.split_H2):
        fit, val = split(train)
        P.SMOOTH_SCALE = scale
        Xt = P.build_oof(fit); yt = fit[TARGET].to_numpy(); Xv = P.build_fit_transform(fit, val)
        pr.append(seed_avg(Xt, yt, Xv)); ys.append(val[TARGET].to_numpy()); P.SMOOTH_SCALE = 1.0
    b, a = np.polyfit(np.concatenate(pr), np.concatenate(ys), 1)
    P.SMOOTH_SCALE = scale
    Xtr = P.build_oof(train); ytr = train[TARGET].to_numpy(); Xte = P.build_fit_transform(train, test)
    raw = seed_avg(Xtr, ytr, Xte); P.SMOOTH_SCALE = 1.0
    return np.clip(a + b * raw, 0, 1)


print("Training final ensemble (3 calibrated models)...")
m1 = calibrated_pred(lambda s: cv.lgbm_factory({**LOWREG, "random_state": s}), 0.5)
m2 = calibrated_pred(lambda s: cv.lgbm_factory({**TUNED, "random_state": s}), 1.0)
c1 = calibrated_pred(lambda s: cv.cat_factory(dict(random_seed=s, depth=8, l2_leaf_reg=3.0, iterations=1000)),
                     0.5, seeds=[42, 101])
final = np.clip(ENS_W[0] * m1 + ENS_W[1] * m2 + ENS_W[2] * c1, 0, 1)

sub = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": final})
assert sub.shape == (41778, 2) and list(sub.columns) == ["Index", "demand"]
assert sub["demand"].notna().all() and sub["demand"].between(0, 1).all()
sub.to_csv(f"{ROOT}/submissions/submission_highscore.csv", index=False)
print(f"wrote submissions/submission_highscore.csv  mean={final.mean():.4f}  weights={ENS_W}")
