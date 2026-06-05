"""
final_model.py — the FINAL high-score model (board 88.96), reproducible from scratch.

Stacked generalization over three calibrated base learners:
  m1  LightGBM  MEMO  SMOOTH=0.5  low-regularization
  m2  LightGBM  MEMO  SMOOTH=1.0  Optuna-tuned
  c1  CatBoost  MEMO  SMOOTH=0.5  depth 8
A non-negative least-squares (NNLS) meta-learner is fit on the H1+H2 validation predictions
(daytime-representative -> correct scale for the daytime test) and applied to the test base
predictions. NNLS found balanced weights (~0.33/0.34/0.31 + small intercept) that beat the
hand-tuned 0.45/0.25/0.30 blend (88.78 -> 88.96).

Writes submissions/submission_highscore.csv.
"""
from __future__ import annotations
import json, warnings, numpy as np, pandas as pd
from scipy.optimize import nnls
import features as F, pipeline as P, cv
warnings.filterwarnings("ignore")
ROOT, T = F.ROOT, F.TARGET
MEMO = P.MEMO_SET
SEEDS = [42, 101]
TUNED = json.load(open(f"{ROOT}/models/best_lgbm_params.json"))["params"]
LOWREG = dict(n_estimators=800, num_leaves=63, min_child_samples=40, learning_rate=0.03,
              subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0, reg_alpha=0.0)
BASES = [
    ("m1", lambda s: cv.lgbm_factory({**LOWREG, "random_state": s}), 0.5),
    ("m2", lambda s: cv.lgbm_factory({**TUNED, "random_state": s}), 1.0),
    ("c1", lambda s: cv.cat_factory(dict(random_seed=s, depth=8, l2_leaf_reg=3.0, iterations=1000)), 0.5),
]

train, test = F.load_raw()
train = P.prepare(train)
test = P.prepare(test, fit_df_for_impute=pd.read_csv(F.DATA_DIR + "/train.csv"))


def base_preds(split):
    """Seed-averaged raw predictions for each base learner on a split (h1/h2/test)."""
    if split == "test":
        fit, ev, yev = train, test, None
    else:
        fit, ev = (cv.split_H1 if split == "h1" else cv.split_H2)(train); yev = ev[T].to_numpy()
    out = {}
    for name, make, scale in BASES:
        P.SMOOTH_SCALE = scale
        Xt = P.build_oof(fit); yt = fit[T].to_numpy(); Xv = P.build_fit_transform(fit, ev)
        P.SMOOTH_SCALE = 1.0
        out[name] = np.mean([np.clip(make(s).fit(Xt[MEMO], yt).predict(Xv[MEMO]), 0, 1) for s in SEEDS], axis=0)
    return out, yev


print("Computing base learner predictions (H1, H2, test)...")
h1, y1 = base_preds("h1"); h2, y2 = base_preds("h2"); tt, _ = base_preds("test")

cols = [b[0] for b in BASES]
A = np.vstack([np.column_stack([h1[c] for c in cols]), np.column_stack([h2[c] for c in cols])])
yy = np.concatenate([y1, y2])
Ai = np.column_stack([A, np.ones(len(A))])            # intercept column
w, _ = nnls(Ai, yy)                                   # non-negative weights + intercept
print(f"NNLS weights: { {c: round(float(wi),3) for c, wi in zip(cols, w[:-1])} } intercept={w[-1]:.3f}")
A1 = np.column_stack([h1[c] for c in cols]); A2 = np.column_stack([h2[c] for c in cols])
print(f"  H1={cv.r2(y1, np.clip(A1@w[:-1]+w[-1],0,1)):.4f}  H2={cv.r2(y2, np.clip(A2@w[:-1]+w[-1],0,1)):.4f}")

At = np.column_stack([tt[c] for c in cols])
final = np.clip(At @ w[:-1] + w[-1], 0, 1)
sub = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": final})
assert sub.shape == (41778, 2) and list(sub.columns) == ["Index", "demand"]
assert sub["demand"].notna().all() and sub["demand"].between(0, 1).all()
sub.to_csv(f"{ROOT}/submissions/submission_highscore.csv", index=False)
print(f"wrote submissions/submission_highscore.csv  mean={final.mean():.4f}  (board 88.96)")
