"""
stack2.py — daytime-calibrated stacking. Fit the meta-learner on the H1+H2 validation set
(the daytime-representative data) so the blend is correctly scaled for the daytime test,
instead of the full-train OOF (which under-predicts daytime). Board-ranked.
"""
from __future__ import annotations
import json, warnings, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from scipy.optimize import nnls
import features as F, pipeline as P, cv
warnings.filterwarnings("ignore")
ROOT, T = F.ROOT, F.TARGET
MEMO = P.MEMO_SET
TUNED = json.load(open(f"{ROOT}/models/best_lgbm_params.json"))["params"]
LOWREG = dict(n_estimators=800, num_leaves=63, min_child_samples=40, learning_rate=0.03,
              subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0, reg_alpha=0.0)
AGGR = dict(n_estimators=1500, num_leaves=255, min_child_samples=5, learning_rate=0.03,
            reg_lambda=0.0, reg_alpha=0.0)
BASES = [
    ("m1", lambda s: cv.lgbm_factory({**LOWREG, "random_state": s}), 0.5),
    ("m2", lambda s: cv.lgbm_factory({**TUNED, "random_state": s}), 1.0),
    ("c1", lambda s: cv.cat_factory(dict(random_seed=s, depth=8, l2_leaf_reg=3.0, iterations=1000)), 0.5),
    ("a1", lambda s: cv.lgbm_factory({**AGGR, "random_state": s}), 0.15),
]
SEEDS = [42, 101]

train, test = F.load_raw()
train = P.prepare(train)
test = P.prepare(test, fit_df_for_impute=pd.read_csv(F.DATA_DIR + "/train.csv"))

# base predictions on H1 val, H2 val, and test
def base_preds(split):
    if split == "test":
        fit, ev, yev = train, test, None
    else:
        fit, ev = (cv.split_H1 if split == "h1" else cv.split_H2)(train); yev = ev[T].to_numpy()
    res = {}
    for name, make, scale in BASES:
        P.SMOOTH_SCALE = scale
        Xt = P.build_oof(fit); yt = fit[T].to_numpy(); Xv = P.build_fit_transform(fit, ev)
        P.SMOOTH_SCALE = 1.0
        res[name] = np.mean([np.clip(make(s).fit(Xt[MEMO], yt).predict(Xv[MEMO]), 0, 1) for s in SEEDS], axis=0)
    return res, yev

print("base preds H1/H2/test ...")
h1, y1 = base_preds("h1")
h2, y2 = base_preds("h2")
tt, _ = base_preds("test")

def save(p, name):
    s = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": np.clip(p, 0, 1)})
    assert s.shape == (41778, 2) and s["demand"].notna().all()
    s.to_csv(f"{ROOT}/submissions/{name}", index=False); print(f"  wrote {name:30s} mean={s['demand'].mean():.4f}")

def stack(cols, tag):
    A1 = np.column_stack([h1[c] for c in cols]); A2 = np.column_stack([h2[c] for c in cols])
    At = np.column_stack([tt[c] for c in cols])
    A = np.vstack([A1, A2]); yy = np.concatenate([y1, y2])
    print(f"\n[{tag}] {cols}")
    # NNLS with intercept
    Ai = np.column_stack([A, np.ones(len(A))]); w, _ = nnls(Ai, yy)
    pt = At @ w[:-1] + w[-1]
    print(f"  NNLS  w={np.round(w,3)} H1={cv.r2(y1, np.clip(A1@w[:-1]+w[-1],0,1)):.4f} H2={cv.r2(y2, np.clip(A2@w[:-1]+w[-1],0,1)):.4f}")
    rg = Ridge(alpha=1.0, positive=True).fit(A, yy)
    print(f"  Ridge w={np.round(rg.coef_,3)} H1={cv.r2(y1, np.clip(rg.predict(A1),0,1)):.4f} H2={cv.r2(y2, np.clip(rg.predict(A2),0,1)):.4f}")
    lr = LinearRegression().fit(A, yy)
    print(f"  Lin   w={np.round(lr.coef_,3)} H1={cv.r2(y1, np.clip(lr.predict(A1),0,1)):.4f} H2={cv.r2(y2, np.clip(lr.predict(A2),0,1)):.4f}")
    return {f"cstack_nnls_{tag}": np.clip(pt, 0, 1),
            f"cstack_ridge_{tag}": np.clip(rg.predict(At), 0, 1),
            f"cstack_lin_{tag}": np.clip(lr.predict(At), 0, 1)}

subs = {}
subs.update(stack(["m1", "m2", "c1"], "3"))
subs.update(stack(["m1", "m2", "c1", "a1"], "4"))
for n, p in subs.items():
    save(p, f"submission_{n}.csv")
print("\ndone")
