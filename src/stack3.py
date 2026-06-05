"""
stack3.py — richer daytime-calibrated stack. Add genuinely-diverse bases (XGBoost, 2nd CatBoost)
and let the meta-learner optimally weight them over subsets. Pushes past 88.96. Board-ranked.
"""
from __future__ import annotations
import json, warnings, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.optimize import nnls
import features as F, pipeline as P, cv
warnings.filterwarnings("ignore")
ROOT, T = F.ROOT, F.TARGET
MEMO = P.MEMO_SET
TUNED = json.load(open(f"{ROOT}/models/best_lgbm_params.json"))["params"]
LOWREG = dict(n_estimators=800, num_leaves=63, min_child_samples=40, learning_rate=0.03,
              subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0, reg_alpha=0.0)
BASES = [
    ("m1", lambda s: cv.lgbm_factory({**LOWREG, "random_state": s}), 0.5),
    ("m2", lambda s: cv.lgbm_factory({**TUNED, "random_state": s}), 1.0),
    ("c1", lambda s: cv.cat_factory(dict(random_seed=s, depth=8, l2_leaf_reg=3.0, iterations=1000)), 0.5),
    ("xg", lambda s: cv.xgb_factory(dict(random_state=s, max_depth=7, min_child_weight=10,
                                         n_estimators=900, subsample=0.9, colsample_bytree=0.8)), 0.5),
    ("c6", lambda s: cv.cat_factory(dict(random_seed=s, depth=6, l2_leaf_reg=5.0, iterations=1200)), 0.6),
]
SEEDS = [42, 101]

train, test = F.load_raw()
train = P.prepare(train)
test = P.prepare(test, fit_df_for_impute=pd.read_csv(F.DATA_DIR + "/train.csv"))

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
        print(f"    {split}:{name} done")
    return res, yev

print("computing base preds...")
h1, y1 = base_preds("h1"); h2, y2 = base_preds("h2"); tt, _ = base_preds("test")

def save(p, name):
    s = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": np.clip(p, 0, 1)})
    assert s.shape == (41778, 2) and s["demand"].notna().all()
    s.to_csv(f"{ROOT}/submissions/{name}", index=False); print(f"  wrote {name:28s} mean={s['demand'].mean():.4f}")

def stack(cols, tag):
    A1 = np.column_stack([h1[c] for c in cols]); A2 = np.column_stack([h2[c] for c in cols])
    At = np.column_stack([tt[c] for c in cols]); A = np.vstack([A1, A2]); yy = np.concatenate([y1, y2])
    Ai = np.column_stack([A, np.ones(len(A))]); w, _ = nnls(Ai, yy)
    rg = Ridge(alpha=1.0, positive=True).fit(A, yy)
    print(f"[{tag}] NNLS w={dict(zip(cols, np.round(w[:-1],3)))} int={w[-1]:.3f} "
          f"H1={cv.r2(y1, np.clip(A1@w[:-1]+w[-1],0,1)):.4f} H2={cv.r2(y2, np.clip(A2@w[:-1]+w[-1],0,1)):.4f}")
    return {f"r3_nnls_{tag}": np.clip(At @ w[:-1] + w[-1], 0, 1),
            f"r3_ridge_{tag}": np.clip(rg.predict(At), 0, 1)}

subs = {}
subs.update(stack(["m1", "m2", "c1"], "base"))          # reproduce 88.96
subs.update(stack(["m1", "m2", "c1", "xg"], "xg"))
subs.update(stack(["m1", "m2", "c1", "c6"], "c6"))
subs.update(stack(["m1", "m2", "c1", "xg", "c6"], "all"))
for n, p in subs.items():
    save(p, f"submission_{n}.csv")
print("\ndone")
