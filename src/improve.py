"""
improve.py — test the user's improvement ideas, board-ranked:
  #1 per-road-type affine calibration (Highway/Street/Residential each get own a,b)
  #5 two-stage residual modeling (GBM predicts demand - base_lookup; final = base + residual)
  combo: residual + per-road calibration
Builds candidate submissions; prints H1/H2 (imperfect daytime proxies, for reference only).
"""
from __future__ import annotations
import json, warnings, numpy as np, pandas as pd
import features as F, pipeline as P, cv
warnings.filterwarnings("ignore")
ROOT, T = F.ROOT, F.TARGET
MEMO = P.MEMO_SET
SEEDS = [42, 101]
TUNED = json.load(open(f"{ROOT}/models/best_lgbm_params.json"))["params"]
LOWREG = dict(n_estimators=800, num_leaves=63, min_child_samples=40, learning_rate=0.03,
              subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0, reg_alpha=0.0)
# ensemble members: (label, factory(seed), scale)
MEMBERS = [
    ("lgb_lr", lambda s: cv.lgbm_factory({**LOWREG, "random_state": s}), 0.5, 0.45),
    ("lgb_tn", lambda s: cv.lgbm_factory({**TUNED, "random_state": s}), 1.0, 0.25),
    ("cat",    lambda s: cv.cat_factory(dict(random_seed=s, depth=8, l2_leaf_reg=3.0, iterations=1000)), 0.5, 0.30),
]

train, test = F.load_raw()
train = P.prepare(train)
test = P.prepare(test, fit_df_for_impute=pd.read_csv(F.DATA_DIR + "/train.csv"))

# ---- cache built matrices per (split, scale) ----
MAT = {}
def mats(split, scale):
    if (split, scale) in MAT: return MAT[(split, scale)]
    P.SMOOTH_SCALE = scale
    if split == "test":
        Xfit = P.build_oof(train); yfit = train[T].to_numpy()
        Xev = P.build_fit_transform(train, test); yev = None; rt = test["RoadType_ord"].to_numpy()
    else:
        fit, val = (cv.split_H1 if split == "h1" else cv.split_H2)(train)
        Xfit = P.build_oof(fit); yfit = fit[T].to_numpy()
        Xev = P.build_fit_transform(fit, val); yev = val[T].to_numpy(); rt = val["RoadType_ord"].to_numpy()
    P.SMOOTH_SCALE = 1.0
    MAT[(split, scale)] = (Xfit, yfit, Xev, yev, rt)
    return MAT[(split, scale)]

def member_pred(make, scale, split, residual=False):
    """seed-averaged member prediction for a split. residual: train on demand-base, return base+pred."""
    Xfit, yfit, Xev, yev, rt = mats(split, scale)
    base_fit = Xfit["d48_shift_adj"].to_numpy(); base_ev = Xev["d48_shift_adj"].to_numpy()
    tgt = (yfit - base_fit) if residual else yfit
    ps = []
    for s in SEEDS:
        m = make(s); m.fit(Xfit[MEMO], tgt)
        p = m.predict(Xev[MEMO])
        ps.append((base_ev + p) if residual else p)
    return np.clip(np.mean(ps, axis=0), 0, 1)

def ensemble(split, residual=False):
    out = np.zeros(len(mats(split, 0.5)[2]))
    for _, make, scale, w in MEMBERS:
        out += w * member_pred(make, scale, split, residual)
    return out

# ---- calibration helpers ----
def fit_global(pred, y):
    b, a = np.polyfit(pred, y, 1); return a, b
def fit_road(pred, y, rt):
    cal = {}
    for r in (0, 1, 2):
        m = rt == r
        if m.sum() > 40:
            b, a = np.polyfit(pred[m], y[m], 1); cal[r] = (a, b)
    return cal
def apply_road(pred, rt, cal, glob):
    out = glob[0] + glob[1] * pred
    for r, (a, b) in cal.items():
        out[rt == r] = a + b * pred[rt == r]
    return np.clip(out, 0, 1)

def save(p, name):
    s = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": np.clip(p, 0, 1)})
    assert s.shape == (41778, 2) and s["demand"].notna().all()
    s.to_csv(f"{ROOT}/submissions/{name}", index=False); print(f"  wrote {name:34s} mean={s['demand'].mean():.4f}")

for residual in (False, True):
    tag = "residual" if residual else "standard"
    print(f"\n=== {tag} ensemble ===")
    e1 = ensemble("h1", residual); y1 = mats("h1", 0.5)[3]; rt1 = mats("h1", 0.5)[4]
    e2 = ensemble("h2", residual); y2 = mats("h2", 0.5)[3]; rt2 = mats("h2", 0.5)[4]
    et = ensemble("test", residual); rtt = mats("test", 0.5)[4]
    pool_p = np.concatenate([e1, e2]); pool_y = np.concatenate([y1, y2]); pool_rt = np.concatenate([rt1, rt2])
    glob = fit_global(pool_p, pool_y)
    road = fit_road(pool_p, pool_y, pool_rt)
    print(f"  global cal:  a={glob[0]:.4f} b={glob[1]:.4f} | "
          f"H1={cv.r2(y1, np.clip(glob[0]+glob[1]*e1,0,1)):.4f} H2={cv.r2(y2, np.clip(glob[0]+glob[1]*e2,0,1)):.4f}")
    for r, (a, b) in sorted(road.items()):
        print(f"  road {r} cal: a={a:.4f} b={b:.4f}")
    # per-road validation metric
    h1_rc = apply_road(e1, rt1, road, glob); h2_rc = apply_road(e2, rt2, road, glob)
    print(f"  road cal:    H1={cv.r2(y1,h1_rc):.4f} H2={cv.r2(y2,h2_rc):.4f}")
    if residual:
        save(np.clip(glob[0]+glob[1]*et, 0, 1), "submission_residual.csv")
        save(apply_road(et, rtt, road, glob), "submission_residual_roadcal.csv")
    else:
        save(apply_road(et, rtt, road, glob), "submission_roadcal.csv")
print("\ndone")
