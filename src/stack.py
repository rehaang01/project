"""
stack.py — stacked generalization over the 3 base learners + an aggressive-overfit base.

Proper stacking: outer 5-fold OOF base predictions on train -> fit meta-learner -> apply to test
base predictions. Compares NNLS / Ridge / Linear metas (suited to collinear base learners).
Also builds a maximally-aggressive memorization base to probe the overfit ceiling. Board-ranked.
"""
from __future__ import annotations
import json, warnings, numpy as np, pandas as pd
from sklearn.model_selection import KFold
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
            subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0, reg_alpha=0.0)
# base learners: (name, make(seed), scale)
BASES = [
    ("m1", lambda s: cv.lgbm_factory({**LOWREG, "random_state": s}), 0.5),
    ("m2", lambda s: cv.lgbm_factory({**TUNED, "random_state": s}), 1.0),
    ("c1", lambda s: cv.cat_factory(dict(random_seed=s, depth=8, l2_leaf_reg=3.0, iterations=1000)), 0.5),
    ("a1", lambda s: cv.lgbm_factory({**AGGR, "random_state": s}), 0.15),  # aggressive overfit base
]
SEED = 42

train, test = F.load_raw()
train = P.prepare(train)
test = P.prepare(test, fit_df_for_impute=pd.read_csv(F.DATA_DIR + "/train.csv"))
y = train[T].to_numpy()

# ---- OOF base predictions on train (outer 5-fold) ----
print("Generating OOF base predictions (5-fold)...")
oof = {n: np.zeros(len(train)) for n, _, _ in BASES}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for fi, (tr_i, va_i) in enumerate(kf.split(train)):
    trn = train.iloc[tr_i]; val = train.iloc[va_i]
    cache = {}
    for name, make, scale in BASES:
        if scale not in cache:
            P.SMOOTH_SCALE = scale
            cache[scale] = (P.build_oof(trn), P.build_fit_transform(trn, val)); P.SMOOTH_SCALE = 1.0
        Xt, Xv = cache[scale]
        m = make(SEED); m.fit(Xt[MEMO], y[tr_i])
        oof[name][va_i] = np.clip(m.predict(Xv[MEMO]), 0, 1)
    print(f"  fold {fi+1}/5 done")

# ---- base predictions on test (fit on full train, seed-averaged) ----
print("Fitting base learners on full train -> test...")
tst = {}
for name, make, scale in BASES:
    P.SMOOTH_SCALE = scale
    Xtr = P.build_oof(train); Xte = P.build_fit_transform(train, test)
    P.SMOOTH_SCALE = 1.0
    tst[name] = np.mean([np.clip(make(s).fit(Xtr[MEMO], y).predict(Xte[MEMO]), 0, 1) for s in (42, 101)], axis=0)

def save(p, name):
    s = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": np.clip(p, 0, 1)})
    assert s.shape == (41778, 2) and s["demand"].notna().all()
    s.to_csv(f"{ROOT}/submissions/{name}", index=False); print(f"  wrote {name:30s} mean={s['demand'].mean():.4f}")

# ---- aggressive base alone (calibrated) ----
b, a = np.polyfit(oof["a1"], y, 1)
save(a + b * tst["a1"], "submission_aggressive.csv")
print(f"  aggressive base OOF R2 = {cv.r2(y, oof['a1']):.4f}")

# ---- meta-learners over the 3 core bases (m1,m2,c1) and the 4-base set ----
def build_meta(cols, tag):
    A = np.column_stack([oof[c] for c in cols]); At = np.column_stack([tst[c] for c in cols])
    print(f"\n[{tag}] bases={cols}  OOF R2: " + ", ".join(f"{c}={cv.r2(y,oof[c]):.4f}" for c in cols))
    # NNLS (with intercept column)
    A1 = np.column_stack([A, np.ones(len(A))]); w, _ = nnls(A1, y)
    p_nnls = At @ w[:-1] + w[-1]
    print(f"  NNLS   w={np.round(w,3)}  OOF R2={cv.r2(y, A1@w):.4f}")
    # Ridge
    rg = Ridge(alpha=1.0, positive=True).fit(A, y)
    print(f"  Ridge+ w={np.round(rg.coef_,3)} b={rg.intercept_:.3f}  OOF R2={cv.r2(y, rg.predict(A)):.4f}")
    # unconstrained linear (most aggressive)
    lr = LinearRegression().fit(A, y)
    print(f"  Linear w={np.round(lr.coef_,3)} b={lr.intercept_:.3f}  OOF R2={cv.r2(y, lr.predict(A)):.4f}")
    return {f"stack_nnls_{tag}": np.clip(p_nnls, 0, 1),
            f"stack_ridge_{tag}": np.clip(rg.predict(At), 0, 1),
            f"stack_linear_{tag}": np.clip(lr.predict(At), 0, 1)}

subs = {}
subs.update(build_meta(["m1", "m2", "c1"], "3"))
subs.update(build_meta(["m1", "m2", "c1", "a1"], "4"))
for name, p in subs.items():
    save(p, f"submission_{name}.csv")
print("\ndone")
