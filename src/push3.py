"""memo_plus: add multiplicative E[demand|geohash,time] features (push toward ~0.90 ceiling)."""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
import features as F, pipeline as P, cv
warnings.filterwarnings("ignore")
ROOT, TARGET = F.ROOT, F.TARGET
FEATS = P.MEMO_PLUS
SCALE = 0.5
SEEDS = [42, 101, 202]
LOWREG = dict(n_estimators=800, num_leaves=63, min_child_samples=40, learning_rate=0.03,
              subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0, reg_alpha=0.0)

train, test = F.load_raw()
train = P.prepare(train)
test = P.prepare(test, fit_df_for_impute=pd.read_csv(F.DATA_DIR + "/train.csv"))


def seed_avg(Xtr, ytr, Xte):
    ps = []
    for s in SEEDS:
        p = dict(LOWREG); p["random_state"] = s
        m = cv.lgbm_factory(p); m.fit(Xtr[FEATS], ytr); ps.append(np.clip(m.predict(Xte[FEATS]), 0, 1))
    return np.mean(ps, axis=0)


def fit_calib():
    pr, ys = [], []
    for split in (cv.split_H1, cv.split_H2):
        fit, val = split(train)
        P.SMOOTH_SCALE = SCALE
        Xt = P.build_oof(fit); yt = fit[TARGET].to_numpy(); Xv = P.build_fit_transform(fit, val)
        pr.append(seed_avg(Xt, yt, Xv)); ys.append(val[TARGET].to_numpy())
        P.SMOOTH_SCALE = 1.0
    b, a = np.polyfit(np.concatenate(pr), np.concatenate(ys), 1)
    print(f"  H1 cal={cv.r2(ys[0], np.clip(a+b*pr[0],0,1)):.4f}  H2 cal={cv.r2(ys[1], np.clip(a+b*pr[1],0,1)):.4f}")
    return a, b


def save(p, name):
    s = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": np.clip(p, 0, 1)})
    assert s.shape == (41778, 2) and list(s.columns) == ["Index", "demand"] and s["demand"].notna().all()
    s.to_csv(f"{ROOT}/submissions/{name}", index=False); print(f"  wrote {name:30s} mean={s['demand'].mean():.4f}")


print("Building memo_plus (multiplicative features):")
P.SMOOTH_SCALE = SCALE
Xtr = P.build_oof(train); ytr = train[TARGET].to_numpy(); Xte = P.build_fit_transform(train, test)
P.SMOOTH_SCALE = 1.0
raw = seed_avg(Xtr, ytr, Xte)
a, b = fit_calib()
memo_plus = np.clip(a + b * raw, 0, 1)
save(memo_plus, "submission_memo_plus.csv")

# ensemble with the diverse models
cat = pd.read_csv(f"{ROOT}/submissions/submission_cat_memo.csv")["demand"].to_numpy()
memo_lr = pd.read_csv(f"{ROOT}/submissions/submission_memo_s50_lowreg.csv")["demand"].to_numpy()
save(0.34 * memo_plus + 0.33 * memo_lr + 0.33 * cat, "submission_ens_plus.csv")
print("done")
