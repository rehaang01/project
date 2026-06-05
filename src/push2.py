"""Push past the 88.4 plateau: add estimator diversity (CatBoost) + ensemble the best memo models.
Honest ceiling ~0.90 (derived from raw-lookup 79.6). Board-ranked."""
from __future__ import annotations
import json, warnings, numpy as np, pandas as pd
import features as F, pipeline as P, cv
warnings.filterwarnings("ignore")
ROOT, TARGET = F.ROOT, F.TARGET
MEMO = P.MEMO_SET
TUNED = json.load(open(f"{ROOT}/models/best_lgbm_params.json"))["params"]
SCALE = 0.5

train, test = F.load_raw()
train = P.prepare(train)
test = P.prepare(test, fit_df_for_impute=pd.read_csv(F.DATA_DIR + "/train.csv"))


def calib_for(factory):
    pr, ys = [], []
    for split in (cv.split_H1, cv.split_H2):
        fit, val = split(train)
        P.SMOOTH_SCALE = SCALE
        Xt = P.build_oof(fit); yt = fit[TARGET].to_numpy(); Xv = P.build_fit_transform(fit, val)
        m = factory(); m.fit(Xt[MEMO], yt)
        pr.append(np.clip(m.predict(Xv[MEMO]), 0, 1)); ys.append(val[TARGET].to_numpy())
        P.SMOOTH_SCALE = 1.0
    b, a = np.polyfit(np.concatenate(pr), np.concatenate(ys), 1)
    return a, b


def predict_test(factory):
    P.SMOOTH_SCALE = SCALE
    Xtr = P.build_oof(train); ytr = train[TARGET].to_numpy(); Xte = P.build_fit_transform(train, test)
    m = factory(); m.fit(Xtr[MEMO], ytr)
    raw = np.clip(m.predict(Xte[MEMO]), 0, 1)
    P.SMOOTH_SCALE = 1.0
    a, b = calib_for(factory)
    return np.clip(a + b * raw, 0, 1)


def save(p, name):
    s = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": np.clip(p, 0, 1)})
    assert s.shape == (41778, 2) and list(s.columns) == ["Index", "demand"] and s["demand"].notna().all()
    s.to_csv(f"{ROOT}/submissions/{name}", index=False); print(f"  wrote {name:34s} mean={s['demand'].mean():.4f}")


# CatBoost memo (different estimator -> diversity)
print("Building CatBoost memo + ensembles:")
cat = predict_test(lambda: cv.cat_factory(dict(depth=8, l2_leaf_reg=3.0, iterations=1000)))
save(cat, "submission_cat_memo.csv")

# load the two best existing LightGBM memo predictions
memo      = pd.read_csv(f"{ROOT}/submissions/submission_memo.csv")["demand"].to_numpy()           # 88.32
memo_lr   = pd.read_csv(f"{ROOT}/submissions/submission_memo_s50_lowreg.csv")["demand"].to_numpy() # 88.40

save(0.5 * memo_lr + 0.5 * cat,                 "submission_ens_lgb_cat.csv")
save(0.45 * memo_lr + 0.25 * memo + 0.30 * cat, "submission_ens3.csv")
print("done")
