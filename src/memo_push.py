"""Build aggressive-memorization candidates to climb the board (daytime demand repeats from day48).
Offline CV cannot rank the day48->day49 daytime lookup, so these are board-ranked."""
from __future__ import annotations
import warnings, numpy as np, pandas as pd
import features as F, pipeline as P, cv
warnings.filterwarnings("ignore")
ROOT, TARGET = F.ROOT, F.TARGET
MEMO = P.MEMO_SET
SEEDS = [42, 101, 202]
TUNED = __import__("json").load(open(f"{ROOT}/models/best_lgbm_params.json"))["params"]
LOWREG = dict(n_estimators=800, num_leaves=63, min_child_samples=40, learning_rate=0.03,
              subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0, reg_alpha=0.0)

train, test = F.load_raw()
train = P.prepare(train)
test = P.prepare(test, fit_df_for_impute=pd.read_csv(F.DATA_DIR + "/train.csv"))


def seed_avg(Xtr, ytr, Xte, cols, params):
    ps = []
    for s in SEEDS:
        p = dict(params); p["random_state"] = s
        m = cv.lgbm_factory(p); m.fit(Xtr[cols], ytr); ps.append(np.clip(m.predict(Xte[cols]), 0, 1))
    return np.mean(ps, axis=0)


def fit_calib(params):
    pr, ys = [], []
    for split in (cv.split_H1, cv.split_H2):
        fit, val = split(train)
        Xt = P.build_oof(fit); yt = fit[TARGET].to_numpy(); Xv = P.build_fit_transform(fit, val)
        pr.append(seed_avg(Xt, yt, Xv, MEMO, params)); ys.append(val[TARGET].to_numpy())
    b, a = np.polyfit(np.concatenate(pr), np.concatenate(ys), 1)
    return a, b


def save(p, name):
    s = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": np.clip(p, 0, 1)})
    assert s.shape == (41778, 2) and list(s.columns) == ["Index", "demand"] and s["demand"].notna().all()
    s.to_csv(f"{ROOT}/submissions/{name}", index=False); print(f"  wrote {name:34s} mean={s['demand'].mean():.4f}")


def memo_submission(scale, params, name, calibrate=True):
    P.SMOOTH_SCALE = scale
    Xtr = P.build_oof(train); ytr = train[TARGET].to_numpy(); Xte = P.build_fit_transform(train, test)
    raw = seed_avg(Xtr, ytr, Xte, MEMO, params)
    if calibrate:
        a, b = fit_calib(params); raw = a + b * raw
    save(raw, name)
    P.SMOOTH_SCALE = 1.0


print("Building memorization candidates:")
memo_submission(0.5, TUNED,  "submission_memo_s50.csv")
memo_submission(0.5, LOWREG, "submission_memo_s50_lowreg.csv")
memo_submission(0.3, LOWREG, "submission_memo_s30_lowreg.csv")

# pure day48 (geohash,hour) mean lookup (averages daytime slots -> low variance)
def tmin(s): h, m = s.split(":"); return int(h) * 60 + int(m)
tr2, te2 = F.load_raw(); tr2["tmin"] = tr2["timestamp"].map(tmin); te2["tmin"] = te2["timestamp"].map(tmin)
tr2["hour"] = tr2["tmin"] // 60; te2["hour"] = te2["tmin"] // 60
d48 = tr2[tr2.day == 48]; gm = d48["demand"].mean()
gh_hr = d48.groupby(["geohash", "hour"])["demand"].mean()
geo = d48.groupby("geohash")["demand"].mean()
gehr = pd.Series(te2.set_index(["geohash", "hour"]).index.map(gh_hr), index=te2.index)
geohour = gehr.fillna(te2["geohash"].map(geo)).fillna(gm).to_numpy()
save(geohour, "submission_geohour.csv")

# blend: aggressive memo GBM + clean geohash-hour lookup
P.SMOOTH_SCALE = 0.5
Xtr = P.build_oof(train); ytr = train[TARGET].to_numpy(); Xte = P.build_fit_transform(train, test)
memo_gbm = seed_avg(Xtr, ytr, Xte, MEMO, LOWREG)
a, b = fit_calib(LOWREG); memo_gbm = a + b * memo_gbm
P.SMOOTH_SCALE = 1.0
save(0.5 * memo_gbm + 0.5 * geohour, "submission_memo_geohour_blend.csv")
print("done")
