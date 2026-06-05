"""Generate candidate submissions to A/B on the board (the only daytime-forward signal).

The board told us segmentation overfit our night-based CV. These variants test the hypotheses our
offline CV cannot, especially: does day-48 daytime memorization help the daytime test?
"""
from __future__ import annotations
import json, warnings
import numpy as np, pandas as pd
import features as F, pipeline as P, cv

warnings.filterwarnings("ignore")
ROOT, TARGET = F.ROOT, F.TARGET
ROBUST, MEMO = P.ROBUST_SET, P.MEMO_SET
SEEDS = [42, 101, 202, 303, 404]
params = json.load(open(f"{ROOT}/models/best_lgbm_params.json"))["params"]


def seed_avg(Xtr, ytr, Xte, cols):
    ps = []
    for s in SEEDS:
        p = dict(params); p["random_state"] = s
        m = cv.lgbm_factory(p); m.fit(Xtr[cols], ytr)
        ps.append(np.clip(m.predict(Xte[cols]), 0, 1))
    return np.mean(ps, axis=0)


def fit_calib(train, cols):
    preds, ys = [], []
    for split in (cv.split_H1, cv.split_H2):
        fit, val = split(train)
        Xtr = P.build_oof(fit); ytr = fit[TARGET].to_numpy()
        Xva = P.build_fit_transform(fit, val)
        p = seed_avg(Xtr, ytr, Xva, cols)
        preds.append(p); ys.append(val[TARGET].to_numpy())
    Pp = np.concatenate(preds); Y = np.concatenate(ys)
    b, a = np.polyfit(Pp, Y, 1)
    return a, b, cv.r2(ys[0], np.clip(a + b * preds[0], 0, 1)), cv.r2(ys[1], np.clip(a + b * preds[1], 0, 1))


def save(test, preds, name):
    sub = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": np.clip(preds, 0, 1)})
    assert sub.shape == (41778, 2) and list(sub.columns) == ["Index", "demand"]
    assert sub["demand"].notna().all() and sub["demand"].between(0, 1).all()
    sub.to_csv(f"{ROOT}/submissions/{name}", index=False)
    print(f"  wrote {name:32s} mean={sub['demand'].mean():.4f}")


train, test = F.load_raw()
train = P.prepare(train)
test = P.prepare(test, fit_df_for_impute=pd.read_csv(F.DATA_DIR + "/train.csv"))
Xtr = P.build_oof(train); ytr = train[TARGET].to_numpy()
Xte = P.build_fit_transform(train, test)

print("Building candidates (with calibration fit on H1+H2):")
# A) FULL MEMO — all road types use memorization (tests daytime-memorization hypothesis)
a, b, h1, h2 = fit_calib(train, MEMO)
memo_te = seed_avg(Xtr, ytr, Xte, MEMO)
save(test, a + b * memo_te, "submission_memo.csv")
print(f"     memo_full  CV H1={h1:.4f} H2={h2:.4f} (NOTE: H1 is night, may under-rate daytime memo)")

# B) BLEND structural(robust) + affine day-48 slot lookup (hedged memorization for daytime)
a, b, h1, h2 = fit_calib(train, ROBUST)
rob_te = seed_avg(Xtr, ytr, Xte, ROBUST)
rob_cal = a + b * rob_te
lookup = np.clip(Xte["d48_shift_adj"].to_numpy(), 0, 1)  # 0.032 + 1.255*d48_slot
for w in (0.20, 0.35):
    save(test, (1 - w) * rob_cal + w * lookup, f"submission_blend{int(w*100)}.csv")

print("\nReady. submission_robust.csv (CV H1=0.790) already exists as the conservative pick.")
