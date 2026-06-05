"""Diverse ensemble library to push past 88.78. Ensemble diversity is the proven lever.
Adds XGBoost + a 2nd CatBoost family, then blends all strong calibrated members. Board-ranked."""
from __future__ import annotations
import json, warnings, numpy as np, pandas as pd
import features as F, pipeline as P, cv
warnings.filterwarnings("ignore")
ROOT, TARGET = F.ROOT, F.TARGET
TUNED = json.load(open(f"{ROOT}/models/best_lgbm_params.json"))["params"]

train, test = F.load_raw()
train = P.prepare(train)
test = P.prepare(test, fit_df_for_impute=pd.read_csv(F.DATA_DIR + "/train.csv"))


def calibrated_test_pred(factory, feats, scale, seeds=(42, 101)):
    """Seed-averaged test prediction, affine-calibrated on pooled H1+H2."""
    def seed_avg(Xtr, ytr, Xte):
        ps = []
        for s in seeds:
            m = factory(s); m.fit(Xtr[feats], ytr); ps.append(np.clip(m.predict(Xte[feats]), 0, 1))
        return np.mean(ps, axis=0)
    # calibration
    pr, ys = [], []
    for split in (cv.split_H1, cv.split_H2):
        fit, val = split(train)
        P.SMOOTH_SCALE = scale
        Xt = P.build_oof(fit); yt = fit[TARGET].to_numpy(); Xv = P.build_fit_transform(fit, val)
        pr.append(seed_avg(Xt, yt, Xv)); ys.append(val[TARGET].to_numpy())
        P.SMOOTH_SCALE = 1.0
    b, a = np.polyfit(np.concatenate(pr), np.concatenate(ys), 1)
    # full fit
    P.SMOOTH_SCALE = scale
    Xtr = P.build_oof(train); ytr = train[TARGET].to_numpy(); Xte = P.build_fit_transform(train, test)
    raw = seed_avg(Xtr, ytr, Xte)
    P.SMOOTH_SCALE = 1.0
    return np.clip(a + b * raw, 0, 1)


def save(p, name):
    s = pd.DataFrame({"Index": test["Index"].to_numpy(), "demand": np.clip(p, 0, 1)})
    assert s.shape == (41778, 2) and list(s.columns) == ["Index", "demand"] and s["demand"].notna().all()
    s.to_csv(f"{ROOT}/submissions/{name}", index=False); print(f"  wrote {name:30s} mean={s['demand'].mean():.4f}")


LOWREG = dict(n_estimators=800, num_leaves=63, min_child_samples=40, learning_rate=0.03,
              subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=0.0, reg_alpha=0.0)

print("Building new diverse members:")
xgb_memo = calibrated_test_pred(lambda s: cv.xgb_factory(dict(random_state=s, max_depth=7,
              min_child_weight=10, n_estimators=900, subsample=0.9, colsample_bytree=0.8)),
              P.MEMO_SET, 0.5); save(xgb_memo, "submission_xgb_memo.csv")
xgb_plus = calibrated_test_pred(lambda s: cv.xgb_factory(dict(random_state=s, max_depth=7,
              min_child_weight=10, n_estimators=900, subsample=0.9, colsample_bytree=0.8)),
              P.MEMO_PLUS, 0.5); save(xgb_plus, "submission_xgb_plus.csv")
cat6 = calibrated_test_pred(lambda s: cv.cat_factory(dict(random_seed=s, depth=6, l2_leaf_reg=5.0,
              iterations=1200)), P.MEMO_SET, 0.6, seeds=(42,)); save(cat6, "submission_cat6_memo.csv")

# ---- assemble library from strong calibrated members (existing + new) ----
def load(n): return pd.read_csv(f"{ROOT}/submissions/{n}")["demand"].to_numpy()
lib = {
    "lgb_memo":        load("submission_memo.csv"),            # 88.32
    "lgb_lowreg":      load("submission_memo_s50_lowreg.csv"), # 88.40
    "lgb_plus":        load("submission_memo_plus.csv"),       # 88.41
    "cat_memo":        load("submission_cat_memo.csv"),
    "cat6_memo":       cat6,
    "xgb_memo":        xgb_memo,
    "xgb_plus":        xgb_plus,
}
M = np.column_stack(list(lib.values()))
print("\nLibrary members:", list(lib.keys()))
print("avg pairwise corr:", round(np.corrcoef(M.T)[np.triu_indices(len(lib),1)].mean(), 4))

# blends to try on the board
save(M.mean(axis=1), "submission_blend_all.csv")                                  # equal weight all 7
# family-balanced: average within family, then across families (LGB/CAT/XGB)
fam = {
    "LGB": np.mean([lib["lgb_memo"], lib["lgb_lowreg"], lib["lgb_plus"]], axis=0),
    "CAT": np.mean([lib["cat_memo"], lib["cat6_memo"]], axis=0),
    "XGB": np.mean([lib["xgb_memo"], lib["xgb_plus"]], axis=0),
}
save(np.mean(list(fam.values()), axis=0), "submission_blend_family.csv")          # 3 families equal
# diversity-weighted: best LGB + both other families
save(0.4*lib["lgb_lowreg"] + 0.3*fam["CAT"] + 0.3*fam["XGB"], "submission_blend_div.csv")
print("done")
