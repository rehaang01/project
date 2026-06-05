"""Targeted diagnostics to resolve: (A) best achievable H1, (B) memorization value on H2,
(C) overfit gap, (D) blend of structural model + day48 lookup."""
import numpy as np
import features as F
import pipeline as P
import cv

TARGET = F.TARGET


def lgbm(**kw):
    return lambda: cv.lgbm_factory(kw)


def main():
    train, _ = F.load_raw()
    train = P.prepare(train)
    fit1, val1 = cv.split_H1(train)
    Xtr = P.build_oof(fit1); ytr = fit1[TARGET].to_numpy()
    Xva = P.build_fit_transform(fit1, val1); yva = val1[TARGET].to_numpy()

    def h1(cols, **kw):
        m = cv.lgbm_factory(kw); m.fit(Xtr[cols], ytr)
        tr_r2 = cv.r2(ytr, np.clip(m.predict(Xtr[cols]), 0, 1))
        va_r2 = cv.r2(yva, np.clip(m.predict(Xva[cols]), 0, 1))
        return tr_r2, va_r2

    print("(A) H1 — can a GBM beat RoadType_mean=0.755? (train_R2 / H1)")
    sets = {
        "te_road only":           ["te_road"],
        "road+hour":              ["te_road", "hour", "sin_t", "cos_t"],
        "road+hour+lanes":        ["te_road", "hour", "sin_t", "cos_t", "NumberofLanes"],
        "road_hour interaction":  ["te_road_hour", "te_road", "hour", "sin_t", "cos_t", "NumberofLanes"],
        "robust_only":            F.BASE_NUMERIC + ["te_p5", "te_p4", "te_road", "te_road_hour"],
    }
    for nm, cols in sets.items():
        tr, va = h1(cols)
        print(f"  {nm:24s} {tr:.3f} / {va:.4f}")

    print("\n(A2) robust_only under stronger regularization:")
    rob = F.BASE_NUMERIC + ["te_p5", "te_p4", "te_road", "te_road_hour"]
    for kw in [dict(n_estimators=300), dict(n_estimators=300, num_leaves=15, min_child_samples=400),
               dict(n_estimators=150, num_leaves=15, learning_rate=0.05, min_child_samples=400),
               dict(n_estimators=600, num_leaves=15, min_child_samples=300, reg_lambda=5)]:
        tr, va = h1(rob, **kw)
        print(f"  {str(kw):70s} {tr:.3f} / {va:.4f}")

    print("\n(D) Blend structural GBM + day48 slot-affine on H1:")
    m = cv.lgbm_factory(); m.fit(Xtr[rob], ytr)
    p_struct = np.clip(m.predict(Xva[rob]), 0, 1)
    p_slot = np.clip(Xva["d48_shift_adj"].to_numpy(), 0, 1)
    for w in [0.0, 0.2, 0.3, 0.4, 0.5]:
        blend = (1 - w) * p_struct + w * p_slot
        print(f"  w_slot={w:.1f}  H1={cv.r2(yva, blend):.4f}")

    # ---- (B) H2: does memorization help on daytime? ----
    print("\n(B) H2 daytime-block (same-day) — robust vs memorization:")
    fit2, val2 = cv.split_H2(train)
    Xtr2 = P.build_oof(fit2); ytr2 = fit2[TARGET].to_numpy()
    Xva2 = P.build_fit_transform(fit2, val2); yva2 = val2[TARGET].to_numpy()
    for nm, cols in {"robust_only": rob,
                     "robust+geo+geohr": rob + ["te_geo", "te_geohr"],
                     "full": P.FEATURE_NAMES}.items():
        m = cv.lgbm_factory(); m.fit(Xtr2[cols], ytr2)
        print(f"  {nm:20s} H2 = {cv.r2(yva2, np.clip(m.predict(Xva2[cols]),0,1)):.4f}")

    # ---- (C) H3 robust_only ----
    print("\n(C) H3 spatial guardrail robust_only:")
    print(f"  robust_only H3 = {cv.eval_H3(cv.lgbm_factory, train, feat_cols=rob):.4f}")


if __name__ == "__main__":
    main()
