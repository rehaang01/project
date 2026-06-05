import numpy as np
import features as F
import pipeline as P
import cv

TARGET = F.TARGET
REG = dict(n_estimators=400, num_leaves=15, min_child_samples=400, learning_rate=0.03,
           subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=1.0)

RB = F.BASE_NUMERIC + ["te_p5", "te_p4", "te_road", "te_road_hour"]
SETS = {
    "robust_base":     RB,
    "robust+p5hr":     RB + ["te_p5_hour"],
    "robust+p5hr+p4hr+rl": RB + ["te_p5_hour", "te_p4_hour", "te_road_lanes"],
    "robust_plus+geo": RB + ["te_p5_hour", "te_p4_hour", "te_road_lanes", "te_geo"],
    "robust_plus+geo+geohr": RB + ["te_p5_hour", "te_p4_hour", "te_road_lanes", "te_geo", "te_geohr"],
}


def main():
    train, _ = F.load_raw(); train = P.prepare(train)
    fit1, val1 = cv.split_H1(train)
    Xtr = P.build_oof(fit1); ytr = fit1[TARGET].to_numpy()
    Xva = P.build_fit_transform(fit1, val1); yva = val1[TARGET].to_numpy()
    fit2, val2 = cv.split_H2(train)
    Xtr2 = P.build_oof(fit2); ytr2 = fit2[TARGET].to_numpy()
    Xva2 = P.build_fit_transform(fit2, val2); yva2 = val2[TARGET].to_numpy()

    print(f"{'set':24s}  H1      H2      (reg params)")
    for nm, cols in SETS.items():
        m = cv.lgbm_factory(REG); m.fit(Xtr[cols], ytr)
        h1 = cv.r2(yva, np.clip(m.predict(Xva[cols]), 0, 1))
        m2 = cv.lgbm_factory(REG); m2.fit(Xtr2[cols], ytr2)
        h2 = cv.r2(yva2, np.clip(m2.predict(Xva2[cols]), 0, 1))
        print(f"  {nm:22s} {h1:.4f}  {h2:.4f}")

    best = SETS["robust+p5hr+p4hr+rl"]
    print(f"\nH3 robust_plus = {cv.eval_H3(lambda: cv.lgbm_factory(REG), train, feat_cols=best):.4f}")


if __name__ == "__main__":
    main()
