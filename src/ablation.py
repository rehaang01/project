"""Feature-subset ablation on H1 (temporal forward) + H3 (spatial guardrail).

Build the OOF train matrix and the valid matrix ONCE per split, then refit LightGBM on
different feature subsets to find what maximizes H1 without collapsing H3.
"""
import numpy as np
import pandas as pd
import features as F
import pipeline as P
import cv

TARGET = F.TARGET

CONFIGS = {
    "robust_only":        F.BASE_NUMERIC + ["te_p5", "te_p4", "te_road", "te_road_hour"],
    "robust+geo":         F.BASE_NUMERIC + ["te_p5", "te_p4", "te_road", "te_road_hour", "te_geo"],
    "robust+geo+geohr":   F.BASE_NUMERIC + ["te_p5", "te_p4", "te_road", "te_road_hour", "te_geo", "te_geohr"],
    "robust+geo+geohr+cnt": F.BASE_NUMERIC + ["te_p5", "te_p4", "te_road", "te_road_hour", "te_geo", "te_geohr", "geo_count"],
    "add_slot":           F.BASE_NUMERIC + ["te_p5", "te_p4", "te_road", "te_road_hour", "te_geo", "te_geohr", "te_slot"],
    "add_shift":          F.BASE_NUMERIC + ["te_p5", "te_p4", "te_road", "te_road_hour", "te_geo", "te_geohr", "d48_shift_adj"],
    "full":               P.FEATURE_NAMES,
    "no_latlon_full":     [c for c in P.FEATURE_NAMES if c not in ("lat", "lon")],
}


def eval_config_H1(Xtr_full, ytr, Xva_full, yva, cols):
    m = cv.lgbm_factory()
    m.fit(Xtr_full[cols], ytr)
    p = np.clip(m.predict(Xva_full[cols]), 0, 1)
    return cv.r2(yva, p)


def main():
    train, _ = F.load_raw()
    train = P.prepare(train)

    # ---- H1 matrices (build once) ----
    fit, valid = cv.split_H1(train)
    Xtr = P.build_oof(fit)
    ytr = fit[TARGET].to_numpy()
    Xva = P.build_fit_transform(fit, valid)
    yva = valid[TARGET].to_numpy()

    print(f"{'config':24s}  H1")
    print("-" * 40)
    results = {}
    for name, cols in CONFIGS.items():
        r = eval_config_H1(Xtr, ytr, Xva, yva, cols)
        results[name] = r
        print(f"{name:24s}  {r:.4f}")

    # ---- H3 for the most promising configs ----
    best_for_h3 = ["robust+geo+geohr+cnt", "no_latlon_full", "full"]
    print("\nH3 (spatial guardrail) for selected configs:")
    for name in best_for_h3:
        r3 = cv.eval_H3(cv.lgbm_factory, train, feat_cols=CONFIGS[name])
        print(f"  {name:24s}  H3 = {r3:.4f}")


if __name__ == "__main__":
    main()
