"""Find the best per-RoadType segmentation: feature set per segment x (global-route vs segment-specific)."""
import numpy as np, pandas as pd, json
import features as F, pipeline as P, cv
__import__('warnings').filterwarnings('ignore')

BEST = json.load(open('../models/best_lgbm_params.json'))['params']
ROBUST = P.ROBUST_SET
GEO = P.ROBUST_SET + ['te_geo', 'te_geohr']
MEMO = P.MEMO_SET

# configs: per segment-code {0:res,1:street,2:highway} -> feature list
CONFIGS = {
    "all_robust":         {0: ROBUST, 1: ROBUST, 2: ROBUST},
    "res_memo":           {0: MEMO,   1: ROBUST, 2: ROBUST},
    "res_geo":            {0: GEO,    1: ROBUST, 2: ROBUST},
    "res_memo_st_geo":    {0: MEMO,   1: GEO,    2: ROBUST},
    "res_memo_st_memo":   {0: MEMO,   1: MEMO,   2: ROBUST},
}


def eval_config(train, split, cfg, segment_specific):
    fit, val = split(train)
    Xtr = P.build_oof(fit); ytr = fit['demand'].to_numpy()
    Xva = P.build_fit_transform(fit, val); yva = val['demand'].to_numpy()
    rt_tr = fit['RoadType_ord'].to_numpy(); rt_va = val['RoadType_ord'].to_numpy()
    pred = np.zeros(len(val))
    # cache global models by feature-set id
    gcache = {}
    for seg in (0, 1, 2):
        cols = cfg[seg]; mask = rt_va == seg
        if mask.sum() == 0:
            continue
        if segment_specific:
            tmask = rt_tr == seg
            m = cv.lgbm_factory(BEST); m.fit(Xtr[cols][tmask], ytr[tmask])
        else:
            key = tuple(cols)
            if key not in gcache:
                mm = cv.lgbm_factory(BEST); mm.fit(Xtr[cols], ytr); gcache[key] = mm
            m = gcache[key]
        pred[mask] = np.clip(m.predict(Xva[cols][mask]), 0, 1)
    return cv.r2(yva, pred)


def main():
    train, _ = F.load_raw(); train = P.prepare(train)
    print(f"{'config':20s} {'mode':10s}  H1      H2")
    for cname, cfg in CONFIGS.items():
        for ss in (False, True):
            h1 = eval_config(train, cv.split_H1, cfg, ss)
            h2 = eval_config(train, cv.split_H2, cfg, ss)
            print(f"  {cname:18s} {'segspec' if ss else 'global':10s} {h1:.4f}  {h2:.4f}")


if __name__ == "__main__":
    main()
