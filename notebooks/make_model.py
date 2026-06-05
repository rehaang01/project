"""Generate notebooks/02_model.ipynb (modeling story + reproducible key results)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook(); C = []

C.append(new_markdown_cell(
    "# Traffic Demand Prediction — Modeling\n\n"
    "Validation-first workflow. Headline = **H1 temporal-forward R2** (train day 48 -> day 49); "
    "H2 = daytime block; H3 = spatial guardrail (unseen geohashes). All target-encodings are "
    "out-of-fold (leak-free). See `reports/approach.md` for the full write-up."
))

C.append(new_code_cell(
    "import sys; sys.path.append('../src')\n"
    "import numpy as np, pandas as pd, json\n"
    "import features as F, pipeline as P, cv\n"
    "import warnings; warnings.filterwarnings('ignore')\n"
    "train, test = F.load_raw(); train = P.prepare(train)\n"
    "print('train', train.shape, '| test', test.shape)"
))

C.append(new_markdown_cell(
    "## 1. Baseline ladder (forward day48->day49)\n"
    "RoadType (stable structure) transfers best; fine geohash/slot memorization is dragged down by "
    "day-to-day drift. Confirms the structural core, not memorization, is the foundation."
))
C.append(new_code_cell(
    "for name, val in cv.baseline_ladder(train):\n"
    "    print(f'  {name:20s} forward R2 = {val:.4f}')"
))

C.append(new_markdown_cell(
    "## 2. The overfitting fix (the key lesson)\n"
    "Default LightGBM memorizes day 48 (train R2 ~0.90) but generalizes poorly (H1 0.71). "
    "Heavy regularization closes the gap and lifts H1 above the baseline."
))
C.append(new_code_cell(
    "best = json.load(open('../models/best_lgbm_params.json'))['params']\n"
    "fit, val = cv.split_H1(train)\n"
    "Xtr = P.build_oof(fit); ytr = fit['demand'].to_numpy()\n"
    "Xva = P.build_fit_transform(fit, val); yva = val['demand'].to_numpy()\n"
    "import lightgbm as lgb\n"
    "default = cv.lgbm_factory(); default.fit(Xtr[P.ROBUST_SET], ytr)\n"
    "tuned = cv.lgbm_factory(best); tuned.fit(Xtr[P.ROBUST_SET], ytr)\n"
    "for nm, m in [('default', default), ('tuned (Optuna)', tuned)]:\n"
    "    tr = cv.r2(ytr, np.clip(m.predict(Xtr[P.ROBUST_SET]),0,1))\n"
    "    va = cv.r2(yva, np.clip(m.predict(Xva[P.ROBUST_SET]),0,1))\n"
    "    print(f'  {nm:16s} train R2 = {tr:.3f}   H1 = {va:.4f}')"
))

C.append(new_markdown_cell(
    "## 3. Segmentation: memorization helps Residential, hurts Highway/Street\n"
    "Highway is ~71% of total variance but within-Highway forward R2 ~ 0 (unpredictable drift). "
    "Within Residential (80% of rows), the geohash signal is stable and helps. So we route by RoadType."
))
C.append(new_code_cell(
    "rt = val['RoadType_ord'].to_numpy()\n"
    "p_rob  = np.clip(tuned.predict(Xva[P.ROBUST_SET]),0,1)\n"
    "memo_m = cv.lgbm_factory(best); memo_m.fit(Xtr[P.MEMO_SET], ytr)\n"
    "p_memo = np.clip(memo_m.predict(Xva[P.MEMO_SET]),0,1)\n"
    "for name,code in [('Residential',0),('Street',1),('Highway',2)]:\n"
    "    m = rt==code\n"
    "    print(f'  within-{name:11s} robust R2={cv.r2(yva[m],p_rob[m]):7.3f}  memo R2={cv.r2(yva[m],p_memo[m]):7.3f}')\n"
    "seg = p_rob.copy(); seg[rt==0] = p_memo[rt==0]\n"
    "print(f'\\n  overall  robust={cv.r2(yva,p_rob):.4f}  full-memo={cv.r2(yva,p_memo):.4f}  SEGMENTED={cv.r2(yva,seg):.4f}')"
))

C.append(new_markdown_cell(
    "## 4. Final model metrics (segmented + seed-averaged + calibrated)\n"
    "Produced by `src/finalize.py`. H1 and H2 land within 0.01 — the signature of a non-overfit model."
))
C.append(new_code_cell(
    "fm = json.load(open('../models/final_metrics.json'))\n"
    "print('FINAL  H1 (forward) raw=%.4f -> calibrated=%.4f' % (fm['H1_raw'], fm['H1_cal']))\n"
    "print('       H2 (daytime) raw=%.4f -> calibrated=%.4f' % (fm['H2_raw'], fm['H2_cal']))\n"
    "print('       H3 (spatial guardrail)        = %.4f' % fm['H3'])\n"
    "print('       calibration a=%.4f b=%.4f' % (fm['a'], fm['b']))\n"
    "sub = pd.read_csv('../submissions/submission.csv')\n"
    "print('\\nsubmission.csv shape', sub.shape, '| mean pred %.4f' % sub['demand'].mean()); sub.head()"
))

C.append(new_markdown_cell(
    "### Conclusion\n"
    "Honest forward-in-time R2 ≈ **0.823** (expected leaderboard ≈ 82). Built for generalization, not "
    "for the leaked public 100s. Every gain (regularization, segmentation, calibration) was validated on "
    "two independent held-out axes."
))

nb['cells'] = C
nbf.write(nb, '02_model.ipynb')
print('wrote 02_model.ipynb with', len(C), 'cells')
