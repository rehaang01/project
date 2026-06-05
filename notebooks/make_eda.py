"""Generate notebooks/01_eda.ipynb (EDA + drift checks) and execute it.

Run:  ../fpml/bin/python make_eda.py
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
C = []

C.append(new_markdown_cell(
    "# Traffic Demand Prediction — Exploratory Data Analysis\n\n"
    "Regression target `demand` in (0,1]; score = `max(0, 100*R2)`.\n\n"
    "**Structural spine (verified):** train = day 48 (full, 96 slots) + day 49 (9 night slots); "
    "test = day 49 daytime (02:15-13:45, 47 slots). The test is the **immediate forward continuation** "
    "of the city for **mostly-seen** geohashes (1180/1190 seen). So this is a spatio-temporal forecast."
))

C.append(new_code_cell(
    "import sys; sys.path.append('../src')\n"
    "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
    "import features as F, pipeline as P\n"
    "plt.rcParams['figure.dpi']=110\n"
    "train, test = F.load_raw()\n"
    "train = F.add_base_features(train); test = F.add_base_features(test)\n"
    "print('train', train.shape, '| test', test.shape)\n"
    "train[['geohash','day','timestamp','tmin','demand','RoadType','NumberofLanes']].head()"
))

C.append(new_markdown_cell("## 1. Target distribution (raw + log)"))
C.append(new_code_cell(
    "fig,ax=plt.subplots(1,2,figsize=(11,3.4))\n"
    "ax[0].hist(train['demand'],bins=80,color='steelblue'); ax[0].set_title('demand (raw) — right-skewed')\n"
    "ax[1].hist(np.log1p(train['demand']),bins=80,color='indianred'); ax[1].set_title('log1p(demand)')\n"
    "plt.tight_layout(); plt.show()\n"
    "print(train['demand'].describe())"
))

C.append(new_markdown_cell("## 2. Time-of-day cycle (peak ~11-13h, trough ~19h)"))
C.append(new_code_cell(
    "by_hr = train.groupby('hour')['demand'].mean()\n"
    "plt.figure(figsize=(8,3)); plt.plot(by_hr.index, by_hr.values,'-o',ms=3)\n"
    "plt.axvspan(2.25,13.75,alpha=0.12,color='orange',label='test window (02:15-13:45)')\n"
    "plt.xlabel('hour'); plt.ylabel('mean demand'); plt.legend(); plt.title('Demand by hour'); plt.show()"
))

C.append(new_markdown_cell("## 3. RoadType dominates; lanes help; weather/temp do not"))
C.append(new_code_cell(
    "fig,ax=plt.subplots(1,3,figsize=(13,3.3))\n"
    "train.groupby('RoadType')['demand'].mean().plot.bar(ax=ax[0],color='teal'); ax[0].set_title('mean demand by RoadType')\n"
    "train.groupby('NumberofLanes')['demand'].mean().plot.bar(ax=ax[1],color='slateblue'); ax[1].set_title('by NumberofLanes')\n"
    "train.groupby('Weather')['demand'].mean().plot.bar(ax=ax[2],color='gray'); ax[2].set_title('by Weather (~flat)')\n"
    "plt.tight_layout(); plt.show()\n"
    "print('corr(demand,Temperature)=%.4f  corr(demand,NumberofLanes)=%.4f'%(\n"
    "  train['demand'].corr(train['Temperature']), train['demand'].corr(train['NumberofLanes'])))"
))

C.append(new_markdown_cell("## 4. Spatial structure (decoded geohash) + region clusters"))
C.append(new_code_cell(
    "gm = train.groupby('geohash').agg(lat=('lat','first'),lon=('lon','first'),demand=('demand','mean'))\n"
    "plt.figure(figsize=(5.2,4.6)); s=plt.scatter(gm['lon'],gm['lat'],c=gm['demand'],s=10,cmap='viridis')\n"
    "plt.colorbar(s,label='mean demand'); plt.xlabel('lon'); plt.ylabel('lat'); plt.title('Per-geohash mean demand'); plt.show()\n"
    "print('prefix-4 regions:', train['p4'].nunique(), '| prefix-5 regions:', train['p5'].nunique(),\n"
    "      '| geohashes:', train['geohash'].nunique())"
))

C.append(new_markdown_cell(
    "## 5. Cross-day drift — why an honest R2=1.0 is unreachable\n"
    "Per-(geohash, time-of-day) demand on day 48 vs day 49. If demand repeated exactly, all points "
    "would lie on y=x. They scatter (Pearson ~0.79), so the public 100s indicate a leaked/recoverable target, not honest skill."
))
C.append(new_code_cell(
    "piv = train.pivot_table(index=['geohash','tmin'],columns='day',values='demand')\n"
    "piv = piv.dropna()\n"
    "plt.figure(figsize=(4.6,4.4)); plt.scatter(piv[48],piv[49],s=4,alpha=0.3)\n"
    "lim=[0,piv.values.max()]; plt.plot(lim,lim,'r--',lw=1)\n"
    "plt.xlabel('day 48 demand'); plt.ylabel('day 49 demand')\n"
    "plt.title('cross-day (n=%d), r=%.3f'%(len(piv),piv[48].corr(piv[49]))); plt.show()\n"
    "from numpy import polyfit\n"
    "b,a=polyfit(piv[48],piv[49],1); print('affine day49 ~ %.3f + %.3f*day48'%(a,b))"
))

C.append(new_markdown_cell(
    "## 6. Generalization vs memorization (the modeling thesis)\n"
    "Forward (day48->day49) R2 of single predictors. RoadType (stable structure) transfers far better "
    "than fine geohash/slot memorization, which is dragged down by day-to-day drift."
))
C.append(new_code_cell(
    "import cv\n"
    "tr = P.prepare(F.load_raw()[0])\n"
    "for name,val in cv.baseline_ladder(tr):\n"
    "    print(f'  {name:20s} forward R2 = {val:.4f}')"
))

C.append(new_markdown_cell("## 7. Missingness (small, balanced across train/test)"))
C.append(new_code_cell(
    "miss = pd.DataFrame({'train_%':(train.isna().mean()*100).round(2)})\n"
    "miss['test_%']=(test.reindex(columns=train.columns).isna().mean()*100).round(2)\n"
    "miss.loc[['RoadType','Temperature','Weather']]"
))

C.append(new_markdown_cell(
    "### EDA conclusions\n"
    "1. Regression on a bounded, right-skewed target -> optimize MSE on the raw scale, clip to [0,1].\n"
    "2. Dominant signal = RoadType x time-of-day x region; weather/temperature are ~noise.\n"
    "3. Test is a **forward** forecast -> validate train48->day49 (H1). Fine geohash/slot memorization "
    "overfits day 48 and drifts, so the robust structural core is both safer and higher-scoring forward.\n"
    "4. Honest R2=1.0 is impossible (cross-day drift); we build for a durable top rank, not the leaked 100."
))

nb['cells'] = C
nbf.write(nb, '01_eda.ipynb')
print('wrote 01_eda.ipynb with', len(C), 'cells')
