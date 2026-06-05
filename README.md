# Traffic Demand Prediction

Solution for the **Traffic Demand Prediction** challenge (Flipkart R1). Predict a continuous traffic
`demand` value for 41,778 test rows; scored by `max(0, 100 · R²)`.

## TL;DR

A spatio-temporal forecast: predict day-49 daytime demand from day-48 history, for mostly-seen
locations. The work is **validation-first and anti-overfitting**, with every modeling choice gated on
leak-free held-out metrics.

| Model | What it is | Leaderboard |
|---|---|---|
| Robust core (`submission_robust.csv`) | Regularized LightGBM on road/region/time — durable to new locations | 78.6 |
| Segmented (`submission.csv`) | + Residential-targeted memorization | 78.6 |
| **Memorization (`submission_memo.csv`)** | Honest day-48 → day-49 daytime pattern repetition | **88.3** |

Key insight: **daytime demand repeats strongly from the previous day**, so memorization of the
provided training data (no external data, no leak) is the dominant honest signal. A literal R²=1.0 is
not honestly reachable — the raw exact-day-48 value scores only ~79.6, proving genuine day-to-day drift.

## Approach (short)

1. **EDA & framing** — regression on a bounded right-skewed target; optimize MSE on the raw scale,
   clip to [0,1]. RoadType dominates; clear time-of-day cycle; weather/temperature are ~noise.
2. **Leak-free features** — cyclic time, decoded lat/lon, prefix-5/4 regions, and hierarchical
   out-of-fold smoothed target-encodings of `demand` at slot / geohash×hour / geohash / region / road
   granularities, with coarser-level fallbacks.
3. **Validation** — H1 (day48→day49 temporal forward), H2 (day-48 daytime block), H3 (GroupKFold by
   geohash). The board later showed daytime memorization is far stronger than the night-only H1 could
   reveal — documented honestly in the journey.
4. **Models** — regularized LightGBM (Optuna-tuned), seed-averaged, with affine recalibration.

Full write-up: [`reports/approach.md`](reports/approach.md).

## Repository layout

```
src/         features.py, pipeline.py, cv.py, train.py, finalize.py  (+ experiment scripts)
notebooks/   01_eda.ipynb, 02_model.ipynb
reports/     approach.md
models/      tuned params, calibration & final metrics (JSON)
submissions/ submission.csv, submission_robust.csv, submission_memo.csv
```

## Reproduce

```bash
python3 -m venv fpml
fpml/bin/pip install -r requirements.txt
# place train.csv / test.csv / sample_submission.csv under dataset/
cd src
../fpml/bin/python cv.py         # baselines + H1/H2/H3
../fpml/bin/python train.py      # Optuna tuning + ensemble
../fpml/bin/python finalize.py   # final submission
```

> Note: the `dataset/` (competition data) and the Python virtualenv `fpml/` are not committed.
