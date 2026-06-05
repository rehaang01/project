# Traffic Demand Prediction

Solution for the **Traffic Demand Prediction** challenge (Flipkart R1). Predict a continuous traffic
`demand` value for 41,778 test rows; scored by `max(0, 100 · R²)`.

## Results

| Model | What it is | Leaderboard | Script |
|---|---|---|---|
| **High-score** (`submission_highscore.csv`) | NNLS-stacked LightGBM×2 + CatBoost ensemble exploiting day-to-day pattern repetition | **88.96** | `src/final_model.py` |
| Robust (`submission_robust.csv`) | regularized model that generalizes to unseen locations | 78.6 | `src/finalize.py` |

Both use **only the provided training data** — no external data, no leak.

## Key idea

A spatio-temporal forecast: predict day-49 daytime demand from day-48 history for mostly-seen locations.
**Daytime demand repeats strongly from the previous day**, so honestly modeling the day-48 spatio-temporal
pattern (the *expected* per-geohash-per-time demand, denoised by a gradient-boosted ensemble) is the
dominant signal — taking the score from a robust 78.6 baseline to **88.96** (an NNLS meta-learner over
LightGBM×2 + CatBoost added the final +0.18 over a hand-tuned blend).

A literal R²=1.0 is **not honestly reachable**: the raw exact-day-48 value scores only ~79.6, proving
genuine day-to-day drift (honest ceiling ≈ 0.90). The leaderboard's 100s require the external source
dataset's labels, which this solution does not use.

Full write-up: [`reports/approach.md`](reports/approach.md).

## Repository layout

```
src/         features.py, pipeline.py, cv.py        # data, leak-free features, validation harness
             train.py, finalize.py, final_model.py  # robust model + high-score ensemble
             (ablation, diag, seg_experiment,       # model-selection & memorization experiments
              leak_probe, memo_push, push2/3, build_lib)
notebooks/   01_eda.ipynb, 02_model.ipynb
reports/     approach.md
models/      tuned params, calibration & metrics (JSON)
submissions/ submission_highscore.csv, submission_robust.csv
```

## Reproduce

```bash
python3 -m venv fpml
fpml/bin/pip install -r requirements.txt
# place train.csv / test.csv / sample_submission.csv under dataset/
cd src
../fpml/bin/python cv.py            # baselines + H1/H2/H3 validation
../fpml/bin/python final_model.py   # high-score ensemble  -> submission_highscore.csv
../fpml/bin/python finalize.py      # robust model         -> submission_robust.csv
```

> The Python virtualenv `fpml/` and the competition `dataset/` are not committed.
