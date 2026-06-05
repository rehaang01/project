# Traffic Demand Prediction — Approach & Methodology

**Task:** predict continuous `demand` for 41,778 test rows. **Metric:** `score = max(0, 100·R²)`.
**Result (honest, leak-free CV):** forward-in-time R² ≈ **0.823**; expected leaderboard score ≈ **82**.

---

## 1. Problem framing

A spatio-temporal **regression**. Each row is one `(geohash, day, timestamp)` cell. Because R² on a
fixed test set is maximized by minimizing MSE on the *raw* target, we optimize squared error on the
untransformed `demand` (a log target was tested and rejected — it under-weights the high-demand rows
that dominate the variance). Predictions are clipped to `[0, 1]` (target is bounded, max = 1.0).

## 2. Data structure (the decisive facts)

* **Temporal spine.** Train = day 48 (full, 96 fifteen-minute slots) + day 49 (9 night slots, 00:00–02:00).
  Test = **day 49, 02:15–13:45 (47 slots)** — the immediate forward continuation, **disjoint in time**
  from the day-49 training slots but **fully covered by day 48**. So the task is a genuine *forecast*.
* **Locations are mostly seen.** 1,180 / 1,190 test geohashes appear in train; **99.9% of test rows**
  (41,753 / 41,778) have a seen geohash. Only 25 rows are new locations.
* **Signal hierarchy (verified):** `RoadType` dominates (Highway 0.61 ≫ Street 0.27 ≫ Residential 0.057);
  strong time-of-day cycle (midday peak ~0.116, evening trough ~0.042); `NumberofLanes` mild (corr 0.21);
  **Weather and Temperature are ~noise** (corr ≈ 0.003, flat group means).
* **Day-to-day drift.** Per-geohash mean demand correlates 0.85 across days; the exact same-slot value
  predicts the next day at only R² 0.49 (raw) / 0.63 (affine-corrected). **An honest R²=1.0 is therefore
  impossible** from the features — the leaderboard's perfect 100s indicate a leaked/recoverable public
  target, which we deliberately do **not** chase. We build for a durable top rank instead.
* **Covariate shift.** The test is enriched in Highway rows (10.2% vs 4.6% in train) and is daytime-only,
  so its mean demand (~0.13–0.15) is legitimately higher than the overall train mean (0.094).
* Missingness is small and balanced (RoadType 0.78%, Temperature 3.23%, Weather 1.03%).

## 3. Validation design (the anti-overfitting core)

All choices are made on leak-free held-out metrics, never on the public score. Every target-derived
feature is **out-of-fold** on the model's own training rows and fit-on-the-fit-set for held-out rows.

* **H1 — temporal forward (PRIMARY).** Train day 48 → validate day 49. Mirrors exactly how test
  features are built; captures real day-to-day drift. This is the faithful proxy for the leaderboard.
* **H2 — daytime block.** Hold out the day-48 02:15–13:45 window (the test *hours*). Captures
  hour-generalization for seen geohashes. (Optimistic — it has same-day history the real test lacks.)
* **H3 — spatial GroupKFold by geohash (guardrail).** Predicts entirely unseen locations.

Because the user's later/private re-test is "**same places, new time**", H1 is the headline and H3 is a
guardrail rather than a co-equal objective.

## 4. Features (`src/features.py`, `src/pipeline.py`)

* **Time:** minute-of-day, hour, cyclic sin/cos, absolute-time ordering.
* **Spatial:** decoded lat/lon (pure-python geohash decoder); prefix-5 (56 regions) and prefix-4
  (6 regions) for robust fallbacks.
* **Categoricals:** RoadType ordinal + one-hot Weather + ordinal LargeVehicles/Landmarks + missing flags;
  Temperature median-imputed (+flag); RoadType imputed from the geohash's mode.
* **Hierarchical smoothed target-encodings** of `demand` at several granularities — slot `(geohash,tmin)`,
  `(geohash,hour)`, `geohash`, `p5`, `p4`, `RoadType`, `RoadType×hour` — each with a coarser-level
  fallback to the global mean, computed out-of-fold to prevent leakage. A day-48 affine lookup
  (`0.032 + 1.255·d48_slot`) and geohash frequency are also provided.

## 5. Modeling journey (every step gated on H1/H2/H3)

| Step | What | H1 | H3 |
|---|---|---|---|
| baseline | RoadType group-mean | 0.755 | 0.746 |
| 1 | LightGBM, all features, default params | 0.712 | 0.579 |
| 2 | **diagnose overfit** → heavy regularization (Optuna) | **0.781** | 0.782 |
| 3 | + affine recalibration (corrects daytime under-prediction) | 0.790 | — |
| 4 | **+ Residential-only memorization (segmented model)** | 0.805 | 0.640 |
| 5 | + seed-averaging + recalibration on segmented model | **0.823** | 0.640 |

**Two findings drove the gains:**

1. **Regularization, not features, was the bottleneck.** The default GBM hit train R²=0.90 / H1=0.71
   (massive overfit). Optuna with `num_leaves≈29, min_child_samples=550, λ=6, α=3.8, colsample=0.5`
   lifted H1 to 0.781 with H3=0.782.
2. **Memorization helps Residential but hurts Highway/Street.** On the forward task, fine geohash/slot
   encodings *hurt* overall (drift makes them noisy). Segment analysis revealed why: **Highway is 71% of
   total variance but within-Highway forward R²≈0** (day-to-day variation there is unpredictable), and
   memorization *catastrophically* overfits Street/Highway forward. But within **Residential** (80% of
   rows, low stable demand) memorization *helps* (within-segment R² 0.17→0.36). So the final model
   **routes by road type**: Residential → memorization features, Street/Highway → robust features. This
   improved *both* H1 (0.781→0.805) and H2 (0.792→0.818) — a genuine, non-overfit gain.

**Recalibration.** The model under-predicts the high-demand daytime test window; an affine map
`a + b·pred` (a≈0.012, b≈1.02, fit on pooled H1+H2 predictions) corrects this for a further ~+0.015 R².

## 6. Final model

Per-segment **seed-averaged (5 seeds) LightGBM**, routed by RoadType, with affine recalibration.
Honest CV: **H1 = 0.823, H2 = 0.825** (consistent → small expected CV↔board gap), H3 = 0.640.

The H3 guardrail is intentionally relaxed: it measures *unseen-location* performance, which is
irrelevant when 99.9% of the test (and the stated future test) shares train's locations. A pure-robust
variant (`submissions/submission_robust.csv`, H1≈0.79 / H3≈0.78) is retained as a hedge if locations
ever change.

## 7. Why this won't overfit

* Out-of-fold target encoding (no row sees its own target); heavy regularization tuned to *minimize*
  the train↔H1 gap; the gain appears on two independent held-out axes (H1 and H2); coarse-region
  fallbacks for unseen keys; predictions clipped to the bounded target range. H1, H2 land within 0.01,
  the signature of a calibrated, non-overfit model.

## 8. Reproduce

```bash
python3 -m venv fpml && fpml/bin/pip install -r requirements.txt
cd src
../fpml/bin/python cv.py            # baseline ladder + H1/H2/H3 for the robust core
../fpml/bin/python train.py         # Optuna tuning + ensemble + robust/memo submissions
../fpml/bin/python finalize.py      # FINAL segmented + calibrated -> submissions/submission.csv
```

## 9. Deliverables

* `submissions/submission.csv` — **PRIMARY** (41,778×2, `Index,demand`).
* `submissions/submission_robust.csv` — pure-robust hedge (durable to new locations).
* `notebooks/01_eda.ipynb` (EDA), `notebooks/02_model.ipynb` (modeling).
* `src/` — `features.py`, `pipeline.py`, `cv.py`, `train.py`, `finalize.py`.
* `reports/approach.md` (this file); `requirements.txt`.
