"""Refined memorization: denoise each (geohash, slot) day-48 value with a local rolling time-window
average within the geohash. Keeps slot-specificity (better than hour-block) while cutting the
single-observation noise that capped the raw lookup at 79.6. Pure day48->day49 lookups (no train leak)."""
import numpy as np, pandas as pd
import features as F

tr, te = F.load_raw()
def tmin(s): h, m = s.split(":"); return int(h) * 60 + int(m)
tr["tmin"] = tr["timestamp"].map(tmin); te["tmin"] = te["timestamp"].map(tmin)
d48 = tr[tr.day == 48]
gm = d48["demand"].mean()
geo = d48.groupby("geohash")["demand"].mean()

# geohash x tmin matrix (rows = all 96 slots, cols = geohash), then rolling-average over time.
mat = d48.pivot_table(index="tmin", columns="geohash", values="demand", aggfunc="mean")
full_t = list(range(0, 1440, 15))
mat = mat.reindex(full_t)


def lookup_smoothed(window):
    sm = mat.rolling(window=window, center=True, min_periods=1).mean()
    stacked = sm.stack()  # index (tmin, geohash)
    key = list(zip(te["tmin"], te["geohash"]))
    vals = pd.Series(stacked.reindex(key).to_numpy(), index=te.index)
    return vals.fillna(te["geohash"].map(geo)).fillna(gm).to_numpy()


def save(p, name):
    s = pd.DataFrame({"Index": te["Index"].to_numpy(), "demand": np.clip(p, 0, 1)})
    assert s.shape == (41778, 2) and list(s.columns) == ["Index", "demand"] and s["demand"].notna().all()
    s.to_csv(f"{F.ROOT}/submissions/{name}", index=False)
    print(f"  wrote {name:34s} mean={s['demand'].mean():.4f}")


print("Denoised-slot pure lookups (window in 15-min slots, centered):")
w3 = lookup_smoothed(3)    # +/-15 min
w5 = lookup_smoothed(5)    # +/-30 min
w7 = lookup_smoothed(7)    # +/-45 min
save(w3, "submission_slotsmooth_w3.csv")
save(w5, "submission_slotsmooth_w5.csv")
save(w7, "submission_slotsmooth_w7.csv")

# blend the best denoised lookup with the aggressive memo GBM (if present)
import os
memo_path = f"{F.ROOT}/submissions/submission_memo_s50_lowreg.csv"
if os.path.exists(memo_path):
    memo = pd.read_csv(memo_path)["demand"].to_numpy()
    save(0.5 * memo + 0.5 * w5, "submission_memo_slotsmooth_blend.csv")
print("done")
