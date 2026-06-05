"""Build the day-48 exact-slot lookup submission — the decisive board test of whether the
test target is the (geohash, time-of-day) value from the previous day (the '100' hypothesis)."""
import numpy as np, pandas as pd
import features as F

tr, te = F.load_raw()
def tmin(s): h,m=s.split(':'); return int(h)*60+int(m)
tr['tmin']=tr['timestamp'].map(tmin); te['tmin']=te['timestamp'].map(tmin)

d48 = tr[tr.day==48]
slot = d48.groupby(['geohash','tmin'])['demand'].mean()
geo  = d48.groupby('geohash')['demand'].mean()
gm   = d48['demand'].mean()

look = pd.Series(te.set_index(['geohash','tmin']).index.map(slot), index=te.index)
cover = look.notna().mean()
look_filled = look.fillna(te['geohash'].map(geo)).fillna(gm)
print(f"test rows with exact day48 (geohash,slot) match: {cover*100:.1f}%")

def save(p, name):
    s = pd.DataFrame({'Index': te['Index'].to_numpy(), 'demand': np.clip(p,0,1)})
    assert s.shape==(41778,2) and list(s.columns)==['Index','demand'] and s['demand'].notna().all()
    s.to_csv(f"{F.ROOT}/submissions/{name}", index=False)
    print(f"  wrote {name:30s} mean={s['demand'].mean():.4f}")

save(look_filled, "submission_lookup_raw.csv")                       # pure day48 value
save(0.032 + 1.255*look_filled, "submission_lookup_affine.csv")      # night-drift-corrected
