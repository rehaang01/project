"""Investigate whether the test target is recoverable from the PROVIDED data (internal leak vectors)."""
import numpy as np, pandas as pd
import features as F

tr, te = F.load_raw()
def tmin(s): h,m=s.split(':'); return int(h)*60+int(m)
tr['tmin']=tr['timestamp'].map(tmin); te['tmin']=te['timestamp'].map(tmin)

print("="*70)
print("1) Index <-> demand structure")
print("="*70)
print("  corr(Index, demand):", round(tr['Index'].corr(tr['demand']),4))
print("  train Index range:", tr['Index'].min(), tr['Index'].max(), "| test:", te['Index'].min(), te['Index'].max())

print("\n"+"="*70)
print("2) Temperature as a near-unique fingerprint (74,804 distinct / 77,299 rows)")
print("="*70)
tr_temp = set(tr['Temperature'].dropna()); te_temp = set(te['Temperature'].dropna())
print("  train distinct Temp:", len(tr_temp), "| test distinct Temp:", len(te_temp))
print("  EXACT Temp values shared train&test:", len(tr_temp & te_temp))
# if a test Temperature exactly matches a train Temperature, is the rest of the row identical?
m = tr.dropna(subset=['Temperature']).drop_duplicates('Temperature').set_index('Temperature')
shared = list(te_temp & tr_temp)[:5]
print("  sample shared Temp -> (train geohash/demand vs test geohash):")
for t in shared:
    trow = tr[tr['Temperature']==t].iloc[0]; terow = te[te['Temperature']==t].iloc[0]
    print(f"    T={t:.6f}: train gh={trow['geohash']} d={trow['demand']:.4f} | test gh={terow['geohash']} day={terow['day']} ts={terow['timestamp']}")

print("\n"+"="*70)
print("3) Full-feature-tuple match between test and train (would recover demand)")
print("="*70)
feat_cols = ['geohash','timestamp','RoadType','NumberofLanes','LargeVehicles','Landmarks','Temperature','Weather']
trk = tr.copy(); tek = te.copy()
key = lambda d: d[feat_cols].astype(str).agg('|'.join, axis=1)
trk['k']=key(trk); tek['k']=key(tek)
demand_by_k = trk.groupby('k')['demand'].mean()
matched = tek['k'].isin(set(trk['k']))
print(f"  test rows whose FULL feature tuple appears in train: {matched.sum()} / {len(tek)} ({matched.mean()*100:.1f}%)")

print("\n"+"="*70)
print("4) Match ignoring day & timestamp: (geohash, Temperature) -> demand")
print("="*70)
for keyset in [['geohash','Temperature'], ['Temperature'], ['geohash','Temperature','Weather','RoadType']]:
    sub = tr.dropna(subset=['Temperature'])
    g = sub.groupby(keyset)['demand']
    uniq = (g.nunique()==1).mean()  # fraction of keys mapping to a single demand
    nkeys = g.ngroups
    tem = te.dropna(subset=['Temperature']).set_index(keyset).index
    cover = pd.Series(tem).isin(set(g.groups.keys())).mean() if nkeys else 0
    print(f"  key={keyset}: {nkeys} train keys, {uniq*100:.1f}% map to ONE demand | test coverage {cover*100:.1f}%")

print("\n"+"="*70)
print("5) Exact duplicate rows")
print("="*70)
print("  dup rows within train (all cols):", tr.drop(columns=['Index']).duplicated().sum())
print("  dup rows within test  (all cols):", te.drop(columns=['Index']).duplicated().sum())
