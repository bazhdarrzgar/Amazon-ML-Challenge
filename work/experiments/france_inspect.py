"""Qualitative check of the unseen country: print random France S1 records with their
top-scored candidates (raw text), marking which ones the decision policy selected."""
import sys
sys.path.insert(0, "code/business_entity_resolution/src")
import numpy as np, pandas as pd
from ber.io import load_source
from ber.ids import encode

country = sys.argv[1] if len(sys.argv) > 1 else "France"
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.72
n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
sc = pd.read_parquet("work/test_scores.parquet")
pcol = "p2" if "p2" in sc else "p"
raw = {}
for s in (1, 2, 3):
    d = load_source("dataset", "test", s, "work/cache")
    d = d[d.country == country]
    raw.update(zip(encode(d.entity_id.values), zip(d.entity_id, d.business_name, d.business_address)))
s1 = [c for c in raw if raw[c][0].startswith("S1-")]
rng = np.random.RandomState(1)
sc = sc[sc.s1.isin(set(s1))]
for c in rng.choice(s1, n, replace=False):
    g = sc[sc.s1 == c].sort_values(pcol, ascending=False).head(6)
    print("=" * 100)
    print("S1:", raw[c])
    for r in g.itertuples():
        mark = "MATCH" if getattr(r, pcol) >= thr else "     "
        print(f"  {mark} p1={r.p:.3f} p2={getattr(r, pcol):.3f} {raw[r.cand]}")
