"""Unsupervised diagnostics of test predictions per country (no labels needed).

Compares, per country, the predicted match cardinality, the fraction of S1
records predicted empty, the fraction of pool records claimed, and the score
distribution of the best candidate. Used to sanity-check the unseen country
(France) against the countries seen in training.

    python -m ber.diagnose --scores work/test_scores.parquet --threshold 0.7
"""
import argparse

import numpy as np
import pandas as pd

from .ids import encode
from .io import load_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--scores", default="work/test_scores.parquet")
    ap.add_argument("--pcol", default=None)
    ap.add_argument("--threshold", type=float, required=True)
    args = ap.parse_args()
    sc = pd.read_parquet(args.scores)
    pcol = args.pcol or ("p2" if "p2" in sc else "p")
    rows = []
    for s in (1, 2, 3):
        d = load_source(args.data_dir, "test", s, f"{args.work_dir}/cache")[["entity_id", "country"]]
        d["code"] = encode(d.entity_id.values)
        rows.append(d.assign(src=s))
    ents = pd.concat(rows)
    cty = dict(zip(ents.code.values, ents.country.values))
    sc["country"] = sc.s1.map(cty)
    s1 = ents[ents.src == 1]
    sel = sc[sc[pcol] >= args.threshold]
    out = []
    for c, g in s1.groupby("country"):
        n = len(g)
        sc_c = sc[sc.country == c]
        sel_c = sel[sel.country == c]
        k = sel_c.groupby("s1").size().reindex(g.code.values).fillna(0).values
        top = sc_c.groupby("s1")[pcol].max().reindex(g.code.values).fillna(0).values
        pool_n = int((ents[(ents.src > 1)].country == c).sum())
        out.append({
            "country": c, "S1": n, "pool": pool_n, "pool_per_S1": pool_n / n,
            "cand_per_S1": len(sc_c) / n,
            "pred_per_S1": k.mean(), "frac_empty": (k == 0).mean(),
            "frac_pool_claimed": len(sel_c) / pool_n,
            "top1_p_median": float(np.median(top)), "top1_p<0.5": float((top < 0.5).mean()),
            "k=1": (k == 1).mean(), "k=2": (k == 2).mean(), "k=3": (k == 3).mean(), "k>=4": (k >= 4).mean(),
            "multi_owner_cands": int(sel_c.cand.duplicated().sum()),
        })
    pd.set_option("display.width", 250)
    print(pd.DataFrame(out).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
