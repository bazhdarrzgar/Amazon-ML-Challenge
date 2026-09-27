"""Experiment: candidate-generation recall on the holdout fold.

Queries = holdout-fold S1 entities of one or more countries; retrieval runs
against the *full* same-country S2+S3 training pool (so matches of other S1
entities act as realistic distractors). Reports entity-level candidate recall
per channel and for unions at several K.

    python -m ber.bench_blocking --countries US India --k-name 50 --k-addr 50 --k-comb 100
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from .candidates import DEFAULT_CFG, generate
from .folds import HOLDOUT, fold_array
from .io import load_ground_truth


def load_norm(work, split):
    s1 = pd.read_parquet(f"{work}/cache/norm_{split}_s1.parquet")
    pool = pd.concat(
        [pd.read_parquet(f"{work}/cache/norm_{split}_s2.parquet"),
         pd.read_parquet(f"{work}/cache/norm_{split}_s3.parquet")],
        ignore_index=True,
    )
    return s1, pool


def recall_table(pairs, s1c, poolc, gt, ks):
    """Entity-level candidate recall for channel/K combinations."""
    q_ids = s1c.entity_id.values
    p_ids = poolc.entity_id.values
    pairs = pairs.assign(s1=q_ids[pairs.q.values], pid=p_ids[pairs.p.values])
    truth = set((k, m) for k in pairs.s1.unique() for m in gt[k])
    n_true = sum(len(gt[k]) for k in s1c.entity_id.values[np.unique(pairs.q.values)])
    rows = []
    key = pairs.s1.values + "|" + pairs.pid.values
    is_true = np.fromiter(((a, b) in truth for a, b in zip(pairs.s1.values, pairs.pid.values)), bool, len(pairs))
    pairs = pairs.assign(y=is_true)
    for spec in ks:
        m = np.zeros(len(pairs), bool)
        for ch, k in spec.items():
            m |= pairs[ch + "_rank"].values < k
        sub = pairs[m]
        rows.append({
            "spec": json.dumps(spec),
            "pair_recall": sub.y.sum() / max(n_true, 1),
            "s2_recall": sub.y[sub.pid.str.startswith("S2")].sum() / max(1, sum(1 for k in sub.s1.unique() for x in gt[k] if x[:2] == "S2")),
            "cand_per_s1": len(sub) / pairs.q.nunique(),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--countries", nargs="+", default=["US", "India"])
    ap.add_argument("--k-name", type=int, default=50)
    ap.add_argument("--k-addr", type=int, default=50)
    ap.add_argument("--k-comb", type=int, default=100)
    ap.add_argument("--w-name", type=float, default=0.5)
    ap.add_argument("--max-df", type=float, default=0.05)
    ap.add_argument("--sample", type=float, default=1.0, help="fraction of holdout S1 used as queries")
    args = ap.parse_args()

    gt = load_ground_truth(args.data_dir, f"{args.work_dir}/cache")
    s1, pool = load_norm(args.work_dir, "train")
    s1["fold"] = fold_array(s1.entity_id.values)
    cfg = {**DEFAULT_CFG, "k_name": args.k_name, "k_addr": args.k_addr, "k_comb": args.k_comb,
           "w_name": args.w_name, "max_df": args.max_df}
    rng = np.random.RandomState(0)
    for c in args.countries:
        t = time.time()
        s1c = s1[s1.country == c].reset_index(drop=True)
        poolc = pool[pool.country == c].reset_index(drop=True)
        qmask = (s1c.fold.values == HOLDOUT) & (rng.rand(len(s1c)) < args.sample)
        print(f"== {c}: S1={len(s1c)} pool={len(poolc)} queries={qmask.sum()}", flush=True)
        pairs, _ = generate(s1c, poolc, cfg, query_mask=qmask)
        specs = []
        for k in (1, 3, 5, 10, 20, 30, 50, 100):
            for ch in ("name", "addr", "comb"):
                if k <= cfg["k_" + ch]:
                    specs.append({ch: k})
        for kn, ka, kc in [(10, 10, 20), (20, 20, 40), (30, 30, 50), (50, 50, 100), (20, 10, 50), (10, 10, 50)]:
            if kn <= cfg["k_name"] and ka <= cfg["k_addr"] and kc <= cfg["k_comb"]:
                specs.append({"name": kn, "addr": ka, "comb": kc})
        tab = recall_table(pairs, s1c, poolc, gt, specs)
        pd.set_option("display.width", 200)
        print(tab.to_string(index=False), flush=True)
        print(f"   ({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
