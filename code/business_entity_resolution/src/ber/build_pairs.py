"""Step 2: candidate generation + pair features for one split.

For each country label present in the split's Source-1 file:
  1. retrieve candidate pairs (``candidates.generate``)
  2. compute label-free context features over the full candidate graph
  3. compute pair features in blocks of S1 records and write parquet parts

Output: WORK/pairs/{split}/{country}/part_XXXX.parquet with columns
  s1_id, cand_id, [label, fold for train], features...

    python -m ber.build_pairs --split train
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from .candidates import DEFAULT_CFG, add_name_block, build_vectors, generate, name_block_channel
from .features import ambiguity_features, context_features, pair_features, rowwise_dot, word_idf_matrix
from .folds import fold_array
from .io import load_ground_truth


def load_norm(work, split):
    s1 = pd.read_parquet(f"{work}/cache/norm_{split}_s1.parquet")
    pool = pd.concat(
        [pd.read_parquet(f"{work}/cache/norm_{split}_s2.parquet"),
         pd.read_parquet(f"{work}/cache/norm_{split}_s3.parquet")],
        ignore_index=True,
    )
    return s1, pool


def process_country(s1c, poolc, out_dir, cfg, gt=None, block=150_000, max_rows=None, cand_cache=None,
                    name_block=False):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    if cand_cache and os.path.exists(cand_cache):
        pairs = pd.read_parquet(cand_cache)
        vecs = build_vectors(s1c, poolc, {**DEFAULT_CFG, **cfg})
        print(f"  loaded cached candidates {cand_cache}: {len(pairs)} pairs ({time.time()-t0:.0f}s)", flush=True)
    else:
        pairs, vecs = generate(s1c, poolc, cfg)
        if cand_cache:
            os.makedirs(os.path.dirname(cand_cache), exist_ok=True)
            pairs.to_parquet(cand_cache, index=False)
    if name_block:
        nb_cache = cand_cache.replace(".parquet", "_nb.parquet") if cand_cache else None
        if nb_cache and os.path.exists(nb_cache):
            nb = pd.read_parquet(nb_cache)
        else:
            nb = name_block_channel(s1c, poolc)
            if nb_cache:
                nb.to_parquet(nb_cache, index=False)
        pairs, n_new = add_name_block(pairs, nb, vecs)
        del nb
        print(f"  + name-block channel: {n_new} new pairs -> {len(pairs)} ({len(pairs)/len(s1c):.1f}/S1)", flush=True)
    # name / address cosines for every pair, then label-free graph/context features
    t = time.time()
    qi, pi = pairs["q"].values.astype(np.int64), pairs["p"].values.astype(np.int64)
    for ch in ("name", "addr"):
        pairs["cos_" + ch] = rowwise_dot(vecs[ch][0], vecs[ch][1], qi, pi)
    ctx = pd.concat([context_features(pairs), ambiguity_features(s1c, poolc, pairs)], axis=1)
    print(f"  context features ({time.time()-t:.0f}s)", flush=True)
    t = time.time()
    both_n = [c or f for c, f in zip(np.concatenate([s1c.n_core.values, poolc.n_core.values]),
                                     np.concatenate([s1c.n_full.values, poolc.n_full.values]))]
    Xn = word_idf_matrix(both_n)
    Xa = word_idf_matrix(list(np.concatenate([s1c.a_core.values, poolc.a_core.values])))
    n1 = len(s1c)
    word_mats = {"name": (Xn[:n1], Xn[n1:]), "addr": (Xa[:n1], Xa[n1:])}
    print(f"  word idf matrices ({time.time()-t:.0f}s)", flush=True)

    s1_ids = s1c.entity_id.values
    pool_ids = poolc.entity_id.values
    owner = None
    if gt is not None:
        owner = {}
        for k in s1_ids:
            for m in gt[k]:
                owner[m] = k
    q_all = pairs["q"].values
    n_s1 = len(s1c) if max_rows is None else min(max_rows, len(s1c))
    starts = list(range(0, n_s1, block))
    for part, start in enumerate(tqdm(starts, desc=f"  Featurising blocks", unit="block", leave=False)):
        t = time.time()
        end = min(start + block, n_s1)
        lo, hi = np.searchsorted(q_all, [start, end])
        pb = pairs.iloc[lo:hi].reset_index(drop=True)
        F = pair_features(pb, s1c, poolc, vecs, word_mats)
        F = pd.concat([F, ctx.iloc[lo:hi].reset_index(drop=True)], axis=1)
        F.insert(0, "cand_id", pool_ids[pb.p.values])
        F.insert(0, "s1_id", s1_ids[pb.q.values])
        if owner is not None:
            F["label"] = np.fromiter((owner.get(c) == s for s, c in zip(F.s1_id.values, F.cand_id.values)),
                                     np.int8, len(F))
            F["fold"] = fold_array(F.s1_id.values)
        F.to_parquet(os.path.join(out_dir, f"part_{part:04d}.parquet"), index=False)
        print(f"  ✓ block {part+1}/{len(starts)} ({start:,}-{end:,}): {len(F):,} pairs ({time.time()-t:.1f}s)", flush=True)
    # S1 records of this country with no candidate at all (still need an output row)
    print(f"  ✓ country {out_dir} completed in {time.time()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--max-rows", type=int, default=None, help="debug: featurise only first N S1 rows")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--name-block", action="store_true", help="add the exact-name-block address channel")
    args = ap.parse_args()
    s1, pool = load_norm(args.work_dir, args.split)
    gt = load_ground_truth(args.data_dir, f"{args.work_dir}/cache") if args.split == "train" else None
    countries = args.countries or sorted(s1.country.unique())
    base = os.path.join(args.work_dir, "pairs", args.out_name or args.split)
    step_num = "2/6" if args.split == "train" else "3/6"
    for c in tqdm(countries, desc=f"Step {step_num}: Candidate pairs ({args.split})", unit="country"):
        print(f"\n== Processing {args.split} / {c} (S1: {len(s1[s1.country==c]):,}, Pool: {len(pool[pool.country==c]):,}) ==", flush=True)
        s1c = s1[s1.country == c].reset_index(drop=True)
        poolc = pool[pool.country == c].reset_index(drop=True)
        cache = os.path.join(args.work_dir, "cand", args.split, f"{c}.parquet")
        process_country(s1c, poolc, os.path.join(base, c), DEFAULT_CFG, gt, max_rows=args.max_rows,
                        cand_cache=cache, name_block=args.name_block)


if __name__ == "__main__":
    main()
