"""Step 5: score the test candidate pairs and write the two submission files.

    python -m ber.predict --models work/models/cv1 --pairs work/pairs/test --out output

Two-stage inference (identical to the validated holdout protocol):
  1. stage-1 probability = mean of the 5 cross-fitted fold models
  2. group / competition features from stage-1 probabilities over the whole
     test candidate graph (``cv.group_features``)
  3. stage-2 probability, then the decision policy stored in decision.json
     (threshold and optional one-owner-per-candidate constraint)

* ``candidate_pairs.tsv``  = exactly the pairs scored by the model
* ``matching_results.tsv`` = the selected subset
Every test S1 entity gets exactly one row in both files (empty when none).
"""
import argparse
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from .cv import group_features, part_files
from .ids import decode, encode
from .io import load_source, write_id_lists


def stage1_scores(files, models, feats):
    rows = []
    for f in tqdm(files, desc="  Scoring stage-1 test parts", unit="part"):
        d = pd.read_parquet(f, columns=["s1_id", "cand_id"] + feats)
        X = d[feats].values.astype(np.float32)
        p = np.mean([m.predict(X) for m in models], axis=0).astype(np.float32)
        rows.append(pd.DataFrame({"s1": encode(d.s1_id.values), "cand": encode(d.cand_id.values), "p": p}))
    return pd.concat(rows, ignore_index=True)


def stage2_scores(files, model, feats1, G):
    out, pos = [], 0
    for f in tqdm(files, desc="  Scoring stage-2 test parts", unit="part"):
        d = pd.read_parquet(f, columns=feats1)
        n = len(d)
        X = np.hstack([d.values.astype(np.float32), G.values[pos:pos + n]])
        out.append(model.predict(X).astype(np.float32))
        pos += n
    assert pos == len(G)
    return np.concatenate(out)


def decide(sc, pcol, threshold, owner):
    d = sc[sc[pcol] >= threshold]
    if owner:
        best = sc.groupby("cand")[pcol].max()
        d = d[d[pcol].values >= best.reindex(d.cand.values).values - 1e-9]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--models", default="work/models/cv1")
    ap.add_argument("--pairs", default="work/pairs/test")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    t0 = time.time()
    with open(os.path.join(args.models, "decision.json")) as f:
        dec = json.load(f)
    with open(os.path.join(args.models, "stage1.json")) as f:
        feats1 = json.load(f)["features"]
    files = part_files(args.pairs)
    m1 = [lgb.Booster(model_file=p) for p in sorted(glob.glob(os.path.join(args.models, "model_s1_f*.txt")))]
    print(f"{len(files)} pair files, {len(m1)} stage-1 models", flush=True)
    sc = stage1_scores(files, m1, feats1)
    pcol = "p"
    if dec.get("stage") == 2:
        m2 = lgb.Booster(model_file=os.path.join(args.models, "model_s2.txt"))
        G = group_features(sc)
        sc["p2"] = stage2_scores(files, m2, feats1, G)
        pcol = "p2"
    sc.to_parquet(os.path.join(args.work_dir, "test_scores.parquet"), index=False)

    s1_ids = load_source(args.data_dir, "test", 1, f"{args.work_dir}/cache").entity_id.values
    sel = decide(sc, pcol, dec["threshold"], dec.get("owner", False))
    # candidate lists / matches keyed by S1 id string, keeping score order
    sc = sc.sort_values(["s1", pcol], ascending=[True, False])
    cand_lists = sc.groupby("s1")["cand"].apply(list)
    sel = sel.sort_values(["s1", pcol], ascending=[True, False])
    match_lists = sel.groupby("s1")["cand"].apply(list)
    s1_codes = encode(s1_ids)
    cand_map = dict(zip(cand_lists.index.values, cand_lists.values))
    match_map = dict(zip(match_lists.index.values, match_lists.values))
    write_id_lists(os.path.join(args.out, "candidate_pairs.tsv"), s1_ids,
                   [decode(cand_map.get(c, [])) for c in s1_codes], "candidate_entity_ids")
    write_id_lists(os.path.join(args.out, "matching_results.tsv"), s1_ids,
                   [decode(match_map.get(c, [])) for c in s1_codes], "matched_entity_ids")
    print(f"wrote {len(s1_ids)} rows | S1 with >=1 match: {len(match_map)} "
          f"({len(match_map)/len(s1_ids):.3f}) | matched ids: {len(sel)} | "
          f"mean matches per S1: {len(sel)/len(s1_ids):.3f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
