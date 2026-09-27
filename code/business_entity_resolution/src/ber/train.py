"""Step 3: train the pairwise matcher and evaluate on the holdout fold.

    python -m ber.train --pairs work/pairs/train --tag exp01

* Training rows: pairs of S1 entities in folds != HOLDOUT (optionally a
  sample of S1 entities for speed); early stopping on a slice of fold 4
  (never on the holdout).
* Holdout evaluation: official macro-F0.5 over *all* holdout S1 entities
  (entities without candidates count as empty predictions).
"""
import argparse
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .decide import expected_f_sets, owner_filter, sweep_thresholds, threshold_sets
from .folds import HOLDOUT, fold_array
from .io import load_ground_truth, load_source
from .metric import detailed_report

NON_FEATURES = {"s1_id", "cand_id", "label", "fold"}

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 10,
    "seed": 42,
}


def load_pairs(pairs_dir, s1_filter=None, columns=None):
    parts = sorted(glob.glob(os.path.join(pairs_dir, "*", "part_*.parquet")))
    dfs = []
    for f in parts:
        d = pd.read_parquet(f, columns=columns)
        if s1_filter is not None:
            d = d[s1_filter(d)]
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--pairs", default="work/pairs/train")
    ap.add_argument("--tag", default="exp")
    ap.add_argument("--train-frac", type=float, default=0.35, help="fraction of non-holdout S1 used")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--drop", nargs="*", default=[])
    args = ap.parse_args()
    out = os.path.join(args.work_dir, "models", args.tag)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    rng_keep = lambda d: ((d.fold != HOLDOUT) & (  # noqa: E731
        (pd.util.hash_array(d.s1_id.values.astype(object)) % 1000) < args.train_frac * 1000)) | (d.fold == HOLDOUT)
    df = load_pairs(args.pairs, rng_keep)
    feats = [c for c in feature_cols(df) if c not in args.drop]
    print(f"loaded {len(df)} pairs, {len(feats)} features ({time.time()-t0:.0f}s)", flush=True)

    tr = df[df.fold != HOLDOUT]
    va = df[df.fold == HOLDOUT]
    es_mask = (tr.fold == 4) & ((pd.util.hash_array(tr.s1_id.values.astype(object)) % 10) == 0)
    fit, es = tr[~es_mask], tr[es_mask]
    print(f"train {len(fit)} (pos {fit.label.mean():.3f}) | early-stop {len(es)} | holdout {len(va)}", flush=True)
    dtr = lgb.Dataset(fit[feats].astype(np.float32), fit.label.values, free_raw_data=True)
    des = lgb.Dataset(es[feats].astype(np.float32), es.label.values, reference=dtr)
    t = time.time()
    model = lgb.train(LGB_PARAMS, dtr, args.rounds, valid_sets=[des],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    print(f"trained {model.best_iteration} rounds ({time.time()-t:.0f}s)", flush=True)
    model.save_model(os.path.join(out, "model.txt"), num_iteration=model.best_iteration)

    va = va.copy()
    va["p"] = model.predict(va[feats].astype(np.float32), num_iteration=model.best_iteration)
    va[["s1_id", "cand_id", "label", "p"]].to_parquet(os.path.join(out, "holdout_preds.parquet"), index=False)

    # truth for ALL holdout S1 entities (including ones without candidates)
    gt = load_ground_truth(args.data_dir, f"{args.work_dir}/cache")
    s1 = load_source(args.data_dir, "train", 1, f"{args.work_dir}/cache")
    hold_ids = s1.entity_id.values[fold_array(s1.entity_id.values) == HOLDOUT]
    truth = {k: gt[k] for k in hold_ids}
    cand_recall = va.label.sum() / sum(len(v) for v in truth.values())
    print(f"holdout S1={len(hold_ids)} candidate pair recall={cand_recall:.4f}", flush=True)

    grid = np.round(np.arange(0.2, 0.95, 0.05), 2)
    res = {"cand_recall": cand_recall, "best_iter": model.best_iteration}
    for owner in (False, True):
        sw = sweep_thresholds(va, truth, hold_ids, grid, owner=owner)
        best = sw.loc[sw.macro_f05.idxmax()]
        print(f"threshold sweep owner={owner}:\n" + sw[["t", "macro_f05", "pair_precision", "pair_recall",
                                                        "singleton_acc"]].to_string(index=False), flush=True)
        res[f"thr_owner{int(owner)}"] = best.to_dict()
    t = time.time()
    ef = detailed_report(expected_f_sets(va), truth, hold_ids)
    print(f"expected-F decision: {ef} ({time.time()-t:.0f}s)", flush=True)
    res["expected_f"] = ef
    ef2 = detailed_report(expected_f_sets(owner_filter(va)), truth, hold_ids)
    print(f"expected-F + owner (within-holdout only): {ef2}", flush=True)
    res["expected_f_owner"] = ef2

    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("top features by gain:\n" + (imp / imp.sum()).head(25).round(4).to_string(), flush=True)
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump({"features": feats, "params": LGB_PARAMS, "train_frac": args.train_frac, **res},
                  f, indent=1, default=float)
    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
