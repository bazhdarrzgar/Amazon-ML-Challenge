"""Cross-fitted two-stage matcher.

Stage 1  pairwise LightGBM on pair features. 5 fold models; model k is trained
         on (a sample of) S1 entities outside fold k and predicts fold k, giving
         out-of-fold (OOF) probabilities for every training pair. At test time
         the 5 models are averaged.
Stage 2  LightGBM on stage-1 features + *group* features computed from the
         stage-1 probabilities of the whole candidate graph:
           - S1 side : rank / gap to best / sum / count>0.5 of p in the S1 group
           - candidate side (each S2/S3 belongs to <=1 S1): best competing p
             from other S1 records, margin, rank, #competitors above 0.5
         Trained on folds != HOLDOUT (OOF inputs), evaluated on HOLDOUT.

    python -m ber.cv stage1 --tag cv1
    python -m ber.cv stage2 --tag cv1
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

from .fasteval import FastEval
from .folds import HOLDOUT, N_FOLDS, fold_array
from .ids import encode
from .io import load_ground_truth, load_source
from .metric import detailed_report
from .train import LGB_PARAMS, NON_FEATURES

KEY = ["s1", "cand"]


def params(args):
    return {**LGB_PARAMS, "learning_rate": args.lr}


def part_files(pairs_dir):
    return sorted(glob.glob(os.path.join(pairs_dir, "*", "part_*.parquet")))


def sample_mask(s1_codes, frac, salt=0):
    """Deterministic S1-level sample (all pairs of a sampled S1 are kept)."""
    h = (s1_codes * 2654435761 + salt) % 1000003
    return h < frac * 1000003


def read_part(f, feats=None):
    d = pd.read_parquet(f)
    d.insert(0, "s1", encode(d.s1_id.values))
    d.insert(1, "cand", encode(d.cand_id.values))
    d = d.drop(columns=["s1_id", "cand_id"])
    return d


def holdout_truth(data_dir, work_dir):
    gt = load_ground_truth(data_dir, f"{work_dir}/cache")
    s1 = load_source(data_dir, "train", 1, f"{work_dir}/cache")
    ids = s1.entity_id.values[fold_array(s1.entity_id.values) == HOLDOUT]
    truth = {int(encode([k])[0]): [int(x) for x in encode(gt[k])] if gt[k] else [] for k in ids}
    return truth, list(truth)


def evaluate(df, truth, ids, pcol="p", label="", owner_full=None):
    """Decision-policy comparison on the holdout (df = holdout pairs, s1/cand codes)."""
    fe = FastEval(df, truth, ids, pcol=pcol)
    out = {}
    grid = np.round(np.arange(0.20, 0.96, 0.01), 2)
    curve = [(t, *fe.score(fe.threshold(t))) for t in grid]
    t_best, f_best, info = max(curve, key=lambda r: r[1])
    print(f"[{label}] threshold: best t={t_best} F={f_best:.5f} " + " ".join(f"{k}={v:.4f}" for k, v in info.items()),
          flush=True)
    print("   curve: " + " ".join(f"{t:.2f}:{f:.5f}" for t, f, _ in curve[::5]), flush=True)
    out["thr"] = {"t": float(t_best), "macro_f05": f_best, **info}
    out["thr_curve"] = [(float(t), f) for t, f, _ in curve]
    if owner_full is not None:
        # one-owner constraint using OOF scores of *all* S1 records competing for a candidate
        best_other = owner_full.reindex(fe.cand).values
        own = fe.p >= best_other - 1e-9
        res = [(t, *fe.score(fe.threshold(t) & own)) for t in grid]
        t2, f2, info2 = max(res, key=lambda r: r[1])
        print(f"[{label}] owner+threshold: best t={t2} F={f2:.5f} " + " ".join(f"{k}={v:.4f}" for k, v in info2.items()),
              flush=True)
        out["owner_thr"] = {"t": float(t2), "macro_f05": f2, **info2}
    fb = max(((t, u, fe.score(fe.fallback(t, u))[0]) for t in grid[::2] for u in np.arange(0.1, t, 0.1)),
             key=lambda r: r[2])
    print(f"[{label}] threshold+top1-fallback: best t={fb[0]} t_fb={fb[1]:.1f} F={fb[2]:.5f}", flush=True)
    out["fallback"] = {"t": float(fb[0]), "t_fb": float(fb[1]), "macro_f05": fb[2]}
    return out


# --------------------------------------------------------------------------- stage 1
def stage1(args):
    out = os.path.join(args.work_dir, "models", args.tag)
    os.makedirs(out, exist_ok=True)
    files = part_files(args.pairs)
    t0 = time.time()
    # 1) sampled training rows (all folds) for fitting the 5 fold models
    feats = None
    Xs, ys, fs, ss = [], [], [], []
    for f in tqdm(files, desc="  Loading training parts", unit="part"):
        d = read_part(f)
        if feats is None:
            feats = [c for c in d.columns if c not in NON_FEATURES | set(KEY) and c not in args.drop]
        d = d[sample_mask(d.s1.values, args.train_frac)]
        Xs.append(d[feats].values.astype(np.float32))
        ys.append(d.label.values)
        fs.append(d.fold.values)
        ss.append(d.s1.values)
        del d
    X = np.vstack(Xs)
    del Xs
    y, fold, s1c = np.concatenate(ys), np.concatenate(fs), np.concatenate(ss)
    print(f"stage1: {len(X):,} sampled rows, {len(feats)} feats ({time.time()-t0:.1f}s)", flush=True)
    es_flag = sample_mask(s1c, 0.05, salt=7)
    full = lgb.Dataset(X, y, free_raw_data=False, params={"max_bin": 255})
    full.construct()
    models = []
    for k in tqdm(range(N_FOLDS), desc="  Training 5 fold models", unit="fold"):
        t = time.time()
        idx_fit = np.flatnonzero((fold != k) & ~es_flag)
        idx_es = np.flatnonzero((fold != k) & es_flag)
        dfit = full.subset(idx_fit)
        des = full.subset(idx_es)
        m = lgb.train(params(args), dfit, args.rounds, valid_sets=[des],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        m.save_model(os.path.join(out, f"model_s1_f{k}.txt"), num_iteration=m.best_iteration)
        models.append(m)
        print(f"  ✓ Fold model {k}: {m.best_iteration} iterations, es logloss "
              f"{m.best_score['valid_0']['binary_logloss']:.5f} ({time.time()-t:.1f}s)", flush=True)
    del full, X
    imp = pd.Series(np.mean([m.feature_importance("gain") for m in models], 0), index=feats)
    print("top gain:\n" + (imp / imp.sum()).sort_values(ascending=False).head(20).round(4).to_string())
    # 2) OOF predictions for every pair
    oof = []
    for f in tqdm(files, desc="  Generating OOF predictions", unit="part"):
        d = read_part(f)
        p = np.zeros(len(d), np.float32)
        Xd = d[feats].values.astype(np.float32)
        for k in range(N_FOLDS):
            m = d.fold.values == k
            if m.any():
                p[m] = models[k].predict(Xd[m], num_iteration=models[k].best_iteration)
        oof.append(pd.DataFrame({"s1": d.s1.values, "cand": d.cand.values, "label": d.label.values,
                                 "fold": d.fold.values, "p": p}))
    oof = pd.concat(oof, ignore_index=True)
    oof.to_parquet(os.path.join(out, "oof_stage1.parquet"), index=False)
    with open(os.path.join(out, "stage1.json"), "w") as fh:
        json.dump({"features": feats, "params": params(args), "train_frac": args.train_frac}, fh, indent=1)
    print(f"OOF written: {len(oof):,} pairs ({time.time()-t0:.1f}s)", flush=True)
    report(oof, args, "stage1")


def report(oof, args, label, pcol="p"):
    truth, ids = holdout_truth(args.data_dir, args.work_dir)
    hold = oof[oof.fold == HOLDOUT]
    n_true = sum(len(v) for v in truth.values())
    print(f"[{label}] holdout S1={len(ids)} pairs={len(hold)} candidate recall={hold.label.sum()/n_true:.4f}")
    # best S1 score per candidate across ALL S1 (OOF) for the owner constraint
    best = oof.groupby("cand")[pcol].max()
    res = evaluate(hold, truth, ids, pcol=pcol, label=label, owner_full=best)
    out = os.path.join(args.work_dir, "models", args.tag)
    with open(os.path.join(out, f"eval_{label}.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


# --------------------------------------------------------------------------- stage 2
def group_features(sc, pcol="p"):
    """Stage-2 features from stage-1 probabilities over the whole candidate graph."""
    g = pd.DataFrame({"s1": sc.s1.values, "cand": sc.cand.values, "p": sc[pcol].values})
    G = pd.DataFrame(index=g.index)
    gq = g.groupby("s1")["p"]
    G["g_p"] = g.p.values
    G["g_q_max"] = gq.transform("max").values
    G["g_q_gap"] = G.g_q_max.values - g.p.values
    G["g_q_sum"] = gq.transform("sum").values
    G["g_q_n50"] = g.assign(x=(g.p > 0.5).astype(np.int16)).groupby("s1")["x"].transform("sum").values
    G["g_q_rank"] = gq.rank(ascending=False, method="first").values.astype(np.int16)
    gp = g.groupby("cand")["p"]
    top1 = gp.transform("max").values
    masked = np.where(g.p.values >= top1, -1.0, g.p.values)
    second = g.assign(m=masked).groupby("cand")["m"].transform("max").values
    best_other = np.where(g.p.values >= top1, second, top1)
    G["g_c_best_other"] = best_other.astype(np.float32)
    G["g_c_margin"] = (g.p.values - best_other).astype(np.float32)
    G["g_c_rank"] = gp.rank(ascending=False, method="first").values.astype(np.int16)
    G["g_c_n50"] = g.assign(x=(g.p > 0.5).astype(np.int16)).groupby("cand")["x"].transform("sum").values
    G["g_c_sum_other"] = (gp.transform("sum").values - g.p.values).astype(np.float32)
    return G.astype(np.float32)


def stage2(args):
    out = os.path.join(args.work_dir, "models", args.tag)
    with open(os.path.join(out, "stage1.json")) as fh:
        feats1 = json.load(fh)["features"]
    oof = pd.read_parquet(os.path.join(out, "oof_stage1.parquet"))
    t0 = time.time()
    G = group_features(oof)
    gcols = list(G.columns)
    Gv = G.values.astype(np.float32)
    del G
    fold = oof.fold.values
    keep_tr = (fold != HOLDOUT) & sample_mask(oof.s1.values, args.train_frac, salt=3)
    keep_ho = fold == HOLDOUT
    cols = feats1 + gcols
    files = part_files(args.pairs)
    # --- training matrix, preallocated and filled part by part (same row order as the OOF file)
    Xtr = np.empty((int(keep_tr.sum()), len(cols)), np.float32)
    pos = w = 0
    for f in tqdm(files, desc="  Building stage-2 train matrix", unit="part"):
        d = pd.read_parquet(f, columns=feats1)
        n = len(d)
        m = keep_tr[pos:pos + n]
        k = int(m.sum())
        Xtr[w:w + k, : len(feats1)] = d.values[m]
        Xtr[w:w + k, len(feats1):] = Gv[pos:pos + n][m]
        pos += n
        w += k
        del d
    assert pos == len(oof) and w == len(Xtr)
    ytr = oof.label.values[keep_tr]
    es_flag = sample_mask(oof.s1.values[keep_tr], 0.05, salt=11)
    print(f"stage2: train rows {len(Xtr):,}, feats {len(cols)} ({time.time()-t0:.1f}s)", flush=True)
    dtr = lgb.Dataset(Xtr[~es_flag], ytr[~es_flag], feature_name=cols, free_raw_data=True)
    des = lgb.Dataset(Xtr[es_flag], ytr[es_flag], reference=dtr)
    dtr.construct()
    des.construct()
    del Xtr
    print("Training Stage-2 reranking LightGBM model...", flush=True)
    m = lgb.train(params(args), dtr, args.rounds, valid_sets=[des], callbacks=[lgb.early_stopping(50, verbose=False)])
    m.save_model(os.path.join(out, "model_s2.txt"), num_iteration=m.best_iteration)
    print(f"✓ Stage-2 model trained: {m.best_iteration} iterations ({time.time()-t0:.1f}s)", flush=True)
    imp = pd.Series(m.feature_importance("gain"), index=cols)
    print("top gain:\n" + (imp / imp.sum()).sort_values(ascending=False).head(25).round(4).to_string())
    # --- holdout predictions, streamed
    p2 = np.empty(int(keep_ho.sum()), np.float32)
    pos = w = 0
    for f in tqdm(files, desc="  Predicting holdout stage-2", unit="part"):
        d = pd.read_parquet(f, columns=feats1)
        n = len(d)
        mh = keep_ho[pos:pos + n]
        k = int(mh.sum())
        if k:
            X = np.hstack([d.values[mh].astype(np.float32), Gv[pos:pos + n][mh]])
            p2[w:w + k] = m.predict(X, num_iteration=m.best_iteration)
        pos += n
        w += k
    ho = oof.loc[keep_ho, ["s1", "cand", "label", "fold", "p"]].copy()
    ho["p2"] = p2
    ho.to_parquet(os.path.join(out, "holdout_stage2.parquet"), index=False)
    with open(os.path.join(out, "stage2.json"), "w") as fh:
        json.dump({"features": cols, "group_features": gcols}, fh, indent=1)
    truth, ids = holdout_truth(args.data_dir, args.work_dir)
    best = oof.groupby("cand")["p"].max()  # (stage-1 OOF) competitor scores, for reference only
    res = evaluate(ho, truth, ids, pcol="p2", label="stage2")
    with open(os.path.join(out, "eval_stage2.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    decision = {"stage": 2, "policy": "threshold", "threshold": float(res["thr"]["t"]), "owner": False,
                "holdout_macro_f05": float(res["thr"]["macro_f05"])}
    with open(os.path.join(out, "decision.json"), "w") as fh:
        json.dump(decision, fh, indent=1)
    print("decision:", decision, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2", "report1"])
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--pairs", default="work/pairs/train")
    ap.add_argument("--tag", default="cv1")
    ap.add_argument("--train-frac", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--drop", nargs="*", default=[])
    args = ap.parse_args()
    if args.stage == "stage1":
        stage1(args)
    elif args.stage == "report1":
        report(pd.read_parquet(os.path.join(args.work_dir, "models", args.tag, "oof_stage1.parquet")), args, "stage1")
    else:
        stage2(args)


if __name__ == "__main__":
    main()
