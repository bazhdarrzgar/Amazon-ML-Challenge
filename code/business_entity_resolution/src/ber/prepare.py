"""Step 1 of the pipeline: learn transliteration dictionaries + normalise all sources.

Usage:
    python -m ber.prepare --data-dir DATA --work-dir WORK [--splits train test]

Outputs (in WORK):
    models/translit.json                   learned dictionaries
    cache/norm_{split}_s{1,2,3}.parquet    normalised views of every record
"""
import argparse
import json
import os
import time
from multiprocessing import get_context

import pandas as pd
from tqdm import tqdm

from .folds import HOLDOUT, fold_of
from .io import load_ground_truth, load_source
from .normalize import Normalizer, learn_addr_comp_map
from .translit import has_native, learn_translit_dict

_NORM = None


def _init_worker(tmap, amap):
    global _NORM
    _NORM = Normalizer(tmap, amap)


def _norm_chunk(rows):
    out = []
    for name, addr, country in rows:
        d = _NORM.name(name)
        d.update(_NORM.address(addr, country))
        out.append(d)
    return out


def learn_dictionaries(data_dir, cache_dir, include_holdout=False):
    """Learn name / address transliteration maps from training ground truth.

    By default the holdout fold's S1 entities are excluded so validation is not
    contaminated.
    """
    gt = load_ground_truth(data_dir, cache_dir)
    s1 = load_source(data_dir, "train", 1, cache_dir)
    keep = {k for k in gt if include_holdout or fold_of(k) != HOLDOUT}
    owner = {m: k for k, v in gt.items() if k in keep for m in v}
    nm1 = dict(zip(s1.entity_id, s1.business_name))
    ad1 = dict(zip(s1.entity_id, s1.business_address))
    name_pairs, addr_pairs = [], []
    for s in (2, 3):
        d = load_source(data_dir, "train", s, cache_dir)
        for i, n, a, c in zip(d.entity_id, d.business_name, d.business_address, d.country):
            o = owner.get(i)
            if o is None:
                continue
            if has_native(n):
                name_pairs.append((nm1[o], n))
            if has_native(a):
                addr_pairs.append((ad1[o], c, a))
    tmap = learn_translit_dict(name_pairs)
    amap = learn_addr_comp_map(addr_pairs)
    return tmap, amap


def normalise_frame(df, tmap, amap, workers=8, chunk=20000):
    rows = list(zip(df.business_name.values, df.business_address.values, df.country.values))
    chunks = [rows[i : i + chunk] for i in range(0, len(rows), chunk)]
    with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(tmap, amap)) as pool:
        res = list(tqdm(pool.imap(_norm_chunk, chunks, chunksize=1), total=len(chunks),
                        desc=f"  Chunk normalisation ({len(df):,} rows)", leave=False, unit="chunk"))
    out = pd.DataFrame([d for r in res for d in r])
    out.insert(0, "country", df.country.values)
    out.insert(0, "entity_id", df.entity_id.values)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--include-holdout", action="store_true",
                    help="learn dictionaries on all training entities (final run)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cache = os.path.join(args.work_dir, "cache")
    os.makedirs(os.path.join(args.work_dir, "models"), exist_ok=True)

    t = time.time()
    dict_path = os.path.join(args.work_dir, "models", "translit.json")
    print("Learning transliteration dictionaries...", flush=True)
    tmap, amap = learn_dictionaries(args.data_dir, cache, args.include_holdout)
    with open(dict_path, "w") as f:
        json.dump({"name": tmap, "addr": amap}, f, ensure_ascii=False, indent=0)
    print(f"Dictionaries learned: name={len(tmap):,} addr={len(amap):,} ({time.time()-t:.1f}s)", flush=True)

    tasks = [(split, s) for split in args.splits for s in (1, 2, 3)]
    for split, s in tqdm(tasks, desc="Step 1/6: Normalising sources", unit="source"):
        t = time.time()
        df = load_source(args.data_dir, split, s, cache)
        out = normalise_frame(df, tmap, amap, args.workers)
        out.to_parquet(os.path.join(cache, f"norm_{split}_s{s}.parquet"), index=False)
        print(f"  ✓ {split} source {s}: {len(out):,} rows written ({time.time()-t:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
