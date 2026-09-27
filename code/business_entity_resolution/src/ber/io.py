"""Data loading / writing helpers.

All challenge files are TAB separated. Fields are read literally
(``quoting=QUOTE_NONE``, ``na_filter=False``) so that strings such as "NA",
"null" or stray quote characters are preserved exactly as supplied and empty
fields become "" instead of NaN.
"""
import csv
import os

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path):
    """Read a challenge TSV literally (all columns as str, empty -> "")."""
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        na_filter=False,
        keep_default_na=False,
        encoding="utf-8",
    )


def load_source(data_dir, split, source, cache_dir=None):
    """Load ``{split}_source{source}.tsv``; cache as parquet when cache_dir given."""
    name = f"{split}_source{source}"
    if cache_dir is not None:
        pq = os.path.join(cache_dir, name + ".parquet")
        if os.path.exists(pq):
            return pd.read_parquet(pq)
    df = read_tsv(os.path.join(data_dir, split, name + ".tsv"))
    assert list(df.columns) == SOURCE_COLS, df.columns
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(os.path.join(cache_dir, name + ".parquet"), index=False)
    return df


def load_ground_truth(data_dir, cache_dir=None):
    """Return dict: S1 id -> list of matched S2/S3 ids (empty list = singleton)."""
    df = None
    if cache_dir is not None:
        pq = os.path.join(cache_dir, "train_ground_truth.parquet")
        if os.path.exists(pq):
            df = pd.read_parquet(pq)
    if df is None:
        df = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            df.to_parquet(os.path.join(cache_dir, "train_ground_truth.parquet"), index=False)
    return {
        s1: (m.split(",") if m else [])
        for s1, m in zip(df["source1_entity_id"].values, df["matched_entity_ids"].values)
    }


def write_id_lists(path, s1_ids, id_lists, value_col):
    """Write a submission-style TSV (``source1_entity_id``, ``value_col``).

    ``id_lists`` is aligned with ``s1_ids``; each entry is an iterable of ids.
    Order is preserved and duplicates are dropped.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{value_col}\n")
        for s1, ids in zip(s1_ids, id_lists):
            seen = []
            for i in ids:
                if i not in seen:
                    seen.append(i)
            f.write(f"{s1}\t{','.join(seen)}\n")
