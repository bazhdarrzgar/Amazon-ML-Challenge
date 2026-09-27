"""Candidate generation (blocking): bidirectional TF-IDF retrieval per country.

Every Source-1 record is compared only with Source-2/3 records carrying the
same country label (0 cross-country matches in 7.6M training pairs). Inside a
country we build sparse TF-IDF vectors (fitted transductively on all records
of that country; no labels involved):

  name : word unigrams+bigrams of the core name (max_df 1%)  +  char 4-grams of
         the compact core name (max_df 0.2%); block L2-normalised
  addr : word unigrams+bigrams of the core address (max_df 0.2%)
  comb : [sqrt(w)*name, sqrt(1-w)*addr]   ->  cosine = w*cos_name + (1-w)*cos_addr

Very frequent n-grams are dropped (max_df): they carry little identity
information and make the sparse product quadratic.

Retrieval channels (multi-threaded sparse top-n product, sparse_dot_topn):
  forward : each S1 -> top ``k_fwd`` pool records by cos_comb
  reverse : each pool record -> top ``k_rev`` S1 records by cos_comb
            (each S2/S3 record belongs to at most one S1, so its true owner is
            almost always among its nearest S1 records)
The union of both channels is the candidate set.

Measured on 5k holdout India queries (EXP-B2): forward@20 recall 94.8%,
reverse top-1 94.9% (4.7 cand/S1), forward@20 + reverse top-3 96.9% (26 cand/S1).
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

DEFAULT_CFG = {
    "k_fwd": 20,
    "k_rev": 3,
    "w_name": 0.5,
    "name_word_max_df": 0.01,
    "name_char_max_df": 0.002,
    "addr_word_max_df": 0.002,
    "threads": 10,
}


def _fit(texts, **kw):
    vec = TfidfVectorizer(lowercase=False, sublinear_tf=True, dtype=np.float32, min_df=2, **kw)
    return vec.fit_transform(texts).tocsr()


def _l2(X):
    n = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    n[n == 0] = 1.0
    return (sp.diags((1.0 / n).astype(np.float32)) @ X).tocsr()


def name_word_text(df):
    return [c or f for c, f in zip(df["n_core"].values, df["n_full"].values)]


def name_char_text(df):
    return [" " + (c or f).replace(" ", "") + " " for c, f in zip(df["n_core"].values, df["n_full"].values)]


def build_vectors(s1, pool, cfg):
    """TF-IDF matrices (S1 rows first, then pool rows) for one country."""
    both = pd.concat([s1[["n_core", "n_full", "a_core"]], pool[["n_core", "n_full", "a_core"]]], ignore_index=True)
    nw = _fit(name_word_text(both), analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2),
              max_df=cfg["name_word_max_df"])
    nc = _fit(name_char_text(both), analyzer="char", ngram_range=(4, 4), max_df=cfg["name_char_max_df"])
    name = _l2(sp.hstack([nw, nc], format="csr"))
    addr = _fit(list(both["a_core"].values), analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2),
                max_df=cfg["addr_word_max_df"])
    w = cfg["w_name"]
    comb = sp.hstack([name * np.float32(np.sqrt(w)), addr * np.float32(np.sqrt(1 - w))], format="csr")
    n1 = len(s1)
    return {k: (X[:n1], X[n1:]) for k, X in (("name", name), ("addr", addr), ("comb", comb))}


def _topn(A, B, k, threads):
    return sp_matmul_topn(A, B.T.tocsr(), top_n=k, threshold=1e-4, sort=True, n_threads=threads).tocsr()


def _rank_within_rows(C):
    """0-based rank of every stored value inside its row.

    ``sp_matmul_topn(sort=True)`` stores each row in descending order, so the
    rank is simply the position inside the row (verified by the assertion).
    """
    starts = np.repeat(C.indptr[:-1], np.diff(C.indptr))
    rank = (np.arange(C.nnz) - starts).astype(np.int16)
    if C.nnz > 1:
        same_row = np.diff(starts) == 0
        assert np.all(np.diff(C.data)[same_row] <= 1e-6), "top-n rows are not sorted"
    return rank


def generate(s1, pool, cfg=None, verbose=True):
    """Candidate pairs for one country.

    Returns (pairs, vecs): pairs has S1 row ``q``, pool row ``p``, ``cos_comb``,
    ``fwd_rank`` / ``rev_rank`` (99 when not retrieved by that channel).
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    t0 = time.time()
    vecs = build_vectors(s1, pool, cfg)
    if verbose:
        print(f"  vectorised {len(s1)} S1 x {len(pool)} pool in {time.time()-t0:.0f}s", flush=True)
    A, B = vecs["comb"]
    t = time.time()
    F = _topn(A, B, cfg["k_fwd"], cfg["threads"])
    fwd = pd.DataFrame({
        "q": np.repeat(np.arange(F.shape[0], dtype=np.int32), np.diff(F.indptr)),
        "p": F.indices.astype(np.int32),
        "cos_comb": F.data.astype(np.float32),
        "fwd_rank": _rank_within_rows(F),
    })
    if verbose:
        print(f"  forward top{cfg['k_fwd']}: {len(fwd)} pairs ({time.time()-t:.0f}s)", flush=True)
    t = time.time()
    R = _topn(B, A, cfg["k_rev"], cfg["threads"])
    rev = pd.DataFrame({
        "p": np.repeat(np.arange(R.shape[0], dtype=np.int32), np.diff(R.indptr)),
        "q": R.indices.astype(np.int32),
        "cos_rev": R.data.astype(np.float32),
        "rev_rank": _rank_within_rows(R),
    })
    if verbose:
        print(f"  reverse top{cfg['k_rev']}: {len(rev)} pairs ({time.time()-t:.0f}s)", flush=True)
    pairs = fwd.merge(rev, on=["q", "p"], how="outer")
    pairs["cos_comb"] = pairs["cos_comb"].fillna(pairs["cos_rev"]).astype(np.float32)
    pairs = pairs.drop(columns="cos_rev")
    pairs["fwd_rank"] = pairs["fwd_rank"].fillna(99).astype(np.int16)
    pairs["rev_rank"] = pairs["rev_rank"].fillna(99).astype(np.int16)
    pairs = pairs.sort_values(["q", "cos_comb"], ascending=[True, False], kind="stable").reset_index(drop=True)
    if verbose:
        print(f"  union: {len(pairs)} pairs, {len(pairs)/max(len(s1),1):.1f} per S1", flush=True)
    return pairs, vecs


def name_block_channel(s1, pool, k=2, max_block=100, chunk=2_000_000, verbose=True):
    """Exact-name blocks re-ranked by full-vocabulary address similarity.

    The TF-IDF retrieval prunes very frequent address tokens (city names), so
    when a core name is shared by many S1 records the right one can be missed.
    Here every pool record whose core name exactly equals the core name of
    1..``max_block`` S1 records is compared with all of them using an
    *unpruned* word TF-IDF of the address; the top-``k`` S1 records are kept.

    Returns all block pairs with ``nb_sim`` (address cosine) and ``nb_rank``
    (rank of the S1 inside the block for that pool record) plus a boolean
    ``nb_keep`` for the top-k pairs that join the candidate set.
    """
    t = time.time()
    vec = TfidfVectorizer(analyzer=str.split, lowercase=False, sublinear_tf=True, min_df=1, dtype=np.float32)
    X = vec.fit_transform(list(s1["a_core"].values) + list(pool["a_core"].values)).tocsr()
    A, B = X[: len(s1)], X[len(s1):]
    core1 = pd.Series(np.arange(len(s1)), index=s1["n_core"].values)
    core1 = core1[core1.index != ""]
    sizes = core1.groupby(level=0).size()
    ok = sizes[sizes <= max_block].index
    blocks = core1[core1.index.isin(ok)].groupby(level=0).apply(lambda s: s.values)
    pc = pool["n_core"].values
    has = pd.Series(pc).isin(blocks.index).values & (pool["a_core"].values != "")
    p_idx = np.flatnonzero(has)
    b_of_p = blocks.reindex(pc[p_idx]).values
    lens = np.fromiter((len(b) for b in b_of_p), np.int64, len(p_idx))
    P = np.repeat(p_idx, lens).astype(np.int32)
    Q = np.concatenate(b_of_p).astype(np.int32) if len(b_of_p) else np.zeros(0, np.int32)
    sim = np.empty(len(P), np.float32)
    for s in range(0, len(P), chunk):
        e = min(s + chunk, len(P))
        sim[s:e] = np.asarray(A[Q[s:e]].multiply(B[P[s:e]]).sum(axis=1)).ravel()
    df = pd.DataFrame({"q": Q, "p": P, "nb_sim": sim})
    df = df.sort_values(["p", "nb_sim"], ascending=[True, False], kind="stable")
    df["nb_rank"] = df.groupby("p").cumcount().astype(np.int16)
    # pessimistic rank for ties: number of block members with sim >= own sim, minus 1
    df["nb_keep"] = (df["nb_rank"] < k) & (df["nb_sim"] > 0)
    if verbose:
        print(f"  name-block channel: {len(p_idx)} pool records in blocks, {len(df)} block pairs, "
              f"{int(df.nb_keep.sum())} kept ({time.time()-t:.0f}s)", flush=True)
    return df.reset_index(drop=True)


def add_name_block(pairs, nb, vecs):
    """Union the name-block top-k pairs into ``pairs`` and attach nb features to all pairs."""
    keep = nb[nb["nb_keep"]][["q", "p"]]
    new = keep.merge(pairs[["q", "p"]], on=["q", "p"], how="left", indicator=True)
    new = new[new["_merge"] == "left_only"][["q", "p"]]
    if len(new):
        A, B = vecs["comb"]
        qi, pi = new["q"].values.astype(np.int64), new["p"].values.astype(np.int64)
        cos = np.asarray(A[qi].multiply(B[pi]).sum(axis=1)).ravel().astype(np.float32)
        new = new.assign(cos_comb=cos, fwd_rank=np.int16(99), rev_rank=np.int16(99))
        pairs = pd.concat([pairs, new], ignore_index=True)
    pairs = pairs.merge(nb[["q", "p", "nb_sim", "nb_rank"]], on=["q", "p"], how="left")
    pairs["nb_sim"] = pairs["nb_sim"].fillna(-1).astype(np.float32)
    pairs["nb_rank"] = pairs["nb_rank"].fillna(99).astype(np.int16)
    pairs = pairs.sort_values(["q", "cos_comb"], ascending=[True, False], kind="stable").reset_index(drop=True)
    return pairs, len(new)
