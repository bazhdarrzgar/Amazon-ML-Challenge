"""Pairwise features for (Source-1 record, candidate record) pairs.

All features are country-agnostic (no country one-hot) so that an unseen
country such as France is scored by the same learned similarity function.

Groups
  * name string similarity   : rapidfuzz ratio / token-set / token-sort /
    partial / Jaro-Winkler on core, full and compact names; alias-part max
  * name token overlap       : IDF-weighted Jaccard / containment (word level)
  * TF-IDF cosines           : char-3gram name / address / combined
  * legal-form agreement     : shared / conflicting canonical legal forms
  * address                  : string similarity, IDF token overlap, number
    agreement (first number, set overlap), state agreement, emptiness
  * record meta              : candidate source (S2/S3), native script,
    domain-style name, alias present, lengths
  * retrieval context        : per-channel ranks, gap to best candidate,
    candidate count per S1, competition from other S1 records for the same
    candidate (how many S1 retrieved it, best competing score)
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import TfidfVectorizer

WORKERS = -1


def _cp(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=WORKERS, dtype=np.float32)


def rowwise_dot(A, B, qi, pi, chunk=2_000_000):
    """sum_j A[qi,j]*B[pi,j] for aligned index arrays (row-normalised -> cosine)."""
    out = np.empty(len(qi), np.float32)
    for s in range(0, len(qi), chunk):
        e = min(s + chunk, len(qi))
        out[s:e] = np.asarray(A[qi[s:e]].multiply(B[pi[s:e]]).sum(axis=1)).ravel()
    return out


def word_idf_matrix(texts):
    """Binary word matrix weighted by IDF (values = idf of the token)."""
    vec = TfidfVectorizer(analyzer=str.split, lowercase=False, use_idf=True, norm=None,
                          binary=True, min_df=1, dtype=np.float32)
    X = vec.fit_transform(texts).tocsr()
    return X


def weighted_overlap(Xa, Xb, qi, pi, chunk=2_000_000):
    """IDF-weighted intersection, and each side's total weight, per pair."""
    inter = np.empty(len(qi), np.float32)
    for s in range(0, len(qi), chunk):
        e = min(s + chunk, len(qi))
        A = Xa[qi[s:e]]
        B = Xb[pi[s:e]]
        inter[s:e] = np.asarray(A.multiply(B > 0).sum(axis=1)).ravel()
    wa = np.asarray(Xa.sum(axis=1)).ravel()[qi]
    wb = np.asarray(Xb.sum(axis=1)).ravel()[pi]
    return inter, wa.astype(np.float32), wb.astype(np.float32)


def _nums(x):
    """Space-separated digit runs -> list with leading zeros stripped ("0031" -> "31")."""
    return [t.lstrip("0") or "0" for t in x.split()] if x else []


def _num_feats(na, nb):
    """Number-token agreement (house / unit / plot numbers, postal codes).

    Hard negatives in the data are "sister" records at the same building with a
    substituted number, while true matches may *add* noise numbers (PMB, H.NO
    prefixes) or zero-pad -> both containment directions are measured.
    """
    n = len(na)
    first_eq = np.zeros(n, np.int8)
    jac = np.zeros(n, np.float32)
    a_in_b = np.zeros(n, np.float32)
    b_in_a = np.zeros(n, np.float32)
    n_shared = np.zeros(n, np.int8)
    a_missing = np.zeros(n, np.int8)
    b_extra = np.zeros(n, np.int8)
    first_a_in_b = np.zeros(n, np.int8)
    min_rel_diff = np.zeros(n, np.float32)
    for i in range(n):
        xs, ys = _nums(na[i]), _nums(nb[i])
        if not xs or not ys:
            first_eq[i] = jac[i] = a_in_b[i] = b_in_a[i] = first_a_in_b[i] = -1
            min_rel_diff[i] = -1
            a_missing[i] = len(xs)
            b_extra[i] = len(ys)
            continue
        first_eq[i] = xs[0] == ys[0]
        sx, sy = set(xs), set(ys)
        inter = len(sx & sy)
        n_shared[i] = min(inter, 127)
        jac[i] = inter / len(sx | sy)
        a_in_b[i] = inter / len(sx)
        b_in_a[i] = inter / len(sy)
        a_missing[i] = min(len(sx - sy), 127)
        b_extra[i] = min(len(sy - sx), 127)
        first_a_in_b[i] = xs[0] in sy
        if xs[0] in sy:
            min_rel_diff[i] = 0.0
        else:
            try:
                a0 = int(xs[0][:9])
                min_rel_diff[i] = min(abs(a0 - int(y[:9])) / max(a0, int(y[:9]), 1) for y in sy)
            except ValueError:
                min_rel_diff[i] = 1.0
    return first_eq, jac, a_in_b, b_in_a, n_shared, a_missing, b_extra, first_a_in_b, min_rel_diff


def _legal_feats(la, lb):
    """Legal-form agreement. "X Private Limited" vs "X Limited" are typically
    *different* (sister) entities in this data -> exact-set and symmetric
    difference features, not only overlap."""
    n = len(la)
    shared = np.zeros(n, np.int8)
    conflict = np.zeros(n, np.int8)
    equal = np.zeros(n, np.int8)
    symdiff = np.zeros(n, np.int8)
    na_ = np.zeros(n, np.int8)
    nb_ = np.zeros(n, np.int8)
    pvt_mismatch = np.zeros(n, np.int8)
    for i in range(n):
        x, y = la[i], lb[i]
        sx = set(x.split()) if x else set()
        sy = set(y.split()) if y else set()
        na_[i], nb_[i] = len(sx), len(sy)
        equal[i] = sx == sy
        symdiff[i] = len(sx ^ sy)
        pvt_mismatch[i] = ("pvt" in sx) != ("pvt" in sy)
        if not sx or not sy:
            shared[i] = -1 if (not sx and not sy) else 0
            continue
        shared[i] = len(sx & sy) > 0
        conflict[i] = len(sx & sy) == 0
    return shared, conflict, equal, symdiff, na_, nb_, pvt_mismatch


def _alias_max(core_a, alias_b, full_b):
    """Max token-set similarity between S1 core name and any alias part of b."""
    out = np.full(len(core_a), -1.0, np.float32)
    idx = [i for i, x in enumerate(alias_b) if x]
    if not idx:
        return out
    aa, bb, owner = [], [], []
    for i in idx:
        for part in alias_b[i].split("|"):
            aa.append(core_a[i])
            bb.append(part)
            owner.append(i)
    sc = _cp(aa, bb, fuzz.token_set_ratio)
    s = pd.Series(sc).groupby(np.asarray(owner)).max()
    out[s.index.values] = s.values
    return out


def pair_features(pairs, s1, pool, vecs, word_mats):
    """Compute the feature frame for candidate ``pairs`` (columns q, p, channel scores/ranks).

    s1 / pool: normalised frames for one country (row positions = q / p).
    vecs: TF-IDF matrices from ``candidates.build_vectors``.
    word_mats: dict with 'name' and 'addr' IDF word matrices ((Xa, Xb) tuples).
    """
    qi = pairs["q"].values.astype(np.int64)
    pi = pairs["p"].values.astype(np.int64)
    F = pd.DataFrame(index=pairs.index)

    # retrieval-channel cosines (name / addr recomputed for every pair)
    for ch in ("name", "addr"):
        if "cos_" + ch in pairs:
            F["cos_" + ch] = pairs["cos_" + ch].values
        else:
            A, B = vecs[ch]
            F["cos_" + ch] = rowwise_dot(A, B, qi, pi)
    F["cos_comb"] = pairs["cos_comb"].values
    F["fwd_rank"] = pairs["fwd_rank"].values
    F["rev_rank"] = pairs["rev_rank"].values
    for c in ("nb_sim", "nb_rank"):  # optional name-block channel features
        if c in pairs:
            F[c] = pairs[c].values

    col = lambda df, c, idx: df[c].values[idx].tolist()  # noqa: E731
    a_core, b_core = col(s1, "n_core", qi), col(pool, "n_core", pi)
    a_full, b_full = col(s1, "n_full", qi), col(pool, "n_full", pi)
    a_cmp, b_cmp = col(s1, "n_compact", qi), col(pool, "n_compact", pi)
    F["n_ratio"] = _cp(a_core, b_core, fuzz.ratio)
    F["n_tset"] = _cp(a_core, b_core, fuzz.token_set_ratio)
    F["n_tsort"] = _cp(a_core, b_core, fuzz.token_sort_ratio)
    F["n_full_ratio"] = _cp(a_full, b_full, fuzz.ratio)
    F["n_partial_cmp"] = _cp(a_cmp, b_cmp, fuzz.partial_ratio)
    F["n_jw_cmp"] = _cp(a_cmp, b_cmp, JaroWinkler.normalized_similarity)
    F["n_exact_core"] = np.fromiter((x == y and x != "" for x, y in zip(a_core, b_core)), np.int8, len(qi))
    F["n_alias_max"] = _alias_max(a_core, col(pool, "n_alias", pi), b_full)
    F["n_len_a"] = np.fromiter((len(x) for x in a_core), np.int16, len(qi))
    F["n_len_b"] = np.fromiter((len(x) for x in b_core), np.int16, len(qi))
    F["n_ntok_a"] = np.fromiter((x.count(" ") + 1 if x else 0 for x in a_core), np.int8, len(qi))
    F["n_ntok_b"] = np.fromiter((x.count(" ") + 1 if x else 0 for x in b_core), np.int8, len(qi))
    F["n_first_tok_eq"] = np.fromiter(
        ((x.split(" ", 1)[0] == y.split(" ", 1)[0]) if x and y else False for x, y in zip(a_core, b_core)),
        np.int8, len(qi))
    Xa, Xb = word_mats["name"]
    inter, wa, wb = weighted_overlap(Xa, Xb, qi, pi)
    F["n_wjac"] = inter / np.maximum(wa + wb - inter, 1e-6)
    F["n_wcont_a"] = inter / np.maximum(wa, 1e-6)
    F["n_wcont_b"] = inter / np.maximum(wb, 1e-6)
    F["n_idf_a"] = wa
    F["n_idf_inter"] = inter
    shared, conflict, equal, symdiff, lna, lnb, pvt_mm = _legal_feats(col(s1, "n_legal", qi), col(pool, "n_legal", pi))
    F["legal_shared"] = shared
    F["legal_conflict"] = conflict
    F["legal_equal"] = equal
    F["legal_symdiff"] = symdiff
    F["legal_n_a"] = lna
    F["legal_n_b"] = lnb
    F["legal_pvt_mismatch"] = pvt_mm
    F["b_domain"] = (pool["n_domain"].values[pi] != "").astype(np.int8)
    F["b_native_name"] = pool["n_native"].values[pi].astype(np.int8)
    F["b_alias"] = (pool["n_alias"].values[pi] != "").astype(np.int8)
    F["b_is_s3"] = np.fromiter((x.startswith("S3") for x in pool["entity_id"].values[pi]), np.int8, len(qi))

    # address
    a_addr, b_addr = col(s1, "a_core", qi), col(pool, "a_core", pi)
    F["a_ratio"] = _cp(a_addr, b_addr, fuzz.ratio)
    F["a_tset"] = _cp(a_addr, b_addr, fuzz.token_set_ratio)
    F["a_tsort"] = _cp(a_addr, b_addr, fuzz.token_sort_ratio)
    F["a_empty_b"] = np.fromiter((x == "" for x in b_addr), np.int8, len(qi))
    F["a_len_b"] = np.fromiter((len(x) for x in b_addr), np.int16, len(qi))
    Xa, Xb = word_mats["addr"]
    inter, wa, wb = weighted_overlap(Xa, Xb, qi, pi)
    F["a_wjac"] = inter / np.maximum(wa + wb - inter, 1e-6)
    F["a_wcont_a"] = inter / np.maximum(wa, 1e-6)
    F["a_wcont_b"] = inter / np.maximum(wb, 1e-6)
    fe, jac, ainb, bina, nsh, amiss, bextra, fainb, mrd = _num_feats(col(s1, "a_nums", qi), col(pool, "a_nums", pi))
    F["num_first_eq"] = fe
    F["num_jac"] = jac
    F["num_a_in_b"] = ainb
    F["num_b_in_a"] = bina
    F["num_shared"] = nsh
    F["num_a_missing"] = amiss
    F["num_b_extra"] = bextra
    F["num_first_a_in_b"] = fainb
    F["num_min_rel_diff"] = mrd
    sa, sb = s1["a_state"].values[qi], pool["a_state"].values[pi]
    F["state_eq"] = np.where((sa == "") | (sb == ""), -1, (sa == sb).astype(np.int8)).astype(np.int8)
    return F


def context_features(pairs, score_cols=("cos_comb", "cos_name", "cos_addr")):
    """Label-free group features over the whole candidate graph of a country.

    S1 side       : max score in the S1's candidate list, gap to it, rank, size.
    candidate side: each S2/S3 record belongs to <= 1 S1, so we measure how
                    strongly *other* S1 records compete for the same candidate
                    (best other score, margin, rank, number of S1 retrieving it).
    """
    q = pairs["q"].values
    p = pairs["p"].values
    F = pd.DataFrame(index=pairs.index)
    F["ctx_q_n"] = pd.Series(q).groupby(q).transform("size").values.astype(np.int16)
    F["ctx_p_n"] = pd.Series(p).groupby(p).transform("size").values.astype(np.int16)
    for col in score_cols:
        tag = col.replace("cos_", "")
        s = pairs[col].values.astype(np.float32)
        df = pd.DataFrame({"q": q, "p": p, "s": s})
        gq = df.groupby("q")["s"]
        qmax = gq.transform("max").values
        F[f"ctx_{tag}_q_max"] = qmax
        F[f"ctx_{tag}_q_gap"] = qmax - s
        F[f"ctx_{tag}_q_rank"] = gq.rank(ascending=False, method="first").values.astype(np.int16)
        gp = df.groupby("p")["s"]
        top1 = gp.transform("max").values
        df["m"] = np.where(s >= top1, -1.0, s)
        second = df.groupby("p")["m"].transform("max").values
        other = np.where(s >= top1, second, top1).astype(np.float32)
        F[f"ctx_{tag}_p_best_other"] = other
        F[f"ctx_{tag}_p_margin"] = (s - other).astype(np.float32)
        F[f"ctx_{tag}_p_rank"] = gp.rank(ascending=False, method="first").values.astype(np.int16)
    return F


def ambiguity_features(s1, pool, pairs):
    """How many S1 records (same country) share the exact core name / core
    address of each side. A candidate with an empty address whose name is shared
    by several S1 records is inherently ambiguous; a unique name is not."""
    name_cnt = s1["n_core"].value_counts()
    addr_cnt = s1.loc[s1["a_core"] != "", "a_core"].value_counts()
    qi, pi = pairs["q"].values, pairs["p"].values
    F = pd.DataFrame(index=pairs.index)
    F["amb_name_a"] = name_cnt.reindex(s1["n_core"].values).fillna(0).values[qi].astype(np.int16)
    F["amb_name_b"] = name_cnt.reindex(pool["n_core"].values).fillna(0).values[pi].astype(np.int16)
    F["amb_addr_a"] = addr_cnt.reindex(s1["a_core"].values).fillna(0).values[qi].astype(np.int16)
    F["amb_addr_b"] = addr_cnt.reindex(pool["a_core"].values).fillna(0).values[pi].astype(np.int16)
    pool_name_cnt = pool["n_core"].value_counts()
    F["amb_pool_name_b"] = pool_name_cnt.reindex(pool["n_core"].values).fillna(0).values[pi].astype(np.int32)
    return F
