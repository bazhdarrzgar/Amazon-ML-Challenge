"""Set-level decision layer: turn pair probabilities into a match set per S1.

Policies (all evaluated with the official macro-F0.5 on validation):
  * threshold   : keep candidates with p >= t
  * owner       : each S2/S3 record may belong to at most one S1 -> keep a pair
                  only if it is the best-scoring S1 for that candidate
  * expected-F  : choose, per S1, the top-k prefix (k = 0..n) of candidates
                  sorted by p that maximises the expected F0.5 under an
                  independence assumption (Jansche 2007; Nan Ye et al. 2012).
"""
import numpy as np
import pandas as pd

from .metric import detailed_report


def owner_filter(df, pcol="p"):
    """Keep only the highest-probability S1 for every candidate id."""
    best = df.groupby("cand_id")[pcol].transform("max")
    return df[df[pcol] >= best]


def threshold_sets(df, t, pcol="p"):
    sel = df[df[pcol] >= t]
    return sel.groupby("s1_id")["cand_id"].apply(list).to_dict()


def _expected_f05_prefix(p, max_k=12):
    """Return (best_k, best_value) for probabilities ``p`` sorted descending.

    Truth count T = X_in + X_out where X_in ~ PoissonBinomial(top-k) and
    X_out ~ PoissonBinomial(rest). E[F] for k>0 = E[1.25*X_in / (0.25*T + k)],
    for k=0: P(T=0).  Computed exactly by dynamic programming.
    """
    n = len(p)
    best_k, best_v = 0, float(np.prod(1 - p))
    if n == 0:
        return 0, 1.0
    kmax = min(n, max_k)
    # distribution of X_out for each suffix: compute suffix PMFs lazily
    for k in range(1, kmax + 1):
        pin, pout = p[:k], p[k:]
        din = np.array([1.0])
        for x in pin:
            din = np.convolve(din, [1 - x, x])
        dout = np.array([1.0])
        for x in pout:
            if x > 1e-4:
                dout = np.convolve(dout, [1 - x, x])
        i = np.arange(len(din))[:, None]
        o = np.arange(len(dout))[None, :]
        T = i + o
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(i > 0, 1.25 * i / (0.25 * T + k), 0.0)
        v = float((din[:, None] * dout[None, :] * f).sum())
        if v > best_v:
            best_k, best_v = k, v
    return best_k, best_v


def expected_f_sets(df, pcol="p", min_p=0.01):
    """Per-S1 expected-F0.5 optimal prefix of candidates sorted by probability."""
    out = {}
    d = df[df[pcol] >= min_p].sort_values(["s1_id", pcol], ascending=[True, False])
    for s1, g in d.groupby("s1_id", sort=False):
        p = g[pcol].values.astype(np.float64)
        k, _ = _expected_f05_prefix(p)
        if k:
            out[s1] = list(g["cand_id"].values[:k])
    return out


def evaluate_sets(pred, truth, ids):
    return detailed_report(pred, truth, ids)


def sweep_thresholds(df, truth, ids, grid, pcol="p", owner=False):
    rows = []
    d = owner_filter(df, pcol) if owner else df
    for t in grid:
        rep = detailed_report(threshold_sets(d, t, pcol), truth, ids)
        rep["t"] = t
        rows.append(rep)
    return pd.DataFrame(rows)
