"""Vectorised macro-F0.5 evaluation of set-selection policies.

For one S1 entity with truth size |T|, predicted size |P| and TP correct ids,
F0.5 = 1.25*TP / (0.25*|T| + |P|)  (TP > 0),  1 if |T| = |P| = 0, else 0.
A policy is a boolean mask over candidate pairs, so any policy can be scored
in milliseconds with per-entity counts (np.bincount). Verified against the
reference implementation in ``metric.py`` (see ``check``).
"""
import numpy as np
import pandas as pd


class FastEval:
    def __init__(self, pairs, truth, ids, pcol="p", s1col="s1", ccol="cand", label="label"):
        """pairs: holdout candidate pairs with probability and 0/1 label.
        truth: dict S1 id -> list of true ids (all evaluated S1, incl. no-candidate ones)."""
        self.ids = list(ids)
        pos = {k: i for i, k in enumerate(self.ids)}
        d = pairs[pairs[s1col].isin(pos)]
        self.g = d[s1col].map(pos).values.astype(np.int64)
        self.p = d[pcol].values.astype(np.float64)
        self.y = d[label].values.astype(np.int64)
        self.cand = d[ccol].values
        self.T = np.array([len(truth[k]) for k in self.ids], dtype=np.float64)
        self.n = len(self.ids)
        # rank of each pair inside its S1 (0 = highest p)
        order = np.lexsort((-self.p, self.g))
        rank = np.empty(len(order), np.int64)
        gs = self.g[order]
        starts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
        run = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
        rank[order] = run
        self.rank = rank
        self.top1 = np.zeros(self.n)
        np.maximum.at(self.top1, self.g, self.p)

    def score(self, mask):
        P = np.bincount(self.g[mask], minlength=self.n).astype(np.float64)
        TP = np.bincount(self.g[mask], weights=self.y[mask], minlength=self.n)
        f = np.where(TP > 0, 1.25 * TP / (0.25 * self.T + np.maximum(P, 1e-9)), 0.0)
        f = np.where((self.T == 0) & (P == 0), 1.0, f)
        return float(f.mean()), {
            "precision": float(TP.sum() / max(P.sum(), 1)),
            "recall": float(TP.sum() / max(self.T.sum(), 1)),
            "singleton_acc": float(((P == 0) & (self.T == 0)).sum() / max((self.T == 0).sum(), 1)),
            "nonsing_empty": int(((P == 0) & (self.T > 0)).sum()),
        }

    # ---- policies --------------------------------------------------------
    def threshold(self, t):
        return self.p >= t

    def fallback(self, t, t_fb):
        """p >= t, plus the top-1 candidate when nothing clears t but p1 >= t_fb."""
        none = self.top1[self.g] < t
        return (self.p >= t) | (none & (self.rank == 0) & (self.p >= t_fb))

    def relative(self, t, ratio):
        """p >= t and p >= ratio * (best p of the S1)."""
        return (self.p >= t) & (self.p >= ratio * self.top1[self.g])


def check(pairs, truth, ids, t=0.5, pcol="p"):
    """Assert FastEval agrees with the reference metric for a threshold policy."""
    from .metric import macro_f05
    fe = FastEval(pairs, truth, ids, pcol=pcol)
    f_fast, _ = fe.score(fe.threshold(t))
    sel = pairs[pairs[pcol] >= t].groupby("s1")["cand"].apply(list).to_dict()
    f_ref = macro_f05(sel, truth, ids)
    assert abs(f_fast - f_ref) < 1e-9, (f_fast, f_ref)
    return f_fast
