"""Official challenge metric: macro-averaged F_0.5 over Source 1 entities.

Per S1 entity with truth set T and predicted set P:
  * T empty and P empty         -> 1.0  (correct singleton)
  * T empty and P non-empty     -> 0.0  (false merge on a singleton)
  * T non-empty and P empty     -> 0.0  (recall 0)
  * otherwise precision = |P&T|/|P|, recall = |P&T|/|T|,
    F0.5 = 1.25*p*r / (0.25*p + r)   (0 when |P&T| == 0)
The score is the plain mean over all evaluated S1 entities.
"""
import numpy as np

BETA2 = 0.25


def f05(pred, truth):
    """F_0.5 for one entity. ``pred`` / ``truth`` are iterables of ids."""
    pred, truth = set(pred), set(truth)
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred_map, truth_map, ids=None):
    """Mean F_0.5 over ``ids`` (default: every key of ``truth_map``).

    Entities missing from ``pred_map`` are treated as an empty prediction.
    """
    if ids is None:
        ids = list(truth_map)
    if len(ids) == 0:
        return float("nan")
    return float(np.mean([f05(pred_map.get(i, ()), truth_map[i]) for i in ids]))


def detailed_report(pred_map, truth_map, ids=None):
    """Macro F0.5 plus diagnostic breakdowns (singletons, pair counts, ...)."""
    if ids is None:
        ids = list(truth_map)
    scores, tp, fp, fn = [], 0, 0, 0
    sing_n = sing_ok = 0
    nonsing_empty = 0
    for i in ids:
        p, t = set(pred_map.get(i, ())), set(truth_map[i])
        scores.append(f05(p, t))
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
        if not t:
            sing_n += 1
            sing_ok += int(not p)
        elif not p:
            nonsing_empty += 1
    scores = np.asarray(scores)
    return {
        "n": len(ids),
        "macro_f05": float(scores.mean()),
        "pair_precision": tp / max(tp + fp, 1),
        "pair_recall": tp / max(tp + fn, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "singletons": sing_n,
        "singleton_acc": sing_ok / max(sing_n, 1),
        "nonsingleton_predicted_empty": nonsing_empty,
    }
