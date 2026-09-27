"""Deterministic entity-level folds for Source-1 records.

Every S1 entity is assigned to one of ``N_FOLDS`` folds by a stable hash of its
id, so all of its candidate pairs / labels always land in the same fold.
Fold ``HOLDOUT`` is the clean validation fold: nothing supervised (model,
thresholds, transliteration dictionaries) is fitted on it during development.
"""
import zlib

import numpy as np

N_FOLDS = 5
HOLDOUT = 0


def fold_of(entity_id):
    return zlib.crc32(entity_id.encode("utf-8")) % N_FOLDS


def fold_array(entity_ids):
    return np.fromiter((fold_of(e) for e in entity_ids), dtype=np.int8, count=len(entity_ids))
