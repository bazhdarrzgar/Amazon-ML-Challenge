"""Compact integer encoding of entity ids ("S2-681193310" -> 2*10**10 + 681193310)."""
import numpy as np
import pandas as pd

BASE = 10 ** 10


def encode(ids):
    s = pd.Series(ids, dtype=object).str
    src = s.slice(1, 2).astype(np.int64).values
    num = s.slice(3).astype(np.int64).values
    return src * BASE + num


def decode(codes):
    codes = np.asarray(codes, dtype=np.int64)
    return [f"S{c // BASE}-{c % BASE}" for c in codes]
