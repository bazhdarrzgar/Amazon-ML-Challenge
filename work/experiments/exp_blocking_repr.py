"""EXP-B1: speed/recall of retrieval representations on a small holdout query sample."""
import sys, time, json
sys.path.insert(0, "code/business_entity_resolution/src")
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from ber.io import load_ground_truth
from ber.folds import fold_array, HOLDOUT
from ber.bench_blocking import load_norm

country = sys.argv[1] if len(sys.argv) > 1 else "India"
nq = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
gt = load_ground_truth("dataset", "work/cache")
s1, pool = load_norm("work", "train")
s1 = s1[s1.country == country].reset_index(drop=True)
pool = pool[pool.country == country].reset_index(drop=True)
fold = fold_array(s1.entity_id.values)
rng = np.random.RandomState(0)
q = rng.choice(np.flatnonzero(fold == HOLDOUT), nq, replace=False)
pid = {e: i for i, e in enumerate(pool.entity_id.values)}
truth = [set(pid[m] for m in gt[e]) for e in s1.entity_id.values[q]]
n_true = sum(len(t) for t in truth)
print(country, "S1", len(s1), "pool", len(pool), "queries", nq, "true pairs", n_true, flush=True)

name_c = [" " + (c.replace(" ", "") or f.replace(" ", "")) + " " for c, f in zip(s1.n_core, s1.n_full)]
name_p = [" " + (c.replace(" ", "") or f.replace(" ", "")) + " " for c, f in zip(pool.n_core, pool.n_full)]
namew_c = [c or f for c, f in zip(s1.n_core, s1.n_full)]
namew_p = [c or f for c, f in zip(pool.n_core, pool.n_full)]
addr_c, addr_p = list(s1.a_core), list(pool.a_core)

def run(label, texts_c, texts_p, **kw):
    t = time.time()
    vec = TfidfVectorizer(lowercase=False, sublinear_tf=True, dtype=np.float32, min_df=2, **kw)
    X = vec.fit_transform(texts_c + texts_p).tocsr()
    A, B = X[: len(texts_c)][q], X[len(texts_c):]
    tv = time.time() - t
    t = time.time()
    C = sp_matmul_topn(A, B.T.tocsr(), top_n=50, threshold=1e-4, sort=True, n_threads=10).tocsr()
    tq = time.time() - t
    res = {}
    for k in (5, 10, 20, 50):
        hit = 0
        for i in range(nq):
            row = C.indices[C.indptr[i]:C.indptr[i + 1]][:k]
            hit += len(truth[i] & set(row.tolist()))
        res[k] = round(hit / n_true, 4)
    print(f"{label:40s} vocab={X.shape[1]:>8d} nnz/row={X.nnz/X.shape[0]:.1f} vec={tv:.0f}s query={tq:.1f}s "
          f"-> est full {tq/nq*len(s1)/60:.1f} min | recall@5/10/20/50 {res}", flush=True)
    return X

for md in (0.0005, 0.002):
    run(f"name char3 max_df={md}", name_c, name_p, analyzer="char", ngram_range=(3, 3), max_df=md)
run("name char4 max_df=0.002", name_c, name_p, analyzer="char", ngram_range=(4, 4), max_df=0.002)
run("name word1-2 max_df=0.01", namew_c, namew_p, analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), max_df=0.01)
for md in (0.0005, 0.002, 0.01):
    run(f"addr word1-2 max_df={md}", addr_c, addr_p, analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), max_df=md)
run("addr char_wb3 max_df=0.002", addr_c, addr_p, analyzer="char_wb", ngram_range=(3, 3), max_df=0.002)
