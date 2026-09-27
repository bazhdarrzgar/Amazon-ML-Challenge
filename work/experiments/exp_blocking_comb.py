"""EXP-B2: combined name+address channel, channel unions and reverse (pool->S1) retrieval."""
import sys, time
sys.path.insert(0, "code/business_entity_resolution/src")
import numpy as np, pandas as pd, scipy.sparse as sp
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
n1 = len(s1)

def fit(texts, **kw):
    v = TfidfVectorizer(lowercase=False, sublinear_tf=True, dtype=np.float32, min_df=2, **kw)
    return v.fit_transform(texts).tocsr()

t = time.time()
nw = fit([c or f for c, f in zip(list(s1.n_core) + list(pool.n_core), list(s1.n_full) + list(pool.n_full))],
         analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), max_df=0.01)
nc = fit([" " + (c.replace(" ", "") or f.replace(" ", "")) + " " for c, f in
          zip(list(s1.n_core) + list(pool.n_core), list(s1.n_full) + list(pool.n_full))],
         analyzer="char", ngram_range=(4, 4), max_df=0.002)
aw = fit(list(s1.a_core) + list(pool.a_core), analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), max_df=0.002)
print(f"vectorised {time.time()-t:.0f}s", flush=True)

def norm_rows(X):
    n = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()); n[n == 0] = 1
    return sp.diags(1 / n) @ X

name = norm_rows(sp.hstack([nw, nc]).tocsr())
mats = {"name": name, "addr": aw}
for w in (0.3, 0.5):
    mats[f"comb{w}"] = sp.hstack([name * np.sqrt(w), aw * np.sqrt(1 - w)]).tocsr()

res = {}
for ch, X in mats.items():
    A, B = X[:n1][q], X[n1:]
    t = time.time()
    C = sp_matmul_topn(A, B.T.tocsr(), top_n=50, threshold=1e-4, sort=True, n_threads=10).tocsr()
    tq = time.time() - t
    res[ch] = [C.indices[C.indptr[i]:C.indptr[i + 1]] for i in range(nq)]
    rec = {k: round(sum(len(truth[i] & set(res[ch][i][:k].tolist())) for i in range(nq)) / n_true, 4) for k in (5, 10, 20, 50)}
    print(f"{ch:10s} query={tq:.1f}s est full={tq/nq*n1/60:.1f}min recall@5/10/20/50 {rec}", flush=True)

# reverse channel: every pool record -> top-k S1 (comb0.5), keep pairs whose S1 is a query
X = mats["comb0.5"]
t = time.time()
R = sp_matmul_topn(X[n1:], X[:n1].T.tocsr(), top_n=5, threshold=1e-4, sort=True, n_threads=10).tocsr()
print(f"reverse channel (all {len(pool)} pool records, top5) {time.time()-t:.0f}s", flush=True)
Rt = R.T.tocsr()  # S1 x pool, holds reverse scores; rank of S1 within pool record's list
rev = {}
Rc = R.tocoo()
rank = np.zeros(R.nnz, np.int8)
for r in range(R.shape[0]):
    s, e = R.indptr[r], R.indptr[r + 1]
    order = np.argsort(-R.data[s:e])
    rank[s + order] = np.arange(e - s)
rev_pairs = pd.DataFrame({"p": np.repeat(np.arange(R.shape[0]), np.diff(R.indptr)), "q": R.indices, "rk": rank})
qpos = {v: i for i, v in enumerate(q)}
rev_pairs = rev_pairs[rev_pairs.q.isin(qpos)]
revsets = {k: [set() for _ in range(nq)] for k in (1, 3, 5)}
for p_, q_, rk in rev_pairs.itertuples(index=False):
    for k in (1, 3, 5):
        if rk < k:
            revsets[k][qpos[q_]].add(p_)
for k in (1, 3, 5):
    rec = sum(len(truth[i] & revsets[k][i]) for i in range(nq)) / n_true
    print(f"reverse top{k}: recall={rec:.4f} cand/S1={sum(len(s) for s in revsets[k])/nq:.1f}")

def union(spec, rk=0):
    hit = cand = 0
    for i in range(nq):
        S = set()
        for ch, k in spec.items():
            S |= set(res[ch][i][:k].tolist())
        if rk:
            S |= revsets[rk][i]
        hit += len(truth[i] & S)
        cand += len(S)
    return round(hit / n_true, 4), round(cand / nq, 1)

for spec in [{"comb0.5": 20}, {"comb0.5": 50}, {"comb0.3": 50},
             {"name": 10, "addr": 10, "comb0.5": 20}, {"name": 20, "addr": 20, "comb0.5": 30},
             {"name": 30, "addr": 30, "comb0.5": 50}, {"name": 50, "addr": 50, "comb0.5": 50, "comb0.3": 50}]:
    for rk in (0, 3, 5):
        print("union", spec, "rev", rk, "-> recall, cand/S1 =", union(spec, rk), flush=True)
