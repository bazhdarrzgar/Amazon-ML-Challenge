# Amazon ML Challenge 2026: Business Entity Resolution

### Team: Blacksmith
**Project Owners & Authors:**
* **SOLOMON BELAY MENGESHA** — *Tezpur University*
* **BASHDAR RZGAR FATIH** — *Tezpur University*

> **Official Methodology Document:** See [Documentation_template.md](Documentation_template.md) for full statistical EDA, blocking design, model hyperparameters, ablation results, and error analysis.

---

## Business Entity Resolution — Reproducible Pipeline

Blocking + two-stage gradient-boosted matcher for the ML Challenge 2026
(Source 1 → Source 2/3 entity resolution, macro-F0.5).

```
raw TSVs ─► normalise (+ learned transliteration) ─► per-country bidirectional TF-IDF retrieval
        ─► pair features (string, token-IDF, number, legal-form, graph/competition, ambiguity)
        ─► stage-1 LightGBM (5 cross-fitted fold models, OOF)
        ─► stage-2 LightGBM (stage-1 probabilities aggregated over the candidate graph)
        ─► threshold chosen on the holdout fold ─► matching_results.tsv / candidate_pairs.tsv
```

No external data, APIs, geocoding or pretrained language models are used. The
only "models" are LightGBM gradient-boosted trees trained from scratch on the
provided training labels (MIT-licensed LightGBM library; a few MB, far below
the 8B-parameter limit). All libraries are permissively licensed
(BSD / MIT / Apache-2.0).

## 1. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.9 tested
pip install -r requirements.txt
# macOS only (OpenMP runtime for LightGBM):
brew install libomp
```

Hardware used: Apple M5, 10 CPU cores, 16 GB RAM, no GPU. Peak RAM ≈ 10 GB.
Disk: ≈ 30 GB for intermediate parquet files.

## 2. Data placement

Default layout (the challenge's `student_resource/` folder):

```
student_resource/
├── dataset/train/{train_source1,2,3,train_ground_truth}.tsv
├── dataset/test/{test_source1,2,3}.tsv
├── utils/validate_submission.py
└── code/business_entity_resolution/        <- this folder
```

All TSVs are read with an explicit tab separator, `quoting=QUOTE_NONE` and
`na_filter=False` (literal strings such as "NA"/"null" are preserved).

## 3. One-command reproduction

```bash
cd code/business_entity_resolution
PYTHON=../../.venv/bin/python bash run_all.sh ../../dataset ../../work ../../output
```

Arguments: dataset dir, work dir (intermediate files), output dir.
Wall-clock on the hardware above ≈ 3–4 h (retrieval ≈ 1.5 h, features ≈ 1 h,
training ≈ 1 h, inference ≈ 15 min).

## 4. Step by step

All commands run with `PYTHONPATH=src`.

| Step | Command | Output |
|---|---|---|
| 1. Normalise + learn transliteration dictionaries (training folds only) | `python -m ber.prepare --data-dir DATA --work-dir WORK` | `WORK/cache/norm_*.parquet`, `WORK/models/translit.json` |
| 2. Candidates + features (train) | `python -m ber.build_pairs --split train --name-block` | `WORK/cand/train/*.parquet`, `WORK/pairs/train/<country>/part_*.parquet` |
| 3. Candidates + features (test) | `python -m ber.build_pairs --split test --name-block` | `WORK/pairs/test/...` |
| 4. Stage-1 cross-fit | `python -m ber.cv stage1 --tag final` | `WORK/models/final/model_s1_f{0..4}.txt`, `oof_stage1.parquet`, `eval_stage1.json` |
| 5. Stage-2 + decision | `python -m ber.cv stage2 --tag final` | `model_s2.txt`, `eval_stage2.json`, `decision.json` |
| 6. Inference | `python -m ber.predict --models WORK/models/final --pairs WORK/pairs/test --out OUTPUT` | `OUTPUT/matching_results.tsv`, `OUTPUT/candidate_pairs.tsv` |
| 7. Validate | `python3 ../../utils/validate_submission.py --matching OUTPUT/matching_results.tsv --candidate OUTPUT/candidate_pairs.tsv --test-dir DATA/test` | `PASS` |

Unit tests: `python tests/test_metric.py && python tests/test_normalize.py`.

## 5. Source layout (`src/ber/`)

| Module | Purpose |
|---|---|
| `io.py` | literal TSV loading, parquet cache, submission writer |
| `translit.py` | Indic-script → Latin (rule-based + dictionary learned from training matches) |
| `normalize.py` | name views (core / legal forms / alias parts / domain), address views (admin components removed, abbreviations canonicalised, numbers) |
| `folds.py` | deterministic S1-level folds (crc32 % 5); fold 0 = clean holdout |
| `candidates.py` | per-country TF-IDF vectors; forward (S1→pool) and reverse (pool→S1) top-k retrieval; exact-name-block address channel |
| `features.py` | pair, context (graph competition) and ambiguity features |
| `build_pairs.py` | orchestration of retrieval + features in memory-bounded blocks |
| `metric.py` | official macro-F0.5 (unit-tested against the README example) |
| `decide.py` | set-level decision policies (threshold, owner constraint, expected-F) |
| `train.py` | single-model development loop |
| `cv.py` | stage-1 cross-fitting (OOF), stage-2 group model, holdout evaluation, decision.json |
| `predict.py` | two-stage test inference and submission writing |
| `bench_blocking.py` | blocking recall benchmark |

## 6. Determinism and leakage control

* Folds are a deterministic hash of the S1 id; all pairs of an S1 entity stay in one fold.
* Fold 0 is the holdout. The transliteration dictionary, the stage-2 model and
  the decision threshold are fitted without fold-0 labels; fold-0 stage-1
  scores come from the fold model that never saw fold 0.
* TF-IDF / IDF statistics are unsupervised and fitted per split on that split's own records
  (train on train, test on test) — no labels are involved.
* LightGBM seeds are fixed (`seed=42`); `sparse_dot_topn` retrieval is exact (deterministic).
* Every test S1 id is written exactly once in both output files; empty lists are empty fields.
