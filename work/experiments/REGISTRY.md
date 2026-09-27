# Experiment Registry — Business Entity Resolution

All scores are **measured** on the clean holdout fold (fold 0 of 5, crc32(S1 id) % 5)
unless stated otherwise. Nothing supervised is fitted on fold 0 during development.

## Environment (Phase 0, 2026-09-25 00:30 IST)
- Apple M5, 10 cores (4P+6E), 16 GB RAM, ~265 GB free disk, no CUDA GPU.
- Python 3.9.6 venv: numpy 1.26.4, pandas 2.2.2, scipy 1.13.1, scikit-learn 1.6.1,
  lightgbm 4.6.0 (needs `brew install libomp`), rapidfuzz 3.13.0, pyarrow 21.0.0,
  indic-transliteration 2.3.82 (MIT), sparse_dot_topn 1.2.0 (Apache-2.0).

## Measured data facts (Phase 2)
| Fact | Value |
|---|---|
| train S1 / S2 / S3 rows | 2,206,821 / 5,034,616 / 5,285,603 |
| test S1 / S2 / S3 rows | 1,732,544 / 4,887,273 / 5,082,316 |
| train countries (S1) | US 1,323,633 (60%), India 883,188 (40%) |
| test countries (S1) | India 809,986 (47%), US 663,106 (38%), France 259,452 (15%) |
| GT pairs | 7,638,365 (S2 3,693,619 / S3 3,944,746) |
| singleton S1 | 5.58% ; 1 match 5.40% ; >=2 matches 89.0% ; mean 3.46 ; max 11 |
| each S2/S3 id matched to at most one S1 | TRUE (max reuse = 1) |
| fraction of S2 / S3 records matched to some S1 | 73.4% / 74.6% |
| cross-country matched pairs | 0 / 7.6M |
| duplicated lower-case S1 names | 30% of S1 rows share a name (e.g. "primary care group" x253) |
| native-script names (S2 / S3) | ~9.3% / ~5.3% (10 Indic scripts); S1 always Latin |
| empty address (S2 / S3) | ~3.3% ; "null"-like tokens ~0.8% |
| learned translit dictionary coverage of test native name tokens | 96.4% (662 entries) |

## Experiments
| ID | Date | Hypothesis | Config | Result | Decision |
|---|---|---|---|---|---|
| B0 | 09-25 00:55 | char-3gram TF-IDF (max_df 5%) name/addr/comb top-K is feasible | India, 44k holdout queries | name channel 200 s / 44k queries -> ~3 h for full train | **Rejected** (too slow: common 3-grams have ~200k postings) |
| B1 | 09-25 01:00 | pruned / word-level representations are fast with good recall | India, 5k holdout queries, top-50 | recall@50: name char3(0.2%) 0.42, name char4 0.61, name word1-2 0.64, **addr word1-2 (0.2%) 0.874** (3 min full), addr char_wb3 0.54 | Keep word1-2 addr + (word1-2 + char4) name |
| B2 | 09-25 01:10 | combined name+addr channel + reverse (pool->S1) retrieval raises recall cheaply | India & US, 5k holdout queries each | India: comb0.5@20 0.948; reverse top1 0.949 @4.7 cand; **comb@20+rev3 0.969 @26 cand**; +name10+addr10 0.973 @33; all@50 0.980 @108. US: comb@20+rev3 **0.981 @25.7**, all@50 0.988 @105 | **Adopt comb@20 + reverse top-3** (w_name 0.5). Extra channels = +0.3-0.5% recall for +25% pairs -> revisit later |
| M0 | 09-25 01:50 | Non-ML baselines (Phase 6) | India holdout (176,658 S1), candidates comb@20+rev3 (cand recall 0.9675) | cos_comb>=0.55: **0.7644**; +"best S1 for candidate" rule: **0.7674** (P 0.809 R 0.841, singleton acc 0.19); fuzzy rule name_tset>=80 & addr_tset>=90: 0.7568 | Reference point |
| M1 | 09-25 02:00 | LightGBM pairwise classifier on 50 pair+context features beats rules | India only, 35% of non-holdout S1 (6.4M pairs), lr .05, 127 leaves, 1781 it | India holdout: threshold t=0.70 **0.9730** (P 0.993 R 0.941 singleton 0.966); expected-F decision 0.9723; oracle(perfect classifier on candidates) 0.9886 | Keep. Error analysis -> M2 features |

### Error analysis of M1 (India holdout, t=0.7): 4,217 FP pairs, 16,133 FN pairs among candidates
- FP-1 "sister company" hard negatives: same name with legal form changed (Private Limited <-> Limited, +LLP) at the same building with a substituted house/unit number (C-205 vs C-207, Flat-004 vs FLAT-7); these records are unmatched. Canonical legal sets collapsed "pvt ltd" vs "ltd" into an overlap.
- FP-2 empty-address candidate whose exact name belongs to 2+ S1 records -> 50/50 ambiguity.
- FP-3 alias-name record at an address shared by two S1 businesses.
- FN-1 empty-address candidates of *unique* names scored low (0.03-0.5): no feature told the model the name is unique.
- FN-2 number noise: zero padding ("00345" vs "345"), prefixed numbers ("H.NO 8 15/9") broke first-number equality.
- FN-3 core token dropped + number truncated ("Delhi Tara" -> "Delhi ... Center", Unit 1206 -> 120).
Fixes (M2 feature set, 79 features): legal equality / symmetric-difference / pvt-mismatch; leading-zero-stripped numbers with containment both ways, missing/extra counts, relative numeric difference; per-candidate best-other-S1 margins separately for name, address and combined cosine; exact-name / exact-address ambiguity counts across S1 and pool.
| B3 | 09-25 02:30 | exact-core-name blocking recovers missed pairs with degraded addresses | India full train candidates (100,417 missed true pairs = 3.28%; 31% have empty candidate address; 40% share the exact core name) | exact-core blocking limited to names held by <=5 S1: recovers 34 pairs (0.03% of misses) for +2.4 pairs/S1; <=10 S1: 431 pairs for +3.0/S1 | **Rejected**: every exact-name miss belongs to a name shared by >5 S1 entities -> inherently ambiguous; unique names are already caught by reverse retrieval |
| M2 | 09-25 02:40 | Error-analysis features (legal equality/pvt-mismatch, zero-stripped number containment both ways, per-candidate best-other margins for name/addr/comb, exact name/address ambiguity counts) fix FP-1/FP-2/FN-1/FN-2 | India only, same split/params as M1, 77 features | India holdout t=0.70: **0.98015** (P 0.9966 R 0.9508 singleton 0.988) vs M1 0.97298 (+0.0072). Calibration near perfect (bin mean p vs rate within 0.02). Loss at t=0.7: partial 0.0146 (FN pairs 28.8k, 65% of them not in candidates), non-singleton empty 0.0046, singleton FP 0.0007 | **Keep** |
| D1 | 09-25 03:30 | Smarter set decisions beat a global threshold (expected-F, top-1 fallback, relative-to-best) | M2 India holdout, vectorised evaluator (verified == reference metric) | threshold 0.98015; best fallback (t .74, t_fb .70) 0.98023; relative 0.98021; expected-F (M1) -0.0007 vs threshold | **Keep global threshold** (differences <1e-4, not worth complexity) |
| B4 | 09-25 03:00 | Exact-name blocks re-ranked by *unpruned* address TF-IDF recover misses caused by max_df pruning city names | India train, 40,139 exact-name misses (26k with non-empty address) | pessimistic tie handling: true S1 ranked top-2 inside its name block for 92.6% -> recovers 24k of 100k India misses (+0.8% recall) | Implemented as opt-in channel (`--name-block`) -> evaluate end-to-end as v3 |
| M3 | 09-25 04:15 | Full-data stage-1 cross-fit (US+India) with M2 features | candidates comb@20+rev3 (holdout cand recall 0.9763), 5 fold models, 20% S1 sample (11.4M rows), lr .08, 127 leaves, 694-989 it | **Full holdout (441,611 S1): 0.98152 @ t=0.72** (P 0.9963 R 0.9546 singleton 0.9823, 1,676 non-singletons empty); owner constraint w/ OOF competitors 0.98170; fallback 0.98151 | Keep; stage-2 next |
