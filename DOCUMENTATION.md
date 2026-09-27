# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Blacksmith  
**Team Members:** Solomon Belay Mengesha, Bashdar Rzgar Fatih (Tezpur University)  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We present an end-to-end, reproducible machine learning pipeline for large-scale business entity resolution across noisy, heterogeneous data sources (Source 1 reference against Source 2 and Source 3). Our solution integrates country-partitioned bidirectional TF-IDF candidate blocking and exact-name address routing (~97.63% blocking recall across ~11.46M pairs) with a two-stage LightGBM architecture. Stage 1 employs 5-fold cross-fitted pairwise GBDT models, while Stage 2 leverages graph-level competition features (measuring mutual exclusivity and competitor margins across the candidate graph). Optimized directly for the competition metric, our system achieves a holdout **Macro $F_{0.5}$ score of 0.9828** (Precision: 99.70%, Recall: 95.72%, Singleton Accuracy: 99.07%) while remaining completely self-contained, GPU-free, and compliant with all competition rules (no external APIs, under 8B parameters, permissive open-source stack).

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory data analysis across the 3 sources revealed distinct challenge-specific characteristics:
- **Scale and Partitions:** The training set contains 2.21M Source 1 records, 5.03M Source 2 records, and 5.29M Source 3 records, generating 7.64M ground-truth matching pairs. The test set comprises 1.73M S1 records, 4.89M S2 records, and 5.08M S3 records.
- **Zero Cross-Country Matches:** 100% of the 7.64M ground-truth matches occur within the same country label. Country labels thus act as an exact partitioning boundary.
- **Open Country Set (France Test Generalization):** Training data covers only the United States (60%) and India (40%), whereas the test set introduces France (15% of test S1 records). All normalisation, feature extraction, and models are country-agnostic to guarantee seamless generalization without hardcoded rules.
- **Noise Patterns in Names:** Significant legal suffix variance (Corp, Inc, Pvt, Ltd, LLC), OCR character substitutions ("5ervices", "lnc", "c0m"), noise prefixes (`--`, `>>`, `@`), generic business terms, domain-name representations, DBA/trade name aliases, and 10 distinct native Indic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam, Odia, Gurmukhi).
- **Noise Patterns in Addresses:** Component reordering, standard vs. abbreviated road suffixes (St/Street, Rd/Road, R./Rue, Av/Avenue), postal PIN code omissions, municipal numbering formats, and landmark references.
- **Cardinality & Singletons:** S1 records exhibit 1-to-many matches (mean 3.46, max 11), while S2 and S3 records belong to at most one S1 entity. 5.6% of S1 entities are singletons (zero matches). Correctly predicting empty match lists for singletons is vital under Macro $F_{0.5}$.

### 2.2 Solution Strategy

**Approach Type:** High-Recall Bidirectional Blocking + Two-Stage Gradient Boosted Trees (LightGBM) + Graph Competition Modeling + Macro $F_{0.5}$ Optimal Decision Policy.

**Core Innovation:** 
1. **Bidirectional Candidate Generation:** Forward top-20 ($S1 \rightarrow \text{Pool}$) plus reverse top-3 ($\text{Pool} \rightarrow S1$) TF-IDF n-gram retrieval combined with an exact-name-block address channel, guaranteeing high recall while bounding candidate pairs.
2. **Two-Stage Graph Competition Features:** Because each Source 2/3 record can belong to at most one Source 1 entity, candidate decisions are not independent. Stage 2 explicitly featurises the rival margin ($p - p_{\text{best\_other}}$), competitor rank, and cluster-level probability aggregates, resolving ambiguity among sister entities.
3. **Learned Cross-Script Transliteration:** A hybrid rule-based and vocabulary-learned transliteration engine mapping 10 Indic scripts to Latin without external lookup.

---

## 3. Candidate Generation (Blocking)

To avoid quadratic pairwise comparison ($1.73\text{M} \times 9.97\text{M} \approx 1.7 \times 10^{13}$ pairs), we perform country-partitioned blocking:

- **Blocking Keys and Channels Used:**
  1. *Forward Channel ($S1 \rightarrow \text{Pool}$):* Sparse TF-IDF word unigrams/bigrams (sublinear TF, max_df 1%) and character 4-grams (max_df 0.2%) for names, and word unigrams/bigrams for addresses. Multi-threaded top-$k$ sparse dot product (`sparse_dot_topn`) retrieves the top 20 candidates per S1 entity by combined cosine similarity:
     $$\text{cos\_comb} = 0.5 \cdot \text{cos}_{\text{name}} + 0.5 \cdot \text{cos}_{\text{addr}}$$
  2. *Reverse Channel ($\text{Pool} \rightarrow S1$):* Top-3 S1 records retrieved for every Source 2 and Source 3 record. Because true owners are almost always among the nearest S1 neighbors of a pool record, this channel recovers matches missed in the forward direction.
  3. *Exact-Name-Block Address Channel:* For entities sharing high-frequency names, address tokens determine identity. Pool records sharing exact normalized core names with 1–100 S1 records are ranked using unpruned address TF-IDF (top-2 kept).
- **Candidate Pairs Generated:**
  - Training: ~11,461,646 candidate pairs (~26 candidates per S1 entity).
  - Test: ~8,920,000 candidate pairs across India, US, and France.
- **How True Matches Were Retained:**
  The union of forward, reverse, and exact-name-block channels achieved **97.63% candidate blocking recall** on the holdout split, retaining true matches even when records suffered from heavy address corruption or spelling variations.

---

## 4. Matching Model

### Features Used (88 Pairwise & Graph Features)
- **Name Features:**
  - RapidFuzz similarity ratios on core, full, and compact strings: Levenshtein ratio, token-sort ratio, token-set ratio, partial ratio, Jaro-Winkler distance.
  - IDF-weighted token intersection, weighted Jaccard, and asymmetric token containment.
  - First token match boolean, length differences, token count ratios.
  - Legal suffix classification: shared legal forms, conflicting legal forms (e.g., Ltd vs Pvt Ltd), private company mismatch flags.
  - Domain indicator and native Indic script detection booleans.
- **Address Features:**
  - RapidFuzz Levenshtein ratio, token-set ratio, and token-sort ratio on normalized address strings.
  - IDF-weighted token overlap and asymmetric containment.
  - House/Building number agreement: first number equality, number set Jaccard, number containment ($A \subseteq B$ and $B \subseteq A$), count of missing/extra numbers, minimum relative numerical difference.
  - Administrative state code agreement (-1 missing, 0 conflict, 1 match).
- **Graph & Context Features (Stage 1 & 2):**
  - Retrieval context: forward rank, reverse rank, cosine similarity score, score gap to top candidate.
  - Competition features: best competing score from rival S1 records ($g_{\text{c\_best\_other}}$), margin over rival ($g_{\text{c\_margin}} = p - p_{\text{best\_other}}$), rival rank, candidate cluster degree, number of competing candidates with $p > 0.5$.
  - Corpus ambiguity features: frequency of core names and core addresses within each country partition.

### Model Architecture
- **Model Type:** Two-Stage Gradient Boosted Decision Trees (LightGBM, MIT licensed).
  - *Stage 1:* 5-fold cross-fitted pairwise LightGBM classifier (`learning_rate=0.08`, `num_leaves=63`, `max_depth=8`, binary logloss objective) generating Out-Of-Fold (OOF) predicted probabilities for all candidate pairs.
  - *Stage 2:* Group-level LightGBM model trained on Stage-1 features augmented with whole-graph competition features, capturing inter-candidate dependencies.
- **Threshold Selection Method:**
  Grid search over prediction probabilities on the holdout fold (Fold 0, strictly isolated from model training) directly optimizing the competition's macro-averaged $F_{0.5}$ metric:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  Optimal threshold selected: **$t = 0.72$**.

---

## 5. Results & Error Analysis

### Holdout Validation Performance

| Stage / Policy | Macro $F_{0.5}$ | Precision | Recall | Singleton Accuracy |
|---|---|---|---|---|
| Stage 1 (Pairwise LightGBM, $t=0.72$) | 0.98152 | 0.9963 | 0.9546 | 98.23% |
| **Stage 2 (Competition LightGBM, $t=0.72$)** | **0.98283** | **0.9970** | **0.9572** | **99.07%** |
| Stage 2 with Top-1 Fallback ($t=0.74, t_{\text{fb}}=0.50$) | 0.98299 | 0.9961 | 0.9591 | 98.85% |

- **Macro $F_{0.5}$ Score:** **0.9828** on clean 441,611 holdout S1 entities and 11.46M candidate pairs.
- **Common False Positives (Wrong Merges):**
  - Distinct business franchises or legal entities operating in identical multi-tenant commercial centers or shopping malls (sharing exact address and postal code) whose trade names exhibit substantial lexical overlap.
  - Common brand branches differentiated only by minor suite numbers omitted in noisy source records.
- **Common False Negatives (Missed Matches):**
  - Entities with extreme abbreviations combined with completely absent address strings in both Source 2 and Source 3 (e.g., empty string or `"null"`), where insufficient signal exists to pass the high-precision threshold $t=0.72$.
  - Rare transliterations of localized rural landmarks that were unseen in the training ground truth dictionary.

---

## 6. Conclusion

Team **Blacksmith** designed and validated a robust, production-grade business entity resolution pipeline tailored specifically for the Amazon ML Challenge 2026. By pairing multi-channel bidirectional TF-IDF blocking with two-stage competition-aware gradient boosting, the system reaches an outstanding **Macro $F_{0.5}$ of 0.9828** and **99.70% precision**. The solution strictly adheres to all competition constraints: it runs entirely on CPU within 3–4 hours, requires zero external APIs or lookups, maintains an ultra-compact footprint (<10 MB total models), and generalizes seamlessly to unseen test countries.

---

## Appendix

### A. Code Artefacts
The reproducible codebase is organized under `code/business_entity_resolution/`:
- `src/ber/prepare.py`: Normalization, address abbreviation mapping, and Indic transliteration learning.
- `src/ber/candidates.py`: Multi-threaded bidirectional sparse TF-IDF and exact-name blocking.
- `src/ber/features.py`: String similarity, numerical analysis, IDF overlap, and graph competition features.
- `src/ber/build_pairs.py`: Blocked feature assembly and parquet streaming.
- `src/ber/cv.py`: Stage 1 5-fold cross-fitting, Stage 2 group reranking, and threshold optimization.
- `src/ber/predict.py`: Two-stage inference pipeline generating `matching_results.tsv` and `candidate_pairs.tsv`.
- `src/ber/pipeline_runner.py`: End-to-end master runner with real-time `tqdm` progress tracking across all stages.
- `run.sh` / `run_all.sh`: Single-command reproduction script.

### B. Execution Command
To reproduce the complete pipeline from scratch:
```bash
# From repository root:
bash run.sh dataset work output
```
Submission outputs generated:
- `output/matching_results.tsv` (Final entity matches)
- `output/candidate_pairs.tsv` (Blocking candidate pairs)
