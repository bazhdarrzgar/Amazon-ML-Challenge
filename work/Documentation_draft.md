# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** Solomon Belay Mengesha, Bashdar Rzgar Fatih (Tezpur University)
**Submission Date:** [Date]

---

## 1. Executive Summary
RESULTS_SUMMARY

---

## 2. Methodology

### 2.1 Problem Analysis
All statistics below were measured on the supplied files (no sampling unless stated).

| Fact | Value |
|---|---|
| train S1 / S2 / S3 records | 2,206,821 / 5,034,616 / 5,285,603 |
| test S1 / S2 / S3 records | 1,732,544 / 4,887,273 / 5,082,316 |
| train countries (S1) | US 60.0%, India 40.0% |
| test countries (S1) | India 46.8%, US 38.3%, **France 15.0% (unseen in training)** |
| ground-truth pairs | 7,638,365 (S2 3.69M, S3 3.94M) |
| S1 cardinality | singleton 5.6%, 1 match 5.4%, ≥2 matches 89.0%; mean 3.46, max 11 |
| each S2/S3 record matched to at most one S1 | yes (0 re-used ids) |
| S2 / S3 records that match some S1 | 73.4% / 74.6% (the rest are distractors) |
| cross-country matches | 0 of 7.6M |
| S1 names shared by several S1 records | 30% of S1 rows (e.g. "primary care group" ×253) |
| native-script (10 Indic scripts) names in S2 / S3 | ≈9.3% / ≈5.3% (S1 always Latin) |
| empty addresses in S2 / S3 | ≈3.3% (+≈0.8% "null"/"N/A" placeholders) |

Noise patterns observed by diffing the tokens of matched pairs:
* **Names** – legal suffixes dropped / swapped / moved to the front ("PC Viveros & Echeverry"), dotted forms
  (L.L.C.), OCR-style digit swaps ("5ervices", "lnc", "c0m"), junk prefixes ("--", ">>", "***", "#", "@"),
  titles ("Mr", "M/s", "Shri", "Smt", "Dr"), appended phone / store numbers, generic descriptors added
  ("Center", "Services", "Partners"), domain names ("anchorsmartcommunities.com", even partial
  "cementprivate.com"), alias constructions ("Rizanex f/k/a …", "… née …", "dba"), unrelated alias names with
  identical address ("Beloumbra"), truncations, and full transliteration into Devanagari, Kannada, Tamil,
  Telugu, Bengali, Gujarati, Malayalam, Odia and Gurmukhi.
* **Addresses** – street-type abbreviations (Street/St, Road/Rd, …), US state code ↔ full name, Indian
  state name ↔ code (MH, DL, KA…) ↔ native script, reordered components, dropped components, PMB / PO Box /
  "H.NO" prefixes, zero padding ("00345"), perturbed house numbers, "null" placeholders, empty addresses.
  France (test only): R./RUE, AV/Avenue, BD/Boulevard, ALL/Allée, ST/Saint, department vs region labels.
* **Hard negatives** – "sister" records: the same name with a different legal form
  (Private Limited ↔ Limited) at the same building with a substituted unit number; these are *not* matches.

### 2.2 Solution Strategy
**Approach Type:** Blocking (bidirectional TF-IDF retrieval) + two-stage gradient-boosted pairwise classifier +
graph/competition features + F0.5-tuned threshold.
**Core Innovation:** CORE_INNOVATION

---

## 3. Candidate Generation (Blocking)
BLOCKING_SECTION

---

## 4. Matching Model
MODEL_SECTION

---

## 5. Results & Error Analysis
RESULTS_SECTION

---

## 6. Conclusion
CONCLUSION

---

## Appendix

### A. Code Artefacts
CODE_SECTION

### B. Additional Results
ADDITIONAL_RESULTS
