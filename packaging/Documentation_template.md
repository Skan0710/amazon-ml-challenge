# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Nth Times the Charm
**Team Members:** Dhruva Gupta (Team Leader), Agneesh Mondal, Anirudh Shenoy, Omkar Dabholkar
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We normalise every record once into several script-aware representations: names, Indic-to-Latin transliteration, phonetic keys, and parsed addresses.

We then generate candidates with a union of seven capped blocking strategies. These keep **97.46%** of true pairs while producing about 101 candidates per Source-1 entity, with no S1 × S2/S3 product ever formed.

Each candidate pair is scored by a gradient-boosted classifier on 47 interpretable features. A single probability threshold (0.80), chosen by the competition's **macro F0.5** on a held-out split of Source-1 entities, decides the matches. A match is never forced, so singletons correctly receive empty lists.

On held-out training entities this gives **macro F0.5 = 0.948** (pair precision 0.978, recall 0.902). Every heavy step runs in DuckDB over Parquet in bounded memory on an 8 GB laptop.

---

## 2. Methodology

### 2.1 Problem Analysis

The exploratory analysis covered all training data:
* **Row counts:** S1 2,206,821 · S2 5,034,616 · S3 5,285,603 records.
* **Ground truth:** 7,638,365 true pairs, with a mean of 3.46 matches per S1 entity (median 3, max 11).
* **Singletons:** 5.6% of S1 entities have no match.
* **Clusters are disjoint:** no target record belongs to two S1 entities.
* **Country:** agrees in 100% of true pairs.

Key observations:
* **Names are noisy.** Typos and OCR digits (`5ervices`), accents, junk prefixes (`***`), legal-form variants and reordering (`Pvt Ltd` ↔ `Private Limited`, `Inc` moved to the front), website names (`zionterm.com`), alias wrappers (`X a/k/a Y`, `dba`, `formerly known as`) and fully replaced names.
* **Indian names appear in nine Indic scripts.** 23% of S2-India names are in Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada or Malayalam. 7.2% of true pairs are Indic vs Latin script.
* **Addresses are inconsistent.**
  * S2 is upper-case, uses state codes, zero-pads house numbers and writes native-script state names.
  * S3 spells US states out and uses 2-letter codes for Indian states.
  * Both contain `null`/`N/A`/`##` noise, shuffled components, truncation and altered house numbers.
  * About 3.3% of S2/S3 addresses are missing.
* **Exact keys are not enough.** Exact or normalised name/address keys recover at most **32.5%** of true pairs, so token, n-gram and structured blocking is required.
* **Test set.** The test data adds a third country, **France**, which is absent from training. Country is therefore treated as an open set everywhere.

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier, with a per-entity decision rule.

**Core Innovation:**
1. **Script-aware normalisation.** One offset-table transliterator covers all nine Brahmic scripts, alongside a loose phonetic key, so Indic and Latin spellings of the same business become comparable.
2. **A "compound" blocking key.** State + (house number or locality) + phonetic name token is specific even when every part is common. It is the single most efficient blocker: 79.6% recall at 17 candidates per S1.
3. **Bounded-memory execution.** Everything is streamed and chunked through DuckDB, with frequency caps, raw-pair explosion guards, disk guards and atomic output publication.

---

## 3. Candidate Generation (Blocking)

Every key is partitioned by country and capped by its target-side frequency. Frequencies are computed from the source records only, never from the ground truth. Keys that exceed their cap are narrowed by state or dropped.

Targets whose address has no state are also reachable through a per-country "no-state" bucket.

| strategy | key | recall alone (100k S1 validation) |
|---|---|---:|
| compound | (state, house number or place, phonetic name token), ≤ 30 targets | 79.6% |
| structured | (state, house number); refined by place or street for large blocks | 67.7% |
| phonetic | sorted phonetic name tokens | 60.0% |
| exact name | compact core name, alias sides, website label | 57.9% |
| rare address tokens | the 2 rarest address tokens within the state | 51.3% |
| character 3-grams | the 8 rarest grams of the compact name, ≥ 3 shared, top 20 | 49.9% |
| rare name tokens | the 2 rarest name tokens within the state | 44.6% |

- **Blocking keys used:** the seven strategies above, unioned and deduplicated. A per-entity cap of 200 candidates keeps pairs found by more strategies first.
- **Candidate pairs generated:** 171,756,016 on the test set (1,732,544 Source-1 entities; 99.1 per entity). On training, 99,514 sampled S1 entities gave 10,063,806 pairs (101 per entity; P95 200; max 200).
- **How we ensured true matches were not lost:**
  * Each strategy was chosen by measured recall against the training ground truth: leave-one-out and greedy recall per candidate on a 10k pilot, confirmed on 100k entities.
  * The union reaches **97.46%** recall on 99,514 held-out S1 entities (S2 97.66%, S3 97.28%; US 98.75%, India 95.52%), against 32.5% for exact keys.
  * The no-state bucket recovers targets with missing addresses.
  * The remaining misses are mostly fully replaced names or records with different states on each side.

---

## 4. Matching Model

**Features used (47, computed in DuckDB SQL per chunk of pairs; NULL = not comparable):**
- **Name features (19):**
  * raw lower-case equality; compact-key equality; overlap of any name key (core, alias or website);
  * Jaro-Winkler on the compact key, the core name and the raw name; normalised Levenshtein;
  * token Jaccard, shared count and containment; phonetic equality and phonetic-token Jaccard; character-3-gram Jaccard;
  * length and token-count differences;
  * Indic-script flags for each side and a script-mismatch flag;
  * log frequency of the target name (genericness).
- **Address features (14):**
  * missing address on S1, target or both;
  * state equality; house-number equality and log absolute difference;
  * street equality and Jaro-Winkler;
  * place Jaccard and shared count; address-token Jaccard and shared count;
  * address-number Jaccard and shared count.
- **Other (14):**
  * target source (S2/S3);
  * one indicator per blocking strategy (7) and the number of strategies that found the pair;
  * candidates per entity;
  * rank and gap-to-best of name similarity and of address similarity within the entity's candidate list.

The most important features (permutation importance, drop in average precision on validation) are: shared address numbers 0.216, address-token Jaccard 0.026, name genericness 0.025, name 3-gram Jaccard 0.016 and the number of agreeing strategies 0.012.

**Model type:** scikit-learn `HistGradientBoostingClassifier` (BSD-3 licence, far below the 8B-parameter limit).
* Settings: learning rate 0.08, 63 leaves, min 40 samples per leaf, L2 1.0; stopped early at 275 iterations.
* Training data:
  * all 267,972 positive pairs;
  * 578,438 hard negatives (similar name, same phonetic key, same house number, similar address, or found by ≥ 3 strategies), sampled at 10%;
  * 136,761 easy negatives, sampled at 10%;
  * each row weighted by the inverse of its sampling rate, so the training distribution matches the full candidate distribution.
* Train/validation split: **by Source-1 entity** (80/20, deterministic hash), so no entity appears on both sides.

**Threshold selection method:**
* The metric is **macro F0.5** exactly as defined by the competition: per Source-1 entity, including singletons and entities with no candidates, averaged over all entities.
* It was computed on 19,957 validation entities for a grid of threshold × top-k × relative-to-best rules.
* A plain threshold of **0.80** won. The curve is flat between 0.75 and 0.85, all within 0.001. Top-k limits hurt (top-3: 0.911), because an entity can have up to 11 true matches.
* A match is never forced: an entity whose candidates all score below 0.80 gets an empty list.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9476** on 19,957 held-out training S1 entities.
  * Pair precision 0.9783, pair recall 0.9021, false positives 1,393, false negatives 6,814.
  * Macro F0.5 over non-singletons only: 0.9502.
  * Singletons: 981 of 1,088 (90.2%) correctly predicted empty.
  * The 10k pilot gave 0.9381.
  * Baselines on the same data: predicting all candidates scores 0.058; predicting nothing scores 0.055.
- **Common false positives (wrong merges):**
  * Near-duplicate "decoys": the same business name at a nearby address with a slightly different house number (e.g. `4217` vs `4228 Silverthorne Dr`).
  * Same address with a lightly modified name (`… Capital LLC` vs `… Capital Harbor LLC`).
  * Most of the 107 singletons given a match are such decoys, scoring above 0.8.
- **Common false negatives (missed matches):**
  * About 2.5 points of recall are lost at blocking: fully replaced names (`Consolidated Education Systems` ↔ `Ciraaria`), and records with different states on each side (`TG` vs `AP` for Hyderabad).
  * The rest are rejected by the precision-oriented threshold. These are typically transliterations that differ in every token, or a heavily truncated address combined with a changed name.

**Real test inference** (no labels exist, so no score can be computed; the figures below are descriptive):

| | value |
|---|---:|
| Source-1 entities | 1,732,544 |
| candidate pairs | 171,756,016 |
| predicted matched pairs | 5,403,410 |
| entities with an empty (no-match) list | 117,592 (6.8%; 9,037 of them had no candidate at all) |
| India / US: entities / mean predicted matches / empty lists | 809,986 / 3.13 / 52,430 · 663,106 / 3.29 / 37,882 |
| France (unseen country): entities / mean predicted matches / empty lists | 259,452 / 2.67 / 27,280 |
| runtime / peak RAM / peak temporary disk | ≈ 5.8 h of compute across resumable runs (candidates 2 h 13 min; scoring ≈ 3.5 h; writing 2.7 min; validation 6.6 min) / 2.16 GB (pipeline process tree) / 5.0 GB (working files + outputs) |
| official `validate_submission.py` | PASS — sharded over both full files (38 shards) and on the full matching file with `--check-ids` |

---

## 6. Conclusion

Careful, script-aware normalisation plus a union of capped, frequency-aware blocking keys turns an intractable 2.2M × 10.3M comparison into about 101 candidates per entity while keeping 97.5% of true matches.

A precision-oriented gradient-boosted matcher with a validated threshold then reaches macro F0.5 ≈ 0.948 while leaving 90% of singletons correctly empty.

The main lessons:
* Always optimise and validate against the exact per-entity metric.
* Engineer every stage for bounded memory before scaling up. Pilots at 10k and 100k entities exposed every scaling issue before the full run.

---

## Appendix

### A. Code Artefacts

The runnable code is under `code/business_entity_resolution/`. `README.md` there gives exact commands, and `requirements.txt` pins versions.

| file | role |
|---|---|
| `src/normalization.py` | name/address normalisation, script detection, transliteration, phonetic keys, state tables |
| `src/candidate_generation.py` | streaming feature materialisation, blocking statistics, 7 blocking strategies, recall evaluation |
| `src/matching.py` | pairwise features, training, scoring, decision rule, macro-F0.5 evaluation, chunked writers, atomic publication |
| `src/build_train_artifacts.py` | training-side blocking artefacts |
| `src/run_phase4.py` | `pilot` (train + choose rule on held-out entities) and `test` (final inference) |
| `src/preflight_phase4.py` | environment checks + 1,000-entity dry run on the real test files |
| `src/validate_final_submission.py`, `src/run_official_validator_sharded.py` | streaming validator; bounded-memory runner for the unmodified official validator |
| `models/` | the exact model and rule used |

**Entry point to reproduce `output/`:**

```
python src/run_phase4.py test --dataset-dir <test> --model models/phase4_model.joblib --rule models/phase4_rule.json --chunks 220 --infer-chunks 440 --infer-threads 2 --resumable-inference --min-free-gb 3.0 --min-free-write-gb 4.0 --i-approve-full-scale
```

### B. Additional Results

| stage | 10k pilot | 100k pilot |
|---|---:|---:|
| S1 entities (train / validation) | 7,977 / 1,952 | 79,557 / 19,957 |
| unique candidate pairs | 1,005,935 | 10,063,806 |
| candidate recall ceiling | 97.23% | 97.46% |
| validation macro F0.5 | 0.9381 | 0.9476 |
| pair precision / recall | 0.9825 / 0.8761 | 0.9783 / 0.9021 |
| peak RAM | 1.91 GB | 1.87 GB |
| runtime | 348 s | 993 s |

**Threshold curve** (100k validation, macro F0.5): 0.50 → 0.9376 · 0.60 → 0.9425 · 0.70 → 0.9460 · 0.75 → 0.9471 · **0.80 → 0.9476** · 0.85 → 0.9473 · 0.90 → 0.9438 · 0.95 → 0.9329.

**Engineering safeguards** (each covered by automated tests):
* **Streaming and chunking:**
  * normalisation is streamed from DuckDB batches into Parquet;
  * candidates are generated in about 7.9k-entity chunks, with a raw-pair explosion guard of 50M;
  * features, scoring and decisions are computed per entity chunk;
  * output files are written in 25k-entity batches;
  * both validators stream.
* **Invariants:**
  * duplicate-pair and ID-format checks run before writing;
  * a no-forced-match gate checks that every match is a candidate at or above the threshold;
  * outputs are written as `*.tmp` and renamed only after both validators pass.
* **Tests:** 105 automated tests cover normalisation, blocking, features, the metric, writers, validators, atomic publication and end-to-end inference.
