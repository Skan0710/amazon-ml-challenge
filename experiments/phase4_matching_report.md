# Phase 4 — Pairwise features, matching model, decision rule, submission validation

**Scope:** training data only; the test set was not used for training or tuning.

**Artefacts**
* Code: `src/matching.py`, `scripts/run_phase4.py`, `scripts/validate_final_submission.py`, `tests/test_matching.py`.
* Metrics: `output/pilots/{10k,100k}/phase4_metrics.json`, together with `rule_search_val.csv`, `singletons_val.csv` and `validation_report.txt`.
* Chosen model and rule: `models/phase4_model.joblib` and `models/phase4_rule.json`.

**Metric.** The official `README.md` defines the score as F0.5 **per S1 entity**, macro-averaged over **all** S1 entities:
* a singleton scores 1.0 for an empty prediction and 0.0 for any prediction;
* an entity with true matches scores 0.0 for an empty prediction.

Every "macro F0.5" below is exactly this metric, computed on held-out **validation S1 entities**.

## 1. Data

| | 10k pilot | 100k pilot |
|---|---:|---:|
| S1 entities (`hash(id) % m = 0`) | 9,929 | 99,514 |
| train / validation S1 (80/20 by S1 entity, salted hash) | 7,977 / 1,952 | 79,557 / 19,957 |
| singleton S1 (all / validation) | 528 / 99 | 5,530 / 1,088 |
| true pairs | 34,587 | 344,494 |
| raw candidate rows (sum over the 7 strategies) | 1,282,798 | 12,856,820 |
| duplicates removed (union, plus the per-S1 cap of 200) | 276,863 | 2,793,014 |
| **unique candidate pairs** | **1,005,935** | **10,063,806** |
| duplicates remaining | 0 | 0 |
| candidate recall ceiling (Phase 3) | 97.23% | 97.46% |
| train pairs (positive / negative) | 808,266 (27,047 / 781,219) | 8,040,679 (267,972 / 7,772,707) |
| validation pairs (positive / negative) | 197,669 (6,581 / 191,088) | 2,023,127 (67,785 / 1,955,342) |

The class balance is about **1 positive to 29 negatives** among candidates.

## 2. Negative sampling

All positives are kept.

**Hard negatives** are known non-matches that look similar. A negative counts as hard if any of these hold:
* name Jaro-Winkler ≥ 0.85;
* name-token containment ≥ 0.5;
* the same phonetic key;
* the same house number;
* address-token Jaccard ≥ 0.5;
* found by ≥ 3 blocking strategies.

This covers same name with a different address, same address with a different name, and similar names with different house numbers.

**Easy negatives** are all the others.

Each kept row gets weight 1/(sampling rate), so the weighted training set reproduces the full candidate distribution. Sampling is a deterministic hash of the pair.

| pilot | hard-negative rate | easy-negative rate | training rows (pos / hard / easy) |
|---|---:|---:|---|
| 10k | 1.0 | 0.1 | 620,275 (27,047 / 579,210 / 14,018) |
| 100k | 0.1 | 0.1 | 983,171 (267,972 / 578,438 / 136,761) |

At 100k, hard negatives were sampled at 10% so the training matrix fits in the roughly 2 GB of RAM that was available.

## 3. Features (47, computed in DuckDB SQL)

Features are stored as FLOAT32 per S1 chunk. NULL means "not comparable" (e.g. one side has no house number), and HistGradientBoosting handles NULL natively, so a missing value never looks like similarity.

* **Name (19):**
  * exact raw lower-case equality; compact core-name key equality; overlap of any name key (core, alias or website label);
  * Jaro-Winkler on the compact key, the core name and the raw lower-case name; normalised Levenshtein;
  * token Jaccard, shared-token count and containment; phonetic-key equality and phonetic-token Jaccard; character 3-gram Jaccard;
  * length and token-count differences;
  * Indic script on either side and a script-mismatch flag;
  * log frequency of the target's name key (flags generic names).
* **Address (14):**
  * missing address on S1, target or both;
  * state and house-number equality; log absolute house-number difference;
  * street equality and Jaro-Winkler;
  * place Jaccard and shared-place count; address-token Jaccard and shared count;
  * number Jaccard and shared count.
* **Structure (14):**
  * target source (S2/S3);
  * one indicator per Phase 3 strategy (7) and the number of strategies that found the pair;
  * log of the number of candidates for the S1;
  * rank and gap-to-best of name Jaro-Winkler and of address Jaccard within the S1's candidate list.

Country equality isn't a feature, because candidates are always within the same country. Nothing is one-hot encoded by country, so France or any other unseen label passes through unchanged.

**Permutation importance** (drop in average precision on 100k validation pairs):

| feature | 10k | 100k |
|---|---:|---:|
| `f_num_jacc` (shared address numbers) | 0.186 | 0.216 |
| `f_addr_tok_jacc` | 0.018 | 0.026 |
| `f_key_freq_log` (name genericness) | 0.023 | 0.025 |
| `f_gram_jacc` (name 3-grams) | 0.007 | 0.016 |
| `f_n_strategies` | 0.014 | 0.012 |
| `f_street_jw` | 0.009 | 0.011 |
| `f_raw_jw` | 0.008 | 0.010 |
| `f_name_lev` | 0.007 | 0.006 |
| `f_hn_absdiff_log` | 0.006 | 0.006 |
| `f_core_jw` | 0.004 | 0.005 |

Address-number agreement dominates. Many name features are correlated with each other, so their individual permutation importance understates their combined value.

## 4. Model

* **Model:** `sklearn.ensemble.HistGradientBoostingClassifier` (BSD-3 licence, well within the 8B-parameter limit; no new dependency, and xgboost/lightgbm aren't installed).
* **Settings:** `max_iter=400, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0`, early stopping on 10% of the training rows (patience 30).
* **Training:**

  | | rows | iterations | time |
  |---|---:|---:|---:|
  | 10k | 620,275 | 400 | 60 s |
  | 100k | 983,171 | 275 (early stop) | 50 s |

* **Validation scoring:** 2,023,127 pairs at 157k pairs per second.

## 5. Evaluation (100k validation: 19,957 S1 entities, 69,596 true pairs)

### 5.1 Threshold analysis (threshold only)

| threshold | macro F0.5 | pair P | pair R | FP | FN | predicted | empty S1 | singletons empty (of 1,088) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.8694 | 0.859 | 0.957 | 10,934 | 2,981 | 77,549 | 607 | 568 |
| 0.30 | 0.9196 | 0.926 | 0.943 | 5,222 | 3,961 | 70,857 | 853 | 781 |
| 0.50 | 0.9376 | 0.954 | 0.931 | 3,156 | 4,823 | 67,929 | 969 | 875 |
| 0.60 | 0.9425 | 0.962 | 0.924 | 2,518 | 5,288 | 66,826 | 1,028 | 913 |
| 0.70 | 0.9460 | 0.970 | 0.915 | 1,968 | 5,884 | 65,680 | 1,088 | 952 |
| 0.75 | 0.9471 | 0.974 | 0.910 | 1,689 | 6,293 | 64,992 | 1,113 | 965 |
| **0.80** | **0.9476** | **0.978** | **0.902** | **1,393** | **6,814** | **64,175** | **1,150** | **981** |
| 0.85 | 0.9473 | 0.983 | 0.891 | 1,076 | 7,567 | 63,105 | 1,202 | 1,006 |
| 0.90 | 0.9438 | 0.987 | 0.873 | 772 | 8,839 | 61,529 | 1,264 | 1,025 |
| 0.95 | 0.9329 | 0.992 | 0.838 | 459 | 11,277 | 58,778 | 1,411 | 1,058 |

**Reference points on the same validation set**
* Predicting every candidate: macro F0.5 0.058 (pair P 0.034, R 0.974).
* Predicting nothing: macro F0.5 0.055.

The full 14-threshold grid is in `rule_search_val.csv`.

### 5.2 Threshold vs top-k vs relative rule

| rule family | best macro F0.5 |
|---|---:|
| threshold only (0.80) | **0.9476** |
| threshold + relative-to-best 0.5 | 0.9476 (identical) |
| threshold + top-6 | 0.9470 |
| threshold + top-4 | 0.9361 |
| threshold + top-3 | 0.9112 |
| threshold + top-2 | 0.8470 |

A top-k limit hurts, because true matches per S1 go up to 11 (median 3). A relative-to-best condition adds nothing.

**Chosen:** `DecisionRule(threshold=0.80)`. It is the best on validation, and 0.75–0.85 differ by less than 0.001, so the choice is stable. A match is never forced: an S1 whose candidates all score below 0.80 gets an empty list.

### 5.3 Final validation result (chosen rule)

| | 10k (threshold 0.75) | **100k (threshold 0.80)** |
|---|---:|---:|
| **macro F0.5** | 0.9381 | **0.9476** |
| macro F0.5, non-singletons | 0.9397 | 0.9502 |
| pair precision | 0.9825 | 0.9783 |
| pair recall | 0.8761 | 0.9021 |
| false positives | 106 | 1,393 |
| false negatives | 840 | 6,814 |
| predicted pairs | 6,047 | 64,175 |
| S1 predicted empty | 117 | 1,150 |
| non-singletons predicted empty | 27 | 169 |

About 2.5 points of the recall gap come from the blocking ceiling (97.46%). The rest is the precision-favouring threshold.

## 6. Singleton analysis (100k validation)

| | count |
|---|---:|
| singleton S1 | 1,088 |
| predicted empty (score 1.0) | **981 (90.2%)** |
| predicted non-empty (score 0.0) | 107 |
| singletons without any candidate | 0 |

**Highest candidate probability per singleton:** median 0.085, p90 0.79, p99 0.97. Most singletons have no convincing candidate. The 107 forced-looking cases are near-duplicate decoys (same name with a nearby address) that the model scores above 0.80. Per-singleton rows are in `output/pilots/100k/singletons_val.csv`.

## 7. Submission checks (pilot outputs for the validation S1 entities)

**Setup.** Outputs are written for the validation S1 entities with original ID strings and are compared against a pseudo `test_source1.tsv` containing exactly those rows. The **official `utils/validate_submission.py` runs unmodified**, from `~/student_resource`.

| check | 10k | 100k |
|---|---|---|
| expected / present / missing S1 | 1,952 / 1,952 / 0 | 19,957 / 19,957 / 0 |
| duplicate S1 rows | 0 | 0 |
| duplicate candidate pairs | 0 | 0 |
| duplicate IDs within a list | 0 | 0 |
| matches ⊆ candidates | yes | yes |
| empty `matched_entity_ids` rows | 117 | 1,150 |
| streaming inference vs batch path (pairs differing) | 0 | 0 |
| `scripts/validate_final_submission.py` | **PASS** | **PASS** |
| official `validate_submission.py` | **PASS** | **PASS** |

**Schema.** Headers are exactly `source1_entity_id` with `matched_entity_ids` / `candidate_entity_ids`, tab-separated, UTF-8, `\n` line endings, and comma-joined IDs with no quotes or spaces. An empty field means no match.

**Writers.** `write_*_tsv` refuse to write (`SubmissionError`) on:
* a duplicate S1 row;
* a duplicate pair;
* an unknown target;
* a malformed target ID.

A written row count different from the S1 count is also an error.

**IDs.** Output IDs come from the original strings (`text_*` tables), never from the integer encoding. The encoding is verified injective per file.

## 8. Performance

| | 10k | 100k |
|---|---:|---:|
| total runtime | 348 s | 993 s |
| candidates (Phase 3) | 38.5 s | 365.5 s |
| features | 25.7 s (39k pairs/s) | 279.7 s (36k pairs/s) |
| training | 70.9 s | 56.1 s |
| validation scoring | 1.8 s | 12.9 s (157k pairs/s) |
| streaming inference (features + score + decide) | 11.5 s for 198k pairs | 106.3 s for 2.02M pairs |
| peak RSS (process + children) | 1.91 GB | 1.87 GB |
| disk: work dir (DB + features) | 67 MB | 522 MB (features 250 MB) |
| disk: text tables (one-off, all 12.5M train records) | 283 MB | shared |
| disk: outputs | 2.7 MB | 27.4 MB |

### Full-scale estimate for the test set (NOT RUN)

This extrapolates linearly from the 100k run: 1,732,544 test S1 records, at about 101 candidates per S1, gives **about 175M pairs**.

| step | estimate |
|---|---|
| normalisation + statistics | ~10 min |
| candidates | ~1.8 h |
| streaming inference (~52 µs per pair) | ~2.5 h |
| **total** | **~4.5 h** |
| peak RAM | bounded per chunk, as in the pilots (≈ 2 GB) |
| disk: features + text + statistics | ~1.5 GB |
| disk: candidate Parquet | ~0.8 GB |
| disk: `candidate_pairs.tsv` | **~2.3 GB** |
| disk: `matching_results.tsv` | ~0.1 GB |
| disk: total | ~4.7 GB, against 5.5 GB free now |

The disk margin is tight: free space should be ≥ 8 GB before the run.

The French share of the test set is unknown, and no French labels exist.

## 9. Known limitations and next steps

* **French recall is unmeasured.** France has no state tables, so the `structured` and `compound` blockers don't fire for it.
* **Target exclusivity isn't applied.** Each target belongs to at most one S1 in the training ground truth, and enforcing that could remove decoy false positives. It needs a global pass over all scored pairs, so it can only be evaluated in the full run.
* **The threshold was chosen on a 20k-S1 validation split.** The plateau between 0.75 and 0.85 is flat to within 0.001, so the risk of overfitting it is low.
* **Features:** the top feature is address-number agreement. Legal-form agreement and alias features on the target's raw name could still help the Indic/Latin subset.
