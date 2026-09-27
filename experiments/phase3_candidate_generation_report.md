# Phase 3 — Candidate generation (blocking) report

*Amazon ML Challenge 2026 · S1 → S2 ∪ S3 entity resolution*

**Scope:** training data only; the test set was not read.

**Artefacts**
* `src/candidate_generation.py`: the pipeline.
* `tests/test_candidate_generation.py`: 19 tests.
* `notebooks/03_candidate_generation_validation.ipynb`: executed; produces every number in this report.
* `experiments/phase3_candidate_generation_results.json`: machine-readable metrics.

This is a new implementation. No earlier Phase 3 code was read or reused. Phase 1 and Phase 2 code is unchanged; Phase 3 calls `normalize_name` and `normalize_address` from `src/normalization.py`.

---

## 1. Objective

For every S1 record, produce a **small set of S2/S3 candidates that contains almost all of its true matches**, without forming S1 × S2 or S1 × S3.

Phase 1 showed that exact or normalised name/address keys reach at most 32.5% recall. This phase targets **≈ 97–99% recall** at a candidate volume that Phase 4 feature computation can handle.

## 2. Computational constraints (measured on this machine)

| resource | value | consequence |
|---|---|---|
| RAM | 8.6 GB total, **~2.1 GB available** at start | DuckDB capped at `memory_limit=1500MB`, 4 threads; ≤ 4 normalisation workers |
| Disk | ~6.8–10 GB free (macOS also grew 3 GB of swap files during the work) | spill capped at `max_temp_directory_size=2GB`; `check_disk(≥ 4 GB)` before every heavy step and every chunk |
| Data | S1 2,206,821 · S2 5,034,616 · S3 5,285,603 | only integer ids + blocking fields are materialised; never raw text; never pandas on a full source |

No step ever loaded a full source into pandas. The biggest pandas objects were the 100k-row normalisation batches.

## 3. Architecture

```
TSV ──DuckDB batches (100k)──► 4 workers: Phase 2 normalize_name/normalize_address ──► Parquet parts (zstd)
                                                                                         features_s{1,2,3}/
features ──DuckDB aggregations (targets only)──► blocking statistics  (blocking_stats.duckdb, 350 MB)
query S1 chunk (≤10k) ──bounded SQL joins per strategy──► cand(s1, t, mask)  ──► evaluate / Parquet part
```

**Integer ids.** `s1 = n` for `S1-n`, and `t = src·10⁹ + n` for S2/S3 (every numeric part is < 10⁹; verified).

**Blocking fields per record.** `blocking_record()` is built on the Phase 2 representations:
* **Name keys** (`name_keys`): compact core name, alias sides and website label.
* **Name tokens:** core-name tokens with stop words and legal forms removed.
* **Phonetic key** (`name_phon`): the *sorted* phonetic tokens, so word order doesn't matter.
* **Address fields:** `state`, `hn` (house-number core), `street` and `places`.
* **Address tokens** (`addr_tokens`): alphabetic, length ≥ 3, state excluded.
* **Numbers:** every number in the address.

**Country** is the first partition of every key, normalised by lower-casing. Nothing is hard-coded to US or India; any new country becomes its own partition.

**Targets are streamed, never indexed.** No inverted index of the 10.3M targets is stored:
* each strategy streams the target keys from Parquet;
* those keys are hash-joined against the small query-side key set;
* only the per-key frequency tables (`st_*`, 0.08–5.3M rows each) are stored.

**Safety mechanisms in code**
1. Every key has a frequency cap. An over-cap key is either narrowed with state or dropped, and the number dropped is reported.
2. `CandidateExplosion` guard: before a strategy runs, its raw pair count is estimated exactly from the statistics. If it exceeds `max_raw_pairs` (50M), the strategy is **not run** and is reported as stopped.
3. Chunked generation, `generate_candidates_chunked`: S1 records are processed in 10k chunks, with a salted `hash(id, 1)` that is independent of any sampling hash. Each chunk is appended to a table or written as its own Parquet part.
4. DuckDB memory and spill limits, plus a disk check before each chunk.

## 4. Blocking strategies tested

| bit | strategy | key (all within country) | cap / rule |
|---|---|---|---|
| 1 | `name` | compact core name / alias side / website label | ≤ 100 targets nationally; else (state, key) ≤ 100; else dropped |
| 2 | `phonetic` | sorted phonetic tokens | ≤ 150 nationally; else (state, key) ≤ 150 |
| 4 | `structured` | (state, house number) | ≤ 100; else refined to (state, hn, place) or (state, hn, street) ≤ 100 |
| 8 | `rare_name` | the 2 rarest name tokens within (state) | token df ≤ 50 |
| 16 | `rare_phon` | the 2 rarest phonetic tokens within (state) | df ≤ 50 |
| 32 | `rare_addr` | the 2 rarest address tokens within (state) | df ≤ 100 |
| 64 | `ngram` | the 8 rarest character 3-grams of the compact name, within (state) | posting df ≤ 1000, ≥ 3 shared grams, top 20 per S1 |
| 128 | `compound` | (state, house number **or** place, phonetic name token) | ≤ 30 targets; frequencies computed only for the chunk's own keys |

**No-state fallback.** Every state-partitioned key is also looked up in the country's **no-state bucket** (`state = ''`), which holds targets with a missing or unparsed address, under the same caps.

**Common tokens** such as `india` (47,601 targets in MH), `services`, `center`, `partners` and `limited` exceed every token cap, so they are never used as rare tokens.

## 5. Results

**Sampling.** Samples are deterministic: `hash(id) % 220 = 0` gives 9,929 S1 records and 34,587 true pairs; `hash(id) % 22 = 0` gives 99,514 S1 records and 344,494 true pairs. Candidates are always retrieved from **all 10,320,219 targets**, so the counts are realistic.

**Metrics.** Candidate statistics are per S1 record, and S1 records with 0 candidates are included. Recall is measured against every true pair of the sampled S1 records.

### 5.1 Pilot 1 — initial design (10k S1)

| strategy | recall | candidates | avg/S1 | P95 | max | seconds |
|---|---:|---:|---:|---:|---:|---:|
| name | 57.41% | 151,719 | 15.3 | 77 | 100 | 1.1 |
| phonetic | 57.31% | 89,162 | 9.0 | 33 | 50 | 0.9 |
| structured | 66.40% | 122,993 | 12.4 | 42 | 92 | 1.9 |
| rare_name | 41.91% | 94,198 | 9.5 | 43 | 95 | 0.9 |
| rare_phon | 19.58% | 59,124 | 6.0 | 40 | 93 | 0.7 |
| rare_addr | 41.17% | 92,293 | 9.3 | 42 | 92 | 1.0 |
| ngram | 24.61% | 25,042 | 2.5 | 20 | 20 | 18.0 |
| **union** | **93.81%** | 480,502 | 48.4 | 111 | 250 | 24.4 |

**What the 2,140 missed pairs had in common:**
* 63.6% are India.
* **28.7% have a target with no state**, which is why the no-state fallback was added.
* 15.4% share state + house number, but inside blocks with a median of **2,801** targets (p90 16,497).
* 85.7% share at least one phonetic token.
* 5.1% have a different state on each side.

### 5.2 Parameter sweep (10k S1)

The base for this sweep is the initial design plus the no-state fallback: **94.84%** recall at 60.1 candidates per S1. Each row shows what one change **adds** to that union.

| strategy | change | Δ recall (pts) | + cands/S1 | pts per 10 cands |
|---|---|---:|---:|---:|
| compound | new, cap 30 | **+1.84** | 12.2 | **1.51** |
| compound | new, cap 60 | +2.09 | 24.3 | 0.86 |
| ngram | m 8, df cap 1000 | +0.52 | 8.1 | 0.64 |
| rare_addr | df cap 100 | +0.52 | 11.4 | 0.46 |
| ngram | min_shared 2 | +0.22 | 5.5 | 0.40 |
| phonetic | cap 150 | +0.22 | 7.2 | 0.31 |
| structured | caps 100 | +0.20 | 10.2 | 0.20 |
| structured | caps 200 | +0.38 | 29.5 | 0.13 |
| rare_name | df cap 100 | +0.38 | 29.0 | 0.13 |
| ngram | top_k 50 | +0.02 | 1.4 | 0.14 |
| name | caps 300 | +0.04 | 9.3 | 0.04 |
| rare_name / rare_phon / rare_addr | k = 3 | 0.00–0.01 | 0.1–0.6 | ~0 |

**Adopted:** compound (cap 30), n-gram m = 8 with df cap 1000, address-token cap 100, phonetic cap 150, and structured caps of 100.

### 5.3 Final configuration (10k S1), with leave-one-out and greedy selection

| strategy | recall alone | avg/S1 | P95 | max | sec | recall lost if removed |
|---|---:|---:|---:|---:|---:|---:|
| compound | 79.42% | 16.6 | 44 | 112 | 7.7 | 1.18 |
| structured | 67.16% | 22.6 | 83 | 163 | 1.8 | 1.42 |
| rare_addr | 51.46% | 21.3 | 88 | 189 | 1.1 | 1.16 |
| phonetic | 60.01% | 22.5 | 104 | 269 | 0.8 | 0.59 |
| ngram | 50.08% | 14.0 | 20 | 20 | 19.6 | 0.37 |
| name | 57.61% | 16.1 | 78 | 119 | 1.1 | 0.31 |
| rare_name | 44.43% | 16.1 | 61 | 148 | 0.9 | 0.26 |
| rare_phon | 20.83% | 10.9 | 56 | 146 | 0.9 | **0.00** |

**Greedy order**, adding the strategy with the best recall gained per candidate at each step:

| step | cumulative recall | avg candidates per S1 |
|---|---:|---:|
| compound | 79.42% | 16.6 |
| + ngram | 87.58% | 28.6 |
| + name | 91.49% | 41.9 |
| + rare_addr | 94.82% | 60.9 |
| + structured | 96.34% | 80.3 |
| + phonetic | 96.99% | 92.2 |
| + rare_name | 97.33% | 103.8 |
| + rare_phon | 97.33% | 108.3 (no gain) |

**Per-S1 cap.** When the cap applies, pairs found by more strategies are kept first, then pairs from higher-priority strategies.

| per-S1 cap | recall | avg | P95 | max | India | US |
|---|---:|---:|---:|---:|---:|---:|
| none | 97.33% | 103.8 | 213 | 425 | 95.39% | 98.65% |
| 300 | 97.33% | 103.7 | 213 | 300 | 95.38% | 98.65% |
| **200** | **97.23%** | 101.3 | 200 | 200 | 95.17% | 98.62% |
| 150 | 96.98% | 95.2 | 150 | 150 | 94.68% | 98.54% |
| 100 | 96.07% | 79.3 | 100 | 100 | 92.82% | 98.27% |

### 5.4 100k validation (99,514 S1 · 344,494 true pairs · 10 chunks of ~10k)

| strategy | recall alone | candidates | avg/S1 | P95 | max | seconds | recall lost if removed |
|---|---:|---:|---:|---:|---:|---:|---:|
| compound | 79.56% | 1,653,446 | 16.6 | 44 | 135 | 59.4 | 1.20 |
| structured | 67.68% | 2,228,998 | 22.4 | 82 | 252 | 13.1 | 1.46 |
| rare_addr | 51.26% | 2,116,453 | 21.3 | 88 | 194 | 8.5 | 1.12 |
| phonetic | 60.00% | 2,273,096 | 22.8 | 105 | 277 | 6.6 | 0.60 |
| ngram | 49.94% | 1,383,038 | 13.9 | 20 | 20 | 153.9 | 0.40 |
| rare_name | 44.57% | 1,586,154 | 15.9 | 61 | 184 | 6.7 | 0.36 |
| name | 57.86% | 1,615,635 | 16.2 | 78 | 119 | 8.5 | 0.31 |
| **union, uncapped** | **97.58%** | 10,311,647 | 103.6 | 212 | 635 | 256.7 | |
| **union, cap 200 (final)** | **97.46%** | 10,063,806 | 101.1 | 200 | 200 | | |

**More detail on the final (cap 200) result:**
* By source: S2 97.66%, S3 97.28%.
* By country: **US 98.75%**, **India 95.52%**.
* S1 records without any candidate: 0.00%.
* Pair precision: 3.3%. That's expected at this stage; Phase 4 filters the candidates.

**Resources:** peak memory (process and children) 1.6 GB, a 0.32 GB work database (deleted afterwards), no spill left behind, and free disk unchanged.

The 100k results match the 10k pilot within 0.25 points, so the pilot generalises.

## 6. Runtime, memory and disk summary

| step | time | peak RSS | disk |
|---|---:|---:|---:|
| Streaming normalisation, all 12.5M records (4 workers) | 490 s (S1 77 s · S2 202 s · S3 211 s) | 1.03 GB | 950 MB Parquet (77 B/record) |
| Blocking statistics (10 tables) | built incrementally across runs; the recorded rebuilds took 3.0 s (`rare_addr`), 25.6 s (`ngram`), 3.3 s (`name`) and 3.0 s (`phonetic`). No single end-to-end timing exists, because the first build was stopped by the §7 error. | ≤ 1.5 GB (DuckDB cap) | `blocking_stats.duckdb` 350 MB |
| 10k S1, 7 strategies | 34 s | 1.3 GB | tens of MB |
| 100k S1, 7 strategies, 10 chunks | 257 s | 1.6 GB | 0.32 GB work DB, deleted |

**Full-scale estimate** (linear from the 100k run; **not executed**):
* 2,206,821 S1 records → **≈ 223M candidate pairs**.
* **≈ 1.0 GB** of Parquet at 4.65 bytes per pair.
* **≈ 1.6 h** on 4 threads, in 220 chunks of 10k.
* `ngram` accounts for 60% of that time (154 of 257 s at 100k) and contributes 0.40 recall points (leave-one-out).

## 7. What went wrong during development (and what was changed)

| event | what exploded | numbers | fix |
|---|---|---|---|
| Building `st_rare_addr` | `SELECT DISTINCT id, country, state, token` over ~50M target address tokens. The DISTINCT needs a hash table keyed by record id, which overflowed the 1.5 GB memory limit and then the 2 GB spill cap. | 1.8 GB spill; DuckDB stopped cleanly with `OutOfMemoryException` and nothing was left on disk | Tokens are already unique per record, so the DISTINCT was removed; n-grams are deduplicated with `list_distinct`. The aggregation now keeps only one counter per (country, state, token): 0.99M rows, built in 3 s. |
| Loading ground-truth ids | 32-bit multiplication `3 · 10⁹` overflowed | | Cast to BIGINT; added a test that loads a real TSV |
| First 100k run | Chunks used `hash(id) % 10` on a sample chosen by `hash(id) % 22 = 0`. Every sampled hash is even, so the odd chunks were empty and each real chunk held ~20k S1. That pushed the `ngram` raw-pair estimate to ~70M, so the guard **stopped `ngram`** in every chunk. | 5 chunks of ~20k; ngram skipped | Salted chunk hash `hash(id, 1)`, plus a regression test. With 10 chunks of 10k, `ngram` stays at ~35M raw pairs, under the 50M limit. |
| Notebook sweep | One n-gram variant exceeded the stricter 30M limit I had set in the sweep | 35.2M estimated raw pairs | The guard refused to run it, as designed. The sweep now records "stopped" and uses the pipeline's 50M limit. |

## 8. Failure cases (final configuration, 10k pilot)

923 true pairs are missed (2.67% of 34,587).

| pattern | share | examples |
|---|---:|---|
| India | 69.9% | |
| Target has no state (missing or unparsed address) | 21.3% | `orthopedic physicians of toledo` ↔ `orthopedic physicians` (no address) |
| State differs between sides | 11.7% | Hyderabad written as `tg` on one side and `ap` on the other |
| No shared name token **and** no shared phonetic token | 21.8% | random replacement names: `grace community church` ↔ `nexxylo`, `systems nirala foods of delhi` ↔ `iridova` |

**Other remaining types**
* Transliterations whose phonetic keys still differ in every token, inside very large city blocks: `great ventures` ↔ `gret bhenycharas`; `guru software` ↔ `guru saphatoyyar` with house numbers 101 vs 48; `southern business` ↔ `sadarn bijnes`.
* Generic names with extra words and no address on the target: `eye group` ↔ `eye group services`. The name key is over its cap nationally, and the target has no state.

## 9. Final chosen blocking combination

`DEFAULT_STRATEGIES = compound, ngram, name, rare_addr, structured, phonetic, rare_name`, with `BlockingConfig()` defaults (§4) and `per_s1_cap = 200`.

| sample | recall | avg / P95 / max candidates per S1 |
|---|---:|---|
| 10k | 97.23% | 101 / 200 / 200 |
| **100k** | **97.46%** | 101 / 200 / 200 |

A cheaper option from the greedy table is to drop `phonetic` and `rare_name`: 96.34% recall at 80 candidates per S1. Dropping `ngram` saves about 60% of generation time for about 0.4 points.

## 10. Rejected strategies and settings

* **`rare_phon`** (phonetic rare tokens): 0.00 points lost when removed, at a cost of about 11 candidates per S1. Compound and phonetic keys already cover it.
* **Exact-key-only blocking** (Phase 1): at most 32.5% recall.
* **More rare tokens per record** (k = 3 or 4): no gain. The rarest two already carry the signal.
* **Higher name caps** (300): +0.04 points for 9 candidates per S1.
* **`ngram_top_k = 50`**: +0.02 points. **`ngram_min_shared = 2`**: dominated by wider postings.
* **Compound cap 60 instead of 30:** +0.25 points for twice the candidates.
* **A full inverted index or full n-gram posting table** of the targets: not built. It would need an estimated 10.3M × ~20 grams ≈ 200M rows, while streaming the target side per chunk costs only about 15 s.
* **A global statistics table for compound keys:** not built. It would need tens of millions of distinct keys, the same failure mode as §7. Compound frequencies are instead computed only for each chunk's keys.
* **MinHash/LSH or FAISS:** not needed. Capped n-gram postings already bound the fuzzy retrieval, and no new dependency was added. `datasketch` isn't installed.

## 11. Recommendations for Phase 4

1. **Generate the full candidate set once** with `generate_candidates_chunked(..., n_chunks=220, out_parquet_dir="data/processed/phase3/candidates")`:
   * about 1.6 h, about 1 GB of Parquet, about 223M pairs;
   * check `df -h` first (need ≥ 4 GB free);
   * consider running it overnight or with `ngram` removed (about 40 min, about 0.4 points lower recall).

   If ~223M pairs is too many for feature computation, use `per_s1_cap = 150` (−0.25 points) or the greedy subset.
2. **Keep `mask` as a feature.** Which strategies found a pair, and how many, is itself strongly predictive: precision rises with the number of agreeing strategies.
3. **Compute features only on candidate pairs**, in chunks aligned with the candidate Parquet parts, joining back to the TSVs by integer id. Useful features:
   * name `token_set` / `token_sort` / Jaro-Winkler on `core_latin`;
   * phonetic similarity;
   * house-number agreement, shared places and state agreement;
   * a missing-address flag;
   * target source (S2 or S3);
   * name frequency (from `st_name` / `st_rare_name`).
4. **Enforce ground-truth structure at inference.** Each target belongs to at most one S1 record, so keep the best-scoring S1 per target. About 26% of targets match nothing.
5. **Evaluate by S1 entity folds**, and report India and US separately. India recall is the weaker side (95.5%), and the remaining misses are mostly random-replacement names and cross-state records that no blocking key can recover.
