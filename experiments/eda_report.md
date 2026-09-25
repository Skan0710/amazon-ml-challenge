# Phase 1 — Training-data EDA report

*Amazon ML Challenge 2026 · business entity resolution (S1 → S2 ∪ S3)*
*Source notebook: `notebooks/01_eda.ipynb` · key numbers: `experiments/eda_summary.json` · training data only; the test set was not read.*

## 1. Row counts and schema

| File | Rows |
| ------------ | ---: |
| Source 1 | 2,206,821 |
| Source 2 | 5,034,616 |
| Source 3 | 5,285,603 |
| Ground Truth | 2,206,821 |

* **Sources 1–3:** `entity_id, business_name, business_address, country`, all VARCHAR, 4 columns.
* **Ground truth:** `source1_entity_id, matched_entity_ids` (comma-separated), VARCHAR, 2 columns.
* **Dialect:** tab-delimited with a header and **no quote character**. The few fields containing literal `"` are kept verbatim.
* Row counts equal raw newline counts minus the header.

## 2. Missing values and duplicates

| | S1 | S2 | S3 |
|---|---:|---:|---:|
| NULL / empty `entity_id`, `business_name`, `country` | 0 | 0 | 0 |
| NULL `business_address` | 0 | 168,967 (3.36%) | 175,916 (3.33%) |
| duplicate `entity_id` | 0 | 0 | 0 |
| exact duplicate records (name+addr+country) | 0 | 25,873 | 18,860 |
| basic-normalised duplicate records | 0 | 47,381 (0.94%) | 32,932 (0.62%) |
| exact duplicate **names** (rows beyond first) | 667,592 (30.3%) | 632,607 | 633,994 |
| exact duplicate **addresses** | 76,215 | 528,388 | 476,923 |

Empty strings never occur; blanks arrive as NULL. No ID is shared across sources.

## 3. Country distribution

| | US | India |
|---|---:|---:|
| S1 | 1,323,633 (59.98%) | 883,188 (40.02%) |
| S2 | 3,016,817 (59.92%) | 2,017,799 (40.08%) |
| S3 | 3,170,056 (59.98%) | 2,115,547 (40.03%) |

* Only two raw values exist (`US`, `India`), and both appear in all sources. No country is missing.
* Country agrees in **100% of true pairs**.

## 4. Ground truth

| match_count | S1 entities | % |
|---|---:|---:|
| 0 | 123,247 | 5.58 |
| 1 | 119,157 | 5.40 |
| 2 | 375,212 | 17.00 |
| 3 | 530,841 | 24.05 |
| 4 | 484,115 | 21.94 |
| 5+ | 574,249 | 26.02 |

* Matches per S1 record: mean 3.46 (3.67 if matched), median 3, max 11.
* 7,638,365 true pairs in total.
* Integrity checks:
  * every matched ID exists in S2 or S3, never in S1
  * no target ID is repeated anywhere in the GT, so clusters are disjoint
  * no duplicate pairs
* Country makes no difference to the match distribution: India 3.465 vs US 3.459 matches per S1.

**S2 vs S3**

| | true pairs | share | S1 entities with ≥1 | source rows matched |
|---|---:|---:|---:|---:|
| S2 | 3,693,619 | 48.36% | 1,919,076 | 73.36% |
| S3 | 3,944,746 | 51.64% | 1,940,545 | 74.63% |

| S1 category | count | % |
|---|---:|---:|
| both S2 and S3 | 1,776,047 | 80.48 |
| S3 only | 164,498 | 7.45 |
| S2 only | 143,029 | 6.48 |
| no matches | 123,247 | 5.58 |

* About **26% of S2/S3 rows match no S1 entity** (1,340,997 in S2 and 1,340,857 in S3).
* Only ~4% of those share a normalised name with any S1 record, and <0.4% share an address. They are mostly unrelated businesses, not engineered decoys.

## 5. Field agreement inside true pairs (all 7.6M pairs)

| | exact name | basic name | alnum name | exact addr | basic addr | alnum addr | target addr NULL | Indic↔Latin name |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S2 | 4.73% | 15.64% | 25.28% | 0.00% | 11.22% | 13.06% | 4.47% | 9.33% |
| S3 | 4.55% | 15.89% | 26.27% | 4.51% | 4.52% | 4.52% | 4.36% | 5.24% |

## 6. Hard positives (RapidFuzz, 100k sampled true pairs)

| % of sampled true pairs | S2 | S3 | all |
|---|---:|---:|---:|
| name token_sort < 50 | 13.78 | 11.36 | 12.53 |
| name token_sort < 70 | 25.70 | 24.97 | 25.32 |
| address token_sort < 50 | 1.35 | 5.78 | 3.64 |
| address missing on one side | 4.54 | 4.31 | 4.42 |
| Indic-script vs Latin name | 9.39 | 5.11 | 7.17 |
| no shared name token | 16.08 | 12.56 | 14.26 |
| no shared address number | 13.61 | 12.78 | 13.18 |
| name < 50 and address < 50 or missing | 0.94 | 1.17 | 1.06 |
| no shared token in name **or** address | 0.00 | 0.00 | 0.00 |

## 7. Blocking: candidates and recall (S1 → S2 ∪ S3)

Candidates are computed exactly as `Σ_k n_S1(k)·n_T(k)`, without materialising any pairs. Recall is measured over all 7,638,365 true pairs.

| strategy | S1 covered | candidates | avg / S1 | p99 | max | recall | pair precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| A exact name | 35.84% | 8,389,568 | 3.80 | 83 | 706 | 4.64% | 4.22% |
| B basic-norm name | 61.67% | 15,852,539 | 7.18 | 138 | 918 | 15.77% | 7.60% |
| C exact address | 8.91% | 204,979 | 0.09 | 1 | 6 | 2.23% | 82.97% |
| D basic-norm address | 22.04% | 690,250 | 0.31 | 3 | 20 | 7.41% | 82.04% |
| E basic-norm name + country | 61.65% | 15,793,889 | 7.16 | 138 | 918 | 15.77% | 7.63% |
| F basic-norm address + country | 22.04% | 690,250 | 0.31 | 3 | 20 | 7.41% | 82.04% |
| B ∪ D | 69.48% | 16,479,837 | 7.47 | 139 | 918 | 22.36% | 10.36% |
| E ∪ F | 69.46% | 16,421,187 | 7.44 | 138 | 918 | 22.36% | 10.40% |
| alnum name *(exploratory)* | 75.13% | 24,486,061 | 11.10 | 191 | 1,071 | 25.79% | 8.04% |
| alnum address *(exploratory)* | 24.15% | 769,813 | 0.35 | 3 | 20 | 8.27% | 82.02% |
| alnum name ∪ alnum addr *(expl.)* | 80.29% | 25,138,139 | 11.39 | 192 | 1,071 | 32.51% | 9.88% |

* Per-source recall for B ∪ D is S2 24.66% and S3 20.21%.
* By country:

  | | India | US |
  |---|---:|---:|
  | S2 | 19.56% | 29.64% |
  | S3 | 16.18% | 24.15% |

* Exact address matches S2 almost never (5 pairs), because S2 is upper-case.
* The heaviest name keys are generic (`eye group` 207×339, `meridian`, `pediatric group`).
* The notebook also has the per-source (S1↔S2, S1↔S3) overlap table.

## 8. Findings

**Data integrity**
* The files are clean. Row counts match raw line counts: S1 2,206,821 · S2 5,034,616 · S3 5,285,603 · GT 2,206,821. Every column is VARCHAR.
* No duplicate or NULL entity IDs, and no ID is shared across sources.
* Only one field is ever missing: `business_address`, in S2 (3.36%) and S3 (3.33%). Names and countries are always present, and S1 is 100% complete.
* The data covers only two countries, **US (~60%) and India (~40%)**, in the same proportions in every source. Country agrees in **100%** of true pairs.

**Ground truth**
* 7,638,365 true pairs, split S2 48.4% / S3 51.6%.
* Every matched ID exists in exactly one target source, and **no target record belongs to two S1 entities**, so the GT defines disjoint clusters.
* Matches per S1 record: mean 3.46, median 3, max 11. 5.6% of S1 records have 0 matches.
* 80.5% of S1 records match both S2 and S3, 6.5% match S2 only and 7.5% match S3 only.
* **~26% of S2/S3 records match no S1 entity** (1.34M in each source). These are distractors.
* Sources also contain internal duplicates: an S1 entity often has 2–5 records in the same target source.

**What makes true matches hard** (100k-pair RapidFuzz sample)
* Exact equality after basic normalisation is rare: name 15.7%, address 7.7% (sample; full data: S2 11.2%, S3 4.5%).
* 12.5% of true pairs have name `token_sort_ratio` < 50.
* 7.2% are **Indic-script vs Latin transliterations** (S2 9.4%, S3 5.1%), covering Devanagari, Gujarati, Telugu, Bengali, Kannada and more.
* 4.4% have the address missing on one side.
* ~1% have both name and address weak (< 50, or missing).
* Name noise:
  * OCR/typo substitutions (`5ervices`, `lmmunopharma`, `Autocmoiteve`)
  * accent injection (`Térm`, `Índia`)
  * leading junk (`***`, `...`, `>>`)
  * doubled spaces; legal-suffix changes, re-ordering and bracketing (`Pvt Ltd` ↔ `Private Limited`, `[LLC]`, `(PLLC)`, `L.L.C.`, `Inc` moved to the front)
  * domain aliases (`zionterm.com`)
  * alias wrappers (`X a/k/a …`, `dba`, `formerly known as`, `t/a`) with a random invented prefix name
  * dropped or added words (`Family Midwest` vs `Family Midwest Associates Inc`)
  * fully random replacement names (`Ciraaria` for `Consolidated Education Systems Inc`)
* Address noise:
  * S2 is UPPER-CASE, uses 2-letter US state codes, zero-pads house numbers (`00123`), injects `##`, and writes Indian state names in native script (`महाराष्ट्र`, `ಕರ್ನಾಟಕ`)
  * S3 spells out US state names (`North Carolina`) but uses 2-letter codes for Indian states (`MH`, `KA`), and injects `null` / `NULL` / `N/A` tokens
  * both sources shuffle comma components, truncate addresses (drop street or locality), abbreviate (`Rd`, `Ave`, `Ln`), use city variants (`MESA CITY`, `… CDP`) and add typos (`COALLTON`)
  * house-number perturbation (`5291` → `291` / `5291b`, `181` → `181B`)
* Addresses carry **no postal codes** in practice. 5- and 6-digit numbers are house numbers.
* Despite this, **100% of sampled true pairs share ≥1 alnum token in name or address**, and 95.6% share an address token. 99.6% have `token_set_ratio ≥ 60` on name or address.

**Hard negatives**
* Names are highly repetitive. 30% of S1 names are exact duplicates of another S1 name, with heavy keys like `eye group` (207 S1 × 339 S2/S3 records). Exact-name blocking therefore produces 15.9M candidates at **7.6% pair precision**.
* Same-name non-matches mostly have clearly different addresses (median addr `token_sort_ratio` 36). The dangerous residue is **same name + near-identical address with a slightly different house number** (`4217` vs `4228 Silverthorne Dr`). Those targets are *unmatched* in the GT, so they are labelled **potential hard negative / needs verification**.
* Same-address non-matches look like **name-perturbed near copies** (`Musgrave Capital LLC` vs `Musgrave Capital Harbor LLC`; `Custom Software Partners Inc` vs `… North Inc`). Most are also unmatched targets.
* However, **unmatched targets are mostly *not* decoys** (§11.1). Only 3.9% (S2) / 4.6% (S3) of unmatched records share a basic-normalised name with any S1 record, versus ~18.5% of matched records. For addresses the figures are 0.34% / 0.00%, versus 11.4% / 4.6%. The ~26% unmatched pool is mainly businesses that are absent from S1, with a small minority of near-copy confounders.

**Blocking**

| strategy (S1 → S2 ∪ S3) | candidates | recall |
|---|---:|---:|
| exact name | 8.4M | 4.6% |
| basic-norm name | 15.9M | 15.8% |
| basic-norm address | 0.69M | 7.4% |
| name ∪ address | 16.5M | 22.4% |
| alnum name ∪ alnum address | 25.1M | 32.5% |

* Every exact or normalised key gets **< 33% recall**.
* Adding **country changes nothing**: there are only 2 countries and they always agree, so it is not a useful blocking key on its own.
* Recall is lower for India (S2 19.6%, S3 16.2%) than for the US (29.6% / 24.2%).
* Address keys are precise (≈ 82% pair precision) but low-recall.

**Runtime and memory**
* Full notebook runs in ≈ 6–7 min on 8 cores (final run 366 s). The blocking evaluation (33 aggregations) takes ≈ 3–4 min.
* Process RSS ≤ 2.9 GB with DuckDB `memory_limit=4GB`.
* The working DuckDB file is ≈ 1.6 GB and is deleted at the end, because the disk is nearly full.
* The first attempts failed with `ENOSPC`. The fixes: a spill cap of `max_temp_directory_size=2GB`, views instead of copies, and sampling IDs before joining.

## 9. Recommendations for Phase 2

These are proposals only. None of them is implemented in Phase 1.

**1. Normalisation: keep several representations and don't collapse to one lossy string.** Keep `raw`, `basic` (lower/trim/whitespace) and the representations below, then use different ones for blocking and for features.
* **Unicode-safe cleanup.**
  * NFKC-normalise.
  * Strip Latin accents **only on Latin characters**. Never apply `strip_accents` or `\p{L}`-only regexes to Indic text, because they delete matras (demo in §4.1).
  * Keep `\p{M}` in token regexes.
* **Name canonicalisation.**
  * Strip leading junk (`***`, `...`, `>>`, `#`) and brackets.
  * Map legal forms to canonical tokens (`pvt`/`private` → `private`, `ltd`/`limited` → `limited`, `l.l.c`/`llc` → `llc`, `inc`/`incorporated` → `inc`, `corp`/`corporation` → `corp`, `co`/`company` → `co`).
  * Keep the legal form as a *separate feature* and also build a `name_core` with it removed. Legal forms move position (`Inc Family …`, `Private Wonderland Energy Ltd`).
* **Alias and domain handling.**
  * Split on `a/k/a | aka | dba | d/b/a | t/a | formerly (known as) | doing business as` and keep the right-hand side as an alternate name.
  * Turn `foo-bar.com` into the tokens of `foobar` and compare it against the concatenated `name_core`. Domains glue the words together (`wenonahsmetalworks.com`).
* **Transliteration of Indic names** (7% of true pairs; 23.5% of S2-India names are in Indic script).
  * Transliterate Indic script to Latin (e.g. `indic-transliteration` / `unidecode`-style ITRANS, or a small lookup built from the GT itself).
  * The GT gives millions of aligned Latin↔Indic name pairs. Common words such as `प्राइवेट लिमिटेड` → `private limited` can be learned directly. Evaluate on a held-out GT split.
* **Address canonicalisation.**
  * Upper-to-lower case.
  * Map full state names ↔ 2-letter codes for both countries, and map native-script Indian state names → English.
  * Standard USPS-style suffixes (`street/st`, `avenue/ave`, `road/rd`, `drive/dr`, `lane/ln`).
  * Remove `null` / `NULL` / `<NULL>` / `N/A` tokens and `##`.
  * Strip leading zeros from numbers.
  * Remove `CDP` / `CITY` suffixes from city names.
  * Treat the address as a **bag of comma components**, because order is shuffled.
* **Structured parts to extract.** Parse `house_number`, `city` and `state` (canonical), and keep the unit / `PMB` / `PO Box` separately. House number and state are cheap, highly discriminative features, and house-number perturbation is exactly what separates the near-duplicate decoys.

**2. Candidate generation: exact keys are not enough (≤ 32.5% recall).** Combine several blockers, with a union target of ≥ 97–99% recall:
* **Rare-token blocking** on `name_core` tokens and address tokens, with document-frequency caps to drop tokens like `group` or `llc`. 100% of sampled true pairs share at least one token.
* **Character n-gram / TF-IDF or MinHash-LSH** on `name_core` and on the address, to survive typos (`Autocmoiteve`, `Allisonville` → `ALLISONVILLE`).
* **Structured keys:** `(state, city, house_number)` and `(state, house_number, street-token)`.
* A **transliterated-name** key for the Indic-script subset.
* **Always partition by country.** It filters nothing within a country, but it halves every join and never loses a true pair.
* Measure recall and candidates per S1 with the §13 evaluator, using a fixed held-out slice of GT S1 IDs.

**3. Pairwise features for the later model.**
* Name similarities (`token_set` / `token_sort` / partial / Jaro-Winkler) on the raw, core and transliterated forms.
* Legal-form agreement.
* Address component overlap.
* House-number equality or numeric distance.
* City and state equality.
* Missing-address indicators.
* Target-source indicator (S2 and S3 noise differs).
* Name frequency / IDF, because common names need address evidence.

**4. Treat the GT as clusters.**
* Each target record belongs to at most one S1 entity. At inference, enforce a one-to-one assignment of target → S1 (keep the best-scoring S1 per target).
* Expect several matches per S1 (median 3; up to 5–6 per source).
* ~26% of targets should be assigned to no S1 at all.

**5. Validation protocol.**
* Split by **S1 entity**, not by pair, so clusters never straddle folds.
* Report per-country and per-source metrics, because India is harder.
* Keep the §11 potential hard negatives (same name + near address; same address + near name) as a stress-test set.

**6. Engineering.**
* Persist `name_n`, `addr_n` and the Phase 2 normalised columns as **Parquet** in `data/processed/`, which is smaller than a DuckDB file.
* Keep `max_temp_directory_size` set, because the disk has ~6 GB free.
* Never materialise full candidate sets as text. Store `(s1_id, target_id)` integer pairs only.
