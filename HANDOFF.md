# Project Handoff

## Current Status
- Phase 1: COMPLETE (Exploratory Data Analysis, distributions, summary reports, and notebook)
- Phase 2: COMPLETE (Text normalization, core name extraction, address parsing, phonetic encoding, 48 unit tests passing)
- Phase 3: COMPLETE (High-recall candidate generation / blocking strategies, streaming Parquet materialization, DuckDB index tables, 20 unit tests passing)
- Phase 4: INCOMPLETE (Feature engineering, HistGradientBoosting model, and pilot validation complete; full-dataset inference, test set submission generation, and target exclusivity remain unfinished)

## What Is Currently Working
- **Phase 1 (EDA):** `scripts/run_eda.py` and `notebooks/01_eda.ipynb` analyze all training datasets (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`). Outputs summary statistics and reports in `experiments/eda_report.md` and `experiments/eda_summary.json`.
- **Phase 2 (Normalization):** `src/normalization.py` provides clean text normalization, legal entity stripping, core name extraction, address decomposition (house number, street, place, state, postal code), phonetic encodings (Double Metaphone / Soundex), and Indic transliteration. All 48 unit tests pass (`tests/test_normalization.py`).
- **Phase 3 (Candidate Generation):** `src/candidate_generation.py` provides 7 blocking strategies (`exact_name`, `name_state`, `structured`, `phonetic`, `rare_name`, `rare_addr`, `compound`), frequency caps to prevent block explosion, Parquet materialization, and evaluation against ground truth. Reaches ~97.5% candidate recall ceiling at ~101 candidates per S1. All 20 unit tests pass (`tests/test_candidate_generation.py`).
- **Phase 4 Components (Pilots & Pipeline):**
  - `src/matching.py`: Full pairwise feature generation (47 features covering name, address, and graph structure), negative sampling (hard vs. easy negatives), HistGradientBoosting classifier training, `DecisionRule` evaluation, streaming chunked inference, and submission TSV writers.
  - `models/phase4_rule.json`: Validated decision rule configuration (`threshold=0.80`).
  - `tests/test_matching.py`: 19 unit tests passing for all Phase 4 feature extraction, modeling, and output formatting logic.
  - `tests/test_production_safety.py`: 18 safety tests passing (verifying memory bounds, leak prevention, DuckDB connection cleanup, SQL injection safety, TSV format compliance, and strict ID validation).
  - Pilot runs on 10k and 100k S1 entities documented in `experiments/phase4_matching_report.md`, achieving macro F0.5 = 0.9476 on held-out validation data.
  - Submission validation scripts: `scripts/validate_final_submission.py` and `scripts/run_official_validator_sharded.py`.

## Phase 4 Status
Phase 4 is **INCOMPLETE**.

### What has been completed:
1. Feature extraction pipeline implemented in DuckDB SQL (47 pairwise features across name similarity, address components, and candidate graph structure).
2. Model training pipeline implemented using `sklearn.ensemble.HistGradientBoostingClassifier` with native handling of missing values and sample weighting.
3. Decision rule tuning completed on 100k pilot validation split, selecting an optimal threshold of `0.80` (favouring high precision as dictated by the F0.5 metric).
4. Streaming inference pipeline designed to process S1 chunks under a strict 2 GB RAM budget.
5. TSV writers for `candidate_pairs.tsv` and `matching_results.tsv` with strict schema validation against official competition requirements.
6. Validation against the official competition validator (`validate_submission.py`) passing cleanly on pilot validation outputs.

### What remains unfinished:
1. **Full Test Inference Not Run:** The pipeline has only been executed on pilot subsets (10k and 100k S1 entities). It has **NOT** been run on the full test set (`test_source1.tsv` with ~1.73M S1 records and an estimated ~175M candidate pairs).
2. **Final Submission Files Not Generated:** `candidate_pairs.tsv` and `matching_results.tsv` have not been generated for the competition test set.
3. **Target Exclusivity Unimplemented:** In the ground truth, each target entity (S2/S3) is linked to at most one S1 entity. A global assignment / 1-to-1 assignment pass (e.g., greedy bipartite matching or score ranking) has not been implemented or evaluated at scale.
4. **French / Unseen Country Generalization Unmeasured:** Structured blockers do not have French postal/state gazetteers, and French performance has not been tested on labelled data.
5. **Disk Space Management:** Full inference produces an estimated ~2.3 GB candidate file and requires ~4.7 GB total working disk space. The host environment needs at least 8 GB free disk before launching the full run.

## Files Changed
- `.gitignore`: Updated with ignore rules for Phase 3 and Phase 4 intermediate Parquet chunks, DuckDB temp spill files, pilot outputs, and large serialized models (`models/*.joblib`).
- `src/candidate_generation.py`: Updated candidate generation module featuring streaming Parquet materialization, chunked candidate generation, and frequency-capped blocking strategies.
- `src/matching.py`: Core Phase 4 matching engine (feature engineering, training, inference, decision rules, TSV formatting).
- `models/phase4_rule.json`: Production decision rule parameters (`threshold: 0.80`).
- `scripts/run_phase4.py`: Execution script for Phase 4 training, pilot validation, and full pipeline inference.
- `scripts/validate_final_submission.py`: Standalone submission verification script confirming format, IDs, schema, and subsets.
- `scripts/run_official_validator_sharded.py`: Memory-efficient sharded runner for the official submission validator.
- `scripts/run_eda.py`: Standalone Phase 1 exploratory data analysis runner.
- `experiments/phase3_candidate_generation_report.md`: Detailed report on Phase 3 blocking performance and recall ceilings.
- `experiments/phase3_candidate_generation_results.json`: Raw metrics from Phase 3 candidate generation runs.
- `experiments/phase4_matching_report.md`: Detailed report on Phase 4 feature importance, pilot benchmarks, threshold search, and runtime estimates.
- `notebooks/03_candidate_generation_validation.ipynb`: Interactive validation notebook for Phase 3 candidate generation.
- `tests/test_candidate_generation.py`: Test suite for candidate generation, blocking keys, caps, and ground-truth evaluation (20 tests).
- `tests/test_matching.py`: Test suite for Phase 4 feature computation, classifier training, and prediction formatting (19 tests).
- `tests/test_production_safety.py`: Test suite verifying memory limits, leak prevention, DuckDB resource cleanup, and injection guards (18 tests).
- `HANDOFF.md`: This project handoff documentation.

## How To Run

### Environment Setup
Activate the project's virtual environment:
```bash
source .venv/bin/activate
```
Verify dependencies:
```bash
python scripts/check_environment.py
```

### Running Tests
Run all unit and safety tests (105 tests across normalization, candidate generation, matching, and safety):
```bash
python -m unittest discover tests
```

### Running Phase 4 Pilot
To run the 10k pilot to verify the end-to-end pipeline:
```bash
python scripts/run_phase4.py --scale 10k --db data/processed/phase4_10k.duckdb
```

To run the 100k pilot:
```bash
python scripts/run_phase4.py --scale 100k --db data/processed/phase4_100k.duckdb
```

### Validating Pilot Submission Outputs
```bash
python scripts/validate_final_submission.py \
  --test-s1 data/raw/train_source1.tsv \
  --candidates output/pilots/100k/candidate_pairs.tsv \
  --matches output/pilots/100k/matching_results.tsv
```

## Environment Variables
- `OFFICIAL_VALIDATOR`: (Optional) Path to the official `validate_submission.py` script provided by the competition organizers (defaults to `~/student_resource/utils/validate_submission.py`).
- `DUCKDB_TEMP_DIRECTORY`: (Optional) Directory for DuckDB disk spillover if working on a drive with limited space.

*Note: No external API keys or secrets are required. All computation runs locally.*

## Known Issues
1. **Full-scale Test Run Untested:** Running full inference on 1.73M test records requires approximately ~4.5 hours of compute and ~5–8 GB of free disk space.
2. **Memory Overhead during DuckDB Parquet Exports:** When exporting multi-million row candidate tables, DuckDB thread limits must be bounded (`SET threads = 4`) to prevent memory spikes beyond 2 GB.
3. **Target Multi-Matching:** The decision rule evaluates pairs independently. Because an S2 or S3 entity can only represent one physical business, multiple S1 queries claiming the same target record creates false positives. Implementing a bipartite resolution pass would improve macro F0.5.
4. **France Address Parsing:** France lacks detailed state gazetteer mappings in `src/normalization.py`, meaning French address records rely primarily on raw token and name blocking rather than structured address hierarchy.

## Next Steps For Member 2
1. **Verify Environment and Test Suite:**
   Run `python -m unittest discover tests` to ensure all 105 tests pass in your local environment.
2. **Review Reports:**
   Read `experiments/phase3_candidate_generation_report.md` and `experiments/phase4_matching_report.md` to understand the feature set, blocking strategies, and threshold choices.
3. **Verify Disk Space:**
   Ensure your system has at least 10 GB of free disk space before attempting full dataset materialization or inference.
4. **Implement Target Exclusivity (Optional Improvement):**
   In `src/matching.py`, consider adding a post-processing step for predictions where if multiple S1 candidates predict the same target `t_id`, only the S1 candidate with the highest predicted probability retains the match (or use 1-to-1 Hungarian/greedy matching).
5. **Run Full Test Inference:**
   Prepare the test input files in `data/raw/` (`test_source1.tsv`, etc.) and execute the full test inference pipeline via `scripts/run_phase4.py` with test-mode flags.
6. **Validate Final Submission Files:**
   Run `scripts/validate_final_submission.py` and the official validator script on the generated `candidate_pairs.tsv` and `matching_results.tsv` before final submission.
