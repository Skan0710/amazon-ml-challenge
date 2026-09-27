# Business Entity Resolution — reproduction guide

End-to-end pipeline: **normalisation → blocking / candidate generation → pairwise features →
gradient-boosted matcher → per-entity decision rule → `candidate_pairs.tsv` + `matching_results.tsv`**.
All heavy steps run in DuckDB over Parquet in bounded memory (tested on an 8 GB laptop).

## Layout

| path | purpose |
|---|---|
| `src/normalization.py` | Phase 2 — script-aware name/address normalisation, Indic transliteration, phonetic keys, state tables (stdlib only) |
| `src/candidate_generation.py` | Phase 3 — streaming feature materialisation, blocking statistics, 7 capped blocking strategies, recall evaluation |
| `src/matching.py` | Phase 4 — 47 pairwise features (DuckDB SQL), model training/scoring, decision rule, macro-F0.5 evaluation, chunked TSV writers, atomic publication |
| `src/build_train_artifacts.py` | builds the training-side Phase 3 artefacts |
| `src/run_phase4.py` | `pilot` = train + choose threshold on a held-out split (training data only); `test` = final inference |
| `src/preflight_phase4.py` | environment checks + small real-test dry run in an isolated directory |
| `src/validate_final_submission.py` | strict, streaming (memory-safe) validator for both output files |
| `src/run_official_validator_sharded.py` | runs the **unmodified** official `utils/validate_submission.py` in bounded memory |
| `src/audit_submission.py` | independent read-only audit of the two output files (coverage, duplicates, consistency, truncation, hashes) |
| `models/phase4_model.joblib`, `models/phase4_rule.json` | the trained model and decision rule used for the submission |

## Environment

Python 3.13 (tested with 3.13.9), macOS arm64, 8 CPUs, 8.6 GB RAM.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Data placement

```
data/raw/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
<TEST_DIR>/test_source1.tsv  test_source2.tsv  test_source3.tsv      (official student_resource/dataset/test)
<OFFICIAL_VALIDATOR> = student_resource/utils/validate_submission.py
```

## Reproduce

```bash
# 1. training-side candidate artefacts (~10 min, ~1.3 GB)
python src/build_train_artifacts.py

# 2. train the matcher and choose the decision rule on a held-out S1 split (training data only, ~17 min)
python src/run_phase4.py pilot --name 100k --mod 22 --chunks 10 --feature-chunks 10 --hard-rate 0.1
#    -> data/processed/phase4/pilot_100k/{model.joblib, rule.json}; copy to models/ to use them
#       (models/ already contains the exact model and rule used for the submission)

# 3. pre-flight: checks + 1,000-entity dry run on the real test files (isolated directory)
TEST_DIR=<TEST_DIR> OFFICIAL_VALIDATOR=<OFFICIAL_VALIDATOR> python src/preflight_phase4.py --n-s1 1000

# 4. final inference on the full test set (resumable; ~2 GB RAM; needs >= 8 GB free disk)
OFFICIAL_VALIDATOR=<OFFICIAL_VALIDATOR> caffeinate -i python src/run_phase4.py test \
    --dataset-dir <TEST_DIR> --model models/phase4_model.joblib --rule models/phase4_rule.json \
    --chunks 220 --infer-chunks 440 --infer-threads 2 --resumable-inference \
    --min-free-gb 3.0 --min-free-write-gb 4.0 --i-approve-full-scale
#    (after an interruption: add --resume-candidates; finished scoring chunks are skipped automatically)
#    -> output/candidate_pairs.tsv, output/matching_results.tsv, output/phase4_metrics.json
```

Step 4 writes `*.tsv.tmp` first, runs the streaming validator and the sharded **official** validator on
them, and only then renames them atomically to the final names — a crash or a failed check can never
leave a partial file under a final name.

## Official validation of the outputs

```bash
cd student_resource
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

On a machine with less than ~24 GB RAM, validate the (≈2.2 GB) candidate file with the bounded-memory
wrapper instead (same official script, sharded by S1):

```bash
python src/run_official_validator_sharded.py --official <OFFICIAL_VALIDATOR> \
    --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir <TEST_DIR> --work-dir /tmp/validator_shards
```
