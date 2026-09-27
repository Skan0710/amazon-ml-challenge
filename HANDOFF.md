# Project Handoff

## Current status

All four phases are complete. The full official test set has been processed and the two submission files are generated and validated. **The final submission package is now being prepared.**

| Phase | Status | Where |
|---|---|---|
| 1 — EDA | complete | `notebooks/01_eda.ipynb`, `experiments/eda_report.md`, `experiments/eda_summary.json` |
| 2 — Normalisation | complete | `src/normalization.py`, `tests/test_normalization.py`, `experiments/phase2_normalization_report.md` |
| 3 — Candidate generation | complete | `src/candidate_generation.py`, `tests/test_candidate_generation.py`, `experiments/phase3_candidate_generation_report.md` |
| 4 — Matching + full test inference | complete | `src/matching.py`, `scripts/run_phase4.py`, `tests/test_matching.py`, `tests/test_production_safety.py`, `experiments/phase4_matching_report.md` |

**Automated tests:** 110/110 pass, split as normalisation 49 (including the module doctests), candidate generation 20, matching 19 and production safety 22.

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

## Final test-set results (measured; see `experiments/phase4_matching_report.md` §10)

| | value |
|---|---:|
| test Source-1 entities | 1,732,544 |
| test S2 / S3 targets | 4,887,273 / 5,082,316 |
| candidate pairs (unique) | 171,756,016 (99.1 per S1; 9,037 S1 have no candidate) |
| predicted matched pairs (threshold 0.80) | 5,403,410 |
| S1 with an empty match list / non-empty | 117,592 / 1,614,952 |
| per country: S1 / mean matches / empty | France 259,452 / 2.67 / 27,280 · India 809,986 / 3.13 / 52,430 · US 663,106 / 3.29 / 37,882 |
| held-out validation (training data, 19,957 S1) | macro F0.5 0.9476 · pair precision 0.978 · pair recall 0.902 |

`output/candidate_pairs.tsv` and `output/matching_results.tsv` were generated successfully and have passed every check:

- Every Source-1 entity appears exactly once in both files, with 0 missing and 0 unexpected entities.
- There are no duplicate Source-1 rows.
- No matching list contains a duplicate target ID.
- There are no duplicate candidate pairs.
- There are no forced matches: every match is a candidate of its S1, with probability ≥ 0.80 (the lowest is 0.8000001).
- Every matched ID exists in the test S2/S3 files.
- Headers are exact, each file ends with a newline, and line count equals rows + 1.
- **Official validation passed.** The official `utils/validate_submission.py` passed:
  - sharded over both full files (38 shards; the unmodified script, via `scripts/run_official_validator_sharded.py`);
  - on the full `matching_results.tsv` with `--check-ids`, run from `student_resource`.
- The test input files' SHA-256 hashes are unchanged.

### Frozen output checksums (SHA-256; recorded in `logs/final_outputs_sha256.txt`, which is not in Git)

```
b37ddd8de5dd0d8224d522ed0a38248b004ec8f962c0ecce5ef80f0d0882580c  output/candidate_pairs.tsv   (2,233,815,424 bytes)
538fd5d07c208d0f3fea1da34eddd632f6c6995790e1681ded7f5c6234ffcde8  output/matching_results.tsv  (92,092,814 bytes)
```

The output files, datasets, Parquet/DuckDB working files and the model binary are git-ignored on purpose. **Do not regenerate or edit the output files.** Verify them with:

```bash
shasum -a 256 -c logs/final_outputs_sha256.txt
.venv/bin/python scripts/audit_submission.py --out output --test-dir ~/student_resource/dataset/test
```

## How the pipeline is run

These are the exact commands; the code README in `packaging/CODE_README.md` has more detail.

```bash
python scripts/build_train_artifacts.py                     # training-side Phase 3 artefacts
python scripts/run_phase4.py pilot --name 100k --mod 22 --chunks 10 --feature-chunks 10 --hard-rate 0.1   # train + choose rule
python scripts/preflight_phase4.py --n-s1 1000              # checks + 1,000-entity real-test dry run
python scripts/run_phase4.py test --dataset-dir <TEST_DIR> --model models/phase4_model.joblib \
    --rule models/phase4_rule.json --chunks 220 --infer-chunks 440 --infer-threads 2 \
    --resumable-inference --min-free-gb 3.0 --min-free-write-gb 4.0 --i-approve-full-scale
```

- The model (`models/phase4_model.joblib`) is a scikit-learn HistGradientBoostingClassifier with 47 features. It is not in Git; it is included in the submission package.
- The rule (`models/phase4_rule.json`) is `threshold = 0.80`.
- The full test run is resumable. After an interruption, add `--resume-candidates`; finished scoring chunks are skipped.
- Outputs are written as `*.tsv.tmp` and renamed only after the streaming validator and the sharded official validator both pass.
- The run needs about 2 GB RAM, about 5 GB of working disk (≥ 8 GB free recommended) and about 6 h of compute on an 8 GB laptop.

## Submission package

`scripts/build_submission_package.py --team <TEAM> --zip` assembles the official layout under `submission/`, which is git-ignored:

```
output/{matching_results.tsv, candidate_pairs.tsv}
code/business_entity_resolution/{src/, models/, README.md, requirements.txt}
Documentation_template.md            (filled methodology, from packaging/)
```

## Known limitations

- **France:** France is absent from the training labels, so its accuracy is unmeasured. French records have no state tables, so the `structured` and `compound` blockers do not fire for them.
- **Long match lists:** 264 test S1 records have more than 11 predicted matches (at most 28), while the training maximum was 11. These are likely generic or chain names.
- **Target exclusivity:** in training each target belongs to at most one S1, but the decision rule is applied per pair. A global one-to-one assignment pass was not implemented.
