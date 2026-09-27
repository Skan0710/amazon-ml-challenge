#!/usr/bin/env python3
"""
Build the TRAINING-side Phase 3 artefacts that model training (``run_phase4.py pilot``) needs:

* ``data/processed/phase3/features_s{1,2,3}/``  — streaming Phase 2 normalisation of the training TSVs
* ``data/processed/phase3/blocking_stats.duckdb`` — target-side key frequencies (no ground truth)

    python scripts/build_train_artifacts.py            # ~8 min normalisation + ~1-2 min statistics

Idempotent: finished sources are skipped (``_SUCCESS`` markers); statistics are rebuilt.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.candidate_generation as C  # noqa: E402

RAW = ROOT / "data" / "raw"
P3 = ROOT / "data" / "processed" / "phase3"


def main():
    t0 = time.time()
    P3.mkdir(parents=True, exist_ok=True)
    for i in (1, 2, 3):
        C.materialize_blocking_features(RAW / f"train_source{i}.tsv", P3 / f"features_s{i}", workers=4)
    con = C.connect(P3 / "blocking_stats.duckdb", memory_limit="1500MB", threads=4, temp_dir=P3 / "duckdb_tmp", max_temp="2GB")
    C.register_features(con, P3)
    C.build_blocking_statistics(con, C.BlockingConfig(),
                                kinds=["name", "phonetic", "structured", "rare_name", "rare_addr", "ngram"])
    con.close()
    print(f"training artefacts ready in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
