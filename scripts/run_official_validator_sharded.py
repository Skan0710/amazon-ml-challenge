#!/usr/bin/env python3
"""
Run the UNMODIFIED official ``utils/validate_submission.py`` in bounded memory.

Why: the official validator keeps every candidate ID of ``candidate_pairs.tsv`` in Python sets
(measured 292 MB for 2.0M ids -> ~16-24 GB for the full ~175M-id test candidate file).

How: every S1 id is assigned to shard ``crc32(s1) % N``. For each shard, the rows of
``test_source1.tsv``, ``matching_results.tsv`` and ``candidate_pairs.tsv`` belonging to it are
streamed into a temporary directory and the official validator is run on that shard exactly as
documented (``--matching --candidate --test-dir``). Because a given S1 id always lands in the same
shard in all three files, every official rule is still applied to every row:
  * header / TSV / UTF-8 / malformed rows           (header copied to every shard)
  * duplicate S1 rows                               (duplicates share a shard)
  * missing required S1 / S1 not in the test set    (each shard has its own required set)
  * duplicate / self / non-S2-S3 ids in a list      (row-local)
  * matched-not-in-candidates warning               (both rows share a shard)
Shard files are deleted after each shard. Exit 0 only if every shard passes.

    python3 scripts/run_official_validator_sharded.py --official ~/student_resource/utils/validate_submission.py \
        --matching output/matching_results.tsv.tmp --candidate output/candidate_pairs.tsv.tmp \
        --test-dir ~/student_resource/dataset/test --work-dir data/processed/phase4/validator_shards
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

TARGET_SHARD_BYTES = 60_000_000      # ~60 MB of candidate file per shard -> ~0.6-0.7 GB official RAM


def shard_of(line: str, n: int) -> int:
    return zlib.crc32(line.split("\t", 1)[0].strip().encode("utf-8")) % n


def write_shard(src: Path, dst: Path, k: int, n: int) -> int:
    """Copy the header and every data line whose S1 falls in shard k. Byte-exact line copies."""
    rows = 0
    with open(src, encoding="utf-8", newline="") as fi, open(dst, "w", encoding="utf-8", newline="") as fo:
        header = fi.readline()
        fo.write(header)
        for line in fi:
            if line.strip() == "":
                continue
            if shard_of(line, n) == k:
                fo.write(line)
                rows += 1
    return rows


def run(official: Path, matching: Path, candidate: Path, test_dir: Path, work_dir: Path, n_shards: int = 0,
        python: str = "python3", log=print) -> dict:
    size = candidate.stat().st_size if candidate and candidate.exists() else matching.stat().st_size
    n = n_shards or max(1, math.ceil(size / TARGET_SHARD_BYTES))
    work_dir.mkdir(parents=True, exist_ok=True)
    results, ok = [], True
    for k in range(n):
        d = work_dir / f"shard_{k:04d}"
        (d / "test").mkdir(parents=True, exist_ok=True)
        s1_rows = write_shard(test_dir / "test_source1.tsv", d / "test" / "test_source1.tsv", k, n)
        m_rows = write_shard(matching, d / "matching_results.tsv", k, n)
        cmd = [python, str(official), "--matching", str(d / "matching_results.tsv"), "--test-dir", str(d / "test")]
        c_rows = None
        if candidate:
            c_rows = write_shard(candidate, d / "candidate_pairs.tsv", k, n)
            cmd += ["--candidate", str(d / "candidate_pairs.tsv")]
        else:  # never let the official default (output/candidate_pairs.tsv relative to cwd) pick up another file
            cmd += ["--candidate", str(d / "no_candidate_file.tsv")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        passed = r.returncode == 0
        ok &= passed
        warnings = [l for l in r.stdout.splitlines() if l.startswith("WARNING") and "ID-existence check is OFF" not in l]
        results.append({"shard": k, "s1_rows": s1_rows, "matching_rows": m_rows, "candidate_rows": c_rows,
                        "exit_code": r.returncode, "extra_warnings": warnings,
                        "output": r.stdout if not passed or warnings else ""})
        if not passed or warnings:
            log(r.stdout)
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    totals = {k: sum((x[k] or 0) for x in results) for k in ("s1_rows", "matching_rows", "candidate_rows")}
    any_warn = any(x["extra_warnings"] for x in results)
    log(f"official validator (sharded x{n}): {'PASS' if ok else 'FAIL'} | totals {totals}"
        + (" | WARNINGS present" if any_warn else ""))
    return {"passed": ok, "shards": n, "totals": totals, "warnings_present": any_warn,
            "failed_shards": [x for x in results if x["exit_code"] != 0]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--official", type=Path, required=True)
    ap.add_argument("--matching", type=Path, required=True)
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=0, help="0 = automatic (~60 MB of candidate file per shard)")
    a = ap.parse_args(argv)
    res = run(a.official, a.matching, a.candidate, a.test_dir, a.work_dir, a.shards)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
