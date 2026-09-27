#!/usr/bin/env python3
"""
Independent, read-only audit of the final submission files (bounded memory, DuckDB streaming).

    .venv/bin/python scripts/audit_submission.py --out output --test-dir ~/student_resource/dataset/test \
        [--pred-parts data/processed/phase4/test_run/pred_parts --threshold 0.8] [--hashes logs/test_sha256.txt]

Checks the published TSVs themselves (not the pipeline's internal tables):
  * exact headers, trailing newline, line count == rows + 1 (no truncation)
  * every test S1 exactly once (missing / unexpected / duplicate S1 rows) in BOTH files
  * no duplicate id inside any list; every list id is S2-/S3- and exists in the test S2/S3 files
  * every matched id is a candidate of the same S1 (matches subset of candidates)
  * candidate pairs unique; totals; empty / non-empty rows; per-country breakdown
  * no forced matches: every prediction part has p >= threshold
  * optional: SHA-256 of the test files equals the recorded values
Exit 0 only if every hard check passes.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--pred-parts", type=Path)
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--hashes", type=Path, help="file with 'sha256  path' lines recorded before the run")
    ap.add_argument("--temp-dir", type=Path, default=Path("data/processed/phase4/audit_tmp"))
    a = ap.parse_args()
    m_path, c_path = a.out / "matching_results.tsv", a.out / "candidate_pairs.tsv"
    R, hard = {}, []

    for name, path, header in (("matching", m_path, "source1_entity_id\tmatched_entity_ids"),
                               ("candidate", c_path, "source1_entity_id\tcandidate_entity_ids")):
        with open(path, "rb") as f:
            first = f.readline().decode("utf-8").rstrip("\n")
            f.seek(-1, 2)
            last_byte = f.read(1)
        lines = sum(1 for _ in open(path, "rb"))
        R[f"{name}_header_ok"] = first == header
        R[f"{name}_ends_with_newline"] = last_byte == b"\n"
        R[f"{name}_lines"] = lines
        R[f"{name}_bytes"] = path.stat().st_size
        if first != header:
            hard.append(f"{name}: bad header {first!r}")
        if last_byte != b"\n":
            hard.append(f"{name}: no trailing newline (truncated?)")

    a.temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='1500MB'"); con.execute("SET threads=2"); con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{a.temp_dir}'"); con.execute("SET max_temp_directory_size='3GB'")
    tsv = lambda p: f"read_csv('{p}', delim='\t', header=true, all_varchar=true, quote='', escape='', columns={{'s1': 'VARCHAR', 'ids': 'VARCHAR'}})"
    src = lambda i: f"read_csv('{a.test_dir / f'test_source{i}.tsv'}', delim='\t', header=true, all_varchar=true, quote='', escape='')"
    con.execute(f"CREATE TABLE req AS SELECT entity_id s1, country FROM {src(1)}")
    con.execute(f"CREATE TABLE m AS SELECT s1, coalesce(ids, '') ids FROM {tsv(m_path)}")
    R["expected_s1"] = con.execute("SELECT COUNT(*) FROM req").fetchone()[0]

    for name, rel in (("matching", "m"), ("candidate", f"(SELECT s1, coalesce(ids, '') ids FROM {tsv(c_path)})")):
        r = con.execute(f"""SELECT COUNT(*), COUNT(DISTINCT s1),
              (SELECT COUNT(*) FROM req ANTI JOIN {rel} x ON x.s1 = req.s1),
              (SELECT COUNT(*) FROM {rel} x ANTI JOIN req ON req.s1 = x.s1),
              COUNT(*) FILTER (WHERE ids = ''), COUNT(*) FILTER (WHERE ids <> ''),
              SUM(CASE WHEN ids = '' THEN 0 ELSE len(string_split(ids, ',')) END),
              SUM(CASE WHEN ids = '' THEN 0 ELSE len(string_split(ids, ',')) - len(list_distinct(string_split(ids, ','))) END),
              COUNT(*) FILTER (WHERE ids <> '' AND NOT regexp_full_match(ids, '(S[23]-[0-9]+)(,S[23]-[0-9]+)*')),
              MAX(CASE WHEN ids = '' THEN 0 ELSE len(string_split(ids, ',')) END)
            FROM {rel}""").fetchone()
        rows, uniq, missing, unexpected, empty, nonempty, ids, dup_ids, malformed, max_len = r
        R[name] = {"rows": rows, "unique_s1": uniq, "duplicate_s1_rows": rows - uniq, "missing_s1": missing,
                   "unexpected_s1": unexpected, "empty_lists": empty, "non_empty_lists": nonempty, "total_ids": int(ids or 0),
                   "duplicate_ids_within_lists": int(dup_ids or 0), "malformed_lists": malformed, "max_list_len": max_len}
        for k in ("duplicate_s1_rows", "missing_s1", "unexpected_s1", "duplicate_ids_within_lists", "malformed_lists"):
            if R[name][k]:
                hard.append(f"{name}: {k} = {R[name][k]}")
        if R[f"{name}_lines"] != rows + 1:
            hard.append(f"{name}: {R[f'{name}_lines']} lines for {rows} rows (+1 header expected)")
    R["candidate"]["unique_pairs"] = R["candidate"]["total_ids"] - R["candidate"]["duplicate_ids_within_lists"]

    # matched ids: exist in S2/S3, and are candidates of the same S1
    con.execute("CREATE TABLE mp AS SELECT s1, unnest(string_split(ids, ',')) t FROM m WHERE ids <> ''")
    con.execute(f"CREATE TABLE tgt AS SELECT entity_id t FROM {src(2)} UNION ALL SELECT entity_id FROM {src(3)}")
    R["matched_ids_not_in_test_s2_s3"] = con.execute("SELECT COUNT(*) FROM mp ANTI JOIN tgt USING (t)").fetchone()[0]
    found = con.execute(f"""SELECT COUNT(*) FROM (
            SELECT c.s1, unnest(string_split(c.ids, ',')) t FROM {tsv(c_path)} c SEMI JOIN (SELECT DISTINCT s1 FROM mp) x ON x.s1 = c.s1 WHERE c.ids <> '') cp
          SEMI JOIN mp ON mp.s1 = cp.s1 AND mp.t = cp.t""").fetchone()[0]
    R["matched_pairs"] = con.execute("SELECT COUNT(*) FROM mp").fetchone()[0]
    R["matched_pairs_not_in_candidates"] = R["matched_pairs"] - found
    for k in ("matched_ids_not_in_test_s2_s3", "matched_pairs_not_in_candidates"):
        if R[k]:
            hard.append(f"{k} = {R[k]}")
    R["per_country"] = con.execute("""SELECT req.country, COUNT(*) s1, COUNT(*) FILTER (WHERE m.ids = '') empty,
            COUNT(*) FILTER (WHERE m.ids <> '') non_empty,
            ROUND(AVG(CASE WHEN m.ids = '' THEN 0 ELSE len(string_split(m.ids, ',')) END), 3) avg_matches
        FROM req JOIN m USING (s1) GROUP BY 1 ORDER BY 1""").df().to_dict(orient="records")
    R["match_list_length_distribution"] = dict(con.execute("""SELECT CASE WHEN ids = '' THEN 0 ELSE len(string_split(ids, ',')) END k, COUNT(*)
        FROM m GROUP BY 1 ORDER BY 1""").fetchall())

    if a.pred_parts:
        n, mn = con.execute(f"SELECT COUNT(*), MIN(p) FROM read_parquet('{a.pred_parts}/pred-*.parquet')").fetchone()
        R["prediction_parts"] = {"pairs": n, "min_probability": mn, "threshold": a.threshold}
        if n != R["matched_pairs"]:
            hard.append(f"prediction parts ({n}) != matched pairs in file ({R['matched_pairs']})")
        if mn is not None and mn < a.threshold:
            hard.append(f"forced/below-threshold prediction: min p {mn} < {a.threshold}")
    if a.hashes and a.hashes.exists():
        rec = dict(reversed(l.split()) for l in a.hashes.read_text().splitlines() if l.strip())
        now = {str(a.test_dir / f"test_source{i}.tsv"): sha256(a.test_dir / f"test_source{i}.tsv") for i in (1, 2, 3)}
        R["test_files_unchanged"] = all(rec.get(k) == v for k, v in now.items())
        if not R["test_files_unchanged"]:
            hard.append("test file hash mismatch")
    con.close()
    R["hard_failures"] = hard
    print(json.dumps(R, indent=1, default=str))
    print("AUDIT PASS" if not hard else f"AUDIT FAIL: {hard}")
    return 0 if not hard else 1


if __name__ == "__main__":
    sys.exit(main())
