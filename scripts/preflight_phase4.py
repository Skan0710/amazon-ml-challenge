#!/usr/bin/env python3
"""
Phase 4 pre-flight: environment checks + a small end-to-end dry run on the REAL test data.

    .venv/bin/python scripts/preflight_phase4.py --n-s1 1000

Checks: Python/packages, Phase 2/3/4 imports, model + rule load and match the feature set,
training/test files readable, output dir writable, free disk, available RAM, official validator.

Dry run: a hash sample of ``--n-s1`` official test S1 records against the FULL official test
S2/S3 files (read through symlinks, never copied or modified), in an isolated directory
``data/processed/phase4/dryrun/`` — never ``output/``. Uses the production code path
(``run_phase4.run_inference``): tmp outputs -> streaming validator + sharded official validator ->
atomic rename. Records peak RAM, peak new disk, runtime and per-country descriptive statistics.
The dry run's normalised targets are kept (``--keep``) so reruns are quick.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST = Path(os.environ.get("TEST_DIR", Path.home() / "student_resource" / "dataset" / "test"))
OFFICIAL = Path(os.environ.get("OFFICIAL_VALIDATOR", Path.home() / "student_resource" / "utils" / "validate_submission.py"))
D = ROOT / "data" / "processed" / "phase4" / "dryrun"


def dsize(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def checks() -> dict:
    import psutil
    out, ok = {}, True
    out["python"] = sys.version.split()[0]
    for m in ("duckdb", "numpy", "pandas", "sklearn", "joblib", "psutil"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception as e:  # noqa: BLE001
            out[m] = f"MISSING ({e})"
            ok = False
    for m in ("src.normalization", "src.candidate_generation", "src.matching"):
        try:
            importlib.import_module(m)
            out[m] = "import ok"
        except Exception as e:  # noqa: BLE001
            out[m] = f"IMPORT FAILED ({e})"
            ok = False
    import joblib
    import src.matching as M
    model = joblib.load(ROOT / "models" / "phase4_model.joblib")
    rule = json.loads((ROOT / "models" / "phase4_rule.json").read_text())
    out["model"] = f"{type(model).__name__}, n_features_in_={model.n_features_in_}, n_iter_={model.n_iter_}"
    out["model_matches_features"] = model.n_features_in_ == len(M.FEATURE_COLUMNS)
    out["rule"] = rule
    ok &= out["model_matches_features"]
    for name, p in [("train_source1", ROOT / "data/raw/train_source1.tsv"), ("test_source1", TEST / "test_source1.tsv"),
                    ("test_source2", TEST / "test_source2.tsv"), ("test_source3", TEST / "test_source3.tsv"),
                    ("official_validator", OFFICIAL)]:
        out[name] = f"{p} ({p.stat().st_size / 1e6:.0f} MB)" if p.exists() and os.access(p, os.R_OK) else f"MISSING {p}"
        ok &= p.exists()
    (ROOT / "output").mkdir(exist_ok=True)
    probe = ROOT / "output" / ".write_probe"
    probe.write_text("x"); probe.unlink()
    out["output_writable"] = True
    out["free_disk_gb"] = round(shutil.disk_usage(ROOT).free / 1e9, 1)
    out["available_ram_gb"] = round(psutil.virtual_memory().available / 1e9, 1)
    out["cpus"] = os.cpu_count()
    out["all_ok"] = ok
    return out


def dry_run(n_s1: int, keep: bool) -> dict:
    import duckdb
    import joblib
    import psutil
    import src.matching as M
    spec = importlib.util.spec_from_file_location("run_phase4", Path(__file__).resolve().parent / "run_phase4.py")
    R4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R4)
    ds = D / "dataset"
    marker = ds / f".n_s1_{n_s1}"
    if not marker.exists():
        shutil.rmtree(D, ignore_errors=True)
        ds.mkdir(parents=True)
        con = duckdb.connect()
        con.execute(f"""COPY (SELECT * FROM read_csv_auto('{TEST / 'test_source1.tsv'}', delim='\t', header=true, all_varchar=true)
                     WHERE hash(entity_id) % {max(1, 1_732_544 // n_s1)} = 0 ORDER BY entity_id LIMIT {n_s1})
                     TO '{ds / 'test_source1.tsv'}' (DELIMITER '\t', HEADER, QUOTE '')""")
        con.close()
        for i in (2, 3):
            os.symlink(TEST / f"test_source{i}.tsv", ds / f"test_source{i}.tsv")
        marker.write_text("")
    work = D / "work"
    for f in ("work.duckdb", "work.duckdb.wal"):
        (work / f).unlink(missing_ok=True)
    for d in ("duckdb_tmp", "candidates", "validator_shards"):
        shutil.rmtree(work / d, ignore_errors=True)
    shutil.rmtree(D / "output", ignore_errors=True)
    base = dsize(work)
    peak = {"rss": 0, "disk": 0, "min_free": 1e18}
    stop = [False]

    def mon():
        p = psutil.Process()
        while not stop[0]:
            try:
                peak["rss"] = max(peak["rss"], p.memory_info().rss + sum(c.memory_info().rss for c in p.children(True)))
                peak["disk"] = max(peak["disk"], dsize(work) + dsize(D / "output") - base)
                peak["min_free"] = min(peak["min_free"], shutil.disk_usage(D).free)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)

    th = threading.Thread(target=mon, daemon=True)
    th.start()
    model = joblib.load(ROOT / "models" / "phase4_model.joblib")
    rule = M.DecisionRule(**json.loads((ROOT / "models" / "phase4_rule.json").read_text()))
    t = time.time()
    try:
        res = R4.run_inference(ds, "test", work, model, rule, D / "output", cand_chunks=1, infer_chunks=1)
    finally:
        stop[0] = True
        th.join()
    res["wall_s"] = round(time.time() - t, 1)
    res["peak_rss_gb"] = round(peak["rss"] / 1e9, 2)
    res["peak_new_disk_mb"] = round(peak["disk"] / 1e6, 1)
    res["min_free_disk_gb"] = round(peak["min_free"] / 1e9, 2)
    res["output_files"] = {p.name: p.stat().st_size for p in sorted((D / "output").iterdir())}
    con = duckdb.connect()
    res["per_country_descriptive_only"] = con.execute(f"""
      WITH s AS (SELECT entity_id, country FROM read_csv_auto('{ds / 'test_source1.tsv'}', delim='\t', header=true, all_varchar=true)),
      c AS (SELECT * FROM read_csv('{D / 'output' / 'candidate_pairs.tsv'}', delim='\t', header=true, all_varchar=true, quote='', escape='')),
      m AS (SELECT * FROM read_csv('{D / 'output' / 'matching_results.tsv'}', delim='\t', header=true, all_varchar=true, quote='', escape=''))
      SELECT s.country, COUNT(*) n_s1,
             ROUND(AVG(coalesce(len(string_split(c.candidate_entity_ids, ',')), 0)), 1) avg_candidates,
             COUNT(*) FILTER (WHERE c.candidate_entity_ids IS NULL) s1_without_candidates,
             ROUND(AVG(coalesce(len(string_split(m.matched_entity_ids, ',')), 0)), 2) avg_matches,
             COUNT(*) FILTER (WHERE m.matched_entity_ids IS NULL) s1_empty_matches
      FROM s LEFT JOIN c ON c.source1_entity_id = s.entity_id LEFT JOIN m ON m.source1_entity_id = s.entity_id
      GROUP BY 1 ORDER BY 1""").df().to_dict(orient="records")
    con.close()
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=1000)
    ap.add_argument("--no-dry-run", action="store_true")
    ap.add_argument("--keep", action="store_true", default=True)
    a = ap.parse_args()
    c = checks()
    print(json.dumps(c, indent=1))
    if not c["all_ok"]:
        raise SystemExit("PRE-FLIGHT CHECKS FAILED")
    if a.no_dry_run:
        return
    res = dry_run(a.n_s1, a.keep)
    spec = importlib.util.spec_from_file_location("run_phase4", Path(__file__).resolve().parent / "run_phase4.py")
    R4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R4)
    (D / "dryrun_metrics.json").write_text(json.dumps(R4.clean({"checks": c, **res}), indent=2, default=str))
    show = {k: res.get(k) for k in ("timings_s", "candidates", "inference", "no_forced_matches", "outputs", "wall_s",
                                    "peak_rss_gb", "peak_new_disk_mb", "min_free_disk_gb", "output_files",
                                    "per_country_descriptive_only")}
    print(json.dumps(R4.clean(show), indent=1, default=str))
    for v in res["publication"]["validators"]:
        print("VALIDATOR ok =", v["ok"], json.dumps(R4.clean(v["details"]) if isinstance(v["details"], dict) else v["details"], default=str)[:1200])
    print("PUBLISHED:", res["publication"]["published"])
    print("DRY RUN PASS")


if __name__ == "__main__":
    main()
