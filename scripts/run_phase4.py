#!/usr/bin/env python3
"""
Phase 4 runner.

Pilot on TRAINING data only (candidates -> features -> model -> rule -> outputs -> validators):

    .venv/bin/python scripts/run_phase4.py pilot --name 10k --mod 220 --chunks 1
    .venv/bin/python scripts/run_phase4.py pilot --name 100k --mod 22 --chunks 10

``--mod m`` selects S1 entities with ``hash(id) % m = 0`` (the Phase 3 samples). S1 entities are
split 80/20 by a salted hash into train / validation; the model and decision rule are fitted on
train and chosen on validation only. Output files are written for the validation S1 entities,
together with a pseudo ``test_source1.tsv`` of exactly those entities, so both the custom and the
**unmodified official** validator can be run on them.

Final test inference (NOT run without explicit approval — it is the full-scale job):

    .venv/bin/python scripts/run_phase4.py test --dataset-dir ~/student_resource/dataset/test \
        --model models/phase4_model.joblib --rule models/phase4_rule.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.candidate_generation as C  # noqa: E402
import src.matching as M  # noqa: E402
from src.candidate_generation import DEFAULT_STRATEGIES, BlockingConfig  # noqa: E402

RAW = ROOT / "data" / "raw"
P3 = ROOT / "data" / "processed" / "phase3"
P4 = ROOT / "data" / "processed" / "phase4"
OFFICIAL_VALIDATOR = Path(os.environ.get("OFFICIAL_VALIDATOR", Path.home() / "student_resource" / "utils" / "validate_submission.py"))
MIN_FREE_GB = 4.0


class Peak:
    """Peak RSS of this process and its children, sampled every 0.2 s."""
    def __enter__(self):
        self.peak, self._stop = 0, False
        def run():
            p = psutil.Process()
            while not self._stop:
                try:
                    self.peak = max(self.peak, p.memory_info().rss + sum(c.memory_info().rss for c in p.children(True)))
                except Exception:
                    pass
                time.sleep(0.2)
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self._t.join()


def dir_mb(p: Path) -> float:
    if not p.exists():
        return 0.0
    if p.is_file():
        return round(p.stat().st_size / 1e6, 1)
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 1)


def open_train_connection(work: Path):
    """DuckDB with Phase 3 features/statistics (read-only) and Phase 4 text tables for TRAIN data."""
    for i in (1, 2, 3):
        M.materialize_text_features(RAW / f"train_source{i}.tsv", P4 / f"text_s{i}.parquet")
    con = C.connect(work, memory_limit="1500MB", threads=4, temp_dir=P4 / "duckdb_tmp", max_temp="2GB")
    con.execute(f"ATTACH '{P3 / 'blocking_stats.duckdb'}' AS st (READ_ONLY)")
    for t in [r[0] for r in con.execute("SELECT table_name FROM duckdb_tables() WHERE database_name='st' AND table_name LIKE 'st_%'").fetchall()]:
        con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM st.{t}")
    C.register_features(con, P3)
    M.register_text(con, P4)
    return con


def run_validators(out: Path, pseudo_s1: Path, log) -> dict:
    res = {}
    cmd = [sys.executable, str(ROOT / "scripts" / "validate_final_submission.py"), "--matching", str(out / "matching_results.tsv"),
           "--candidate", str(out / "candidate_pairs.tsv"), "--source1", str(pseudo_s1)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    res["custom"] = {"exit_code": r.returncode, "output": r.stdout + r.stderr}
    log(r.stdout.strip())
    if OFFICIAL_VALIDATOR.exists():
        cmd = ["python3", str(OFFICIAL_VALIDATOR), "--matching", str(out / "matching_results.tsv"),
               "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", str(pseudo_s1.parent)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        res["official"] = {"exit_code": r.returncode, "output": r.stdout + r.stderr, "command": " ".join(cmd)}
        log(r.stdout.strip())
    else:
        res["official"] = {"exit_code": None, "output": f"official validator not found at {OFFICIAL_VALIDATOR}"}
    return res


def pilot(name: str, mod: int, chunks: int, feature_chunks: int, easy_rate: float, hard_rate: float = 1.0) -> dict:
    t_all = time.time()
    lines: list[str] = []
    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        lines.append(s)
    out = ROOT / "output" / "pilots" / name
    work_dir = P4 / f"pilot_{name}"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)
    free0 = C.check_disk(P4, MIN_FREE_GB)
    R: dict = {"name": name, "sample": f"hash(id) % {mod} = 0", "free_disk_gb_start": round(free0, 1), "timings_s": {}}
    with Peak() as pk:
        # ---------------------------------------------------------------- setup
        t = time.time()
        con = open_train_connection(work_dir / "work.duckdb")
        R["timings_s"]["setup_text_tables"] = round(time.time() - t, 1)
        con.execute(f"CREATE OR REPLACE TABLE q_all AS SELECT * FROM feat_s1 WHERE hash(id) % {mod} = 0")
        con.execute(f"CREATE OR REPLACE TABLE s1_split AS SELECT id, {M.assign_split('id')} AS split FROM q_all")
        con.execute("CREATE OR REPLACE TABLE val_s1 AS SELECT id FROM s1_split WHERE split = 'val'")
        n_gt = C.load_ground_truth_pairs(con, RAW / "train_ground_truth.tsv", s1_filter_sql="SELECT id FROM q_all")
        R["s1"] = {k: v for k, v in zip(("total", "train", "val"), con.execute(
            "SELECT COUNT(*), COUNT(*) FILTER (WHERE split='train'), COUNT(*) FILTER (WHERE split='val') FROM s1_split").fetchone())}
        R["s1"]["singletons"] = con.execute("SELECT COUNT(*) FROM q_all WHERE id NOT IN (SELECT s1 FROM gt_pairs)").fetchone()[0]
        R["true_pairs"] = n_gt
        log(f"[{name}] S1 {R['s1']} | true pairs {n_gt:,}")
        # ---------------------------------------------------------------- candidates (Phase 3, unchanged)
        t = time.time()
        gen = C.generate_candidates_chunked(con, DEFAULT_STRATEGIES, BlockingConfig(), n_chunks=chunks,
                                            min_free_gb=MIN_FREE_GB, log=log)
        R["timings_s"]["candidates"] = round(time.time() - t, 1)
        raw_rows = sum(v["pairs"] for v in gen["per_strategy"].values())
        uniq = con.execute("SELECT COUNT(*) FROM cand").fetchone()[0]
        dup_after = con.execute("SELECT COUNT(*) - COUNT(DISTINCT (s1, t)) FROM cand").fetchone()[0]
        R["candidates"] = {"raw_strategy_rows": int(raw_rows), "unique_pairs": int(uniq),
                           "duplicate_rows_removed_by_union_and_cap": int(raw_rows - uniq),
                           "duplicates_remaining": int(dup_after), "per_strategy": gen["per_strategy"],
                           "recall_ceiling": C.evaluate_candidate_recall(con)}
        if dup_after:
            raise M.SubmissionError(f"{dup_after} duplicate candidate pairs after deduplication")
        log(f"[{name}] candidates raw {raw_rows:,} -> unique {uniq:,} | recall ceiling {R['candidates']['recall_ceiling']['recall']}")
        # ---------------------------------------------------------------- features (+ labels, split)
        t = time.time()
        R["features"] = M.compute_features_chunked(
            con, "cand", work_dir / "features", with_label=True, n_chunks=feature_chunks,
            extra_cols_sql=", (SELECT split FROM s1_split WHERE s1_split.id = s1) AS split", min_free_gb=MIN_FREE_GB, log=log)
        R["timings_s"]["features"] = round(time.time() - t, 1)
        fglob = str(work_dir / "features" / "*.parquet")
        R["features"]["n_features"] = len(M.FEATURE_COLUMNS)
        R["features"]["rows_per_s"] = round(R["features"]["rows"] / max(R["features"]["seconds"], 0.1))
        lab = con.execute(f"""SELECT split, COUNT(*), SUM(label) FROM read_parquet('{fglob}') GROUP BY 1 ORDER BY 1""").fetchall()
        R["labels"] = {s: {"pairs": int(n), "positive": int(p), "negative": int(n - p)} for s, n, p in lab}
        log(f"[{name}] features {R['features']['rows']:,} rows in {R['features']['seconds']}s | labels {R['labels']}")
        # ---------------------------------------------------------------- training
        t = time.time()
        train_df, samp = M.sample_training_rows(con, fglob, hard_rate=hard_rate, easy_rate=easy_rate)
        R["sampling"] = samp
        model, minfo = M.train_model(train_df)
        R["model"] = {"type": "sklearn.ensemble.HistGradientBoostingClassifier", **minfo}
        del train_df
        R["timings_s"]["training"] = round(time.time() - t, 1)
        log(f"[{name}] trained on {minfo['rows']:,} rows ({minfo['positives']:,} positive) in {minfo['seconds']}s, {minfo['n_iter']} iterations")
        # ---------------------------------------------------------------- validation scoring
        t = time.time()
        R["scoring"] = M.score_pairs(con, model, f"SELECT * FROM read_parquet('{fglob}') WHERE split = 'val'")
        R["scoring"]["rows_per_s"] = round(R["scoring"]["rows"] / max(R["scoring"]["seconds"], 0.1))
        R["timings_s"]["scoring_val"] = R["scoring"]["seconds"]
        # ---------------------------------------------------------------- decision rule search (validation only)
        t = time.time()
        ths = [round(x, 2) for x in (0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)]
        grid = M.search_decision_rules(con, ths, top_ks=(None, 2, 3, 4, 6), relatives=(None, 0.5))
        grid.to_csv(out / "rule_search_val.csv", index=False)
        best = grid.sort_values(["macro_f05", "threshold"], ascending=[False, False]).iloc[0]
        rule = M.DecisionRule(float(best.threshold), int(best.top_k) or None, float(best.relative) or None)
        th_only = grid[(grid.top_k == 0) & (grid.relative == 0)]
        R["decision"] = {"chosen_rule": asdict(rule), "chosen_metrics": best.to_dict(),
                         "threshold_only_table": th_only.to_dict(orient="records"),
                         "best_per_family": grid.sort_values("macro_f05", ascending=False)
                             .groupby(["top_k", "relative"]).head(1).to_dict(orient="records")}
        R["timings_s"]["rule_search"] = round(time.time() - t, 1)
        log(f"[{name}] chosen rule {asdict(rule)} -> macro F0.5 {best.macro_f05:.4f} (P {best.pair_precision}, R {best.pair_recall})")
        # ---------------------------------------------------------------- final validation predictions + singletons
        M.apply_decision(con, rule)
        R["validation"] = M.evaluate_predictions(con)
        # batch-consistency check: the streaming inference path must reproduce the same predictions
        con.execute("CREATE OR REPLACE TABLE pred_ref AS SELECT * FROM pred")
        R["streaming_inference_val"] = M.infer_chunked(
            con, "SELECT * FROM cand WHERE s1 IN (SELECT id FROM val_s1)", model, rule, n_chunks=max(2, feature_chunks),
            pred_table="pred_stream", min_free_gb=MIN_FREE_GB, log=log)
        diff = con.execute("""SELECT (SELECT COUNT(*) FROM (SELECT s1, t FROM pred_ref EXCEPT SELECT s1, t FROM pred_stream))
                                   + (SELECT COUNT(*) FROM (SELECT s1, t FROM pred_stream EXCEPT SELECT s1, t FROM pred_ref))""").fetchone()[0]
        R["streaming_inference_val"]["pairs_differing_from_batch_path"] = int(diff)
        if diff:
            raise M.SubmissionError(f"streaming inference differs from batch path on {diff} pairs")
        # reference points
        con.execute("CREATE OR REPLACE TABLE pred_all AS SELECT s1, t, 1.0 p FROM cand WHERE s1 IN (SELECT id FROM val_s1)")
        R["reference"] = {"predict_all_candidates": M.evaluate_predictions(con, pred_table="pred_all"),
                          "predict_nothing": M.evaluate_predictions(con, pred_table="(SELECT * FROM pred_all WHERE FALSE)")}
        sing = M.singleton_report(con)
        sing.to_csv(out / "singletons_val.csv", index=False)
        R["singletons_val"] = {"count": int(len(sing)), "predicted_empty": int(sing.predicted_empty.sum()),
                               "predicted_nonempty": int((~sing.predicted_empty).sum()),
                               "without_candidates": int((sing.n_candidates == 0).sum()),
                               "max_p_quantiles": sing.pmax.dropna().quantile([0.5, 0.9, 0.99]).round(4).to_dict()}
        imp = permutation_importance_sample(con, model, fglob)
        R["feature_importance"] = imp
        # ---------------------------------------------------------------- outputs for validation S1
        t = time.time()
        pseudo = work_dir / "pseudo_test"
        pseudo.mkdir()
        con.execute(f"""COPY (SELECT s.* FROM read_csv_auto('{RAW / 'train_source1.tsv'}', delim='\t', header=true, all_varchar=true) s
                     WHERE CAST(substr(s.entity_id, 4) AS BIGINT) IN (SELECT id FROM val_s1))
                     TO '{pseudo / 'test_source1.tsv'}' (DELIMITER '\t', HEADER, QUOTE '')""")
        universe = "SELECT t.id, t.entity_id, t.rn FROM text_s1 t JOIN val_s1 v ON v.id = t.id"
        R["outputs"] = {
            "candidate_pairs": M.write_candidate_pairs_tsv(con, universe, "SELECT s1, t FROM cand WHERE s1 IN (SELECT id FROM val_s1)",
                                                           out / "candidate_pairs.tsv"),
            "matching_results": M.write_matching_results_tsv(con, universe, "SELECT s1, t FROM pred", out / "matching_results.tsv")}
        R["timings_s"]["write_outputs"] = round(time.time() - t, 1)
        R["validators"] = run_validators(out, pseudo / "test_source1.tsv", log)
        import joblib
        joblib.dump(model, work_dir / "model.joblib")
        (work_dir / "rule.json").write_text(json.dumps(asdict(rule)))
        con.close()
    R["peak_rss_gb"] = round(pk.peak / 1e9, 2)
    R["disk_mb"] = {"work_dir": dir_mb(work_dir), "features": dir_mb(work_dir / "features"), "outputs": dir_mb(out),
                    "text_tables": sum(dir_mb(P4 / f"text_s{i}.parquet") for i in (1, 2, 3))}
    R["free_disk_gb_end"] = round(C.check_disk(P4, 0), 1)
    R["runtime_s"] = round(time.time() - t_all, 1)
    (out / "phase4_metrics.json").write_text(json.dumps(clean(R), indent=2))
    (out / "validation_report.txt").write_text("\n".join(lines) + "\n\n" + R["validators"]["custom"]["output"]
                                               + "\n" + (R["validators"]["official"]["output"] or ""))
    log(f"[{name}] done in {R['runtime_s']}s | peak RSS {R['peak_rss_gb']} GB | disk {R['disk_mb']}")
    shutil.rmtree(P4 / "duckdb_tmp", ignore_errors=True)
    return R


def permutation_importance_sample(con, model, fglob: str, n: int = 100_000) -> list[dict]:
    """Permutation importance (drop in average precision) on a fixed validation sample."""
    from sklearn.inspection import permutation_importance
    df = con.execute(f"""SELECT label, {', '.join(M.FEATURE_COLUMNS)} FROM read_parquet('{fglob}')
                         WHERE split = 'val' ORDER BY hash(s1, t, 11) LIMIT {n}""").df()
    X, y = df[list(M.FEATURE_COLUMNS)].to_numpy("float32"), df["label"].to_numpy()
    r = permutation_importance(model, X, y, scoring="average_precision", n_repeats=3, random_state=0, n_jobs=1)
    out = sorted(({"feature": f, "importance": round(float(m), 5), "std": round(float(s), 5)}
                  for f, m, s in zip(M.FEATURE_COLUMNS, r.importances_mean, r.importances_std)),
                 key=lambda d: -d["importance"])
    return out


def clean(o):
    import math
    import numpy as np
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_inference(dataset_dir: Path, prefix: str, work_dir: Path, model, rule: M.DecisionRule, out_dir: Path,
                  cand_chunks: int, infer_chunks: int, workers: int = 4, min_free_gb: float = MIN_FREE_GB,
                  official_validator: Path = OFFICIAL_VALIDATOR, log=print) -> dict:
    """End-to-end inference on an unlabelled dataset directory (``{prefix}_source{1,2,3}.tsv``):
    Phase 3 features + statistics of THAT dataset -> candidates (Parquet parts) -> streaming
    features/scores/decision -> candidate_pairs.tsv + matching_results.tsv -> both validators.
    Used for the final test inference; unit-tested end-to-end on a synthetic dataset."""
    t0 = time.time()
    work_dir.mkdir(parents=True, exist_ok=True)
    R: dict = {"timings_s": {}}
    for i in (1, 2, 3):
        C.materialize_blocking_features(dataset_dir / f"{prefix}_source{i}.tsv", work_dir / f"features_s{i}",
                                        workers=workers, min_free_gb=min_free_gb, log=log)
        M.materialize_text_features(dataset_dir / f"{prefix}_source{i}.tsv", work_dir / f"text_s{i}.parquet")
    R["timings_s"]["normalisation"] = round(time.time() - t0, 1)
    con = C.connect(work_dir / "work.duckdb", memory_limit="1500MB", threads=4, temp_dir=work_dir / "duckdb_tmp", max_temp="2GB")
    C.register_features(con, work_dir)
    M.register_text(con, work_dir)
    t = time.time()
    C.build_blocking_statistics(con, BlockingConfig(), kinds=["name", "phonetic", "structured", "rare_name", "rare_addr", "ngram"], log=log)
    R["timings_s"]["statistics"] = round(time.time() - t, 1)
    con.execute("CREATE OR REPLACE TABLE q_all AS SELECT * FROM feat_s1")
    t = time.time()
    shutil.rmtree(work_dir / "candidates", ignore_errors=True)
    gen = C.generate_candidates_chunked(con, DEFAULT_STRATEGIES, BlockingConfig(), n_chunks=cand_chunks,
                                        out_parquet_dir=work_dir / "candidates", min_free_gb=min_free_gb, log=log)
    con.execute("DROP TABLE IF EXISTS cand")          # last chunk's table left by the Parquet-mode generator
    con.execute(f"CREATE OR REPLACE VIEW cand AS SELECT * FROM read_parquet('{work_dir / 'candidates'}/*.parquet')")
    R["timings_s"]["candidates"] = round(time.time() - t, 1)
    dup = con.execute("SELECT COUNT(*) - COUNT(DISTINCT (s1, t)) FROM cand").fetchone()[0]
    if dup:
        raise M.SubmissionError(f"{dup} duplicate candidate pairs")
    R["candidates"] = {"raw_strategy_rows": int(sum(v["pairs"] for v in gen["per_strategy"].values())),
                       "unique_pairs": int(con.execute("SELECT COUNT(*) FROM cand").fetchone()[0]), "duplicates_remaining": 0}
    t = time.time()
    R["inference"] = M.infer_chunked(con, "SELECT * FROM cand", model, rule, n_chunks=infer_chunks, min_free_gb=min_free_gb, log=log)
    R["timings_s"]["inference"] = round(time.time() - t, 1)
    R["no_forced_matches"] = M.verify_no_forced_matches(con, "pred", "SELECT s1, t FROM cand", rule)
    # ---- write ONLY temporary files; publish atomically after both validators pass
    out_dir.mkdir(parents=True, exist_ok=True)
    finals = {"candidate": out_dir / "candidate_pairs.tsv", "matching": out_dir / "matching_results.tsv"}
    tmps = {k: v.with_name(v.name + ".tmp") for k, v in finals.items()}
    for p in tmps.values():
        p.unlink(missing_ok=True)
    t = time.time()
    universe = "SELECT id, entity_id, rn FROM text_s1"
    C.check_disk(out_dir, min_free_gb)
    R["outputs"] = {"candidate_pairs": M.write_candidate_pairs_tsv(con, universe, "SELECT s1, t FROM cand", tmps["candidate"]),
                    "matching_results": M.write_matching_results_tsv(con, universe, "SELECT s1, t FROM pred", tmps["matching"])}
    con.close()
    R["timings_s"]["write_outputs"] = round(time.time() - t, 1)
    source1 = dataset_dir / f"{prefix}_source1.tsv"
    t = time.time()

    def custom():
        import importlib.util
        spec = importlib.util.spec_from_file_location("vfs", ROOT / "scripts" / "validate_final_submission.py")
        V = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(V)
        problems, report = V.validate(tmps["matching"], tmps["candidate"], source1)
        log(f"custom streaming validator: {'PASS' if not problems else 'FAIL'} {report}")
        for pr in problems:
            log(f"  - {pr}")
        return not problems, {"problems": problems, "report": report}

    def official():
        if not official_validator.exists():
            return False, f"official validator not found at {official_validator}"
        import importlib.util
        spec = importlib.util.spec_from_file_location("ovs", ROOT / "scripts" / "run_official_validator_sharded.py")
        O = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(O)
        res = O.run(official_validator, tmps["matching"], tmps["candidate"], source1.parent, work_dir / "validator_shards", log=log)
        return res["passed"] and not res["warnings_present"], res

    R["publication"] = M.finalize_outputs({tmps["candidate"]: finals["candidate"], tmps["matching"]: finals["matching"]},
                                          [custom, official])
    R["timings_s"]["validation"] = round(time.time() - t, 1)
    R["runtime_s"] = round(time.time() - t0, 1)
    return R


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--name", required=True)
    p.add_argument("--mod", type=int, required=True)
    p.add_argument("--chunks", type=int, default=1)
    p.add_argument("--feature-chunks", type=int, default=1)
    p.add_argument("--easy-rate", type=float, default=0.1)
    p.add_argument("--hard-rate", type=float, default=1.0)
    t = sub.add_parser("test")
    t.add_argument("--dataset-dir", type=Path, required=True)
    t.add_argument("--model", type=Path, required=True)
    t.add_argument("--rule", type=Path, required=True)
    t.add_argument("--chunks", type=int, default=220)
    t.add_argument("--i-approve-full-scale", action="store_true",
                   help="required: the test run is the full-scale job (~1.7M S1, ~10^8 candidate pairs)")
    a = ap.parse_args()
    if a.cmd == "pilot":
        pilot(a.name, a.mod, a.chunks, a.feature_chunks, a.easy_rate, a.hard_rate)
    else:
        if not a.i_approve_full_scale:
            raise SystemExit("Refusing to run full-scale test inference without --i-approve-full-scale.")
        import joblib
        rule = M.DecisionRule(**json.loads(a.rule.read_text()))
        R = run_inference(a.dataset_dir, "test", P4 / "test_run", joblib.load(a.model), rule, ROOT / "output",
                          cand_chunks=a.chunks, infer_chunks=a.chunks)
        (ROOT / "output" / "phase4_metrics.json").write_text(json.dumps(clean(R), indent=2))


if __name__ == "__main__":
    main()
