"""Unit tests for Phase 4 (src/matching.py, scripts/validate_final_submission.py, scripts/run_phase4.py).

Run from the project root:
    .venv/bin/python -m unittest discover -s tests -t . -v
All tests use small synthetic data through the real code paths (Phase 3 blocking fields,
text tables, feature SQL, writers, validators, end-to-end inference).
"""
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import src.candidate_generation as C
import src.matching as M
from src.candidate_generation import BlockingConfig, blocking_records_chunk, encode_entity_id

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_final_submission.py"
OFFICIAL = Path(os.environ.get("OFFICIAL_VALIDATOR", Path.home() / "student_resource" / "utils" / "validate_submission.py"))

_spec = importlib.util.spec_from_file_location("validate_final_submission", VALIDATOR)
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

HEAD = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
S1_ROWS = [
    ("S1-00001", "Wenonah's Metal Works", "181 Farragut Avenue, Hastings-on-hudson, NY", "US"),
    ("S1-00002", "Reliable Asset Group", "2916 Louisiana Avenue, Halethorpe, MD", "US"),
    ("S1-00003", "Sky Supreme Products Private Limited", "Flat No.1, Cantonment Po, Aurangabad, Maharashtra", "India"),
    ("S1-00004", "Qzxv Unmatched Holdings", "77 Nowhere Road, Tulsa, OK", "US"),        # singleton, no candidates
    ("S1-00005", "Maison Dupont", "12 Rue de la Paix, Paris", "France"),               # unseen country
]
S2_ROWS = [
    ("S2-00010", "WENONAH'S METAL WS", "0181 FARRAGUT AVENUE, HASTINGS-ON-HUDSON, NY", "US"),
    ("S2-00012", "Reliable Asie Group", "Maryland, Halethorpe, null, 2916 Louisiana Avenue", "US"),
    ("S2-00014", "Reliable Asset Group", "4228 Other Street, Halethorpe, MD", "US"),       # decoy: same name
    ("S2-00016", "Maison Dupont", "12 Rue de la Paix, Paris", "France"),
]
S3_ROWS = [
    ("S3-00011", "wenonahsmetalworks.com", "Hastings On Hudson, New York, 181B Farragut Avenue", "US"),
    ("S3-00013", "स्काई सुप्रीम प्रोडक्ट्स प्राइवेट लिमिटेड", "FLAT NO.1, CANTONMENT PO, AURANGABAD, महाराष्ट्र", "India"),
    ("S3-00015", "Wenonah's Metal Works", None, "US"),                                   # missing address
]
GT = {"S1-00001": ["S2-00010", "S3-00011", "S3-00015"], "S1-00002": ["S2-00012"], "S1-00003": ["S3-00013"],
      "S1-00004": [], "S1-00005": ["S2-00016"]}


def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(HEAD)
        for r in rows:
            f.write("\t".join("" if v is None else v for v in r) + "\n")


def make_dataset(d, prefix="train"):
    d = Path(d)
    write_tsv(d / f"{prefix}_source1.tsv", S1_ROWS)
    write_tsv(d / f"{prefix}_source2.tsv", S2_ROWS)
    write_tsv(d / f"{prefix}_source3.tsv", S3_ROWS)
    with open(d / f"{prefix}_ground_truth.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for k, v in GT.items():
            f.write(f"{k}\t{','.join(v)}\n")
    return d


def make_con(d):
    """feat_s1/feat_t (Phase 3 blocking fields), text_s1/text_t, st_name, gt_pairs, cand."""
    d = Path(d)
    con = duckdb.connect()
    for table, rows in (("feat_s1", S1_ROWS), ("feat_t", S2_ROWS + S3_ROWS)):
        df = pd.DataFrame(blocking_records_chunk(rows))
        con.register("df", df)
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
        con.unregister("df")
    for i in (1, 2, 3):
        M.materialize_text_features(d / f"train_source{i}.tsv", d / f"text_s{i}.parquet")
    M.register_text(con, d)
    C.build_blocking_statistics(con, BlockingConfig(), log=lambda *a: None)
    C.load_ground_truth_pairs(con, d / "train_ground_truth.tsv")
    con.execute("CREATE TABLE q AS SELECT * FROM feat_s1")
    C.generate_candidates(con, list(C.DEFAULT_STRATEGIES), BlockingConfig(), log=lambda *a: None)
    return con


class StubModel:
    """Deterministic 'model': probability = name Jaro-Winkler (NaN -> 0)."""
    def predict_proba(self, X):
        p = np.nan_to_num(X[:, M.FEATURE_COLUMNS.index("f_name_jw")], nan=0.0)
        return np.column_stack([1 - p, p])


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = make_dataset(cls.tmp.name)
        cls.con = make_con(cls.d)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.tmp.cleanup()


class TestIdsAndText(Base):
    def test_zero_padded_ids_preserved(self):
        t = self.con.execute("SELECT id, entity_id FROM text_s1 ORDER BY rn").fetchall()
        self.assertEqual(t[0], (1, "S1-00001"))                       # string kept exactly
        tt = dict(self.con.execute("SELECT id, entity_id FROM text_t").fetchall())
        self.assertEqual(tt[encode_entity_id("S2-00010")[1]], "S2-00010")

    def test_id_collision_detected(self):
        with tempfile.TemporaryDirectory() as d:
            write_tsv(Path(d) / "s.tsv", [("S2-7", "A", None, "US"), ("S2-007", "B", None, "US")])
            with self.assertRaises(M.SubmissionError):
                M.materialize_text_features(Path(d) / "s.tsv", Path(d) / "t.parquet")

    def test_bad_id_format_detected(self):
        with tempfile.TemporaryDirectory() as d:
            write_tsv(Path(d) / "s.tsv", [("X9-1", "A", None, "US")])
            with self.assertRaises(M.SubmissionError):
                M.materialize_text_features(Path(d) / "s.tsv", Path(d) / "t.parquet")


class TestCandidatesAndFeatures(Base):
    def test_candidates_unique_and_country_partitioned(self):
        n, d = self.con.execute("SELECT COUNT(*), COUNT(DISTINCT (s1, t)) FROM cand").fetchone()
        self.assertEqual(n, d)
        # France is its own partition: S1-00005 only ever meets the French target
        ts = {r[0] for r in self.con.execute("SELECT t FROM cand WHERE s1 = 5").fetchall()}
        self.assertEqual(ts, {encode_entity_id("S2-00016")[1]})

    def test_dedup_report(self):
        info = {"per_strategy": {"name": {"pairs": 5}, "structured": {"pairs": 4}}}
        self.assertEqual(M.candidate_dedup_report(info, [])["raw_candidate_rows"], 9)

    def test_feature_values(self):
        n = M.compute_features(self.con, "SELECT * FROM cand", with_label=True)
        self.assertEqual(n, self.con.execute("SELECT COUNT(*) FROM cand").fetchone()[0])
        cols = [r[0] for r in self.con.execute("DESCRIBE pair_features").fetchall()]
        self.assertEqual(tuple(cols[3:]), M.FEATURE_COLUMNS)
        f = self.con.execute("SELECT * FROM pair_features WHERE s1 = 5").df().iloc[0]
        self.assertEqual((f.f_key_eq, f.f_name_jw, f.f_name_lc_eq, f.label), (1, 1.0, 1, 1))
        self.assertEqual(f.f_state_eq if not pd.isna(f.f_state_eq) else "null", "null")   # France: no state
        for c in ("f_name_jw", "f_tok_jacc", "f_addr_tok_jacc", "f_gram_jacc", "f_phon_jacc"):
            v = self.con.execute(f"SELECT min({c}), max({c}) FROM pair_features").fetchone()
            self.assertTrue(all(x is None or 0 <= x <= 1.0000001 for x in v), c)

    def test_labels_match_ground_truth(self):
        M.compute_features(self.con, "SELECT * FROM cand", with_label=True)
        got = {(a, b) for a, b in self.con.execute("SELECT s1, t FROM pair_features WHERE label = 1").fetchall()}
        cand = {(a, b) for a, b in self.con.execute("SELECT s1, t FROM cand").fetchall()}
        truth = {(encode_entity_id(k)[1], encode_entity_id(v)[1]) for k, vs in GT.items() for v in vs}
        self.assertEqual(got, truth & cand)

    def test_missing_address_is_not_similarity(self):
        M.compute_features(self.con, "SELECT * FROM cand", with_label=True)
        t15 = encode_entity_id("S3-00015")[1]
        f = self.con.execute(f"SELECT * FROM pair_features WHERE s1 = 1 AND t = {t15}").df().iloc[0]
        self.assertEqual((f.f_b_addr_missing, f.f_a_addr_missing, f.f_both_addr_missing), (1, 0, 0))
        for c in ("f_hn_eq", "f_state_eq", "f_street_eq", "f_addr_tok_jacc"):
            self.assertTrue(pd.isna(f[c]), c)                  # NULL, never 0/1 similarity
        self.assertEqual(f.f_key_eq, 1)                        # the candidate itself is kept

    def test_chunked_features_equal_single(self):
        M.compute_features(self.con, "SELECT * FROM cand", with_label=True, out_table="f_single")
        with tempfile.TemporaryDirectory() as d:
            M.compute_features_chunked(self.con, "cand", d, with_label=True, n_chunks=3, min_free_gb=0, log=lambda *a: None)
            a = self.con.execute("SELECT * FROM f_single ORDER BY s1, t").df()
            b = self.con.execute(f"SELECT * FROM read_parquet('{d}/*.parquet') ORDER BY s1, t").df()
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)


class TestMetricAndDecision(Base):
    def test_entity_f05(self):
        self.assertEqual(M.entity_f05(0, 0, 0), 1.0)          # singleton, empty -> 1
        self.assertEqual(M.entity_f05(0, 2, 0), 0.0)          # singleton, any prediction -> 0
        self.assertEqual(M.entity_f05(2, 0, 0), 0.0)          # missed everything -> 0
        self.assertAlmostEqual(M.entity_f05(2, 3, 2), 0.7142857, places=6)   # README example
        self.assertEqual(M.entity_f05(3, 3, 3), 1.0)

    def test_evaluate_matches_python_reference(self):
        self.con.execute("CREATE OR REPLACE TABLE uni AS SELECT id FROM feat_s1")
        self.con.execute("CREATE OR REPLACE TABLE pr (s1 BIGINT, t BIGINT, p DOUBLE)")
        preds = {1: ["S2-00010", "S2-00014"], 2: [], 3: ["S3-00013"], 4: ["S2-00012"], 5: []}
        for s, ts in preds.items():
            for t in ts:
                self.con.execute("INSERT INTO pr VALUES (?, ?, 1)", [s, encode_entity_id(t)[1]])
        ev = M.evaluate_predictions(self.con, pred_table="pr", universe_table="uni")
        ref = []
        for k, truth in GT.items():
            p = set(preds[int(k[3:])]); tr = set(truth)
            ref.append(M.entity_f05(len(tr), len(p), len(p & tr)))
        self.assertAlmostEqual(ev["macro_f05"], round(sum(ref) / len(ref), 5), places=5)
        self.assertEqual((ev["singletons"], ev["singleton_empty"], ev["singleton_nonempty"]), (1, 0, 1))
        self.assertEqual(ev["false_positives"], 2)

    def test_decision_never_forces_a_match(self):
        self.con.execute("CREATE OR REPLACE TABLE sc AS SELECT s1, t, 0.2 AS p FROM cand")
        n = M.apply_decision(self.con, M.DecisionRule(threshold=0.5, top_k=1), scores_table="sc", out_table="pr2")
        self.assertEqual(n, 0)                                 # weak evidence -> empty, even with top_k
        self.con.execute("CREATE OR REPLACE TABLE sc AS SELECT s1, t, CASE WHEN t % 2 = 0 THEN 0.9 ELSE 0.8 END p FROM cand")
        M.apply_decision(self.con, M.DecisionRule(threshold=0.5, top_k=1), scores_table="sc", out_table="pr2")
        per = self.con.execute("SELECT max(c) FROM (SELECT COUNT(*) c FROM pr2 GROUP BY s1)").fetchone()[0]
        self.assertEqual(per, 1)
        M.apply_decision(self.con, M.DecisionRule(threshold=0.5, relative=0.95), scores_table="sc", out_table="pr2")
        bad = self.con.execute("""SELECT COUNT(*) FROM pr2 JOIN (SELECT s1, max(p) m FROM sc GROUP BY 1) x USING (s1)
                                  WHERE pr2.p < 0.95 * x.m""").fetchone()[0]
        self.assertEqual(bad, 0)                              # kept only within 95% of that S1's best

    def test_streaming_inference_equals_batch(self):
        M.compute_features(self.con, "SELECT * FROM cand", with_label=False, out_table="fb")
        M.score_pairs(self.con, StubModel(), "SELECT * FROM fb", out_table="sb", batch_rows=2)   # tiny batches
        rule = M.DecisionRule(threshold=0.7)
        M.apply_decision(self.con, rule, scores_table="sb", out_table="pb")
        for k in (1, 3):
            M.infer_chunked(self.con, "SELECT * FROM cand", StubModel(), rule, n_chunks=k, pred_table="ps", min_free_gb=0, log=lambda *a: None)
            a = set(self.con.execute("SELECT s1, t FROM pb").fetchall())
            b = set(self.con.execute("SELECT s1, t FROM ps").fetchall())
            self.assertEqual(a, b)


class TestWritersAndValidators(Base):
    def _write(self, d, pred_sql):
        uni = "SELECT id, entity_id, rn FROM text_s1"
        M.write_candidate_pairs_tsv(self.con, uni, "SELECT s1, t FROM cand", Path(d) / "candidate_pairs.tsv")
        return M.write_matching_results_tsv(self.con, uni, pred_sql, Path(d) / "matching_results.tsv")

    def test_outputs_valid_and_complete(self):
        with tempfile.TemporaryDirectory() as d:
            info = self._write(d, "SELECT s1, t FROM cand WHERE s1 IN (1, 5)")
            self.assertEqual(info["rows"], len(S1_ROWS))
            lines = (Path(d) / "matching_results.tsv").read_text(encoding="utf-8").split("\n")
            self.assertEqual(lines[0], "source1_entity_id\tmatched_entity_ids")
            self.assertIn("S1-00004\t", lines)                 # empty list = empty field, row kept
            self.assertTrue(all("[" not in l and "nan" not in l.lower() for l in lines))
            problems, rep = V.validate(Path(d) / "matching_results.tsv", Path(d) / "candidate_pairs.tsv", self.d / "train_source1.tsv",
                                       self.d / "train_source2.tsv", self.d / "train_source3.tsv")
            self.assertEqual(problems, [])
            self.assertEqual(rep["matching_missing_s1"], 0)
            if OFFICIAL.exists():
                shutil_dir = Path(d) / "test"; shutil_dir.mkdir()
                (shutil_dir / "test_source1.tsv").write_text((self.d / "train_source1.tsv").read_text())
                r = subprocess.run(["python3", str(OFFICIAL), "--matching", f"{d}/matching_results.tsv",
                                    "--candidate", f"{d}/candidate_pairs.tsv", "--test-dir", str(shutil_dir)],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout)

    def test_writer_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(M.SubmissionError):          # duplicate S1 in universe
                M.write_matching_results_tsv(self.con, "SELECT id, entity_id, rn FROM text_s1 UNION ALL SELECT id, entity_id, rn FROM text_s1 WHERE id = 1",
                                             "SELECT s1, t FROM cand WHERE FALSE", Path(d) / "m.tsv")
            with self.assertRaises(M.SubmissionError):          # duplicate pair
                M.write_matching_results_tsv(self.con, "SELECT id, entity_id, rn FROM text_s1",
                                             "SELECT s1, t FROM cand UNION ALL SELECT s1, t FROM cand", Path(d) / "m.tsv")
            with self.assertRaises(M.SubmissionError):          # unknown target id
                M.write_matching_results_tsv(self.con, "SELECT id, entity_id, rn FROM text_s1",
                                             "SELECT 1::BIGINT s1, 2999999999::BIGINT t", Path(d) / "m.tsv")

    def _check(self, matching_text, candidate_text=None):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "m.tsv").write_text(matching_text, encoding="utf-8")
            good_c = "source1_entity_id\tcandidate_entity_ids\n" + "".join(
                f"{r[0]}\tS2-00010,S3-00011,S2-00012\n" for r in S1_ROWS)
            (Path(d) / "c.tsv").write_text(candidate_text or good_c, encoding="utf-8")
            return V.validate(Path(d) / "m.tsv", Path(d) / "c.tsv", self.d / "train_source1.tsv")[0]

    def _rows(self, overrides):
        rows = {r[0]: "" for r in S1_ROWS}
        rows.update(overrides)
        return "source1_entity_id\tmatched_entity_ids\n" + "".join(f"{k}\t{v}\n" for k, v in rows.items())

    def test_validator_accepts_good_file(self):
        self.assertEqual(self._check(self._rows({"S1-00001": "S2-00010,S3-00011"})), [])

    def test_validator_detects_every_defect(self):
        cases = {
            "missing S1": self._rows({}).replace("S1-00003\t\n", ""),
            "duplicate S1": self._rows({}) + "S1-00001\t\n",
            "duplicate target": self._rows({"S1-00001": "S2-00010,S2-00010"}),
            "empty element": self._rows({"S1-00001": "S2-00010,,S3-00011"}),
            "python list": self._rows({"S1-00001": "['S2-00010']"}),
            "nan": self._rows({"S1-00001": "nan"}),
            "self match": self._rows({"S1-00001": "S1-00002"}),
            "not a candidate": self._rows({"S1-00001": "S3-00015"}),
            "wrong header": self._rows({}).replace("matched_entity_ids", "Matched_Entity_IDs"),
            "csv": self._rows({}).replace("\t", ","),
            "unexpected S1": self._rows({}) + "S1-99999\t\n",
            "extra column": self._rows({"S1-00001": "S2-00010\textra"}),
        }
        for name, text in cases.items():
            self.assertTrue(self._check(text), f"validator missed: {name}")

    def test_validator_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "m.tsv").write_text(self._rows({}), encoding="utf-8")
            r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", f"{d}/m.tsv", "--candidate", f"{d}/m.tsv",
                                "--source1", str(self.d / "train_source1.tsv")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)                   # candidate file has the wrong header -> FAIL


class TestEndToEndInference(unittest.TestCase):
    def test_run_inference_on_unlabelled_dataset(self):
        spec = importlib.util.spec_from_file_location("run_phase4", ROOT / "scripts" / "run_phase4.py")
        R4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(R4)
        with tempfile.TemporaryDirectory() as d:
            ds = make_dataset(Path(d) / "ds", prefix="test") if (Path(d) / "ds").mkdir() is None else None
            os.remove(ds / "test_ground_truth.tsv")          # unlabelled, like the real test set
            res = R4.run_inference(ds, "test", Path(d) / "work", StubModel(), M.DecisionRule(threshold=0.9),
                                   Path(d) / "out", cand_chunks=2, infer_chunks=2, workers=1, min_free_gb=0, log=lambda *a: None)
            pub = res["publication"]["validators"]
            self.assertTrue(pub[0]["ok"], pub[0])                   # custom streaming validator
            if OFFICIAL.exists():
                self.assertTrue(pub[1]["ok"], pub[1])               # official validator (sharded)
            self.assertFalse(list((Path(d) / "out").glob("*.tmp")))  # temporaries renamed away
            m = (Path(d) / "out" / "matching_results.tsv").read_text().splitlines()
            self.assertEqual(len(m), 1 + len(S1_ROWS))
            self.assertIn("S1-00005\tS2-00016", m)              # France entity matched and present
            self.assertIn("S1-00004\t", m)                      # singleton kept with empty list


if __name__ == "__main__":
    unittest.main()
