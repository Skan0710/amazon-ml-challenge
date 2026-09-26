"""Regression tests for the production-safety fixes (pre-flight audit).

* S1-batched TSV writer (bounded memory, identical output for any batch size, full coverage)
* atomic publication (``*.tsv.tmp`` -> final only after validation; never on failure)
* streaming custom validator: truncation, duplicate S1, duplicate targets, missing S1,
  empty lists, matches-not-in-candidates, zero-padded ids
* sharded run of the unmodified official validator
* no-forced-match gate

Run: .venv/bin/python -m unittest discover -s tests -t . -v
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import src.matching as M
from src.candidate_generation import encode_entity_id
from tests.test_matching import OFFICIAL, S1_ROWS, V, make_con, make_dataset

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ovs", ROOT / "scripts" / "run_official_validator_sharded.py")
O = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(O)

UNI = "SELECT id, entity_id, rn FROM text_s1"
S1_IDS = [r[0] for r in S1_ROWS]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = make_dataset(cls.tmp.name)
        cls.con = make_con(cls.d)
        cls.src1 = cls.d / "train_source1.tsv"

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.tmp.cleanup()

    def write_pair(self, d, pred_sql="SELECT s1, t FROM cand WHERE s1 IN (1, 5)", batch=25_000):
        d = Path(d)
        c = M.write_candidate_pairs_tsv(self.con, UNI, "SELECT s1, t FROM cand", d / "candidate_pairs.tsv.tmp", batch)
        m = M.write_matching_results_tsv(self.con, UNI, pred_sql, d / "matching_results.tsv.tmp", batch)
        return c, m

    def custom(self, m, c):
        return V.validate(m, c, self.src1)


# ------------------------------------------------------------------------------------ writer
class TestBatchedWriter(Base):
    def test_identical_output_for_any_batch_size(self):
        outs = []
        for batch in (1, 2, 3, 1000):
            with tempfile.TemporaryDirectory() as d:
                c, m = self.write_pair(d, batch=batch)
                self.assertEqual(c["batches"], -(-len(S1_ROWS) // batch))
                outs.append(((Path(d) / "candidate_pairs.tsv.tmp").read_bytes(), (Path(d) / "matching_results.tsv.tmp").read_bytes()))
        self.assertTrue(all(o == outs[0] for o in outs))

    def test_every_s1_once_in_file_order_with_empty_lists(self):
        with tempfile.TemporaryDirectory() as d:
            c, m = self.write_pair(d, batch=2)
            lines = (Path(d) / "matching_results.tsv.tmp").read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "source1_entity_id\tmatched_entity_ids")
            self.assertEqual([l.split("\t")[0] for l in lines[1:]], S1_IDS)        # exactly once, original order
            self.assertIn("S1-00004\t", lines)                                       # empty list preserved
            self.assertEqual(m["rows"], len(S1_ROWS))
            self.assertEqual(self.custom(Path(d) / "matching_results.tsv.tmp", Path(d) / "candidate_pairs.tsv.tmp")[0], [])

    def test_writer_raises_on_invariant_violations(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.tsv.tmp"
            for uni, pairs in (
                (UNI + " UNION ALL SELECT id, entity_id, rn FROM text_s1 WHERE id = 3", "SELECT s1, t FROM cand WHERE FALSE"),  # dup S1
                (UNI, "SELECT s1, t FROM cand UNION ALL SELECT s1, t FROM cand"),                                             # dup pair
                (UNI, "SELECT 1::BIGINT s1, 2999999999::BIGINT t"),                                                           # unknown target
                (UNI + " WHERE id <> 1", "SELECT s1, t FROM cand"),                                                           # pair outside universe
            ):
                with self.assertRaises(M.SubmissionError):
                    M.write_matching_results_tsv(self.con, uni, pairs, p, batch_s1=2)


# ------------------------------------------------------------------------------------ atomic publication
class TestAtomicPublication(Base):
    def _validators(self, d):
        m, c = Path(d) / "matching_results.tsv.tmp", Path(d) / "candidate_pairs.tsv.tmp"
        return [lambda: (not self.custom(m, c)[0], "custom")]

    def test_publish_on_success(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_pair(d)
            res = M.finalize_outputs({Path(d) / "candidate_pairs.tsv.tmp": Path(d) / "candidate_pairs.tsv",
                                      Path(d) / "matching_results.tsv.tmp": Path(d) / "matching_results.tsv"}, self._validators(d))
            self.assertTrue((Path(d) / "matching_results.tsv").exists() and (Path(d) / "candidate_pairs.tsv").exists())
            self.assertFalse(list(Path(d).glob("*.tmp")))
            self.assertTrue(res["published"][-1].endswith("matching_results.tsv"))   # scored file renamed last

    def test_no_publication_on_failure_and_old_finals_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "matching_results.tsv").write_text("OLD FILE\n")                     # a pre-existing final
            self.write_pair(d)
            with open(d / "matching_results.tsv.tmp", "a") as f:                      # corrupt: duplicate S1 row
                f.write("S1-00001\t\n")
            with self.assertRaises(M.SubmissionError):
                M.finalize_outputs({d / "candidate_pairs.tsv.tmp": d / "candidate_pairs.tsv",
                                    d / "matching_results.tsv.tmp": d / "matching_results.tsv"}, self._validators(d))
            self.assertEqual((d / "matching_results.tsv").read_text(), "OLD FILE\n")  # not replaced
            self.assertFalse((d / "candidate_pairs.tsv").exists())                    # not created
            self.assertTrue((d / "matching_results.tsv.tmp").exists())                # kept for debugging

    def test_any_failing_validator_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_pair(d)
            with self.assertRaises(M.SubmissionError):
                M.finalize_outputs({Path(d) / "matching_results.tsv.tmp": Path(d) / "matching_results.tsv"},
                                   [lambda: (True, "ok"), lambda: (False, "official failed")])
            self.assertFalse((Path(d) / "matching_results.tsv").exists())


# ------------------------------------------------------------------------------------ streaming validator
class TestStreamingValidator(Base):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.p = Path(self.dir.name)
        self.write_pair(self.p)
        self.m, self.c = self.p / "matching_results.tsv.tmp", self.p / "candidate_pairs.tsv.tmp"

    def tearDown(self):
        self.dir.cleanup()

    def edit(self, path, fn):
        text = path.read_text(encoding="utf-8")
        path.write_text(fn(text), encoding="utf-8")

    def test_valid_passes(self):
        problems, rep = self.custom(self.m, self.c)
        self.assertEqual(problems, [])
        self.assertEqual((rep["matching"]["missing_s1"], rep["matching"]["duplicate_s1_rows"],
                          rep["matching"]["lists_with_duplicate_ids"]), (0, 0, 0))
        self.assertTrue(rep["matched_subset_of_candidates"])

    def test_truncated_by_lines(self):
        self.edit(self.m, lambda t: "\n".join(t.splitlines()[:-2]) + "\n")
        self.assertTrue(any("missing" in p for p in self.custom(self.m, self.c)[0]))

    def test_truncated_mid_line(self):
        self.edit(self.c, lambda t: t[: len(t) - 7])                 # cut inside the last row
        problems = self.custom(self.m, self.c)[0]
        self.assertTrue(any("truncated" in p or "not S2-/S3-" in p or "missing" in p for p in problems), problems)

    def test_duplicated_s1(self):
        self.edit(self.m, lambda t: t + "S1-00002\t\n")
        self.assertTrue(any("duplicate source1_entity_id" in p for p in self.custom(self.m, self.c)[0]))

    def test_duplicated_target_ids(self):
        self.edit(self.m, lambda t: t.replace("S1-00005\tS2-00016", "S1-00005\tS2-00016,S2-00016"))
        self.assertTrue(any("duplicate ids inside a list" in p for p in self.custom(self.m, self.c)[0]))

    def test_empty_lists_are_valid(self):
        self.edit(self.m, lambda t: "source1_entity_id\tmatched_entity_ids\n" + "".join(f"{s}\t\n" for s in S1_IDS))
        self.assertEqual(self.custom(self.m, self.c)[0], [])

    def test_match_not_in_candidates(self):
        self.edit(self.c, lambda t: t.replace("S2-00016", "S3-00013"))   # S1-00005's only candidate replaced
        self.assertTrue(any("not all in their candidate" in p for p in self.custom(self.m, self.c)[0]))

    def test_matched_s1_without_candidate_row(self):
        self.edit(self.c, lambda t: "\n".join(l for l in t.splitlines() if not l.startswith("S1-00005")) + "\n")
        problems = self.custom(self.m, self.c)[0]
        self.assertTrue(any("no candidate row" in p for p in problems) and any("missing" in p for p in problems))

    def test_zero_padded_ids_do_not_collide(self):
        self.assertNotEqual(V.id_code("S2-012"), V.id_code("S2-12"))
        self.assertEqual(V.id_code("S2-12"), 2_000_000_012)
        self.assertEqual(V.id_code("S3-00013"), "S3-00013")


# ------------------------------------------------------------------------------------ official (sharded)
@unittest.skipUnless(OFFICIAL.exists(), "official validator not available")
class TestShardedOfficial(Base):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.p = Path(self.dir.name)
        self.write_pair(self.p)
        (self.p / "test").mkdir()
        (self.p / "test" / "test_source1.tsv").write_text(self.src1.read_text(encoding="utf-8"), encoding="utf-8")
        self.m, self.c = self.p / "matching_results.tsv.tmp", self.p / "candidate_pairs.tsv.tmp"

    def tearDown(self):
        self.dir.cleanup()

    def run_o(self, shards):
        return O.run(OFFICIAL, self.m, self.c, self.p / "test", self.p / "shards", n_shards=shards, log=lambda *a: None)

    def test_pass_for_any_shard_count_and_row_totals_preserved(self):
        for k in (1, 2, 3, 7):
            r = self.run_o(k)
            self.assertTrue(r["passed"], r)
            self.assertEqual(r["totals"], {"s1_rows": len(S1_ROWS), "matching_rows": len(S1_ROWS), "candidate_rows": len(S1_ROWS)})
            self.assertFalse((self.p / "shards").exists())            # shard files cleaned up

    def test_detects_duplicate_and_missing_across_shards(self):
        with open(self.m, "a") as f:
            f.write("S1-00003\t\n")
        self.assertFalse(self.run_o(3)["passed"])
        self.m.write_text("\n".join(l for l in self.m.read_text().splitlines() if not l.startswith("S1-00003")) + "\n")
        self.m.write_text("\n".join(l for l in self.m.read_text().splitlines() if not l.startswith("S1-00002")) + "\n")
        self.assertFalse(self.run_o(3)["passed"])


# ------------------------------------------------------------------------------------ forced matches
class TestNoForcedMatches(Base):
    def test_gate(self):
        rule = M.DecisionRule(threshold=0.8)
        self.con.execute("CREATE OR REPLACE TABLE pg AS SELECT s1, t, 0.9 p FROM cand WHERE s1 = 1")
        self.assertEqual(M.verify_no_forced_matches(self.con, "pg", "SELECT s1, t FROM cand", rule)["predicted_below_threshold"], 0)
        self.con.execute("INSERT INTO pg SELECT s1, t, 0.3 FROM cand WHERE s1 = 2 LIMIT 1")
        with self.assertRaises(M.SubmissionError):
            M.verify_no_forced_matches(self.con, "pg", "SELECT s1, t FROM cand", rule)
        self.con.execute("CREATE OR REPLACE TABLE pg AS SELECT 1::BIGINT s1, 3000000013::BIGINT t, 0.99 p")  # not a candidate of S1-1
        with self.assertRaises(M.SubmissionError):
            M.verify_no_forced_matches(self.con, "pg", "SELECT s1, t FROM cand", rule)


if __name__ == "__main__":
    unittest.main()
