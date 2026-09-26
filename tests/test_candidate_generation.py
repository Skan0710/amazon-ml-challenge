"""Unit tests for src/candidate_generation.py (Phase 3).

Run from the project root:
    .venv/bin/python -m unittest discover -s tests -t . -v
The SQL strategies are exercised on tiny synthetic DuckDB tables built with the same
``blocking_record`` function used on the real data, so the tests cover the real code paths.
"""
import unittest

import duckdb
import pandas as pd

from src import candidate_generation as C
from src.candidate_generation import (
    BlockingConfig, STRATEGY_BITS, blocking_record, blocking_records_chunk, candidate_statistics,
    decode_target_id, encode_entity_id, evaluate_candidate_recall, generate_candidates, normalize_country,
)


def make_con(s1_rows, t_rows, gt_pairs=(), cfg=None):
    """Build feat_s1 / feat_t / q / gt_pairs tables from (entity_id, name, address, country) rows."""
    con = duckdb.connect()
    for table, rows in (("feat_s1", s1_rows), ("feat_t", t_rows)):
        df = pd.DataFrame(blocking_records_chunk(rows))
        con.register("df", df)
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM df")
        con.unregister("df")
    con.execute("CREATE TABLE q AS SELECT * FROM feat_s1")
    gt = [(encode_entity_id(a)[1], encode_entity_id(b)[1], encode_entity_id(b)[0]) for a, b in gt_pairs]
    con.execute("CREATE TABLE gt_pairs (s1 BIGINT, t BIGINT, src TINYINT)")
    if gt:
        con.executemany("INSERT INTO gt_pairs VALUES (?, ?, ?)", gt)
    C.build_blocking_statistics(con, cfg or BlockingConfig(), log=lambda *a: None)
    return con


def pairs(con, strategy=None):
    flt = f"WHERE mask & {STRATEGY_BITS[strategy]} > 0" if strategy else ""
    return {(s1, decode_target_id(t)) for s1, t in con.execute(f"SELECT s1, t FROM cand {flt}").fetchall()}


class TestIdsAndCountry(unittest.TestCase):
    def test_encode_decode(self):
        self.assertEqual(encode_entity_id("S1-965667"), (1, 965667))
        self.assertEqual(encode_entity_id("S2-681193310"), (2, 2_681_193_310))
        self.assertEqual(encode_entity_id("S3-11291185"), (3, 3_011_291_185))
        self.assertEqual(decode_target_id(3_011_291_185), "S3-11291185")
        for bad in ("X2-1", "S2-", "S2-12a", "S2-1000000000"):
            with self.assertRaises(ValueError):
                encode_entity_id(bad)

    def test_country_not_hard_coded(self):
        self.assertEqual(normalize_country(" US "), "us")
        self.assertEqual(normalize_country("India"), "india")
        self.assertEqual(normalize_country("Canada"), "canada")
        self.assertEqual(normalize_country(None), "")
        self.assertEqual(normalize_country(float("nan")), "")


class TestBlockingKeys(unittest.TestCase):
    def test_name_keys_variants(self):
        r = blocking_record("Quodova a/k/a Indian International Private Limited", "1 Main St, Austin, TX", "US")
        self.assertEqual(r["name_keys"].split("|"), ["quodovaindianinternational", "quodova", "indianinternational"])
        w = blocking_record("wenonahsmetalworks.com", None, "US")
        n = blocking_record("*** Wenonah's Metal Works LLC", None, "US")
        self.assertEqual(w["name_key"], n["name_key"])            # website label == compact name

    def test_tokens_and_phonetic(self):
        r = blocking_record("The Fresh Deli & Care Inc", None, "US")
        self.assertEqual(r["name_tokens"], "fresh deli care")    # stop words + legal form removed
        a = blocking_record("Family Midwest Associates", None, "US")
        b = blocking_record("Midwest Family Associates", None, "US")
        self.assertEqual(a["name_phon"], b["name_phon"])          # order-insensitive phonetic key
        x = blocking_record("Global Tech Private Limited", None, "India")
        y = blocking_record("ग्लोबल टेक प्राइवेट लिमिटेड", None, "India")
        self.assertEqual(x["name_phon"], y["name_phon"])          # cross-script phonetic key

    def test_structured_fields(self):
        r = blocking_record("X", "00515 Kitty Hawk Ln, Point Pleasant, West Virginia", "US")
        self.assertEqual((r["state"], r["hn"], r["street"], r["places"]), ("wv", "515", "kitty hawk ln", "pt pleasant"))
        self.assertNotIn("wv", r["addr_tokens"].split())
        i = blocking_record("Y", "H.NO. 69, FARIDABAD, हरियाणा", "India")
        self.assertEqual((i["state"], i["hn"], i["places"]), ("hr", "69", "faridabad"))

    def test_edge_cases(self):
        for name, addr, country in ((None, None, None), ("", "N/A", ""), ("...", "null", "US"), ("Co", "", "India")):
            r = blocking_record(name, addr, country)
            self.assertIsInstance(r, dict)
            self.assertEqual(r["state"], "")
            self.assertEqual(r["hn"], "")
        self.assertEqual(blocking_record("Co", None, "US")["name_keys"], "")   # keys shorter than 3 chars dropped


US = "US"
S1 = [("S1-1", "Wenonah's Metal Works", "181 Farragut Avenue, Hastings-on-hudson, NY", US),
      ("S1-2", "Reliable Asset Group", "2916 Louisiana Avenue, Halethorpe, MD", US),
      ("S1-3", "Sky Supreme Products Private Limited", "Flat No.1, Cantonment Po, Aurangabad, Maharashtra", "India"),
      ("S1-4", "Qzxv Unmatched Holdings", None, US)]
T = [("S2-10", "WENONAH'S METAL WS", "0181 FARRAGUT AVENUE, HASTINGS-ON-HUDSON, NY", US),   # structured
     ("S3-11", "wenonahsmetalworks.com", "Hastings On Hudson, New York, 181B Farragut Avenue", US),  # name key
     ("S2-12", "Reliable Asie Group", "Maryland, Halethorpe, null, 2916 Louisiana Avenue", US),
     ("S3-13", "स्काई सुप्रीम प्रोडक्ट्स प्राइवेट लिमिटेड", "FLAT NO.1, CANTONMENT PO, AURANGABAD, महाराष्ट्र", "India"),
     ("S2-14", "Wenonah's Metal Works", "1 Other Road, Mumbai, Maharashtra", "India"),      # other country!
     ("S2-15", "Unrelated Bakery", "77 Elm Street, Denver, CO", US)]
GT = [("S1-1", "S2-10"), ("S1-1", "S3-11"), ("S1-2", "S2-12"), ("S1-3", "S3-13")]


class TestStrategies(unittest.TestCase):
    def setUp(self):
        self.con = make_con(S1, T, GT)

    def test_country_partitioning(self):
        generate_candidates(self.con, list(STRATEGY_BITS), BlockingConfig(), log=lambda *a: None)
        got = pairs(self.con)
        self.assertNotIn((1, "S2-14"), got)        # same name, different country -> never a candidate
        self.assertFalse(any(t == "S2-15" for _, t in got if _ == 3))

    def test_each_strategy(self):
        generate_candidates(self.con, list(STRATEGY_BITS), BlockingConfig(), log=lambda *a: None)
        self.assertIn((1, "S3-11"), pairs(self.con, "name"))
        self.assertIn((1, "S2-10"), pairs(self.con, "structured"))
        self.assertIn((2, "S2-12"), pairs(self.con, "structured"))
        self.assertIn((3, "S3-13"), pairs(self.con, "phonetic"))

    def test_dedup_and_mask(self):
        generate_candidates(self.con, ["name", "structured", "rare_addr"], BlockingConfig(), log=lambda *a: None)
        n, d = self.con.execute("SELECT COUNT(*), COUNT(DISTINCT (s1, t)) FROM cand").fetchone()
        self.assertEqual(n, d)                      # one row per pair
        m = self.con.execute("SELECT mask FROM cand WHERE s1 = 1 AND t = 2000000010").fetchone()[0]
        self.assertTrue(m & STRATEGY_BITS["structured"] and m & STRATEGY_BITS["rare_addr"])

    def test_recall_evaluation(self):
        generate_candidates(self.con, list(STRATEGY_BITS), BlockingConfig(), log=lambda *a: None)
        ev = evaluate_candidate_recall(self.con)
        self.assertEqual((ev["true_pairs"], ev["recovered"], ev["recall"]), (4, 4, 1.0))
        self.assertEqual(ev["s1_records"], 4)
        ev_name = evaluate_candidate_recall(self.con, mask=STRATEGY_BITS["name"])
        self.assertLess(ev_name["recovered"], 4)
        st = candidate_statistics(self.con)
        self.assertGreaterEqual(st["pct_s1_without_candidates"], 25.0)   # S1-4 has no candidates

    def test_empty_strategy_list(self):
        info = generate_candidates(self.con, [], log=lambda *a: None)
        self.assertEqual(info["_union_pairs"], 0)
        self.assertEqual(evaluate_candidate_recall(self.con)["recall"], 0.0)


class TestGroundTruthLoader(unittest.TestCase):
    def test_load_from_tsv(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gt.tsv")
            with open(path, "w") as f:
                f.write("source1_entity_id\tmatched_entity_ids\n")
                f.write("S1-1\tS2-10, S3-999999999\n")
                f.write("S1-2\t\n")                                  # no matches
                f.write("S1-3\tS3-11\n")
            con = duckdb.connect()
            con.execute("CREATE TABLE ids AS SELECT 1 AS id UNION ALL SELECT 2")
            self.assertEqual(C.load_ground_truth_pairs(con, path), 3)
            self.assertEqual(sorted(con.execute("SELECT s1, t, src FROM gt_pairs").fetchall()),
                             [(1, 2_000_000_010, 2), (1, 3_999_999_999, 3), (3, 3_000_000_011, 3)])
            self.assertEqual(C.load_ground_truth_pairs(con, path, s1_filter_sql="SELECT id FROM ids"), 2)


class TestCapsAndRareTokens(unittest.TestCase):
    def test_name_cap_falls_back_to_state(self):
        # 5 targets called "Eye Group" in 5 states; national cap 3 -> only the same-state one
        t = [(f"S2-{i}", "Eye Group", f"{i} Main St, Springfield, {st}", US)
             for i, st in enumerate(["IL", "OH", "MO", "MA", "OR"], start=1)]
        cfg = BlockingConfig(name_cap=3, name_state_cap=3)
        con = make_con([("S1-1", "Eye Group LLC", "9 Oak Ave, Chicago, IL", US)], t, cfg=cfg)
        C.generate_exact_name_candidates(con, cfg)
        self.assertEqual(con.execute("SELECT t FROM cand_name").fetchall(), [(2_000_000_001,)])
        cfg2 = BlockingConfig(name_cap=3, name_state_cap=0)          # state block too big too -> dropped
        con2 = make_con([("S1-1", "Eye Group LLC", "9 Oak Ave, Chicago, IL", US)], t, cfg=cfg2)
        meta = C.generate_exact_name_candidates(con2, cfg2)
        self.assertEqual(con2.execute("SELECT COUNT(*) FROM cand_name").fetchone()[0], 0)
        self.assertEqual(meta["keys_dropped_over_cap"], 2)          # counted per bucket: IL + no-state

    def test_stateless_fallback(self):
        # target has no address (state = '') -> reachable only through the no-state bucket
        t = [("S3-7", "Zygomatic Holdings", None, US)] + \
            [(f"S2-{i}", f"Zygomatic Other{i}", f"{i} Main St, Austin, TX", US) for i in range(1, 4)]
        s1 = [("S1-1", "Zygomatic Holdings LLC", "5 C St, Dallas, TX", US)]
        for fallback, expected in ((True, {3_000_000_007}), (False, set())):
            cfg = BlockingConfig(stateless_fallback=fallback)
            con = make_con(s1, t, cfg=cfg)
            C.generate_rare_token_candidates(con, cfg, "rare_name")
            got = {r[0] for r in con.execute("SELECT t FROM cand_rare_name").fetchall()}
            self.assertEqual(got & {3_000_000_007}, expected)

    def test_rare_token_selection(self):
        # 'services' is common (4 targets), 'zygomatic' is rare (1 target): with k=1 and cap=2
        # only the rare token is used.
        t = [("S2-1", "Zygomatic Labs", "1 A St, Austin, TX", US)] + \
            [(f"S2-{i}", f"Other{i} Services", f"{i} B St, Austin, TX", US) for i in range(2, 6)]
        cfg = BlockingConfig(token_k=1, token_df_cap=2)
        con = make_con([("S1-1", "Zygomatic Services", "5 C St, Dallas, TX", US)], t, cfg=cfg)
        C.generate_rare_token_candidates(con, cfg, "rare_name")
        self.assertEqual(con.execute("SELECT k FROM qt").fetchall(), [("zygomatic",)])
        self.assertEqual(con.execute("SELECT t FROM cand_rare_name").fetchall(), [(2_000_000_001,)])

    def test_structured_refinement(self):
        # 3 targets at '100' in TX; hn_cap=2 forces refinement by place/street
        t = [("S2-1", "A", "100 Main St, Austin, TX", US), ("S2-2", "B", "100 Oak St, Dallas, TX", US),
             ("S2-3", "C", "100 Elm St, Houston, TX", US)]
        cfg = BlockingConfig(hn_cap=2, hn_refined_cap=5)
        con = make_con([("S1-1", "Z", "100 Oak Street, Plano, Texas", US)], t, cfg=cfg)
        C.generate_structured_candidates(con, cfg)
        self.assertEqual(con.execute("SELECT t FROM cand_structured").fetchall(), [(2_000_000_002,)])

    def test_explosion_guard(self):
        t = [(f"S2-{i}", "Acme", f"{i} Main St, Austin, TX", US) for i in range(1, 30)]
        con = make_con([("S1-1", "Acme", "1 Main St, Austin, TX", US)], t)
        info = generate_candidates(con, ["name"], BlockingConfig(), max_raw_pairs=10, log=lambda *a: None)
        self.assertEqual(info["name"]["status"], "stopped")
        self.assertEqual(info["_union_pairs"], 0)

    def test_chunked_equals_single_run(self):
        con = make_con(S1, T, GT)
        cfg = BlockingConfig(per_s1_cap=None)
        generate_candidates(con, list(STRATEGY_BITS), cfg, log=lambda *a: None)
        single = set(con.execute("SELECT s1, t, mask FROM cand").fetchall())
        con.execute("CREATE TABLE q_all AS SELECT * FROM q")
        C.generate_candidates_chunked(con, list(STRATEGY_BITS), cfg, n_chunks=3, min_free_gb=0, log=lambda *a: None)
        self.assertEqual(set(con.execute("SELECT s1, t, mask FROM cand").fetchall()), single)
        self.assertEqual(con.execute("SELECT COUNT(*) FROM q").fetchone()[0], 4)   # q restored

    def test_chunks_independent_of_hash_sampling(self):
        # a query set sampled with hash(id) % 22 = 0 must still spread over all chunks
        con = duckdb.connect()
        con.execute("CREATE TABLE q_all AS SELECT i AS id FROM range(0, 200000) t(i) WHERE hash(i) % 22 = 0")
        sizes = [r[0] for r in con.execute("SELECT COUNT(*) FROM q_all GROUP BY hash(id, 1) % 10 ORDER BY 1").fetchall()]
        self.assertEqual(len(sizes), 10)
        self.assertGreater(min(sizes), 0.8 * max(sizes))

    def test_per_s1_cap(self):
        t = [(f"S2-{i}", "Acme Widgets", f"{i} Main St, Austin, TX", US) for i in range(1, 6)]
        con = make_con([("S1-1", "Acme Widgets", "3 Main St, Austin, TX", US)], t)
        generate_candidates(con, ["name", "structured"], log=lambda *a: None)
        removed = C.apply_per_s1_cap(con, 2, ["structured", "name"])
        self.assertEqual(removed, 3)
        kept = [r[0] for r in con.execute("SELECT t FROM cand ORDER BY t").fetchall()]
        self.assertIn(2_000_000_003, kept)       # found by both strategies -> kept first


if __name__ == "__main__":
    unittest.main()
