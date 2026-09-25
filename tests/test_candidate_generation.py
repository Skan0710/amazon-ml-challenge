"""Unit tests for src/blocking.py and src/candidate_generation.py (Phase 3).

Every test builds a tiny S1 / target record table with the real ``blocking_record`` and runs
the real DuckDB strategy SQL on it.

Run from the project root:
    python -m pytest tests -q
"""
import unittest

import duckdb
import pyarrow as pa

from src.blocking import (ID_BASE, STRATEGIES, BlockingConfig, blocking_record, entity_rid,
                          rid_to_entity_id, run_strategy)
from src.candidate_generation import _arrow_schema, evaluate, evaluate_pairs, union_candidates


def _con(s1_rows, t_rows):
    """In-memory DuckDB with s1_rec / t_rec built from (entity_id, name, address, country) rows."""
    con = duckdb.connect()
    schema = _arrow_schema()
    s1 = pa.Table.from_pylist([blocking_record(*r) for r in s1_rows], schema=schema)
    t = pa.Table.from_pylist([blocking_record(*r) for r in t_rows], schema=schema)
    con.register("_s1", s1)
    con.register("_t", t)
    con.execute("CREATE TABLE s1_rec AS SELECT * FROM _s1")
    con.execute("CREATE TABLE t_rec AS SELECT * FROM _t")
    return con


def _pairs(con, name, cfg=None):
    run_strategy(con, name, "s1_rec", "t_rec", cfg or BlockingConfig())
    return {(rid_to_entity_id(a), rid_to_entity_id(b)) for a, b in con.execute(f"SELECT s1, t FROM cand_{name}").fetchall()}


class TestIdentifiers(unittest.TestCase):
    def test_roundtrip(self):
        for e in ("S1-965667", "S2-681193310", "S3-0"):
            self.assertEqual(rid_to_entity_id(entity_rid(e)), e)
        self.assertEqual(entity_rid("S2-681193310"), 2 * ID_BASE + 681193310)

    def test_bad_format(self):
        for e in ("X1-5", "S1-", "S1-12a", "S12-3", "S1-" + "9" * 11):
            with self.assertRaises(ValueError):
                entity_rid(e)


class TestBlockingRecord(unittest.TestCase):
    def test_uses_phase2_representations(self):
        r = blocking_record("S1-1", "Wonderland Energy Pvt. Ltd.", "00123 YEAGER RD, COALLTON, WV", "US")
        self.assertEqual(r["name_core"], "wonderland energy")
        self.assertEqual(r["name_canon"], "wonderland energy private limited")
        self.assertEqual((r["addr_state"], r["addr_house"]), ("wv", "123"))
        self.assertIn("#wo", r["name_grams"])
        self.assertEqual(len(r["name_grams"]), len(set(r["name_grams"])))   # distinct

    def test_missing_values_become_none(self):
        r = blocking_record("S2-7", "Acme", None, "India")
        self.assertTrue(r["addr_missing"])
        for col in ("addr_state", "addr_house", "addr_street", "addr_city", "addr_norm", "name_web"):
            self.assertIsNone(r[col], col)
        self.assertEqual(r["addr_places"], [])
        r = blocking_record("S2-8", "   ", "N/A", "US")
        self.assertIsNone(r["name_core"])
        self.assertEqual(r["name_tokens"], [])

    def test_country_kept_verbatim(self):
        self.assertEqual(blocking_record("S1-1", "a", "b", " Canada ")["country"], "Canada")


class TestCountryPartition(unittest.TestCase):
    def test_no_cross_country_pairs(self):
        con = _con([("S1-1", "Sai Traders", "12 MG Road, Pune, Maharashtra", "India")],
                   [("S2-1", "Sai Traders", "12 MG Road, Pune, MH", "India"),
                    ("S2-2", "Sai Traders", "12 Main St, Austin, TX", "US")])
        for name in STRATEGIES:
            pairs = _pairs(con, name)
            self.assertNotIn(("S1-1", "S2-2"), pairs, name)
        self.assertIn(("S1-1", "S2-1"), _pairs(con, "exact_name"))

    def test_new_country_value_works(self):
        con = _con([("S1-1", "Maple Leaf Foods", "5 King St, Toronto, ON", "Canada")],
                   [("S2-1", "Maple Leaf Foods", "5 King St, Toronto, ON", "Canada"),
                    ("S2-2", "Maple Leaf Foods", "5 King St, Toronto, ON", "US")])
        self.assertEqual(_pairs(con, "core_name"), {("S1-1", "S2-1")})


class TestNameStrategies(unittest.TestCase):
    def setUp(self):
        self.con = _con(
            [("S1-1", "Wonderland Energy Pvt. Ltd.", "4 Park St, Kolkata, West Bengal", "India"),
             ("S1-2", "Wenonah's Metal Works", "9 Elm St, Austin, TX", "US"),
             ("S1-3", "Indian International Private Limited", "7 Ring Rd, Delhi, Delhi", "India")],
            [("S2-1", "WONDERLAND ENERGY PRIVATE LIMITED", "4 PARK ST, KOLKATA, পশ্চিমবঙ্গ", "India"),
             ("S3-1", "Private Wonderland Energy Ltd", None, "India"),
             ("S3-2", "Wonderland Energy LLP", "1 Other Rd, Pune, MH", "India"),
             ("S2-2", "wenonahsmetalworks.com", "9 ELM ST, AUSTIN, TX", "US"),
             ("S3-3", "Quodova a/k/a Indian International Private Limited", "Delhi, DL", "India")])

    def test_exact_name(self):
        pairs = _pairs(self.con, "exact_name")
        self.assertIn(("S1-1", "S2-1"), pairs)          # pvt ltd == private limited after canonicalisation
        self.assertNotIn(("S1-1", "S3-2"), pairs)       # different legal form -> not an exact name

    def test_core_name(self):
        pairs = _pairs(self.con, "core_name")
        self.assertTrue({("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-1", "S3-2")} <= pairs)

    def test_website(self):
        self.assertEqual(_pairs(self.con, "website"), {("S1-2", "S2-2")})

    def test_alias(self):
        self.assertIn(("S1-3", "S3-3"), _pairs(self.con, "alias"))


class TestPhonetic(unittest.TestCase):
    def test_cross_script_and_word_order(self):
        con = _con([("S1-1", "Guru Technology Private Limited", "Kolkata, WB", "India"),
                    ("S1-2", "Wonderland Energy LLC", "1 A St, Reno, NV", "US")],
                   [("S2-1", "গুরু টেকনোলজি প্রাইভেট লিমিটেড", "KOLKATA, পশ্চিমবঙ্গ", "India"),
                    ("S3-1", "Energy Wonderland LLC", "1 A St, Reno, Nevada", "US"),
                    ("S3-2", "Garden Tools LLC", "1 A St, Reno, Nevada", "US")])
        pairs = _pairs(con, "phonetic")
        # 'guru' and Bengali গুরু share a phonetic token, although the full keys differ
        # ('guru teknolji' vs 'guru technology'): the shared-phonetic-token leg finds it.
        self.assertIn(("S1-1", "S2-1"), pairs)
        self.assertIn(("S1-2", "S3-1"), pairs)          # word-order swap: whole-key leg
        self.assertNotIn(("S1-2", "S3-2"), pairs)


class TestAddressStrategies(unittest.TestCase):
    def setUp(self):
        self.con = _con(
            [("S1-1", "Alpha", "00123 YEAGER RD, COALLTON, WV", "US"),
             ("S1-2", "Beta", "Friends Colony, Sector 14, Gurgaon, Haryana", "India"),
             ("S1-3", "Gamma", "Coallton, WV", "US")],                       # no house number
            [("S2-1", "Totally Different", "123 Yeager Road, Coalton, West Virginia", "US"),
             ("S2-2", "Other", "123 Main St, Denver, CO", "US"),               # same number, other state
             ("S3-1", "Beta Ent", "SECTOR 14, GURGAON, हरियाणा", "India"),
             ("S3-2", "Delta", "Mack Rd, Coallton, West Virginia", "US")])     # no house number

    def test_state_house(self):
        pairs = _pairs(self.con, "state_house")
        self.assertIn(("S1-1", "S2-1"), pairs)
        self.assertNotIn(("S1-1", "S2-2"), pairs)

    def test_missing_house_number_never_blocks(self):
        # (wv, NULL) must not become a block: S1-3 and S3-2 share only the state
        pairs = _pairs(self.con, "state_house")
        self.assertFalse({p for p in pairs if "S1-3" in p or "S3-2" in p})

    def test_place_number(self):
        self.assertIn(("S1-2", "S3-1"), _pairs(self.con, "place_number"))

    def test_missing_address(self):
        con = _con([("S1-1", "Alpha", None, "US")], [("S2-1", "Beta", None, "US")])
        for name in ("state_house", "place_number"):
            self.assertEqual(_pairs(con, name), set(), name)


class TestRareToken(unittest.TestCase):
    def test_rare_vs_common_tokens(self):
        t_rows = [(f"S2-{i}", f"Sunrise Services {i}", f"{i} Main St, Austin, TX", "US") for i in range(1, 8)]
        t_rows.append(("S3-1", "Quixotic Services", "5 Oak St, Dallas, TX", "US"))
        con = _con([("S1-1", "Quixotic Services Inc", "9 Pine St, Waco, TX", "US")], t_rows)
        cfg = BlockingConfig(rare_token_max_df=3)
        pairs = _pairs(con, "rare_token", cfg)
        # 'services' has df 8 > 3 in the whole of TX -> dropped; 'quixotic' (df 1) is used
        self.assertEqual(pairs, {("S1-1", "S3-1")})

    def test_common_token_refined_by_state(self):
        t_rows = [(f"S2-{i}", f"Sunrise Services {i}", f"{i} Main St, Austin, TX", "US") for i in range(1, 8)]
        t_rows.append(("S3-1", "Sunrise Bakery", "5 Oak St, Reno, NV", "US"))
        con = _con([("S1-1", "Sunrise Cafe", "9 Pine St, Reno, NV", "US")], t_rows)
        pairs = _pairs(con, "rare_token", BlockingConfig(rare_token_max_df=3))
        # 'sunrise' has df 8 overall but only 1 in NV -> the refined (token, state) block is used
        self.assertEqual(pairs, {("S1-1", "S3-1")})

    def test_short_tokens_ignored(self):
        con = _con([("S1-1", "A B Holdings", "1 X St, Reno, NV", "US")],
                   [("S2-1", "A Zed", "2 Y St, Reno, NV", "US")])
        self.assertEqual(_pairs(con, "rare_token"), set())


class TestTrigram(unittest.TestCase):
    def test_typo_retrieved_unrelated_not(self):
        con = _con([("S1-1", "Holloway Peak Seafood", "1 Elm St, Morganton, NC", "US")],
                   [("S2-1", "Hollowya Peak Seafood Inc", "1 ELM ST, MORGANTON, NC", "US"),
                    ("S3-1", "H0lloway Peak Seaf00d", "Morganton, NC", "US"),
                    ("S3-2", "Cedar Ridge Dental", "1 Elm St, Morganton, NC", "US")])
        pairs = _pairs(con, "trigram")
        self.assertIn(("S1-1", "S2-1"), pairs)
        self.assertIn(("S1-1", "S3-1"), pairs)
        self.assertNotIn(("S1-1", "S3-2"), pairs)

    def test_top_k_limits_candidates(self):
        t_rows = [(f"S2-{i}", f"Holloway Peak Seafood {i}", "1 Elm St, Morganton, NC", "US") for i in range(10)]
        con = _con([("S1-1", "Holloway Peak Seafood", "1 Elm St, Morganton, NC", "US")], t_rows)
        cfg = BlockingConfig(trigram_top_k=3, trigram_batches=2)
        self.assertEqual(len(_pairs(con, "trigram", cfg)), 3)


class TestCommonKeyProtection(unittest.TestCase):
    def setUp(self):
        states = ["Texas"] * 6 + ["Nevada"] * 2 + ["Ohio"] * 4
        self.con = _con([("S1-1", "Eye Group", "1 A St, Reno, NV", "US"),
                         ("S1-2", "Eye Group", "1 A St, Dayton, OH", "US")],
                        [(f"S2-{i}", "Eye Group", f"{i} B St, City, {s}", "US") for i, s in enumerate(states)])

    def test_oversized_key_refined_or_dropped(self):
        cfg = BlockingConfig(max_block_name=3)
        run = run_strategy(self.con, "core_name", "s1_rec", "t_rec", cfg)
        pairs = {(a % ID_BASE, b % ID_BASE) for a, b in self.con.execute("SELECT s1, t FROM cand_core_name").fetchall()}
        # the 'eye group' key has 12 targets > 3, so it is refined by state: the NV block (2
        # targets) is kept, the OH (4) and TX (6) blocks are still too large and are dropped.
        # S1-1 is in NV and S1-2 in OH, so only S1-1 gets candidates.
        self.assertEqual(pairs, {(1, 6), (1, 7)})
        self.assertEqual(run["keys_oversized"], 1)
        self.assertEqual(run["refined_blocks_kept"], 1)
        self.assertEqual(run["refined_blocks_dropped"], 2)
        self.assertEqual(run["largest_dropped_block"], 12)

    def test_block_within_cap_is_used_whole(self):
        run_strategy(self.con, "core_name", "s1_rec", "t_rec", BlockingConfig(max_block_name=50))
        self.assertEqual(self.con.execute("SELECT count(*) FROM cand_core_name").fetchone()[0], 24)

    def test_per_s1_candidates_bounded(self):
        run_strategy(self.con, "core_name", "s1_rec", "t_rec", BlockingConfig(max_block_name=3))
        mx = self.con.execute("SELECT coalesce(max(n), 0) FROM (SELECT s1, count(*) n FROM cand_core_name GROUP BY s1)").fetchone()[0]
        self.assertLessEqual(mx, 3)


class TestUnionAndEvaluation(unittest.TestCase):
    def setUp(self):
        self.con = _con([("S1-1", "Wonderland Energy Pvt Ltd", "4 Park St, Kolkata, West Bengal", "India"),
                         ("S1-2", "Blue Fin Sushi", "8 Bay Rd, Tampa, FL", "US")],
                        [("S2-1", "Wonderland Energy Private Limited", "4 PARK ST, KOLKATA, WB", "India"),
                         ("S3-1", "Unrelated Name", "4 Park Street, Kolkata, WB", "India"),
                         ("S3-2", "Red Fox Tavern", "99 Oak Ave, Miami, FL", "US")])
        self.names = ["exact_name", "core_name", "state_house"]
        for n in self.names:
            run_strategy(self.con, n, "s1_rec", "t_rec", BlockingConfig())
        self.con.execute(f"""CREATE TABLE gt AS SELECT * FROM (VALUES
            ({entity_rid('S1-1')}, {entity_rid('S2-1')}), ({entity_rid('S1-1')}, {entity_rid('S3-1')}),
            ({entity_rid('S1-2')}, {entity_rid('S3-2')})) v(s1, t)""")

    def test_deduplication(self):
        n = union_candidates(self.con, self.names)
        rows = self.con.execute("SELECT s1, t, mask FROM cand_all").fetchall()
        self.assertEqual(n, len(rows))
        self.assertEqual(len({(a, b) for a, b, _ in rows}), len(rows))            # one row per pair
        mask = {(a, b): m for a, b, m in rows}[(entity_rid("S1-1"), entity_rid("S2-1"))]
        self.assertEqual(mask, 0b111)                                             # found by all three
        for name in self.names:                                                   # each strategy table distinct
            c, d = self.con.execute(f"SELECT count(*), count(DISTINCT (s1, t)) FROM cand_{name}").fetchone()
            self.assertEqual(c, d)

    def test_recall_precision(self):
        union_candidates(self.con, self.names)
        res = evaluate(self.con, self.names, verbose=False)
        self.assertEqual(res["total_true_pairs"], 3)
        self.assertEqual(res["union"]["true_found"], 2)                           # S3-2 is unreachable
        self.assertAlmostEqual(res["union"]["recall_pct"], 66.667, places=2)
        self.assertEqual(res["union"]["candidates"], 2)
        self.assertEqual(res["union"]["precision_pct"], 100.0)
        self.assertEqual(res["per_strategy"]["exact_name"]["true_found"], 1)
        self.assertEqual([r["recall_pct"] for r in res["incremental"]], [33.333, 33.333, 66.667])
        self.assertEqual(res["unique"]["state_house"]["true_pairs_only_this"], 1)
        # per-S1 statistics count S1 records without candidates as 0
        m = evaluate_pairs(self.con, "SELECT s1, t FROM cand_exact_name")
        self.assertEqual((m["avg_per_s1"], m["max_per_s1"], m["s1_with_candidates_pct"]), (0.5, 1, 50.0))


if __name__ == "__main__":
    unittest.main()
