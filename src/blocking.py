"""
Blocking building blocks for Phase 3 candidate generation.

Two layers
----------
1. **Per-record blocking fields** (:func:`blocking_record`): the compact subset of the
   Phase 2 representations that the blockers (and later feature engineering) need. All
   normalisation is delegated to :mod:`src.normalization`; nothing is re-implemented here.
   Missing / empty values become ``None`` so they can never form a blocking key.

2. **Blocking strategies in DuckDB** (:func:`capped_key_join`, :func:`trigram_join` and the
   ``STRATEGIES`` table). Every strategy turns records into ``(country, key)`` rows for
   S1 and for the targets (S2 ∪ S3), joins them and writes a *deduplicated* pair table
   ``(s1 BIGINT, t BIGINT)``. Country is part of every key, so every strategy is
   country-partitioned (strategy J) and a cross-country pair can never be produced.

Common-key protection
---------------------
For each key the number of distinct *target* records ``n_t`` is counted before joining.

* ``n_t <= max_block``: the block is used as is.
* ``n_t >  max_block``: the key is **refined** with a second field (usually the state; for
  the house-number key the street / places) and the refined block is used only if it now
  has ``<= max_block`` targets.
* otherwise the key is **dropped**, and its size is logged in the returned stats.

So an S1 record can receive at most ``max_block`` candidates from one key, and no key can
produce millions of pairs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .normalization import char_ngrams, normalize_address, normalize_name, transliterate_text

__all__ = [
    "ID_BASE", "entity_rid", "rid_to_entity_id", "blocking_record", "RECORD_SCHEMA",
    "BlockingConfig", "Leg", "capped_key_join", "trigram_join", "STRATEGIES", "strategy_legs",
    "run_strategy",
]

# =============================================================================
# Record identifiers
# =============================================================================

#: ``S<src>-<number>`` is encoded as ``src * ID_BASE + number`` (one int64). Training IDs
#: have at most 9 digits, so there is a 10x safety margin.
ID_BASE = 10 ** 10


def entity_rid(entity_id: str) -> int:
    """``'S2-681193310'`` -> ``20681193310``. Raises ``ValueError`` on any other format."""
    prefix, _, num = entity_id.strip().partition("-")
    if len(prefix) != 2 or prefix[0] not in "sS" or not prefix[1].isdigit() or not num.isdigit():
        raise ValueError(f"unexpected entity_id format: {entity_id!r}")
    n = int(num)
    if n >= ID_BASE:
        raise ValueError(f"entity_id number too large: {entity_id!r}")
    return int(prefix[1]) * ID_BASE + n


def rid_to_entity_id(rid: int) -> str:
    """Inverse of :func:`entity_rid`."""
    return f"S{rid // ID_BASE}-{rid % ID_BASE}"


# =============================================================================
# Per-record blocking fields (built from the Phase 2 representations)
# =============================================================================

#: Column -> DuckDB type of the materialised record table.
RECORD_SCHEMA: dict[str, str] = {
    "rid": "BIGINT", "src": "TINYINT", "country": "VARCHAR",
    "name_canon": "VARCHAR",        # transliterated canonical name (legal forms canonicalised, kept)
    "name_core": "VARCHAR",         # NameRepr.core_latin  (legal forms / junk removed, Latin)
    "name_compact": "VARCHAR",      # NameRepr.compact     (core_latin without spaces / 'and')
    "name_web": "VARCHAR",          # NameRepr.website_label (None unless the name is a domain)
    "name_aliases": "VARCHAR[]",    # NameRepr.aliases     (core_latin of each alias side)
    "name_phon": "VARCHAR",         # NameRepr.phonetic    (phonetic_key of core_latin)
    "name_tokens": "VARCHAR[]",     # distinct core_latin tokens
    "name_grams": "VARCHAR[]",      # distinct char_ngrams(core_latin, 3)
    "name_script": "VARCHAR",       # NameRepr.script
    "name_legal": "VARCHAR",        # '|'.join(NameRepr.legal_forms)
    "addr_state": "VARCHAR",        # AddressRepr.state
    "addr_house": "VARCHAR",        # AddressRepr.house_number_core
    "addr_street": "VARCHAR",       # AddressRepr.street
    "addr_city": "VARCHAR",         # AddressRepr.city
    "addr_places": "VARCHAR[]",     # AddressRepr.places
    "addr_numbers": "VARCHAR[]",    # AddressRepr.numbers
    "addr_norm": "VARCHAR",         # AddressRepr.normalized
    "addr_missing": "BOOLEAN",      # AddressRepr.is_missing
}


def _nz(s: Optional[str]) -> Optional[str]:
    return s if s else None


def _distinct(items) -> list[str]:
    return list(dict.fromkeys(i for i in items if i))


def blocking_record(entity_id: str, name: object, address: object, country: object) -> dict:
    """Blocking fields of one record (see :data:`RECORD_SCHEMA`).

    ``country`` is kept verbatim (only stripped) so that partitioning follows whatever
    country values occur in the data rather than a hard-coded list.
    """
    c = str(country).strip() if country is not None and str(country).strip() else None
    n = normalize_name(name)
    a = normalize_address(address, c)
    return {
        "rid": entity_rid(entity_id),
        "src": int(entity_id.strip()[1]),
        "country": c,
        "name_canon": _nz(transliterate_text(n.canonical)),
        "name_core": _nz(n.core_latin),
        "name_compact": _nz(n.compact),
        "name_web": _nz(n.website_label),
        "name_aliases": _distinct(n.aliases),
        "name_phon": _nz(n.phonetic),
        "name_tokens": _distinct(n.core_latin.split()),
        "name_grams": _distinct(char_ngrams(n.core_latin, 3)),
        "name_script": n.script,
        "name_legal": "|".join(n.legal_forms),
        "addr_state": _nz(a.state),
        "addr_house": _nz(a.house_number_core),
        "addr_street": _nz(a.street),
        "addr_city": _nz(a.city),
        "addr_places": _distinct(a.places),
        "addr_numbers": _distinct(a.numbers),
        "addr_norm": _nz(a.normalized),
        "addr_missing": a.is_missing,
    }


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BlockingConfig:
    """Tunable parameters. Defaults are the values selected in the Phase 3 experiments
    (see ``experiments/phase3_candidate_generation_report.md``)."""
    #: max distinct targets per (country, key) block for the name / phonetic / alias keys
    max_block_name: int = 200
    #: max distinct targets per block for the address keys (E, F)
    max_block_addr: int = 200
    #: max document frequency (distinct targets) for a name token to count as "rare"
    rare_token_max_df: int = 200
    #: minimum token length for rare-token blocking (1-letter initials are never keys)
    rare_token_min_len: int = 2
    #: 3-gram retrieval: number of rarest grams indexed per record (prefix-filter length)
    trigram_prefix: int = 4
    #: 3-gram retrieval: grams whose target df exceeds this are never indexed
    trigram_max_df: int = 20_000
    #: 3-gram retrieval: minimum trigram Jaccard to keep a retrieved pair
    trigram_min_jaccard: float = 0.35
    #: 3-gram retrieval: keep at most this many targets per S1 record (best Jaccard first)
    trigram_top_k: int = 20
    #: number of S1 hash-batches for the 3-gram join (bounds DuckDB memory)
    trigram_batches: int = 8
    #: strategy name -> max_block override
    overrides: dict = field(default_factory=dict)

    def max_block(self, strategy: str) -> int:
        if strategy in self.overrides:
            return self.overrides[strategy]
        if strategy in ("state_house", "place_number"):
            return self.max_block_addr
        if strategy == "rare_token":
            return self.rare_token_max_df
        return self.max_block_name


# =============================================================================
# Capped key join
# =============================================================================

@dataclass(frozen=True)
class Leg:
    """One S1-keys ↔ target-keys join. Both SQL snippets must return
    ``(rid BIGINT, country VARCHAR, key VARCHAR, refine VARCHAR)``; ``refine`` may be NULL
    (then an oversized key is simply dropped)."""
    s1_sql: str
    t_sql: str


def capped_key_join(con, out_table: str, legs: list[Leg], max_block: int) -> dict:
    """Run every leg with common-key protection and write ``out_table(s1, t)`` (distinct).

    Returns statistics about the keys and the capping decisions.
    """
    stats = {"legs": len(legs), "keys": 0, "keys_oversized": 0, "refined_blocks_kept": 0,
             "refined_blocks_dropped": 0, "keys_dropped_no_refine": 0,
             "largest_dropped_block": 0, "max_block": max_block}
    parts = []
    for i, leg in enumerate(legs):
        s, t = f"_blk_s{i}", f"_blk_t{i}"
        con.execute(f"CREATE OR REPLACE TEMP TABLE {s} AS SELECT DISTINCT * FROM ({leg.s1_sql}) "
                    f"WHERE key IS NOT NULL AND key <> '' AND country IS NOT NULL")
        con.execute(f"CREATE OR REPLACE TEMP TABLE {t} AS SELECT DISTINCT * FROM ({leg.t_sql}) "
                    f"WHERE key IS NOT NULL AND key <> '' AND country IS NOT NULL")
        # block sizes (distinct targets per key), only for keys that S1 actually uses
        con.execute(f"""CREATE OR REPLACE TEMP TABLE _blk_kc{i} AS
            SELECT country, key, count(DISTINCT rid) n FROM {t}
            WHERE (country, key) IN (SELECT DISTINCT country, key FROM {s}) GROUP BY ALL""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE _blk_rc{i} AS
            SELECT country, key, refine, count(DISTINCT rid) n FROM {t}
            WHERE refine IS NOT NULL AND (country, key) IN (SELECT country, key FROM _blk_kc{i} WHERE n > {max_block})
            GROUP BY ALL""")
        k = con.execute(f"""SELECT count(*), count(*) FILTER (WHERE n > {max_block}),
                              coalesce(max(n) FILTER (WHERE n > {max_block}), 0) FROM _blk_kc{i}""").fetchone()
        r = con.execute(f"""SELECT count(*) FILTER (WHERE n <= {max_block}), count(*) FILTER (WHERE n > {max_block}),
                              coalesce(max(n) FILTER (WHERE n > {max_block}), 0) FROM _blk_rc{i}""").fetchone()
        nr = con.execute(f"""SELECT count(*) FROM _blk_kc{i} k WHERE n > {max_block} AND NOT EXISTS
                               (SELECT 1 FROM _blk_rc{i} r WHERE r.country = k.country AND r.key = k.key AND r.n <= {max_block})""").fetchone()
        stats["keys"] += k[0]
        stats["keys_oversized"] += k[1]
        stats["refined_blocks_kept"] += r[0]
        stats["refined_blocks_dropped"] += r[1]
        stats["keys_dropped_no_refine"] += nr[0]
        stats["largest_dropped_block"] = max(stats["largest_dropped_block"], k[2])
        parts.append(f"""SELECT s.rid AS s1, t.rid AS t FROM (SELECT DISTINCT rid, country, key FROM {s}) s
                           JOIN _blk_kc{i} k ON k.country = s.country AND k.key = s.key AND k.n <= {max_block}
                           JOIN (SELECT DISTINCT rid, country, key FROM {t}) t
                             ON t.country = s.country AND t.key = s.key""")
        parts.append(f"""SELECT s.rid AS s1, t.rid AS t FROM {s} s
                           JOIN _blk_rc{i} r ON r.country = s.country AND r.key = s.key AND r.refine = s.refine
                                         AND r.n <= {max_block}
                           JOIN {t} t ON t.country = s.country AND t.key = s.key AND t.refine = s.refine""")
    con.execute(f"CREATE OR REPLACE TABLE {out_table} AS SELECT DISTINCT s1, t FROM ({' UNION ALL '.join(parts)})")
    for i in range(len(legs)):
        for tb in (f"_blk_s{i}", f"_blk_t{i}", f"_blk_kc{i}", f"_blk_rc{i}"):
            con.execute(f"DROP TABLE IF EXISTS {tb}")
    return stats


# =============================================================================
# Strategy key definitions (A-H). ``{S1}`` / ``{T}`` are replaced by the S1 / target
# record relations (tables or views with the RECORD_SCHEMA columns).
# =============================================================================

_SORTED_PHON = "array_to_string(list_sort(string_split(name_phon, ' ')), ' ')"
# street and places are both refinements of an oversized (state, house number) block; a record
# with neither still emits its key once (refine NULL) so small blocks never lose it
_ADDR_REFINE = """unnest(CASE WHEN len(list_filter(list_prepend(addr_street, addr_places), x -> x IS NOT NULL)) = 0
                      THEN [NULL::VARCHAR]
                      ELSE list_distinct(list_filter(list_prepend(addr_street, addr_places), x -> x IS NOT NULL)) END)"""


def _k(rel: str, key: str, refine: str = "NULL", where: str = "TRUE") -> str:
    return f"SELECT rid, country, CAST({key} AS VARCHAR) AS key, CAST({refine} AS VARCHAR) AS refine FROM {rel} WHERE {where}"


def strategy_legs(name: str, cfg: BlockingConfig) -> list[Leg]:
    """The join legs of the key-based strategies A-H."""
    S1, T = "{S1}", "{T}"
    if name == "exact_name":          # A: full name, legal forms canonicalised, transliterated
        return [Leg(_k(S1, "name_canon", "addr_state"), _k(T, "name_canon", "addr_state"))]
    if name == "core_name":           # B: core name (legal forms, junk, 'the', 'm/s' removed)
        return [Leg(_k(S1, "name_core", "addr_state"), _k(T, "name_core", "addr_state"))]
    if name == "alias":               # C: any alias side vs the other record's core name / aliases
        s1_variants = f"""SELECT rid, country, v AS key, addr_state AS refine FROM
                          (SELECT rid, country, addr_state, unnest(list_prepend(name_core, name_aliases)) v FROM {S1})"""
        t_aliases = f"SELECT rid, country, unnest(name_aliases) AS key, addr_state AS refine FROM {T}"
        s1_aliases = f"SELECT rid, country, unnest(name_aliases) AS key, addr_state AS refine FROM {S1}"
        return [Leg(s1_variants, t_aliases), Leg(s1_aliases, _k(T, "name_core", "addr_state"))]
    if name == "website":             # D: website label vs the other record's compact name
        return [Leg(_k(S1, "name_compact", "addr_state"), _k(T, "name_web", "addr_state", "name_web IS NOT NULL")),
                Leg(_k(S1, "name_web", "addr_state", "name_web IS NOT NULL"), _k(T, "name_compact", "addr_state"))]
    if name == "state_house":         # E: (state, house number core); missing parts -> no key
        sql = f"""SELECT rid, country, key, CAST(refine AS VARCHAR) AS refine FROM
                  (SELECT rid, country, addr_state || '|' || addr_house AS key, {_ADDR_REFINE} AS refine FROM {{R}}
                   WHERE addr_state IS NOT NULL AND addr_house IS NOT NULL)"""
        return [Leg(sql.replace("{R}", S1), sql.replace("{R}", T))]
    if name == "place_number":        # F: (state, place, number) for every place x number
        sql = """SELECT rid, country, addr_state || '|' || p || '|' || n AS key, NULL AS refine FROM
                 (SELECT rid, country, addr_state, unnest(addr_places) p FROM {R}
                  WHERE addr_state IS NOT NULL AND len(addr_places) > 0 AND len(addr_numbers) > 0) pl
                 JOIN (SELECT rid, unnest(addr_numbers) n FROM {R}) nu USING (rid)"""
        return [Leg(sql.replace("{R}", S1), sql.replace("{R}", T))]
    if name == "phonetic":            # G: whole phonetic key (order-insensitive) + shared phonetic token
        # Exact phonetic-string equality alone is weak across scripts (Phase 2 measured only 3.3%
        # exact core_latin equality for Indic-vs-Latin pairs), so a *shared phonetic token* leg is
        # added. Both legs are capped and state-refined like every other key.
        tok = f"""SELECT rid, country, t AS key, addr_state AS refine FROM
                  (SELECT rid, country, addr_state, unnest(string_split(name_phon, ' ')) t FROM {{R}}
                   WHERE name_phon IS NOT NULL) WHERE length(t) >= {cfg.rare_token_min_len}"""
        return [Leg(_k(S1, _SORTED_PHON, "addr_state", "name_phon IS NOT NULL"),
                    _k(T, _SORTED_PHON, "addr_state", "name_phon IS NOT NULL")),
                Leg(tok.replace("{R}", S1), tok.replace("{R}", T))]
    if name == "rare_token":          # H: any shared core-name token whose block is small enough
        sql = f"""SELECT rid, country, tok AS key, addr_state AS refine FROM
                  (SELECT rid, country, addr_state, unnest(name_tokens) tok FROM {{R}})
                  WHERE length(tok) >= {cfg.rare_token_min_len}"""
        return [Leg(sql.replace("{R}", S1), sql.replace("{R}", T))]
    raise KeyError(name)


# =============================================================================
# I: character 3-gram retrieval (prefix filtering + Jaccard top-k)
# =============================================================================

def trigram_join(con, out_table: str, s1_rel: str, t_rel: str, cfg: BlockingConfig) -> dict:
    """Character-3-gram retrieval on ``name_grams`` without any all-pairs comparison.

    1. ``df(country, gram)`` = number of target records containing the gram.
    2. Every record indexes only its ``trigram_prefix`` *rarest* grams (global order: df, gram),
       and grams with ``df > trigram_max_df`` are never indexed. By the prefix-filtering
       principle, two gram sets that differ in fewer than ``trigram_prefix`` grams on the
       larger side always share an indexed gram, so single typos / swaps are retrieved.
    3. Pairs sharing an indexed gram (same country) are scored with the Jaccard similarity
       of their *full* gram sets; pairs with Jaccard < ``trigram_min_jaccard`` are dropped
       and only the ``trigram_top_k`` best targets per S1 record are kept.
    Step 3 runs in ``trigram_batches`` S1 hash batches so DuckDB memory stays bounded.
    """
    P, B = cfg.trigram_prefix, cfg.trigram_batches
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _blk_gdf AS
        SELECT country, g, count(*) df FROM (SELECT country, unnest(name_grams) g FROM {t_rel}) GROUP BY ALL""")

    def prefix(rel: str, tbl: str) -> None:
        con.execute(f"""CREATE OR REPLACE TEMP TABLE {tbl} AS
            SELECT rid, country, g FROM (
              SELECT r.rid, r.country, r.g, coalesce(d.df, 0) df,
                     row_number() OVER (PARTITION BY r.rid ORDER BY coalesce(d.df, 0), r.g) rn
              FROM (SELECT rid, country, unnest(name_grams) g FROM {rel}) r
              LEFT JOIN _blk_gdf d ON d.country = r.country AND d.g = r.g)
            WHERE rn <= {P} AND df <= {cfg.trigram_max_df}""")
    prefix(s1_rel, "_blk_ps")
    prefix(t_rel, "_blk_pt")
    con.execute(f"CREATE OR REPLACE TABLE {out_table} (s1 BIGINT, t BIGINT)")
    n_raw = 0
    for b in range(B):
        con.execute(f"""CREATE OR REPLACE TEMP TABLE _blk_tp AS
            SELECT DISTINCT s.rid s1, t.rid t FROM _blk_ps s JOIN _blk_pt t ON t.country = s.country AND t.g = s.g
            WHERE hash(s.rid) % {B} = {b}""")
        n_raw += con.execute("SELECT count(*) FROM _blk_tp").fetchone()[0]
        con.execute(f"""INSERT INTO {out_table}
            SELECT s1, t FROM (
              SELECT p.s1, p.t, len(list_intersect(a.name_grams, b.name_grams)) AS inter,
                     len(a.name_grams) + len(b.name_grams) AS tot
              FROM _blk_tp p JOIN {s1_rel} a ON a.rid = p.s1 JOIN {t_rel} b ON b.rid = p.t)
            WHERE inter / (tot - inter) >= {cfg.trigram_min_jaccard}
            QUALIFY row_number() OVER (PARTITION BY s1 ORDER BY inter / (tot - inter) DESC, t) <= {cfg.trigram_top_k}""")
    for tb in ("_blk_gdf", "_blk_ps", "_blk_pt", "_blk_tp"):
        con.execute(f"DROP TABLE IF EXISTS {tb}")
    return {"prefix": P, "max_df": cfg.trigram_max_df, "min_jaccard": cfg.trigram_min_jaccard,
            "top_k": cfg.trigram_top_k, "prefix_pairs_before_scoring": n_raw}


#: Strategy name -> (letter, description), in the order used for incremental recall.
STRATEGIES: dict[str, tuple[str, str]] = {
    "exact_name":   ("A", "exact normalised full name (legal forms canonicalised, transliterated)"),
    "core_name":    ("B", "exact core name (core_latin)"),
    "alias":        ("C", "alias side <-> core name / alias"),
    "website":      ("D", "website label <-> compact name"),
    "state_house":  ("E", "(state, house-number core); street/place refinement for large blocks"),
    "place_number": ("F", "(state, place, number)"),
    "phonetic":     ("G", "order-insensitive phonetic key; state refinement for large blocks"),
    "rare_token":   ("H", "shared core-name token with target df <= cap; state refinement"),
    "trigram":      ("I", "char 3-gram prefix-filter retrieval, Jaccard-scored top-k"),
}


def run_strategy(con, name: str, s1_rel: str, t_rel: str, cfg: BlockingConfig,
                 out_table: Optional[str] = None) -> dict:
    """Build ``cand_<name>(s1, t)`` for one strategy; returns its capping / retrieval stats."""
    out = out_table or f"cand_{name}"
    if name == "trigram":
        return trigram_join(con, out, s1_rel, t_rel, cfg)
    legs = [Leg(l.s1_sql.replace("{S1}", s1_rel).replace("{T}", t_rel),
                l.t_sql.replace("{S1}", s1_rel).replace("{T}", t_rel)) for l in strategy_legs(name, cfg)]
    return capped_key_join(con, out, legs, cfg.max_block(name))
