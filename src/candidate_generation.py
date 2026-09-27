"""
Phase 3 — candidate generation (blocking) for S1 -> S2 ∪ S3 entity resolution.

Architecture
------------
1. **Normalise once, streaming** (:func:`materialize_blocking_features`). Each source TSV is read
   in batches by DuckDB, normalised with the Phase 2 functions in a small process pool, and
   written as compact Parquet parts (integer ids + blocking fields only). Raw text stays in
   the TSVs.
2. **Blocking statistics** (:func:`build_blocking_statistics`). Key and token document
   frequencies are computed with DuckDB aggregations over *source records only* — the
   ground truth is never used to build keys.
3. **Candidate generation** (``generate_*_candidates``). Every strategy is a bounded SQL join
   between the query S1 records and target key postings, partitioned by country (and state
   where useful), with a frequency cap on every key so no block can explode. Strategies
   return ``(s1_id, t_id)`` rows; :func:`generate_candidates` unions them into one
   deduplicated table with a bitmask saying which strategies produced each pair.
4. **Evaluation** (:func:`evaluate_candidate_recall`, :func:`candidate_statistics`) joins the
   candidate table with ground-truth pairs of the same S1 ids.

Ids are integers: ``s1_id`` = numeric part of ``S1-<n>``, ``t_id = src * 1e9 + n`` for S2/S3
(all numeric parts are < 1e9, verified on the training files). Use :func:`encode_entity_id`
/ :func:`decode_target_id` to convert.
"""
from __future__ import annotations

import gc
import os
import shutil
import time
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from src.normalization import normalize_address, normalize_name

__all__ = [
    "ID_BASE", "encode_entity_id", "decode_target_id", "normalize_country",
    "blocking_record", "blocking_records_chunk", "materialize_blocking_features",
    "BlockingConfig", "connect", "register_features", "check_disk",
    "build_blocking_statistics", "build_blocking_indices",
    "generate_exact_name_candidates", "generate_phonetic_candidates",
    "generate_structured_candidates", "generate_rare_token_candidates",
    "generate_ngram_candidates", "generate_compound_candidates", "generate_candidates",
    "generate_candidates_chunked", "apply_per_s1_cap", "DEFAULT_STRATEGIES",
    "load_ground_truth_pairs", "evaluate_candidate_recall", "candidate_statistics",
    "STRATEGY_BITS",
]

ID_BASE = 1_000_000_000
_NAME_STOP = frozenset({"and", "the", "of", "&", "de", "la", "a", "an", "for", "in", "at", "by"})


# =============================================================================
# Ids and countries
# =============================================================================

def encode_entity_id(entity_id: str) -> tuple[int, int]:
    """``'S2-123'`` -> ``(2, 2_000_000_123)``; ``'S1-55'`` -> ``(1, 55)`` (S1 ids are not offset)."""
    prefix, _, num = entity_id.strip().partition("-")
    if not (prefix[:1] in ("S", "s") and prefix[1:].isdigit() and num.isdigit()):
        raise ValueError(f"unexpected entity id: {entity_id!r}")
    src, n = int(prefix[1:]), int(num)
    if n >= ID_BASE:
        raise ValueError(f"numeric id too large for encoding: {entity_id!r}")
    return src, (n if src == 1 else src * ID_BASE + n)


def decode_target_id(t_id: int) -> str:
    """Inverse of :func:`encode_entity_id` for target ids: ``2_000_000_123`` -> ``'S2-123'``."""
    src, n = divmod(int(t_id), ID_BASE)
    return f"S{src}-{n}" if src else f"S1-{n}"


def normalize_country(country: object) -> str:
    """Partition label for a country value: trimmed + lower-cased; missing -> ``''``.

    Nothing is hard-coded to US/India: any country string becomes its own partition.
    """
    if country is None or (isinstance(country, float) and country != country):
        return ""
    return str(country).strip().casefold()


# =============================================================================
# Per-record blocking fields (Python, reuses Phase 2)
# =============================================================================

def _compact(text: str) -> str:
    return "".join(t for t in text.split() if t != "and")


def blocking_record(name: object, address: object, country: object) -> dict:
    """Compact blocking fields of one record, derived only from the record itself.

    Fields
    ------
    name_key      compact core name (``'wenonahsmetalworks'``)
    name_keys     '|'-joined distinct compact variants: core, alias sides, website label
    name_core     core name in Latin script (space separated)
    name_tokens   ' '-joined distinct core/alias tokens (stop words removed, len >= 2)
    name_phon     sorted distinct phonetic tokens (' '-joined) — order-insensitive key
    state, hn     canonical state code, house-number core
    street        canonical street ('farragut ave')
    places        '|'-joined digit-free components (city, district, locality ...)
    addr_tokens   ' '-joined distinct alphabetic address tokens (len >= 3), state excluded
    numbers       ' '-joined distinct zero-stripped digit groups
    """
    c = normalize_country(country)
    n = normalize_name(name)
    a = normalize_address(address, c or None)

    variants = [n.core_latin, *n.aliases]
    keys: list[str] = []
    for v in (*variants, n.website_label or ""):
        k = _compact(v)
        if len(k) >= 3 and k not in keys:
            keys.append(k)
    toks: list[str] = []
    for v in variants:
        for t in v.split():
            if len(t) >= 2 and t not in _NAME_STOP and t not in toks:
                toks.append(t)
    if n.website_label and n.website_label not in toks:
        toks.append(n.website_label)
    phon = sorted({t for t in n.phonetic.split() if len(t) >= 2})

    atoks: list[str] = []
    for comp in a.components:
        if a.state is not None and comp == a.state:
            continue
        for t in comp.split():
            if len(t) >= 3 and t.isalpha() and t not in atoks:
                atoks.append(t)
    return {
        "country": c,
        "name_key": _compact(n.core_latin) if len(_compact(n.core_latin)) >= 3 else "",
        "name_keys": "|".join(keys),
        "name_core": n.core_latin,
        "name_tokens": " ".join(toks),
        "name_phon": " ".join(phon),
        "state": a.state or "",
        "hn": a.house_number_core or "",
        "street": a.street or "",
        "places": "|".join(a.places),
        "addr_tokens": " ".join(atoks),
        "numbers": " ".join(a.numbers),
    }


_FEATURE_COLUMNS = ("country", "name_key", "name_keys", "name_core", "name_tokens", "name_phon",
                    "state", "hn", "street", "places", "addr_tokens", "numbers")


def blocking_records_chunk(rows: Sequence[tuple]) -> dict[str, list]:
    """Normalise ``(entity_id, name, address, country)`` rows into column lists.

    Designed for ``multiprocessing`` workers: returns plain lists (cheap to pickle) and clears
    the Phase 2 LRU caches afterwards so a long-running worker cannot grow without bound.
    """
    from src import normalization as N

    out: dict[str, list] = {"id": [], "src": []}
    out.update({c: [] for c in _FEATURE_COLUMNS})
    for entity_id, name, address, country in rows:
        src, i = encode_entity_id(entity_id)
        rec = blocking_record(name, address, country)
        out["id"].append(i)
        out["src"].append(src)
        for c in _FEATURE_COLUMNS:
            out[c].append(rec[c])
    N._normalize_name_cached.cache_clear()
    N._normalize_address_cached.cache_clear()
    return out


# =============================================================================
# DuckDB connection, disk guard, materialisation
# =============================================================================

@dataclass
class BlockingConfig:
    """Caps and parameters for every strategy. Defaults are the values chosen in the Phase 3
    pilot (see experiments/phase3_candidate_generation_report.md)."""
    name_cap: int = 100             # max targets per (country, name key); bigger -> retry with state
    name_state_cap: int = 100       # max targets per (country, state, name key); bigger -> dropped
    phon_cap: int = 150             # max targets per (country, phonetic key); bigger -> retry with state
    phon_state_cap: int = 150
    hn_cap: int = 100               # max targets per (country, state, house number); bigger -> refine
    hn_refined_cap: int = 100       # max targets per refined (country, state, hn, place|street) key
    token_k: int = 2                # rarest tokens used per S1 record (and per state bucket)
    token_df_cap: int = 50          # max targets per (country, state, name token)
    addr_token_df_cap: int = 100    # max targets per (country, state, address token)
    ngram_n: int = 3                # must match the n used to build st_ngram
    ngram_m: int = 8                # rarest grams used per S1 record (and per state bucket)
    ngram_df_cap: int = 1000        # max targets per (country, state, gram) posting
    ngram_min_shared: int = 3       # min shared selected grams for an n-gram candidate
    ngram_top_k: int = 20           # max n-gram candidates kept per S1 record
    compound_cap: int = 30          # max targets per (country, state, hn|place, phonetic token)
    stateless_fallback: bool = True   # also search targets whose address has no state (state = '')
    per_s1_cap: Optional[int] = 200   # final cap on candidates per S1 (by strategy agreement)


#: Strategies retained after the pilot, in greedy order of recall gained per candidate.
DEFAULT_STRATEGIES: tuple[str, ...] = ("compound", "ngram", "name", "rare_addr", "structured", "phonetic", "rare_name")
#: Priority used when the per-S1 cap has to drop candidates (after strategy-agreement count).
CAP_PRIORITY: tuple[str, ...] = ("name", "structured", "compound", "phonetic", "rare_addr", "rare_name", "ngram", "rare_phon")


def connect(db_path: Optional[str | Path] = None, memory_limit: str = "1500MB", threads: int = 4,
            temp_dir: Optional[str | Path] = None, max_temp: str = "2GB"):
    """DuckDB connection with conservative memory, thread and spill limits."""
    import duckdb

    con = duckdb.connect(str(db_path) if db_path else ":memory:")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={int(threads)}")
    con.execute("SET preserve_insertion_order=false")
    if temp_dir is not None:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{Path(temp_dir)}'")
        con.execute(f"SET max_temp_directory_size='{max_temp}'")
    return con


def check_disk(path: str | Path, min_free_gb: float) -> float:
    """Free disk space (GB) at ``path``; raises ``RuntimeError`` if below ``min_free_gb``."""
    free = shutil.disk_usage(Path(path).resolve().anchor if not Path(path).exists() else path).free / 1e9
    if free < min_free_gb:
        raise RuntimeError(f"only {free:.1f} GB free at {path}; need >= {min_free_gb} GB")
    return free


def materialize_blocking_features(tsv_path: str | Path, out_dir: str | Path, *, workers: int = 4,
                                  batch_rows: int = 100_000, chunk_rows: int = 5_000,
                                  where: Optional[str] = None, min_free_gb: float = 3.0,
                                  overwrite: bool = False, log=print) -> dict:
    """Stream one source TSV into Parquet parts of blocking features.

    * DuckDB reads the TSV in ``batch_rows`` batches (never the whole file in pandas).
    * Each batch is split into ``chunk_rows`` chunks normalised by ``workers`` processes.
    * Each batch is written as ``out_dir/part-XXXXX.parquet`` (zstd) by DuckDB.
    * Idempotent: a ``_SUCCESS`` marker skips finished sources unless ``overwrite``.
    """
    import duckdb

    out_dir = Path(out_dir)
    if (out_dir / "_SUCCESS").exists() and not overwrite:
        log(f"skip {out_dir.name}: already materialised")
        return {"skipped": True}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    check_disk(out_dir, min_free_gb)

    reader = duckdb.connect()
    reader.execute("SET threads=2")
    writer = duckdb.connect()
    writer.execute("SET threads=2")
    sql = (f"SELECT entity_id, business_name, business_address, country FROM "
           f"read_csv_auto('{tsv_path}', delim='\t', header=true, all_varchar=true)")
    if where:
        sql += f" WHERE {where}"
    cur = reader.execute(sql)
    t0, n_rows, part = time.time(), 0, 0
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        while True:
            rows = cur.fetchmany(batch_rows)
            if not rows:
                break
            chunks = [rows[i:i + chunk_rows] for i in range(0, len(rows), chunk_rows)]
            cols: dict[str, list] = {}
            for res in pool.imap(blocking_records_chunk, chunks):
                for k, v in res.items():
                    cols.setdefault(k, []).extend(v)
            df = pd.DataFrame(cols)
            df["id"] = df["id"].astype("int64")
            df["src"] = df["src"].astype("int8")
            writer.register("batch_df", df)
            writer.execute(f"COPY batch_df TO '{out_dir / f'part-{part:05d}.parquet'}' "
                           f"(FORMAT parquet, COMPRESSION zstd)")
            writer.unregister("batch_df")
            n_rows += len(df)
            part += 1
            del rows, chunks, cols, df
            gc.collect()
            if part % 10 == 0:
                check_disk(out_dir, min_free_gb)
                log(f"  {out_dir.name}: {n_rows:,} rows, {time.time() - t0:.0f}s")
    reader.close()
    writer.close()
    size = sum(f.stat().st_size for f in out_dir.glob("*.parquet"))
    (out_dir / "_SUCCESS").write_text(f"{n_rows}\n")
    stats = {"rows": n_rows, "parts": part, "seconds": round(time.time() - t0, 1), "bytes": size}
    log(f"done {out_dir.name}: {stats}")
    return stats


# =============================================================================
# Registration of feature Parquet as DuckDB views
# =============================================================================

STRATEGY_BITS: dict[str, int] = {
    "name": 1,          # exact compact name / alias / website key
    "phonetic": 2,      # exact sorted phonetic key
    "structured": 4,    # state + house number (refined by place / street for big blocks)
    "rare_name": 8,     # k rarest name tokens within (country, state)
    "rare_phon": 16,    # k rarest phonetic tokens within (country, state)
    "rare_addr": 32,    # k rarest address tokens within (country, state)
    "ngram": 64,        # m rarest character n-grams of the compact name, >= t shared, top-K
    "compound": 128,    # (state, house number | place, phonetic name token) — specific combos of common parts
}


class CandidateExplosion(RuntimeError):
    """Raised when a strategy's estimated raw pair count exceeds the configured limit."""


def register_features(con, feature_dir: str | Path) -> None:
    """Create views ``feat_s1`` and ``feat_t`` (S2 ∪ S3) over the materialised Parquet parts."""
    d = Path(feature_dir)
    for i in (1, 2, 3):
        if not (d / f"features_s{i}" / "_SUCCESS").exists():
            raise FileNotFoundError(f"features for S{i} are not materialised in {d}")
    con.execute(f"CREATE OR REPLACE VIEW feat_s1 AS SELECT * FROM read_parquet('{d}/features_s1/*.parquet')")
    con.execute(f"""CREATE OR REPLACE VIEW feat_t AS
        SELECT * FROM read_parquet(['{d}/features_s2/*.parquet', '{d}/features_s3/*.parquet'])""")


def set_query_records(con, s1_ids_sql: str) -> int:
    """Materialise the query S1 records (table ``q``) from a SQL query returning ``id`` values."""
    con.execute(f"CREATE OR REPLACE TABLE q AS SELECT * FROM feat_s1 WHERE id IN ({s1_ids_sql})")
    return con.execute("SELECT COUNT(*) FROM q").fetchone()[0]


def _with_stateless(sql: str, enabled: bool) -> str:
    """Query-side key rows, duplicated into the country's no-state bucket (``state = ''``) so that
    targets with a missing/unparsed address remain reachable by state-partitioned strategies."""
    if not enabled:
        return sql
    return f"SELECT * FROM ({sql}) UNION ALL SELECT id, country, '' AS state, k FROM ({sql}) WHERE state <> ''"


# SQL fragments that explode a record into its keys (target side streamed, never stored)
def _key_sql(kind: str, rel: str, cfg: "BlockingConfig") -> str:
    base = f"SELECT id, country, state, {{expr}} AS k FROM {rel}"
    if kind == "name":
        return f"SELECT id, country, state, unnest(string_split(name_keys, '|')) AS k FROM {rel} WHERE name_keys <> ''"
    if kind == "phonetic":
        return f"SELECT id, country, state, name_phon AS k FROM {rel} WHERE name_phon <> ''"
    if kind == "rare_name":
        return f"SELECT id, country, state, unnest(string_split(name_tokens, ' ')) AS k FROM {rel} WHERE name_tokens <> ''"  # tokens are unique per record already
    if kind == "rare_phon":
        return f"SELECT id, country, state, unnest(string_split(name_phon, ' ')) AS k FROM {rel} WHERE name_phon <> ''"  # tokens are unique per record already
    if kind == "rare_addr":
        return f"SELECT id, country, state, unnest(string_split(addr_tokens, ' ')) AS k FROM {rel} WHERE addr_tokens <> ''"  # tokens are unique per record already
    if kind == "ngram":
        n = int(cfg.ngram_n)
        padded = "('#' || name_key || '#')"
        return (f"SELECT id, country, state, unnest(list_distinct(list_transform(range(1, length({padded}) - {n} + 2), "
                f"i -> substr({padded}, i, {n})))) AS k FROM {rel} WHERE name_key <> ''")
    raise ValueError(kind)


# =============================================================================
# Blocking statistics (document frequencies over TARGET records only)
# =============================================================================

def build_blocking_statistics(con, cfg: Optional["BlockingConfig"] = None, kinds: Iterable[str] = (
        "name", "phonetic", "structured", "rare_name", "rare_phon", "rare_addr", "ngram"), log=print) -> dict:
    """Target-side key frequencies used for caps and rarity (no ground truth involved).

    Tables created (``n`` = number of target records carrying the key):
      st_name(country, k, n), st_name_state(country, state, k, n)      [only for over-cap keys]
      st_phonetic(...), st_phonetic_state(...)
      st_hn(country, state, hn, n), st_hn_refined(country, state, hn, r, n) [only for over-cap hn]
      st_rare_name / st_rare_phon / st_rare_addr / st_ngram (country, state, k, n)
    """
    cfg = cfg or BlockingConfig()
    timings: dict[str, float] = {}
    kinds = list(kinds)
    for kind, cap in (("name", cfg.name_cap), ("phonetic", cfg.phon_cap)):
        if kind not in kinds:
            continue
        t = time.time()
        con.execute(f"CREATE OR REPLACE TABLE st_{kind} AS SELECT country, k, COUNT(*) n FROM ({_key_sql(kind, 'feat_t', cfg)}) GROUP BY ALL")
        con.execute(f"""CREATE OR REPLACE TABLE st_{kind}_state AS
            SELECT t.country, t.state, t.k, COUNT(*) n
            FROM ({_key_sql(kind, 'feat_t', cfg)}) t SEMI JOIN (SELECT country, k FROM st_{kind} WHERE n > {int(cap)}) b
                 ON t.country = b.country AND t.k = b.k
            GROUP BY ALL""")
        timings[kind] = round(time.time() - t, 1)
    if "structured" in kinds:
        t = time.time()
        con.execute("""CREATE OR REPLACE TABLE st_hn AS SELECT country, state, hn, COUNT(*) n
                       FROM feat_t WHERE state <> '' AND hn <> '' GROUP BY ALL""")
        con.execute(f"""CREATE OR REPLACE TABLE st_hn_refined AS
            SELECT country, state, hn, r, COUNT(DISTINCT id) n FROM (
              SELECT t.id, t.country, t.state, t.hn, unnest(list_concat(
                       list_transform(string_split(t.places, '|'), p -> 'p:' || p),
                       CASE WHEN t.street <> '' THEN ['s:' || t.street] ELSE [] END)) AS r
              FROM feat_t t SEMI JOIN (SELECT * FROM st_hn WHERE n > {int(cfg.hn_cap)}) b
                   ON t.country = b.country AND t.state = b.state AND t.hn = b.hn)
            WHERE r NOT IN ('p:', 's:') GROUP BY ALL""")
        timings["structured"] = round(time.time() - t, 1)
    for kind in ("rare_name", "rare_phon", "rare_addr", "ngram"):
        if kind not in kinds:
            continue
        t = time.time()
        con.execute(f"CREATE OR REPLACE TABLE st_{kind} AS SELECT country, state, k, COUNT(*) n FROM ({_key_sql(kind, 'feat_t', cfg)}) GROUP BY ALL")
        timings[kind] = round(time.time() - t, 1)
    sizes = {r[0]: r[1] for r in con.execute(
        "SELECT table_name, estimated_size FROM duckdb_tables() WHERE table_name LIKE 'st_%'").fetchall()}
    log(f"blocking statistics built: {timings}")
    return {"seconds": timings, "rows": sizes}


def build_blocking_indices(con, feature_dir: str | Path, cfg: Optional["BlockingConfig"] = None, **kw) -> dict:
    """Register feature views and build all blocking statistics (the 'indices').

    Target postings are *not* materialised: every strategy streams the target side from
    Parquet and joins it against the (small) query-side key set, so no inverted index of
    10M+ records has to be held in memory or written to disk.
    """
    register_features(con, feature_dir)
    return build_blocking_statistics(con, cfg, **kw)


# =============================================================================
# Candidate strategies — each fills table ``cand_<name>(s1 BIGINT, t BIGINT)``
# =============================================================================

def _guard(con, estimate_sql: str, strategy: str, max_raw_pairs: int) -> int:
    est = con.execute(estimate_sql).fetchone()[0] or 0
    if est > max_raw_pairs:
        raise CandidateExplosion(f"{strategy}: estimated {est:,} raw pairs > limit {max_raw_pairs:,}")
    return int(est)


def _two_tier(con, kind: str, cap: int, state_cap: int, max_raw_pairs: int, stateless: bool = True) -> dict:
    """Exact-key join with a national cap, falling back to (state, key) for over-cap keys
    (plus the no-state bucket when ``stateless``)."""
    cfg = BlockingConfig()
    con.execute(f"CREATE OR REPLACE TEMP TABLE qk AS SELECT DISTINCT * FROM ({_with_stateless(_key_sql(kind, 'q', cfg), stateless)})")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE qk_nat AS
        SELECT DISTINCT qk.id, qk.country, qk.k, s.n FROM qk JOIN st_{kind} s USING (country, k) WHERE s.n <= {int(cap)}""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE qk_st AS
        SELECT qk.id, qk.country, qk.state, qk.k, s.n FROM qk
        JOIN st_{kind} a USING (country, k) JOIN st_{kind}_state s USING (country, state, k)
        WHERE a.n > {int(cap)} AND s.n <= {int(state_cap)}""")
    est = _guard(con, "SELECT (SELECT COALESCE(SUM(n),0) FROM qk_nat) + (SELECT COALESCE(SUM(n),0) FROM qk_st)",
                 kind, max_raw_pairs)
    tk = _key_sql(kind, "feat_t", cfg)
    con.execute(f"""CREATE OR REPLACE TABLE cand_{kind} AS
        SELECT DISTINCT s1, t FROM (
          SELECT a.id s1, t.id t FROM qk_nat a JOIN ({tk}) t ON t.country = a.country AND t.k = a.k
          UNION ALL
          SELECT a.id, t.id FROM qk_st a JOIN ({tk}) t ON t.country = a.country AND t.state = a.state AND t.k = a.k)""")
    dropped = con.execute(f"""SELECT COUNT(*) FROM qk JOIN st_{kind} a USING (country, k)
        LEFT JOIN st_{kind}_state s USING (country, state, k)
        WHERE a.n > {int(cap)} AND (s.n IS NULL OR s.n > {int(state_cap)})""").fetchone()[0]
    return {"estimated_raw_pairs": est, "keys_dropped_over_cap": int(dropped)}


def generate_exact_name_candidates(con, cfg: "BlockingConfig", max_raw_pairs: int = 50_000_000) -> dict:
    """Exact compact name / alias / website-label key within country (state for common names)."""
    return _two_tier(con, "name", cfg.name_cap, cfg.name_state_cap, max_raw_pairs, cfg.stateless_fallback)


def generate_phonetic_candidates(con, cfg: "BlockingConfig", max_raw_pairs: int = 50_000_000) -> dict:
    """Exact sorted-phonetic-token key within country (state for common keys)."""
    return _two_tier(con, "phonetic", cfg.phon_cap, cfg.phon_state_cap, max_raw_pairs, cfg.stateless_fallback)


def generate_structured_candidates(con, cfg: "BlockingConfig", max_raw_pairs: int = 50_000_000) -> dict:
    """(country, state, house number); over-cap blocks are refined by a shared place or street."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE qh AS
        SELECT q.id, q.country, q.state, q.hn, q.places, q.street, s.n FROM q JOIN st_hn s USING (country, state, hn)
        WHERE q.state <> '' AND q.hn <> ''""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE qh_ref AS
        SELECT DISTINCT r.id, r.country, r.state, r.hn, r.r, s.n FROM (
          SELECT id, country, state, hn, unnest(list_concat(
                   list_transform(string_split(places, '|'), p -> 'p:' || p),
                   CASE WHEN street <> '' THEN ['s:' || street] ELSE [] END)) AS r
          FROM qh WHERE n > {int(cfg.hn_cap)}) r
        JOIN st_hn_refined s USING (country, state, hn, r) WHERE s.n <= {int(cfg.hn_refined_cap)}""")
    est = _guard(con, f"""SELECT (SELECT COALESCE(SUM(n),0) FROM qh WHERE n <= {int(cfg.hn_cap)})
                               + (SELECT COALESCE(SUM(n),0) FROM qh_ref)""", "structured", max_raw_pairs)
    con.execute(f"""CREATE OR REPLACE TABLE cand_structured AS
        SELECT DISTINCT s1, t FROM (
          SELECT a.id s1, t.id t FROM (SELECT * FROM qh WHERE n <= {int(cfg.hn_cap)}) a
          JOIN feat_t t ON t.country = a.country AND t.state = a.state AND t.hn = a.hn
          UNION ALL
          SELECT a.id, t.id FROM qh_ref a JOIN (
              SELECT id, country, state, hn, unnest(list_concat(
                       list_transform(string_split(places, '|'), p -> 'p:' || p),
                       CASE WHEN street <> '' THEN ['s:' || street] ELSE [] END)) AS r
              FROM feat_t WHERE state <> '' AND hn <> '') t
          ON t.country = a.country AND t.state = a.state AND t.hn = a.hn AND t.r = a.r)""")
    return {"estimated_raw_pairs": est}


def generate_rare_token_candidates(con, cfg: "BlockingConfig", kind: str = "rare_name",
                                   max_raw_pairs: int = 50_000_000) -> dict:
    """Each S1 record keeps its ``token_k`` rarest tokens whose (country, state) target
    frequency is between 1 and ``token_df_cap``; targets sharing one of them are candidates.

    Common tokens ('company', 'india', 'llc', 'st') are never used, and every posting list is
    capped, so candidates per S1 <= ``token_k * token_df_cap``.
    """
    cfg_k = int(cfg.token_k)
    cap = int(cfg.addr_token_df_cap if kind == "rare_addr" else cfg.token_df_cap)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE qt AS
        SELECT id, country, state, k, n FROM (
          SELECT q.*, s.n, row_number() OVER (PARTITION BY q.id, q.state ORDER BY s.n, q.k) rk
          FROM ({_with_stateless(_key_sql(kind, 'q', cfg), cfg.stateless_fallback)}) q JOIN st_{kind} s USING (country, state, k)
          WHERE s.n <= {cap})
        WHERE rk <= {cfg_k}""")
    est = _guard(con, "SELECT COALESCE(SUM(n),0) FROM qt", kind, max_raw_pairs)
    con.execute(f"""CREATE OR REPLACE TABLE cand_{kind} AS
        SELECT DISTINCT a.id s1, t.id t FROM qt a JOIN ({_key_sql(kind, 'feat_t', cfg)}) t
        ON t.country = a.country AND t.state = a.state AND t.k = a.k""")
    return {"estimated_raw_pairs": est}


def generate_ngram_candidates(con, cfg: "BlockingConfig", max_raw_pairs: int = 50_000_000) -> dict:
    """Character n-gram retrieval on the compact name, bounded at every step.

    1. per S1 record keep the ``ngram_m`` rarest grams with 1 <= df(country, state) <= ``ngram_df_cap``;
    2. stream target grams, join only those postings (<= m * cap raw rows per S1);
    3. keep pairs sharing >= ``ngram_min_shared`` of the selected grams, top ``ngram_top_k`` per S1.
    """
    con.execute(f"""CREATE OR REPLACE TEMP TABLE qg AS
        SELECT id, country, state, k, n FROM (
          SELECT q.*, s.n, row_number() OVER (PARTITION BY q.id, q.state ORDER BY s.n, q.k) rk
          FROM ({_with_stateless(_key_sql('ngram', 'q', cfg), cfg.stateless_fallback)}) q JOIN st_ngram s USING (country, state, k)
          WHERE s.n <= {int(cfg.ngram_df_cap)})
        WHERE rk <= {int(cfg.ngram_m)}""")
    est = _guard(con, "SELECT COALESCE(SUM(n),0) FROM qg", "ngram", max_raw_pairs)
    con.execute(f"""CREATE OR REPLACE TABLE cand_ngram AS
        SELECT s1, t FROM (
          SELECT s1, t, shared, row_number() OVER (PARTITION BY s1 ORDER BY shared DESC, t) rk FROM (
            SELECT a.id s1, t.id t, COUNT(*) shared
            FROM qg a JOIN ({_key_sql('ngram', 'feat_t', cfg)}) t
              ON t.country = a.country AND t.state = a.state AND t.k = a.k
            GROUP BY ALL HAVING COUNT(*) >= {int(cfg.ngram_min_shared)}))
        WHERE rk <= {int(cfg.ngram_top_k)}""")
    return {"estimated_raw_pairs": est}


def _compound_key_sql(rel: str) -> str:
    """(house number | place) x phonetic token keys of a record, within (country, state)."""
    return f"""SELECT id, country, state, a || '#' || p AS k FROM (
        SELECT id, country, state,
               unnest(list_concat(CASE WHEN hn <> '' THEN ['h:' || hn] ELSE [] END,
                                  list_transform(list_filter(string_split(places, '|'), x -> x <> ''), x -> 'p:' || x))) AS a,
               string_split(name_phon, ' ') AS ps
        FROM {rel} WHERE state <> '' AND name_phon <> '' AND (hn <> '' OR places <> ''))
      , unnest(ps) AS u(p)"""


def generate_compound_candidates(con, cfg: "BlockingConfig", max_raw_pairs: int = 50_000_000) -> dict:
    """Specific combinations of individually common parts: (state, house number, phonetic name
    token) and (state, place, phonetic name token).

    A full statistics table of these combinations would have tens of millions of keys, so the
    frequencies are computed *only for the keys of the query records* in a first streaming pass
    (the group-by is bounded by the number of query keys); only keys with at most
    ``compound_cap`` targets are then joined.
    """
    con.execute(f"CREATE OR REPLACE TEMP TABLE qc AS SELECT DISTINCT * FROM ({_compound_key_sql('q')})")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE qc_df AS
        SELECT t.country, t.state, t.k, COUNT(*) n
        FROM ({_compound_key_sql('feat_t')}) t
        SEMI JOIN (SELECT DISTINCT country, state, k FROM qc) q ON q.country = t.country AND q.state = t.state AND q.k = t.k
        GROUP BY ALL""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE qc_ok AS
        SELECT qc.id, qc.country, qc.state, qc.k, d.n FROM qc JOIN qc_df d USING (country, state, k)
        WHERE d.n <= {int(cfg.compound_cap)}""")
    est = _guard(con, "SELECT COALESCE(SUM(n),0) FROM qc_ok", "compound", max_raw_pairs)
    con.execute(f"""CREATE OR REPLACE TABLE cand_compound AS
        SELECT DISTINCT a.id s1, t.id t FROM qc_ok a JOIN ({_compound_key_sql('feat_t')}) t
        ON t.country = a.country AND t.state = a.state AND t.k = a.k""")
    return {"estimated_raw_pairs": est}


_GENERATORS = {
    "name": lambda con, cfg, lim: generate_exact_name_candidates(con, cfg, lim),
    "phonetic": lambda con, cfg, lim: generate_phonetic_candidates(con, cfg, lim),
    "structured": lambda con, cfg, lim: generate_structured_candidates(con, cfg, lim),
    "rare_name": lambda con, cfg, lim: generate_rare_token_candidates(con, cfg, "rare_name", lim),
    "rare_phon": lambda con, cfg, lim: generate_rare_token_candidates(con, cfg, "rare_phon", lim),
    "rare_addr": lambda con, cfg, lim: generate_rare_token_candidates(con, cfg, "rare_addr", lim),
    "ngram": lambda con, cfg, lim: generate_ngram_candidates(con, cfg, lim),
    "compound": lambda con, cfg, lim: generate_compound_candidates(con, cfg, lim),
}


def generate_candidates(con, strategies: Sequence[str], cfg: Optional["BlockingConfig"] = None,
                        max_raw_pairs: int = 50_000_000, log=print) -> dict:
    """Run strategies for the query records in ``q`` and union them into ``cand(s1, t, mask)``.

    ``mask`` is the bitwise OR of :data:`STRATEGY_BITS` of every strategy that produced the pair,
    so single strategies and any union can be evaluated from this one deduplicated table.
    A strategy whose estimated raw pair count exceeds ``max_raw_pairs`` is skipped and reported
    (it is never run).
    """
    cfg = cfg or BlockingConfig()
    info: dict[str, dict] = {}
    parts = []
    for s in strategies:
        t = time.time()
        try:
            meta = _GENERATORS[s](con, cfg, max_raw_pairs)
        except CandidateExplosion as e:
            log(f"STOPPED {s}: {e}")
            info[s] = {"status": "stopped", "reason": str(e)}
            continue
        n = con.execute(f"SELECT COUNT(*) FROM cand_{s}").fetchone()[0]
        info[s] = {"status": "ok", "pairs": int(n), "seconds": round(time.time() - t, 2), **meta}
        log(f"{s:<11} {n:>12,} pairs  {time.time() - t:6.1f}s")
        parts.append(f"SELECT s1, t, {STRATEGY_BITS[s]}::USMALLINT b FROM cand_{s}")
    if not parts:
        con.execute("CREATE OR REPLACE TABLE cand (s1 BIGINT, t BIGINT, mask USMALLINT)")
    else:
        con.execute(f"CREATE OR REPLACE TABLE cand AS SELECT s1, t, bit_or(b) mask FROM ({' UNION ALL '.join(parts)}) GROUP BY ALL")
    for s in strategies:
        con.execute(f"DROP TABLE IF EXISTS cand_{s}")
    info["_union_pairs"] = int(con.execute("SELECT COUNT(*) FROM cand").fetchone()[0])
    return info


def generate_candidates_chunked(con, strategies: Sequence[str] = DEFAULT_STRATEGIES,
                                cfg: Optional["BlockingConfig"] = None, n_chunks: int = 10,
                                query_table: str = "q_all", out_parquet_dir: Optional[str | Path] = None,
                                max_raw_pairs: int = 50_000_000, min_free_gb: float = 3.0, log=print) -> dict:
    """Bounded generation for many S1 records: process ``query_table`` in ``n_chunks`` hash
    chunks (table ``q`` is rebuilt per chunk), apply the per-S1 cap inside each chunk, and either
    append to table ``cand_all`` or, for full-scale runs, write one Parquet part per chunk to
    ``out_parquet_dir`` (never one huge in-memory table). Disk is checked before every chunk.
    On return ``q`` = all query records and ``cand`` = all candidates (table mode only).
    """
    cfg = cfg or BlockingConfig()
    if out_parquet_dir is not None:
        Path(out_parquet_dir).mkdir(parents=True, exist_ok=True)
    else:
        con.execute("CREATE OR REPLACE TABLE cand_all (s1 BIGINT, t BIGINT, mask USMALLINT)")
    per_chunk, per_strategy, t0 = [], {}, time.time()
    for i in range(int(n_chunks)):
        check_disk(out_parquet_dir if out_parquet_dir is not None else ".", min_free_gb)
        t = time.time()
        # salted hash: independent of any `hash(id) % k` sampling used to build the query table
        con.execute(f"CREATE OR REPLACE TABLE q AS SELECT * FROM {query_table} WHERE hash(id, 1) % {int(n_chunks)} = {i}")
        info = generate_candidates(con, strategies, cfg, max_raw_pairs, log=lambda *a: None)
        stopped = [s for s in strategies if info[s]["status"] != "ok"]
        if stopped:
            log(f"chunk {i}: STOPPED {stopped}")
        if cfg.per_s1_cap:
            apply_per_s1_cap(con, cfg.per_s1_cap, CAP_PRIORITY)
        n = con.execute("SELECT COUNT(*) FROM cand").fetchone()[0]
        if out_parquet_dir is not None:
            con.execute(f"COPY cand TO '{Path(out_parquet_dir) / f'part-{i:05d}.parquet'}' (FORMAT parquet, COMPRESSION zstd)")
        else:
            con.execute("INSERT INTO cand_all SELECT * FROM cand")
        for s in strategies:
            d = per_strategy.setdefault(s, {"seconds": 0.0, "pairs": 0, "stopped_chunks": 0})
            d["seconds"] += info[s].get("seconds", 0.0)
            d["pairs"] += info[s].get("pairs", 0)
            d["stopped_chunks"] += info[s]["status"] != "ok"
        per_chunk.append({"chunk": i, "pairs": int(n), "seconds": round(time.time() - t, 1)})
        log(f"chunk {i + 1}/{n_chunks}: {n:,} pairs in {time.time() - t:.0f}s")
    con.execute(f"CREATE OR REPLACE TABLE q AS SELECT * FROM {query_table}")
    if out_parquet_dir is None:
        con.execute("DROP TABLE IF EXISTS cand")
        con.execute("ALTER TABLE cand_all RENAME TO cand")
    for d in per_strategy.values():
        d["seconds"] = round(d["seconds"], 1)
    return {"seconds": round(time.time() - t0, 1), "chunks": per_chunk, "per_strategy": per_strategy}


def apply_per_s1_cap(con, cap: int, priority: Sequence[str]) -> int:
    """Keep at most ``cap`` candidates per S1: pairs found by more strategies first, then by the
    highest-priority strategy (``priority`` order). Rewrites ``cand``; returns rows removed."""
    before = con.execute("SELECT COUNT(*) FROM cand").fetchone()[0]
    prio = " + ".join(f"(CASE WHEN mask & {STRATEGY_BITS[s]} > 0 THEN {2 ** (len(priority) - i)} ELSE 0 END)"
                      for i, s in enumerate(priority))
    con.execute(f"""CREATE OR REPLACE TABLE cand AS SELECT s1, t, mask FROM (
        SELECT *, row_number() OVER (PARTITION BY s1 ORDER BY bit_count(mask) DESC, {prio} DESC, t) rk FROM cand)
        WHERE rk <= {int(cap)}""")
    return int(before - con.execute("SELECT COUNT(*) FROM cand").fetchone()[0])


# =============================================================================
# Evaluation (ground truth used ONLY here)
# =============================================================================

def load_ground_truth_pairs(con, gt_tsv: str | Path, table: str = "gt_pairs", s1_filter_sql: Optional[str] = None) -> int:
    """Load true pairs as integer ids into ``table(s1, t, src)``; optionally only for some S1 ids."""
    where = f"AND s1 IN ({s1_filter_sql})" if s1_filter_sql else ""
    con.execute(f"""CREATE OR REPLACE TABLE {table} AS
        SELECT s1, CAST(substr(m, 2, 1) AS BIGINT) * {ID_BASE} + CAST(substr(m, 4) AS BIGINT) AS t,
               CAST(substr(m, 2, 1) AS TINYINT) AS src
        FROM (SELECT CAST(substr(source1_entity_id, 4) AS BIGINT) s1, trim(u.m) m
              FROM read_csv_auto('{gt_tsv}', delim='\t', header=true, all_varchar=true),
                   unnest(string_split(matched_entity_ids, ',')) AS u(m)
              WHERE matched_entity_ids IS NOT NULL AND trim(u.m) <> '')
        WHERE TRUE {where}""")
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def candidate_statistics(con, mask: Optional[int] = None, cand_table: str = "cand",
                         query_table: str = "q") -> dict:
    """Candidates per S1 over *all* query S1 records (records with 0 candidates count as 0)."""
    flt = f"WHERE c.mask & {int(mask)} > 0" if mask is not None else ""
    r = con.execute(f"""
        WITH per AS (SELECT c.s1, COUNT(*) n FROM {cand_table} c {flt} GROUP BY 1),
        allq AS (SELECT q.id, COALESCE(per.n, 0) n FROM {query_table} q LEFT JOIN per ON per.s1 = q.id)
        SELECT SUM(n), AVG(n), quantile_cont(n, 0.5), quantile_cont(n, 0.95), quantile_cont(n, 0.99), MAX(n),
               AVG((n = 0)::INT), COUNT(*) FROM allq""").fetchone()
    return {"candidates": int(r[0] or 0), "avg_per_s1": round(float(r[1] or 0), 2), "median_per_s1": float(r[2] or 0),
            "p95_per_s1": float(r[3] or 0), "p99_per_s1": float(r[4] or 0), "max_per_s1": int(r[5] or 0),
            "pct_s1_without_candidates": round(100 * float(r[6] or 0), 2), "s1_records": int(r[7])}


def evaluate_candidate_recall(con, mask: Optional[int] = None, cand_table: str = "cand",
                              gt_table: str = "gt_pairs", query_table: str = "q") -> dict:
    """Recall of the candidate set (optionally restricted to pairs whose strategy ``mask``
    intersects) against the true pairs of the query S1 records, plus candidate statistics."""
    flt = f"AND c.mask & {int(mask)} > 0" if mask is not None else ""
    r = con.execute(f"""
        WITH g AS (SELECT g.* , q.country FROM {gt_table} g JOIN {query_table} q ON q.id = g.s1)
        SELECT COUNT(*), COUNT(c.s1),
               COUNT(*) FILTER (WHERE g.src = 2), COUNT(c.s1) FILTER (WHERE g.src = 2),
               COUNT(*) FILTER (WHERE g.src = 3), COUNT(c.s1) FILTER (WHERE g.src = 3),
               list(DISTINCT g.country)
        FROM g LEFT JOIN {cand_table} c ON c.s1 = g.s1 AND c.t = g.t {flt}""").fetchone()
    by_country = {row[0]: {"true_pairs": int(row[1]), "recall": round(row[2] / row[1], 4) if row[1] else None}
                  for row in con.execute(f"""
        WITH g AS (SELECT g.*, q.country FROM {gt_table} g JOIN {query_table} q ON q.id = g.s1)
        SELECT g.country, COUNT(*), COUNT(c.s1) FROM g LEFT JOIN {cand_table} c
        ON c.s1 = g.s1 AND c.t = g.t {flt} GROUP BY 1 ORDER BY 1""").fetchall()}
    out = {"true_pairs": int(r[0]), "recovered": int(r[1]), "recall": round(r[1] / r[0], 4) if r[0] else None,
           "recall_s2": round(r[3] / r[2], 4) if r[2] else None, "recall_s3": round(r[5] / r[4], 4) if r[4] else None,
           "recall_by_country": by_country}
    out.update(candidate_statistics(con, mask, cand_table, query_table))
    out["pair_precision"] = round(out["recovered"] / out["candidates"], 4) if out["candidates"] else None
    return out
