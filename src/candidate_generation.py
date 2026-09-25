"""
Phase 3 — high-recall candidate generation (S1 -> S2 ∪ S3).

Pipeline
--------
1. :func:`materialize` — normalise every training record once with the Phase 2 functions
   (via :func:`src.blocking.blocking_record`) in a process pool and write one Parquet file per
   source to ``data/processed/phase3/`` (gitignored). Only blocking / feature fields are kept.
2. :func:`generate_candidates` — run the blocking strategies A-I (``src.blocking.STRATEGIES``)
   in DuckDB; each writes a deduplicated ``cand_<strategy>(s1, t)`` table.
3. :func:`union_candidates` — ``cand_all(s1, t, mask)``: one row per distinct pair, ``mask`` has
   bit *i* set when strategy *i* produced the pair (deduplication + provenance).
4. :func:`evaluate` — recall / precision / candidates-per-S1 against the training ground truth,
   per strategy, cumulative (incremental) and for the union.

Everything heavy runs inside DuckDB with a memory limit and a bounded spill directory;
Python only touches chunks (materialisation) and small result frames.

Usage (from the project root)::

    python -m src.candidate_generation materialize
    python -m src.candidate_generation run          # strategies + union + evaluation -> JSON

The test set is never read by this module.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Optional

from .blocking import (ID_BASE, RECORD_SCHEMA, STRATEGIES, BlockingConfig, blocking_record,
                       run_strategy)

PROJECT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT / "data" / "raw"
PROC_DIR = PROJECT / "data" / "processed" / "phase3"
TRAIN_FILES = {1: "train_source1.tsv", 2: "train_source2.tsv", 3: "train_source3.tsv"}
GT_FILE = "train_ground_truth.tsv"


# =============================================================================
# DuckDB connection
# =============================================================================

def connect(db_path: Optional[Path] = None, memory_limit: str = "6GB", threads: Optional[int] = None,
            temp_dir: Optional[Path] = None, max_temp: str = "60GB"):
    """DuckDB connection with bounded memory and a bounded spill directory."""
    import duckdb
    con = duckdb.connect(str(db_path) if db_path else ":memory:")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads or os.cpu_count()}")
    con.execute("SET preserve_insertion_order=false")
    if temp_dir is not None:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{Path(temp_dir).as_posix()}'")
        con.execute(f"SET max_temp_directory_size='{max_temp}'")
    return con


def _read_tsv(path: Path) -> str:
    # tab-delimited, header, NO quote character (Phase 1 finding); everything VARCHAR
    return (f"read_csv('{Path(path).as_posix()}', delim='\t', header=true, quote='', escape='', "
            f"all_varchar=true)")


# =============================================================================
# 1. Materialisation
# =============================================================================

def _arrow_schema():
    import pyarrow as pa
    m = {"BIGINT": pa.int64(), "TINYINT": pa.int8(), "VARCHAR": pa.string(), "BOOLEAN": pa.bool_(),
         "VARCHAR[]": pa.list_(pa.string())}
    return pa.schema([(k, m[v]) for k, v in RECORD_SCHEMA.items()])


def _normalize_chunk(rows: tuple[list, list, list, list]) -> list[dict]:
    ids, names, addrs, countries = rows
    return [blocking_record(i, n, a, c) for i, n, a, c in zip(ids, names, addrs, countries)]


def records_glob(src: int, proc_dir: Path = PROC_DIR) -> str:
    """Parquet glob of the materialised records of one source."""
    return (Path(proc_dir) / f"records_s{src}" / "*.parquet").as_posix()


def materialize(raw_dir: Path = RAW_DIR, out_dir: Path = PROC_DIR, workers: Optional[int] = None,
                chunk_rows: int = 50_000, sources: Iterable[int] = (1, 2, 3), verbose: bool = True) -> dict:
    """Normalise every record of the training sources into ``records_s<k>/part-*.parquet``.

    Raw rows are streamed from DuckDB and normalised in waves of ``2 * workers`` chunks, so at
    most a few hundred thousand rows are held in Python at a time. Each wave is written as one
    Parquet part file through DuckDB (pyarrow's Parquet writer is unavailable on this machine).
    """
    import duckdb
    import pyarrow as pa
    out_dir = Path(out_dir)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    schema = _arrow_schema()
    report = {}
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        for src in sources:
            t0 = time.time()
            dest = out_dir / f"records_s{src}"
            if dest.exists():
                for old in dest.glob("*.parquet"):
                    old.unlink()
            dest.mkdir(parents=True, exist_ok=True)
            # separate connections: running a query on the reader's connection would invalidate
            # the streaming result and silently truncate the output
            rcon, con = duckdb.connect(), duckdb.connect()
            rcon.execute("SET threads=4")
            con.execute("SET threads=4")
            expected = rcon.execute(f"SELECT count(*) FROM {_read_tsv(Path(raw_dir) / TRAIN_FILES[src])}").fetchone()[0]
            reader = rcon.execute(f"SELECT entity_id, business_name, business_address, country FROM "
                                  f"{_read_tsv(Path(raw_dir) / TRAIN_FILES[src])}").to_arrow_reader(chunk_rows)
            n = part = 0
            wave: list = []

            def flush(wave: list) -> None:
                nonlocal n, part
                recs = [r for chunk in pool.map(_normalize_chunk, wave) for r in chunk]
                tbl = pa.Table.from_pylist(recs, schema=schema)
                con.register("_wave", tbl)
                con.execute(f"COPY (SELECT * FROM _wave) TO '{(dest / f'part-{part:04d}.parquet').as_posix()}' "
                            f"(FORMAT parquet, COMPRESSION zstd)")
                con.unregister("_wave")
                n += len(recs)
                part += 1

            for batch in reader:
                d = batch.to_pydict()
                wave.append((d["entity_id"], d["business_name"], d["business_address"], d["country"]))
                if len(wave) >= 2 * workers:
                    flush(wave)
                    wave = []
            if wave:
                flush(wave)
            if n != expected:            # never let a truncated materialisation pass silently
                raise RuntimeError(f"S{src}: wrote {n:,} records but the source has {expected:,}")
            mb = sum(f.stat().st_size for f in dest.glob("*.parquet")) / 2**20
            con.close()
            rcon.close()
            report[f"S{src}"] = {"rows": n, "seconds": round(time.time() - t0, 1),
                                 "parts": part, "parquet_mb": round(mb, 1)}
            if verbose:
                print(f"S{src}: {n:,} records in {report[f'S{src}']['seconds']}s "
                      f"-> {dest.name}/ ({part} parts, {round(mb, 1)} MB)", flush=True)
    return report


def register_records(con, proc_dir: Path = PROC_DIR, as_tables: bool = False) -> None:
    """Create ``s1_rec`` and ``t_rec`` (S2 ∪ S3) relations over the materialised Parquet."""
    kind = "TABLE" if as_tables else "VIEW"
    con.execute(f"CREATE OR REPLACE {kind} s1_rec AS SELECT * FROM read_parquet('{records_glob(1, proc_dir)}')")
    con.execute(f"""CREATE OR REPLACE {kind} t_rec AS SELECT * FROM read_parquet(
        ['{records_glob(2, proc_dir)}', '{records_glob(3, proc_dir)}'])""")


def load_ground_truth(con, raw_dir: Path = RAW_DIR, table: str = "gt") -> int:
    """``gt(s1 BIGINT, t BIGINT)``: one row per true pair (targets exploded, IDs encoded)."""
    con.execute(f"""CREATE OR REPLACE TABLE {table} AS
        SELECT DISTINCT
          CAST(substr(source1_entity_id, 2, 1) AS BIGINT) * {ID_BASE} + CAST(split_part(source1_entity_id, '-', 2) AS BIGINT) AS s1,
          CAST(substr(m, 2, 1) AS BIGINT) * {ID_BASE} + CAST(split_part(m, '-', 2) AS BIGINT) AS t
        FROM (SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) m
              FROM {_read_tsv(Path(raw_dir) / GT_FILE)} WHERE matched_entity_ids IS NOT NULL)
        WHERE trim(m) <> ''""")
    return con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


# =============================================================================
# 2-3. Candidate generation and union
# =============================================================================

def generate_candidates(con, cfg: BlockingConfig, strategies: Optional[list[str]] = None,
                        s1_rel: str = "s1_rec", t_rel: str = "t_rec", verbose: bool = True) -> dict:
    """Run each strategy; returns ``{strategy: {"seconds": .., "stats": ..}}``."""
    import psutil
    proc = psutil.Process()
    out = {}
    for name in strategies or list(STRATEGIES):
        t0 = time.time()
        stats = run_strategy(con, name, s1_rel, t_rel, cfg)
        n = con.execute(f"SELECT count(*) FROM cand_{name}").fetchone()[0]
        out[name] = {"seconds": round(time.time() - t0, 1), "pairs": n, "stats": stats,
                     "rss_mb_after": round(proc.memory_info().rss / 2**20)}
        if verbose:
            print(f"  {STRATEGIES[name][0]} {name:<13} {n:>13,} pairs  {out[name]['seconds']:>7.1f}s  {stats}", flush=True)
    return out


def union_candidates(con, strategies: Optional[list[str]] = None, out_table: str = "cand_all") -> int:
    """Deduplicated union ``out_table(s1, t, mask)``; bit i of ``mask`` = strategy i produced it."""
    names = strategies or list(STRATEGIES)
    parts = " UNION ALL ".join(f"SELECT s1, t, {1 << i}::INTEGER AS m FROM cand_{n}" for i, n in enumerate(names))
    con.execute(f"CREATE OR REPLACE TABLE {out_table} AS SELECT s1, t, bit_or(m) AS mask FROM ({parts}) GROUP BY s1, t")
    return con.execute(f"SELECT count(*) FROM {out_table}").fetchone()[0]


# =============================================================================
# 4. Evaluation
# =============================================================================

def _per_s1_stats(con, pair_sql: str, s1_rel: str) -> dict:
    """Candidates per S1 record, over *all* S1 records (records without candidates count as 0)."""
    r = con.execute(f"""
        WITH c AS (SELECT s1, count(*) n FROM ({pair_sql}) GROUP BY s1),
             a AS (SELECT coalesce(c.n, 0) n FROM {s1_rel} s LEFT JOIN c ON c.s1 = s.rid)
        SELECT avg(n), quantile_disc(n, 0.5), quantile_disc(n, 0.95), quantile_disc(n, 0.99), max(n),
               avg((n > 0)::INT)
        FROM a""").fetchone()
    return {"avg_per_s1": round(r[0], 3), "median_per_s1": r[1], "p95_per_s1": r[2], "p99_per_s1": r[3],
            "max_per_s1": r[4], "s1_with_candidates_pct": round(100 * r[5], 3)}


def evaluate_pairs(con, pair_sql: str, gt: str = "gt", s1_rel: str = "s1_rec") -> dict:
    """Recall, precision and volume of a pair relation ``(s1, t)`` (must be distinct)."""
    total_true = con.execute(f"SELECT count(*) FROM {gt}").fetchone()[0]
    n, hit = con.execute(f"""SELECT count(*), count(g.s1) FROM ({pair_sql}) c
                             LEFT JOIN {gt} g ON g.s1 = c.s1 AND g.t = c.t""").fetchone()
    return {"candidates": n, "true_found": hit, "recall_pct": round(100 * hit / total_true, 3),
            "precision_pct": round(100 * hit / n, 3) if n else 0.0, **_per_s1_stats(con, pair_sql, s1_rel)}


def evaluate(con, strategies: Optional[list[str]] = None, gt: str = "gt", s1_rel: str = "s1_rec",
             union_table: str = "cand_all", verbose: bool = True) -> dict:
    """Per-strategy, incremental, unique-contribution and breakdown metrics."""
    names = strategies or list(STRATEGIES)
    total_true = con.execute(f"SELECT count(*) FROM {gt}").fetchone()[0]
    res: dict = {"total_true_pairs": total_true, "per_strategy": {}, "incremental": [], "unique": {}}

    for n in names:
        res["per_strategy"][n] = evaluate_pairs(con, f"SELECT s1, t FROM cand_{n}", gt, s1_rel)
        if verbose:
            m = res["per_strategy"][n]
            print(f"  {n:<13} cand {m['candidates']:>13,}  recall {m['recall_pct']:6.2f}%  "
                  f"avg/S1 {m['avg_per_s1']:8.2f}  p99 {m['p99_per_s1']}", flush=True)

    # true pairs with their strategy mask (0 = missed) -> incremental recall / unique contribution
    con.execute(f"""CREATE OR REPLACE TABLE gt_mask AS
        SELECT g.s1, g.t, coalesce(c.mask, 0) AS mask FROM {gt} g LEFT JOIN {union_table} c ON c.s1 = g.s1 AND c.t = g.t""")
    prefix_masks = [(1 << (i + 1)) - 1 for i in range(len(names))]
    rec = con.execute("SELECT " + ", ".join(f"count(*) FILTER (WHERE mask & {m} <> 0)" for m in prefix_masks)
                      + " FROM gt_mask").fetchone()
    vol = con.execute("SELECT " + ", ".join(f"count(*) FILTER (WHERE mask & {m} <> 0)" for m in prefix_masks)
                      + f" FROM {union_table}").fetchone()
    for i, n in enumerate(names):
        res["incremental"].append({"added": n, "strategies": names[:i + 1], "candidates": vol[i],
                                   "recall_pct": round(100 * rec[i] / total_true, 3),
                                   "avg_per_s1": round(vol[i] / con.execute(f"SELECT count(*) FROM {s1_rel}").fetchone()[0], 3)})
    uniq = con.execute("SELECT " + ", ".join(f"count(*) FILTER (WHERE mask = {1 << i})" for i in range(len(names)))
                       + " FROM gt_mask").fetchone()
    for i, n in enumerate(names):
        res["unique"][n] = {"true_pairs_only_this": uniq[i], "pct_of_true": round(100 * uniq[i] / total_true, 3)}

    res["union"] = evaluate_pairs(con, f"SELECT s1, t FROM {union_table}", gt, s1_rel)

    # breakdowns of union recall: by target source and by country
    res["union_by_source"] = con.execute(f"""
        SELECT 'S' || (t // {ID_BASE}) src, count(*) true_pairs, round(100 * avg((mask <> 0)::INT), 3) recall_pct
        FROM gt_mask GROUP BY ALL ORDER BY 1""").df().to_dict("records")
    res["union_by_country"] = con.execute(f"""
        SELECT s.country, count(*) true_pairs, round(100 * avg((g.mask <> 0)::INT), 3) recall_pct
        FROM gt_mask g JOIN {s1_rel} s ON s.rid = g.s1 GROUP BY ALL ORDER BY 1""").df().to_dict("records")
    # share of S1 records whose true matches are *all* retrieved
    r = con.execute("""SELECT count(*), count(*) FILTER (WHERE missed = 0) FROM
                       (SELECT s1, count(*) FILTER (WHERE mask = 0) missed FROM gt_mask GROUP BY s1)""").fetchone()
    res["s1_all_matches_retrieved_pct"] = round(100 * r[1] / r[0], 3)
    return res


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["materialize", "run"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--memory-limit", default="6GB")
    ap.add_argument("--db", default=str(PROC_DIR / "phase3.duckdb"))
    ap.add_argument("--out", default=str(PROJECT / "experiments" / "phase3_candidate_generation_run.json"))
    args = ap.parse_args(argv)
    if args.command == "materialize":
        print(json.dumps(materialize(workers=args.workers), indent=2))
        return
    cfg = BlockingConfig()
    con = connect(Path(args.db), memory_limit=args.memory_limit, temp_dir=PROC_DIR / "duckdb_tmp")
    t0 = time.time()
    register_records(con)
    load_ground_truth(con)
    gen = generate_candidates(con, cfg)
    union_candidates(con)
    ev = evaluate(con)
    ev["generation"] = gen
    ev["config"] = asdict(cfg)
    ev["total_seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(ev, indent=2, default=str), encoding="utf-8")
    print(json.dumps(ev["union"], indent=2))


if __name__ == "__main__":
    main()
