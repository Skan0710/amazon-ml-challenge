"""
Phase 4 — pairwise features, matching model, decision rule, submission writing.

Pipeline (every step bounded in memory; DuckDB + Parquet, never a full pandas frame)::

    Phase 3 candidates (s1, t, mask)            integer ids, one row per unique pair
      -> pairwise features (DuckDB SQL)         per S1 hash chunk -> Parquet parts
      -> labels (ground truth, training only)
      -> split by S1 entity (train / validation)
      -> negative sampling (all positives + hard negatives + weighted random negatives)
      -> HistGradientBoostingClassifier (scikit-learn; BSD licence, no new dependency)
      -> probability per pair (chunked)
      -> per-S1 decision rule (threshold / top-k / relative-to-best), chosen on validation
         by the competition metric: macro F0.5 over *all* S1 entities incl. singletons
      -> candidate_pairs.tsv + matching_results.tsv (one row per S1, original id strings)
      -> scripts/validate_final_submission.py + official utils/validate_submission.py

Competition metric (README): F0.5 per S1 entity, averaged over all S1 entities. A singleton
scores 1.0 for an empty prediction and 0.0 for any prediction; an entity with true matches
scores 0.0 for an empty prediction.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src.candidate_generation import ID_BASE, STRATEGY_BITS, check_disk

__all__ = [
    "materialize_text_features", "register_text", "candidate_dedup_report",
    "FEATURE_COLUMNS", "feature_sql", "compute_features", "compute_features_chunked",
    "assign_split", "sample_training_rows", "train_model", "score_pairs", "infer_chunked", "ModelConfig",
    "DecisionRule", "apply_decision", "evaluate_predictions", "entity_f05",
    "search_decision_rules", "singleton_report",
    "write_candidate_pairs_tsv", "write_matching_results_tsv", "SubmissionError",
    "verify_no_forced_matches", "finalize_outputs",
]


class SubmissionError(RuntimeError):
    """A hard submission invariant was violated. The pipeline stops; nothing is repaired."""


# =============================================================================
# Text side table: original id strings + raw-name derived columns (DuckDB only)
# =============================================================================

def materialize_text_features(tsv_path: str | Path, out_path: str | Path, overwrite: bool = False) -> dict:
    """Write ``(id, src, rn, entity_id, name_lc, name_indic, addr_missing)`` Parquet for one source.

    * ``entity_id`` keeps the **original id string**; outputs are always written from it (never
      rebuilt from the integer id, so zero padding such as ``S1-00001`` survives).
    * The integer encoding is verified to be injective and to round-trip; otherwise it raises.
    Pure DuckDB (no Python loop over rows).
    """
    import duckdb

    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        return {"skipped": True}
    con = duckdb.connect()
    con.execute("SET threads=2")
    src_sql = f"read_csv_auto('{tsv_path}', delim='\t', header=true, all_varchar=true)"
    con.execute(f"""CREATE TEMP TABLE t AS
        SELECT CAST(substr(trim(entity_id), 2, 1) AS TINYINT) AS src, trim(entity_id) AS entity_id,
               row_number() OVER () AS rn, business_name, business_address
        FROM {src_sql}""")
    bad = con.execute(r"""SELECT COUNT(*) FROM t WHERE NOT regexp_matches(entity_id, '^S[123]-[0-9]{1,9}$')""").fetchone()[0]
    if bad:
        raise SubmissionError(f"{tsv_path}: {bad} entity ids do not match ^S[123]-[0-9]{{1,9}}$")
    con.execute(f"""CREATE TEMP TABLE e AS SELECT *,
        CASE WHEN src = 1 THEN CAST(substr(entity_id, 4) AS BIGINT)
             ELSE CAST(src AS BIGINT) * {ID_BASE} + CAST(substr(entity_id, 4) AS BIGINT) END AS id FROM t""")
    n, n_id, n_str = con.execute("SELECT COUNT(*), COUNT(DISTINCT id), COUNT(DISTINCT entity_id) FROM e").fetchone()
    if not (n == n_id == n_str):
        raise SubmissionError(f"{tsv_path}: id encoding not injective (rows {n}, ids {n_id}, strings {n_str})")
    con.execute(f"""COPY (
        SELECT id, src, rn, entity_id,
               nullif(regexp_replace(trim(lower(business_name)), '\\s+', ' ', 'g'), '') AS name_lc,
               regexp_matches(coalesce(business_name, ''), '[\\x{{0900}}-\\x{{0DFF}}]') AS name_indic,
               (business_address IS NULL OR trim(business_address) = ''
                OR lower(trim(business_address)) IN ('null', 'n/a', 'na', '<null>', 'none', 'nan')) AS addr_missing
        FROM e ORDER BY rn) TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)""")
    con.close()
    return {"rows": int(n), "bytes": out_path.stat().st_size}


def register_text(con, text_dir: str | Path, sources: Sequence[int] = (1, 2, 3), prefix: str = "") -> None:
    """Views ``text_s1`` and ``text_t`` (S2 ∪ S3) over the text Parquet files."""
    d = Path(text_dir)
    con.execute(f"CREATE OR REPLACE VIEW text_s1 AS SELECT * FROM read_parquet('{d}/{prefix}text_s1.parquet')")
    con.execute(f"""CREATE OR REPLACE VIEW text_t AS SELECT * FROM read_parquet(
        ['{d}/{prefix}text_s2.parquet', '{d}/{prefix}text_s3.parquet'])""")


def candidate_dedup_report(gen_info: dict, strategies: Iterable[str]) -> dict:
    """Raw rows (sum over strategies, each already distinct within itself), duplicates removed by
    the union, and unique pairs — measured, never silent."""
    per = gen_info.get("per_strategy") or {s: gen_info[s] for s in strategies if s in gen_info}
    raw = int(sum(v.get("pairs", 0) for v in per.values()))
    return {"raw_candidate_rows": raw, "per_strategy_rows": {s: int(v.get("pairs", 0)) for s, v in per.items()}}


# =============================================================================
# Pairwise features — one SQL statement, evaluated per chunk of candidate pairs
# =============================================================================

#: Feature columns in model order. NULL (NaN) means "not comparable" (e.g. a side without a house
#: number); HistGradientBoosting handles NaN natively, so missing never looks like similarity.
FEATURE_COLUMNS: tuple[str, ...] = (
    # ---- name
    "f_name_lc_eq", "f_key_eq", "f_keys_overlap", "f_name_jw", "f_name_lev", "f_core_jw", "f_raw_jw",
    "f_tok_jacc", "f_tok_shared", "f_tok_contain", "f_phon_eq", "f_phon_jacc", "f_gram_jacc",
    "f_len_diff", "f_ntok_diff", "f_a_indic", "f_b_indic", "f_script_mismatch", "f_key_freq_log",
    # ---- address
    "f_a_addr_missing", "f_b_addr_missing", "f_both_addr_missing", "f_state_eq", "f_hn_eq",
    "f_hn_absdiff_log", "f_street_eq", "f_street_jw", "f_place_jacc", "f_place_shared",
    "f_addr_tok_jacc", "f_addr_tok_shared", "f_num_jacc", "f_num_shared",
    # ---- structure / candidate generation
    "f_src", "m_name", "m_phonetic", "m_structured", "m_rare_name", "m_rare_addr", "m_ngram",
    "m_compound", "f_n_strategies", "f_n_cands_log", "f_name_jw_rank", "f_name_jw_gap",
    "f_addr_jacc_rank", "f_addr_jacc_gap",
)

_MACROS = r"""
CREATE OR REPLACE MACRO _tok(s, sep) AS list_distinct(list_filter(string_split(coalesce(s, ''), sep), x -> x <> ''));
CREATE OR REPLACE MACRO _jacc(a, b) AS CASE WHEN len(a) = 0 OR len(b) = 0 THEN NULL
     ELSE len(list_intersect(a, b))::DOUBLE / len(list_distinct(list_concat(a, b))) END;
CREATE OR REPLACE MACRO _grams(s) AS list_distinct(list_transform(range(1, length('#' || s || '#') - 1),
     i -> substr('#' || s || '#', i, 3)));
"""


def feature_sql(pairs_sql: str, with_label: bool) -> str:
    """SQL producing ``s1, t, [label,] <FEATURE_COLUMNS>`` for the pairs returned by ``pairs_sql``
    (columns ``s1, t, mask``). Needs views feat_s1, feat_t, text_s1, text_t, table st_name and,
    when ``with_label``, table gt_pairs(s1, t)."""
    bits = STRATEGY_BITS
    label = ", (g.s1 IS NOT NULL)::TINYINT AS label" if with_label else ""
    gt_join = "LEFT JOIN gt_pairs g ON g.s1 = p.s1 AND g.t = p.t" if with_label else ""
    return f"""
WITH p AS ({pairs_sql}),
j AS (
  SELECT p.s1, p.t, p.mask{label},
         a.country, a.name_key a_key, b.name_key b_key, a.name_keys a_keys, b.name_keys b_keys,
         a.name_core a_core, b.name_core b_core,
         _tok(a.name_tokens, ' ') a_tok, _tok(b.name_tokens, ' ') b_tok,
         a.name_phon a_phon, b.name_phon b_phon, _tok(a.name_phon, ' ') a_ptok, _tok(b.name_phon, ' ') b_ptok,
         a.state a_state, b.state b_state, a.hn a_hn, b.hn b_hn, a.street a_street, b.street b_street,
         _tok(a.places, '|') a_pl, _tok(b.places, '|') b_pl,
         _tok(a.addr_tokens, ' ') a_at, _tok(b.addr_tokens, ' ') b_at,
         _tok(a.numbers, ' ') a_num, _tok(b.numbers, ' ') b_num,
         ta.name_lc a_lc, tb.name_lc b_lc, ta.name_indic a_indic, tb.name_indic b_indic,
         ta.addr_missing a_miss, tb.addr_missing b_miss, CAST(p.t // {ID_BASE} AS TINYINT) AS src
  FROM p
  JOIN feat_s1 a ON a.id = p.s1 JOIN feat_t b ON b.id = p.t
  JOIN text_s1 ta ON ta.id = p.s1 JOIN text_t tb ON tb.id = p.t
  {gt_join}),
f AS (
  SELECT j.s1, j.t{', j.label' if with_label else ''},
    (a_lc IS NOT NULL AND a_lc = b_lc)::TINYINT f_name_lc_eq,
    (a_key <> '' AND a_key = b_key)::TINYINT f_key_eq,
    (len(list_intersect(_tok(a_keys, '|'), _tok(b_keys, '|'))) > 0)::TINYINT f_keys_overlap,
    CASE WHEN a_key <> '' AND b_key <> '' THEN jaro_winkler_similarity(a_key, b_key) END f_name_jw,
    CASE WHEN a_key <> '' AND b_key <> '' THEN 1 - levenshtein(a_key, b_key)::DOUBLE / greatest(length(a_key), length(b_key)) END f_name_lev,
    CASE WHEN a_core <> '' AND b_core <> '' THEN jaro_winkler_similarity(a_core, b_core) END f_core_jw,
    CASE WHEN a_lc IS NOT NULL AND b_lc IS NOT NULL THEN jaro_winkler_similarity(a_lc, b_lc) END f_raw_jw,
    _jacc(a_tok, b_tok) f_tok_jacc,
    len(list_intersect(a_tok, b_tok)) f_tok_shared,
    CASE WHEN len(a_tok) > 0 AND len(b_tok) > 0 THEN len(list_intersect(a_tok, b_tok))::DOUBLE / least(len(a_tok), len(b_tok)) END f_tok_contain,
    (a_phon <> '' AND a_phon = b_phon)::TINYINT f_phon_eq,
    _jacc(a_ptok, b_ptok) f_phon_jacc,
    CASE WHEN a_key <> '' AND b_key <> '' THEN _jacc(_grams(a_key), _grams(b_key)) END f_gram_jacc,
    abs(length(a_key) - length(b_key)) f_len_diff,
    abs(len(a_tok) - len(b_tok)) f_ntok_diff,
    a_indic::TINYINT f_a_indic, b_indic::TINYINT f_b_indic, (a_indic <> b_indic)::TINYINT f_script_mismatch,
    ln(1 + coalesce(sn.n, 0)) f_key_freq_log,
    a_miss::TINYINT f_a_addr_missing, b_miss::TINYINT f_b_addr_missing, (a_miss AND b_miss)::TINYINT f_both_addr_missing,
    CASE WHEN a_state <> '' AND b_state <> '' THEN (a_state = b_state)::TINYINT END f_state_eq,
    CASE WHEN a_hn <> '' AND b_hn <> '' THEN (a_hn = b_hn)::TINYINT END f_hn_eq,
    CASE WHEN TRY_CAST(a_hn AS BIGINT) IS NOT NULL AND TRY_CAST(b_hn AS BIGINT) IS NOT NULL
         THEN ln(1 + abs(TRY_CAST(a_hn AS BIGINT) - TRY_CAST(b_hn AS BIGINT))) END f_hn_absdiff_log,
    CASE WHEN a_street <> '' AND b_street <> '' THEN (a_street = b_street)::TINYINT END f_street_eq,
    CASE WHEN a_street <> '' AND b_street <> '' THEN jaro_winkler_similarity(a_street, b_street) END f_street_jw,
    _jacc(a_pl, b_pl) f_place_jacc, len(list_intersect(a_pl, b_pl)) f_place_shared,
    _jacc(a_at, b_at) f_addr_tok_jacc, len(list_intersect(a_at, b_at)) f_addr_tok_shared,
    _jacc(a_num, b_num) f_num_jacc, len(list_intersect(a_num, b_num)) f_num_shared,
    src f_src,
    (mask & {bits['name']} > 0)::TINYINT m_name, (mask & {bits['phonetic']} > 0)::TINYINT m_phonetic,
    (mask & {bits['structured']} > 0)::TINYINT m_structured, (mask & {bits['rare_name']} > 0)::TINYINT m_rare_name,
    (mask & {bits['rare_addr']} > 0)::TINYINT m_rare_addr, (mask & {bits['ngram']} > 0)::TINYINT m_ngram,
    (mask & {bits['compound']} > 0)::TINYINT m_compound, bit_count(mask) f_n_strategies
  FROM j LEFT JOIN st_name sn ON sn.country = j.country AND sn.k = j.b_key)
SELECT *,
  ln(COUNT(*) OVER w) f_n_cands_log,
  rank() OVER (PARTITION BY s1 ORDER BY f_name_jw DESC NULLS LAST) f_name_jw_rank,
  max(f_name_jw) OVER w - f_name_jw f_name_jw_gap,
  rank() OVER (PARTITION BY s1 ORDER BY f_addr_tok_jacc DESC NULLS LAST) f_addr_jacc_rank,
  max(f_addr_tok_jacc) OVER w - f_addr_tok_jacc f_addr_jacc_gap
FROM f WINDOW w AS (PARTITION BY s1)
"""


def install_macros(con) -> None:
    for stmt in _MACROS.strip().split(";\n"):
        if stmt.strip():
            con.execute(stmt)


def _select_cols(with_label: bool) -> str:
    """s1, t, [label], features cast to FLOAT (32-bit) — half the size of DOUBLE on disk and in RAM."""
    feats = ", ".join(f"CAST({c} AS FLOAT) AS {c}" for c in FEATURE_COLUMNS)
    return "s1, t, " + ("label, " if with_label else "") + feats


def compute_features(con, pairs_sql: str, with_label: bool, out_table: str = "pair_features") -> int:
    """Materialise features for the pairs of ``pairs_sql`` into DuckDB table ``out_table``."""
    install_macros(con)
    cols = _select_cols(with_label)
    con.execute(f"CREATE OR REPLACE TABLE {out_table} AS SELECT {cols} FROM ({feature_sql(pairs_sql, with_label)})")
    return con.execute(f"SELECT COUNT(*) FROM {out_table}").fetchone()[0]


def compute_features_chunked(con, cand_table: str, out_dir: str | Path, with_label: bool, n_chunks: int,
                             extra_cols_sql: str = "", min_free_gb: float = 3.0, log=print) -> dict:
    """Features for every pair of ``cand_table`` (s1, t, mask), one Parquet part per S1 chunk.

    Chunks are by S1 (salted hash), so all candidates of an S1 — needed by the per-S1 rank
    features — are always in the same chunk. ``extra_cols_sql`` adds columns (e.g. the split).
    """
    install_macros(con)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = _select_cols(with_label)
    t0, rows = time.time(), 0
    for i in range(int(n_chunks)):
        check_disk(out_dir, min_free_gb)
        t = time.time()
        pairs = f"SELECT s1, t, mask FROM {cand_table} WHERE hash(s1, 3) % {int(n_chunks)} = {i}"
        part = out_dir / f"part-{i:05d}.parquet"
        con.execute(f"COPY (SELECT {cols}{extra_cols_sql} FROM ({feature_sql(pairs, with_label)})) "
                    f"TO '{part}' (FORMAT parquet, COMPRESSION zstd)")
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{part}')").fetchone()[0]
        rows += n
        log(f"  features chunk {i + 1}/{n_chunks}: {n:,} rows in {time.time() - t:.1f}s")
    size = sum(p.stat().st_size for p in out_dir.glob("*.parquet"))
    return {"rows": rows, "seconds": round(time.time() - t0, 1), "bytes": size}


def infer_chunked(con, cand_sql: str, model, rule: "DecisionRule", n_chunks: int, pred_table: str = "pred",
                  keep_scores: bool = False, min_free_gb: float = 3.0, log=print) -> dict:
    """Streaming inference for large candidate sets: per S1 chunk -> features (temp table) -> scores
    -> per-S1 decision -> append predictions; the chunk's features are dropped immediately.

    Only the *predicted* pairs (and optionally the scores) are kept, so memory and disk are bounded
    by one chunk. The decision rule is per S1 and chunks are split by S1, so the result is identical
    to scoring everything first (tested)."""
    con.execute(f"CREATE OR REPLACE TABLE {pred_table} (s1 BIGINT, t BIGINT, p DOUBLE)")
    if keep_scores:
        con.execute("CREATE OR REPLACE TABLE scores_all (s1 BIGINT, t BIGINT, p DOUBLE)")
    t0, n_pairs, n_pred = time.time(), 0, 0
    for i in range(int(n_chunks)):
        check_disk(".", min_free_gb)
        t = time.time()
        pairs = f"SELECT s1, t, mask FROM ({cand_sql}) WHERE hash(s1, 3) % {int(n_chunks)} = {i}"
        n = compute_features(con, pairs, with_label=False, out_table="_chunk_features")
        score_pairs(con, model, "SELECT * FROM _chunk_features", out_table="_chunk_scores")
        apply_decision(con, rule, scores_table="_chunk_scores", out_table="_chunk_pred")
        con.execute(f"INSERT INTO {pred_table} SELECT * FROM _chunk_pred")
        if keep_scores:
            con.execute("INSERT INTO scores_all SELECT * FROM _chunk_scores")
        k = con.execute("SELECT COUNT(*) FROM _chunk_pred").fetchone()[0]
        for tname in ("_chunk_features", "_chunk_scores", "_chunk_pred"):
            con.execute(f"DROP TABLE IF EXISTS {tname}")
        n_pairs += n
        n_pred += k
        log(f"  inference chunk {i + 1}/{n_chunks}: {n:,} pairs -> {k:,} matches in {time.time() - t:.1f}s")
    return {"pairs": n_pairs, "predicted": n_pred, "seconds": round(time.time() - t0, 1)}


# =============================================================================
# Split, negative sampling, model
# =============================================================================

def assign_split(s1_col: str = "s1", val_pct: int = 20, salt: int = 42) -> str:
    """SQL expression: ``'val'`` for ~``val_pct``% of S1 entities (deterministic), else ``'train'``.
    Split is by S1 entity, so all candidates of one S1 are on the same side."""
    return f"CASE WHEN hash({s1_col}, {int(salt)}) % 100 < {int(val_pct)} THEN 'val' ELSE 'train' END"


HARD_NEGATIVE_SQL = ("(f_name_jw >= 0.85 OR f_tok_contain >= 0.5 OR f_phon_eq = 1 OR f_hn_eq = 1 "
                     "OR f_addr_tok_jacc >= 0.5 OR f_n_strategies >= 3)")


def sample_training_rows(con, features_glob: str, hard_rate: float = 1.0, easy_rate: float = 0.1,
                         seed: int = 7) -> tuple[pd.DataFrame, dict]:
    """All positives + hard negatives (sampled at ``hard_rate``) + easy negatives (``easy_rate``).

    Hard negatives: known non-matches that look similar (similar/contained name, same phonetic
    key, same house number, similar address, or found by >= 3 strategies). Each kept row gets
    weight 1/rate, so the weighted training set reproduces the full candidate distribution.
    Sampling is a deterministic hash of the pair.
    """
    cols = ", ".join(FEATURE_COLUMNS)
    q = f"""SELECT s1, t, label, {cols},
          CASE WHEN label = 1 THEN 1.0 WHEN {HARD_NEGATIVE_SQL} THEN {1 / hard_rate} ELSE {1 / easy_rate} END AS w,
          CASE WHEN label = 1 THEN 'positive' WHEN {HARD_NEGATIVE_SQL} THEN 'hard_negative' ELSE 'easy_negative' END AS kind
        FROM read_parquet('{features_glob}') WHERE split = 'train' AND (
          label = 1
          OR ({HARD_NEGATIVE_SQL} AND hash(s1, t, {seed}) % 1000000 < {int(hard_rate * 1e6)})
          OR (NOT {HARD_NEGATIVE_SQL} AND hash(s1, t, {seed}) % 1000000 < {int(easy_rate * 1e6)}))"""
    df = con.execute(q).df()
    pop = con.execute(f"""SELECT COUNT(*) FILTER (WHERE label = 1), COUNT(*) FILTER (WHERE label = 0 AND {HARD_NEGATIVE_SQL}),
        COUNT(*) FILTER (WHERE label = 0 AND NOT {HARD_NEGATIVE_SQL}) FROM read_parquet('{features_glob}') WHERE split = 'train'""").fetchone()
    info = {"population": {"positive": int(pop[0]), "hard_negative": int(pop[1]), "easy_negative": int(pop[2])},
            "sampled": df["kind"].value_counts().to_dict(), "hard_rate": hard_rate, "easy_rate": easy_rate}
    return df, info


@dataclass
class ModelConfig:
    """HistGradientBoostingClassifier settings (scikit-learn, BSD-3 licence)."""
    max_iter: int = 400
    learning_rate: float = 0.08
    max_leaf_nodes: int = 63
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 30
    random_state: int = 0


def train_model(train_df: pd.DataFrame, cfg: Optional[ModelConfig] = None):
    """Fit the classifier on sampled rows with sampling weights; returns ``(model, info)``."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    cfg = cfg or ModelConfig()
    X = train_df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    y = train_df["label"].to_numpy(dtype=np.int8)
    w = train_df["w"].to_numpy(dtype=np.float32)
    model = HistGradientBoostingClassifier(**asdict(cfg))
    t = time.time()
    model.fit(X, y, sample_weight=w)
    return model, {"seconds": round(time.time() - t, 1), "n_iter": int(model.n_iter_), "rows": int(len(y)),
                   "positives": int(y.sum()), "config": asdict(cfg)}


def score_pairs(con, model, source_sql: str, out_table: str = "scores", batch_rows: int = 250_000) -> dict:
    """Stream rows of ``source_sql`` (s1, t, [label], features) through the model in batches and
    store ``(s1, t, p)`` in DuckDB — never the whole feature matrix in memory."""
    cur = con.cursor()
    cur.execute(f"SELECT s1, t, {', '.join(FEATURE_COLUMNS)} FROM ({source_sql})")
    con.execute(f"CREATE OR REPLACE TABLE {out_table} (s1 BIGINT, t BIGINT, p DOUBLE)")
    t0, n = time.time(), 0
    while True:
        df = cur.fetch_df_chunk(max(1, batch_rows // 2048))
        if df is None or len(df) == 0:
            break
        X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        out = pd.DataFrame({"s1": df["s1"].to_numpy(), "t": df["t"].to_numpy(), "p": model.predict_proba(X)[:, 1]})
        con.register("score_batch", out)
        con.execute(f"INSERT INTO {out_table} SELECT * FROM score_batch")
        con.unregister("score_batch")
        n += len(df)
    cur.close()
    return {"rows": n, "seconds": round(time.time() - t0, 1)}


# =============================================================================
# Decision rule, metric, search
# =============================================================================

@dataclass(frozen=True)
class DecisionRule:
    """Keep candidate (s1, t) iff ``p >= threshold`` and, optionally, its rank within s1 is
    ``<= top_k`` and ``p >= relative * max_p(s1)``. An S1 with no surviving candidate gets an
    empty list — a match is never forced."""
    threshold: float = 0.5
    top_k: Optional[int] = None
    relative: Optional[float] = None

    def where_sql(self) -> str:
        c = [f"p >= {float(self.threshold)}"]
        if self.top_k:
            c.append(f"rk <= {int(self.top_k)}")
        if self.relative:
            c.append(f"p >= {float(self.relative)} * pmax")
        return " AND ".join(c)


def apply_decision(con, rule: DecisionRule, scores_table: str = "scores", out_table: str = "pred") -> int:
    con.execute(f"""CREATE OR REPLACE TABLE {out_table} AS SELECT s1, t, p FROM (
        SELECT s1, t, p, row_number() OVER (PARTITION BY s1 ORDER BY p DESC, t) rk, max(p) OVER (PARTITION BY s1) pmax
        FROM {scores_table}) WHERE {rule.where_sql()}""")
    return con.execute(f"SELECT COUNT(*) FROM {out_table}").fetchone()[0]


def entity_f05(n_true: int, n_pred: int, tp: int) -> float:
    """Competition F0.5 of one S1 entity (README): singleton -> 1.0 iff nothing predicted."""
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0 or tp == 0:
        return 0.0
    p, r = tp / n_pred, tp / n_true
    return 1.25 * p * r / (0.25 * p + r)


def evaluate_predictions(con, pred_table: str = "pred", universe_table: str = "val_s1",
                         gt_table: str = "gt_pairs") -> dict:
    """Macro F0.5 over every S1 in ``universe_table(id)`` (singletons and S1 without candidates
    included), plus pair-level precision/recall/FP/FN and singleton behaviour."""
    r = con.execute(f"""
      WITH gt AS (SELECT g.s1, g.t FROM {gt_table} g JOIN {universe_table} u ON u.id = g.s1),
      pr AS (SELECT p.s1, p.t FROM {pred_table} p JOIN {universe_table} u ON u.id = p.s1),
      nt AS (SELECT s1, COUNT(*) n FROM gt GROUP BY 1), np AS (SELECT s1, COUNT(*) n FROM pr GROUP BY 1),
      tp AS (SELECT pr.s1, COUNT(*) n FROM pr JOIN gt USING (s1, t) GROUP BY 1),
      e AS (SELECT u.id, coalesce(nt.n, 0) nt, coalesce(np.n, 0) np, coalesce(tp.n, 0) tp
            FROM {universe_table} u LEFT JOIN nt ON nt.s1 = u.id LEFT JOIN np ON np.s1 = u.id LEFT JOIN tp ON tp.s1 = u.id),
      f AS (SELECT *, CASE WHEN nt = 0 THEN (np = 0)::DOUBLE WHEN np = 0 OR tp = 0 THEN 0.0
                           ELSE 1.25 * (tp / np) * (tp / nt) / (0.25 * (tp / np) + (tp / nt)) END f05 FROM e)
      SELECT AVG(f05), COUNT(*), SUM(nt), SUM(np), SUM(tp),
             COUNT(*) FILTER (WHERE nt = 0), COUNT(*) FILTER (WHERE nt = 0 AND np = 0),
             COUNT(*) FILTER (WHERE nt > 0), AVG(f05) FILTER (WHERE nt > 0), COUNT(*) FILTER (WHERE np = 0),
             COUNT(*) FILTER (WHERE nt > 0 AND np = 0)
      FROM f""").fetchone()
    tot_true, tot_pred, tp = int(r[2] or 0), int(r[3] or 0), int(r[4] or 0)
    return {"macro_f05": round(float(r[0]), 5), "s1_entities": int(r[1]),
            "pair_precision": round(tp / tot_pred, 5) if tot_pred else None,
            "pair_recall": round(tp / tot_true, 5) if tot_true else None,
            "true_pairs": tot_true, "predicted_pairs": tot_pred, "true_positives": tp,
            "false_positives": tot_pred - tp, "false_negatives": tot_true - tp,
            "singletons": int(r[5]), "singleton_empty": int(r[6]), "singleton_nonempty": int(r[5] - r[6]),
            "non_singletons": int(r[7]), "macro_f05_non_singletons": round(float(r[8] or 0), 5),
            "s1_predicted_empty": int(r[9]), "non_singleton_predicted_empty": int(r[10])}


def search_decision_rules(con, thresholds: Sequence[float], top_ks: Sequence[Optional[int]] = (None,),
                          relatives: Sequence[Optional[float]] = (None,), **kw) -> pd.DataFrame:
    rows = []
    for rel in relatives:
        for k in top_ks:
            for th in thresholds:
                rule = DecisionRule(th, k, rel)
                apply_decision(con, rule)
                rows.append({"threshold": th, "top_k": k or 0, "relative": rel or 0.0, **evaluate_predictions(con, **kw)})
    return pd.DataFrame(rows)


def singleton_report(con, universe_table: str = "val_s1", gt_table: str = "gt_pairs",
                     scores_table: str = "scores", pred_table: str = "pred") -> pd.DataFrame:
    """Per singleton S1: number of candidates, highest probability, predicted empty or not."""
    return con.execute(f"""
      SELECT u.id s1, coalesce(c.n, 0) n_candidates, c.pmax, coalesce(pr.n, 0) n_predicted, (coalesce(pr.n, 0) = 0) predicted_empty
      FROM {universe_table} u
      LEFT JOIN (SELECT s1, COUNT(*) n, max(p) pmax FROM {scores_table} GROUP BY 1) c ON c.s1 = u.id
      LEFT JOIN (SELECT s1, COUNT(*) n FROM {pred_table} GROUP BY 1) pr ON pr.s1 = u.id
      WHERE u.id NOT IN (SELECT s1 FROM {gt_table})""").df()


# =============================================================================
# Submission writers (one row per S1, original id strings, hard invariant checks)
# =============================================================================

def _write_id_list_tsv(con, universe_sql: str, pairs_sql: str, path: str | Path, header: Sequence[str],
                       batch_s1: int = 25_000) -> dict:
    """Stream ``universe (id, entity_id, rn)`` LEFT JOIN ``pairs (s1, t)`` -> TSV, **in S1 batches**.

    Memory is bounded by one batch of S1 entities (``batch_s1``) and their pairs: no table of all
    pairs is ever built. Every S1 of the universe gets exactly one row, in ``rn`` order (empty list
    if it has no pairs); list ids come from ``text_t.entity_id`` (original strings).

    Hard invariants — any violation raises :class:`SubmissionError`:
      * universe S1 ids unique;                     * no pair whose S1 is outside the universe;
      * no duplicate (s1, t) pair (checked per batch; batches partition S1);
      * every target id resolves to a string matching ``^S[23]-[0-9]+$``;
      * rows written == universe size.
    The caller decides the final file name (see :func:`finalize_outputs`); this function writes
    only to ``path``.
    """
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _u AS
        SELECT id, entity_id, rn, (row_number() OVER (ORDER BY rn) - 1) // {int(batch_s1)} AS b FROM ({universe_sql})""")
    n_u, n_dist, n_batches = con.execute("SELECT COUNT(*), COUNT(DISTINCT id), COALESCE(MAX(b) + 1, 0) FROM _u").fetchone()
    if n_u != n_dist:
        raise SubmissionError(f"{Path(path).name}: {n_u - n_dist} duplicate S1 ids in the universe")
    outside = con.execute(f"SELECT COUNT(*) FROM ({pairs_sql}) p ANTI JOIN _u ON _u.id = p.s1").fetchone()[0]
    if outside:
        raise SubmissionError(f"{Path(path).name}: {outside} pairs whose S1 is not in the universe")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_empty = n_ids = max_list = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for b in range(int(n_batches)):
            con.execute(f"""CREATE OR REPLACE TEMP TABLE _pb AS
                SELECT p.s1, p.t, tt.entity_id tid
                FROM ({pairs_sql}) p SEMI JOIN (SELECT id FROM _u WHERE b = {b}) ub ON ub.id = p.s1
                LEFT JOIN text_t tt ON tt.id = p.t""")
            dup, unres, badid = con.execute(r"""SELECT COUNT(*) - COUNT(DISTINCT (s1, t)),
                    COUNT(*) FILTER (WHERE tid IS NULL),
                    COUNT(*) FILTER (WHERE tid IS NOT NULL AND NOT regexp_matches(tid, '^S[23]-[0-9]+$'))
                FROM _pb""").fetchone()
            if dup or unres or badid:
                raise SubmissionError(f"{path.name}: batch {b}: duplicate_pairs={dup}, "
                                      f"unresolved_target_ids={unres}, bad_target_ids={badid}")
            rows = con.execute(f"""SELECT u.entity_id, coalesce(string_agg(p.tid, ',' ORDER BY p.tid), '') ids, COUNT(p.tid) n
                FROM (SELECT * FROM _u WHERE b = {b}) u LEFT JOIN _pb p ON p.s1 = u.id
                GROUP BY u.entity_id, u.rn ORDER BY u.rn""").fetchall()
            for eid, ids, n in rows:
                f.write(f"{eid}\t{ids}\n")
                n_rows += 1
                n_ids += n
                n_empty += n == 0
                max_list = max(max_list, n)
            con.execute("DROP TABLE IF EXISTS _pb")
    con.execute("DROP TABLE IF EXISTS _u")
    if n_rows != n_u:
        raise SubmissionError(f"{path.name}: wrote {n_rows} rows for {n_u} S1 entities")
    return {"rows": n_rows, "empty_rows": n_empty, "ids": n_ids, "max_list_len": max_list,
            "batches": int(n_batches), "bytes": path.stat().st_size}


def write_candidate_pairs_tsv(con, universe_sql: str, cand_sql: str, path: str | Path, batch_s1: int = 25_000) -> dict:
    """``source1_entity_id  candidate_entity_ids`` — the exact pairs the model scores."""
    return _write_id_list_tsv(con, universe_sql, cand_sql, path, ("source1_entity_id", "candidate_entity_ids"), batch_s1)


def write_matching_results_tsv(con, universe_sql: str, pred_sql: str, path: str | Path, batch_s1: int = 25_000) -> dict:
    """``source1_entity_id  matched_entity_ids`` — empty field for no match (singleton)."""
    return _write_id_list_tsv(con, universe_sql, pred_sql, path, ("source1_entity_id", "matched_entity_ids"), batch_s1)


def verify_no_forced_matches(con, pred_table: str, cand_sql: str, rule: "DecisionRule") -> dict:
    """Hard gate: every predicted pair is a candidate and meets the decision rule's threshold
    (nothing was added to avoid an empty list)."""
    below = con.execute(f"SELECT COUNT(*) FROM {pred_table} WHERE p < {float(rule.threshold)}").fetchone()[0]
    not_cand = con.execute(f"SELECT COUNT(*) FROM {pred_table} pr ANTI JOIN ({cand_sql}) c ON c.s1 = pr.s1 AND c.t = pr.t").fetchone()[0]
    if below or not_cand:
        raise SubmissionError(f"forced/invalid matches: below_threshold={below}, not_a_candidate={not_cand}")
    return {"predicted_below_threshold": 0, "predicted_not_candidate": 0}


def finalize_outputs(tmp_to_final: dict, validators: Sequence) -> dict:
    """Atomic publication. ``tmp_to_final`` maps ``*.tsv.tmp`` -> final path; ``validators`` are
    callables returning ``(ok: bool, details)`` run on the *closed* temporary files. Only if every
    validator passes are the temporary files renamed (``os.replace``, atomic on one filesystem) to
    the final names. On failure nothing final is created or replaced; the temporary files remain
    for debugging and :class:`SubmissionError` is raised."""
    results = []
    for v in validators:
        ok, details = v()
        results.append({"ok": bool(ok), "details": details})
    if not all(r["ok"] for r in results):
        raise SubmissionError(f"validation failed — final outputs NOT written; temporary files kept: "
                              f"{[str(k) for k in tmp_to_final]}; results: {results}")
    for tmp, final in tmp_to_final.items():
        if Path(final).parent != Path(tmp).parent:
            raise SubmissionError("temporary and final files must be in the same directory (atomic rename)")
    # rename the scored file last, so a crash between renames can never leave a new
    # matching_results.tsv next to an old candidate_pairs.tsv
    order = sorted(tmp_to_final.items(), key=lambda kv: "matching_results" in str(kv[1]))
    for tmp, final in order:
        os.replace(tmp, final)
    return {"validators": results, "published": [str(f) for _, f in order]}
