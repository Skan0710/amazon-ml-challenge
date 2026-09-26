#!/usr/bin/env python3
"""
scripts/run_eda.py  — Amazon ML Challenge 2026  Phase 1 EDA
All heavy computation via DuckDB. Pandas only for small result sets.
"""
import sys, time, json, textwrap, re
from pathlib import Path
import duckdb, pandas as pd
import numpy as np
from rapidfuzz import fuzz
import random

random.seed(42)
t0 = time.time()

PROJECT = Path(__file__).resolve().parent.parent
RAW     = PROJECT / "data" / "raw"
EXP     = PROJECT / "experiments"
EXP.mkdir(exist_ok=True)

S1  = str(RAW / "train_source1.tsv")
S2  = str(RAW / "train_source2.tsv")
S3  = str(RAW / "train_source3.tsv")
GT  = str(RAW / "train_ground_truth.tsv")

con = duckdb.connect()

def banner(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def q(sql): return con.execute(sql).df()

# ─────────────────────────────────────────────────────────────
# 1. SCHEMA
# ─────────────────────────────────────────────────────────────
banner("1. SCHEMA")
results = {}
for name, path in [("S1",S1),("S2",S2),("S3",S3),("GT",GT)]:
    df = con.execute(f"DESCRIBE SELECT * FROM read_csv_auto('{path}', delim='\t', header=true) LIMIT 0").df()
    print(f"\n--- {name} ---")
    print(df.to_string(index=False))
    results[f"schema_{name}"] = df.to_dict(orient='records')

# ─────────────────────────────────────────────────────────────
# 2. ROW COUNTS
# ─────────────────────────────────────────────────────────────
banner("2. ROW COUNTS")
t1 = time.time()
counts = {}
for name, path in [("Source1",S1),("Source2",S2),("Source3",S3),("GroundTruth",GT)]:
    n = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{path}', delim='\t', header=true)").fetchone()[0]
    counts[name] = n
    print(f"  {name:<15} {n:>10,}")
print(f"  [row counts: {time.time()-t1:.1f}s]")
results['row_counts'] = counts

# ─────────────────────────────────────────────────────────────
# 3. ENTITY ID UNIQUENESS
# ─────────────────────────────────────────────────────────────
banner("3. ENTITY ID UNIQUENESS")
dup_stats = {}
for name, path in [("Source1",S1),("Source2",S2),("Source3",S3)]:
    t1 = time.time()
    r = con.execute(f"""
        SELECT
            COUNT(*) AS total,
            COUNT(DISTINCT entity_id) AS unique_ids,
            COUNT(*) - COUNT(DISTINCT entity_id) AS dup_rows,
            ROUND(100.0*(COUNT(*) - COUNT(DISTINCT entity_id))/COUNT(*),4) AS dup_pct
        FROM read_csv_auto('{path}', delim='\t', header=true)
    """).df()
    dup_stats[name] = r.iloc[0].to_dict()
    print(f"\n  {name}: {r.to_string(index=False)}  [{time.time()-t1:.1f}s]")
results['id_uniqueness'] = dup_stats

# ─────────────────────────────────────────────────────────────
# 4. MISSING VALUES
# ─────────────────────────────────────────────────────────────
banner("4. MISSING VALUES")
missing_stats = {}
for name, path in [("Source1",S1),("Source2",S2),("Source3",S3)]:
    t1 = time.time()
    r = con.execute(f"""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN entity_id IS NULL THEN 1 ELSE 0 END) AS null_entity_id,
            SUM(CASE WHEN business_name IS NULL THEN 1 ELSE 0 END) AS null_name,
            SUM(CASE WHEN TRIM(COALESCE(business_name,''))='' THEN 1 ELSE 0 END) AS empty_name,
            SUM(CASE WHEN business_address IS NULL THEN 1 ELSE 0 END) AS null_addr,
            SUM(CASE WHEN TRIM(COALESCE(business_address,''))='' THEN 1 ELSE 0 END) AS empty_addr,
            SUM(CASE WHEN country IS NULL THEN 1 ELSE 0 END) AS null_country,
            SUM(CASE WHEN TRIM(COALESCE(country,''))='' THEN 1 ELSE 0 END) AS empty_country
        FROM read_csv_auto('{path}', delim='\t', header=true)
    """).df()
    missing_stats[name] = r.iloc[0].to_dict()
    print(f"\n  {name}: [{time.time()-t1:.1f}s]")
    print(r.T.to_string())
results['missing_values'] = missing_stats

# ─────────────────────────────────────────────────────────────
# 5. COUNTRY DISTRIBUTION
# ─────────────────────────────────────────────────────────────
banner("5. COUNTRY DISTRIBUTION")
country_dist = {}
for name, path in [("Source1",S1),("Source2",S2),("Source3",S3)]:
    r = con.execute(f"""
        SELECT country, COUNT(*) AS cnt,
               ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),2) AS pct
        FROM read_csv_auto('{path}', delim='\t', header=true)
        GROUP BY country ORDER BY cnt DESC
    """).df()
    country_dist[name] = r.to_dict(orient='records')
    print(f"\n  {name}:")
    print(r.to_string(index=False))
results['country_distribution'] = country_dist

all_countries = {}
for name, path in [("S1",S1),("S2",S2),("S3",S3)]:
    cs = con.execute(f"SELECT DISTINCT country FROM read_csv_auto('{path}', delim='\t', header=true)").df()['country'].dropna().tolist()
    all_countries[name] = set(cs)

s1c, s2c, s3c = all_countries['S1'], all_countries['S2'], all_countries['S3']
print(f"\n  Common to all 3: {s1c & s2c & s3c}")
print(f"  S1 only: {s1c - s2c - s3c}")
print(f"  S2 only: {s2c - s1c - s3c}")
print(f"  S3 only: {s3c - s1c - s2c}")
results['country_overlap'] = {
    'common_all': list(s1c & s2c & s3c),
    'S1_only': list(s1c - s2c - s3c),
    'S2_only': list(s2c - s1c - s3c),
    'S3_only': list(s3c - s1c - s2c),
}

# ─────────────────────────────────────────────────────────────
# 6. BUSINESS NAME ANALYSIS (DuckDB-native non-ASCII detection)
# ─────────────────────────────────────────────────────────────
banner("6. BUSINESS NAME ANALYSIS")
name_stats = {}
for src_name, path in [("Source1",S1),("Source2",S2),("Source3",S3)]:
    t1 = time.time()
    # Use ascii() function to detect non-ASCII: if ascii() != business_name then has non-ASCII
    r = con.execute(f"""
        SELECT
            COUNT(*) AS total,
            COUNT(DISTINCT business_name) AS unique_names,
            COUNT(DISTINCT LOWER(TRIM(regexp_replace(business_name,'\\s+',' ')))) AS unique_norm_names,
            MIN(LENGTH(business_name)) AS min_len,
            MAX(LENGTH(business_name)) AS max_len,
            ROUND(AVG(LENGTH(business_name)),1) AS avg_len,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LENGTH(business_name)) AS median_len,
            SUM(CASE WHEN LENGTH(business_name) <= 3 THEN 1 ELSE 0 END) AS very_short,
            SUM(CASE WHEN LENGTH(business_name) >= 200 THEN 1 ELSE 0 END) AS very_long,
            SUM(CASE WHEN LENGTH(business_name) != LENGTH(encode(business_name)) THEN 1 ELSE 0 END) AS has_unicode,
            SUM(CASE WHEN regexp_matches(business_name, '[0-9]') THEN 1 ELSE 0 END) AS has_digits
        FROM read_csv_auto('{path}', delim='\t', header=true)
        WHERE business_name IS NOT NULL AND TRIM(business_name) != ''
    """).df()
    name_stats[src_name] = r.iloc[0].to_dict()
    print(f"\n  {src_name}: [{time.time()-t1:.1f}s]")
    print(r.T.to_string())
results['name_analysis'] = name_stats

# Sample very short names
print("\n  Sample very short names (<=3 chars, S1):")
short = con.execute(f"""
    SELECT entity_id, business_name, country FROM read_csv_auto('{S1}', delim='\t', header=true)
    WHERE LENGTH(business_name)<=3 AND business_name IS NOT NULL AND TRIM(business_name)!=''
    LIMIT 10
""").df()
print(short.to_string(index=False))

# Sample unicode names
print("\n  Sample non-ASCII names (S1):")
uni = con.execute(f"""
    SELECT entity_id, business_name, country FROM read_csv_auto('{S1}', delim='\t', header=true)
    WHERE LENGTH(business_name) != LENGTH(encode(business_name))
    LIMIT 10
""").df()
print(uni.to_string(index=False))

# Most common last-word tokens (legal suffixes)
print("\n  Top last-word tokens (S1 name suffix):")
sfx = con.execute(f"""
    SELECT UPPER(TRIM(list_last(string_split(TRIM(business_name),' ')))) AS suffix,
           COUNT(*) AS cnt
    FROM read_csv_auto('{S1}', delim='\t', header=true)
    WHERE business_name IS NOT NULL AND TRIM(business_name)!=''
    GROUP BY 1 ORDER BY cnt DESC LIMIT 20
""").df()
print(sfx.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 7. ADDRESS ANALYSIS
# ─────────────────────────────────────────────────────────────
banner("7. ADDRESS ANALYSIS")
addr_stats = {}
for src_name, path in [("Source1",S1),("Source2",S2),("Source3",S3)]:
    t1 = time.time()
    r = con.execute(f"""
        SELECT
            COUNT(*) AS total,
            COUNT(DISTINCT business_address) AS unique_addrs,
            COUNT(DISTINCT LOWER(TRIM(regexp_replace(business_address,'\\s+',' ')))) AS unique_norm_addrs,
            MIN(LENGTH(business_address)) AS min_len,
            MAX(LENGTH(business_address)) AS max_len,
            ROUND(AVG(LENGTH(business_address)),1) AS avg_len,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LENGTH(business_address)) AS median_len,
            SUM(CASE WHEN LENGTH(TRIM(COALESCE(business_address,'')))<=5 THEN 1 ELSE 0 END) AS very_short,
            SUM(CASE WHEN LENGTH(business_address)>=500 THEN 1 ELSE 0 END) AS very_long,
            SUM(CASE WHEN regexp_matches(business_address,'[0-9]') THEN 1 ELSE 0 END) AS has_digits,
            SUM(CASE WHEN LENGTH(business_address) != LENGTH(encode(business_address)) THEN 1 ELSE 0 END) AS has_unicode
        FROM read_csv_auto('{path}', delim='\t', header=true)
        WHERE business_address IS NOT NULL AND TRIM(business_address) != ''
    """).df()
    addr_stats[src_name] = r.iloc[0].to_dict()
    print(f"\n  {src_name}: [{time.time()-t1:.1f}s]")
    print(r.T.to_string())
results['address_analysis'] = addr_stats

print("\n  Sample very long addresses (S1, top 5):")
lng = con.execute(f"""
    SELECT entity_id, LENGTH(business_address) AS len,
           LEFT(business_address,120) AS addr_prefix
    FROM read_csv_auto('{S1}', delim='\t', header=true)
    ORDER BY LENGTH(business_address) DESC LIMIT 5
""").df()
print(lng.to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 8. GROUND TRUTH ANALYSIS
# ─────────────────────────────────────────────────────────────
banner("8. GROUND TRUTH ANALYSIS")
t1 = time.time()

gt_counts = con.execute(f"""
    WITH gt AS (
        SELECT source1_entity_id, matched_entity_ids,
            CASE
                WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids) = '' THEN 0
                ELSE len(string_split(TRIM(matched_entity_ids), ','))
            END AS match_count
        FROM read_csv_auto('{GT}', delim='\t', header=true)
    )
    SELECT
        match_count,
        COUNT(*) AS source1_entity_count
    FROM gt
    GROUP BY match_count
    ORDER BY match_count
""").df()
print("\n  Match count distribution:")
print(gt_counts.to_string(index=False))
results['gt_match_distribution'] = gt_counts.to_dict(orient='records')

gt_summary = con.execute(f"""
    WITH gt AS (
        SELECT
            CASE WHEN matched_entity_ids IS NULL OR TRIM(matched_entity_ids)='' THEN 0
                 ELSE len(string_split(TRIM(matched_entity_ids),','))
            END AS match_count
        FROM read_csv_auto('{GT}', delim='\t', header=true)
    )
    SELECT
        COUNT(*) AS total_s1,
        SUM(CASE WHEN match_count=0 THEN 1 ELSE 0 END) AS zero_matches,
        SUM(CASE WHEN match_count>0 THEN 1 ELSE 0 END) AS has_matches,
        ROUND(AVG(match_count),4) AS mean_matches,
        PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY match_count) AS median_matches,
        MAX(match_count) AS max_matches
    FROM gt
""").df()
print(f"\n  GT summary [{time.time()-t1:.1f}s]:")
print(gt_summary.T.to_string())
results['gt_summary'] = gt_summary.iloc[0].to_dict()

# ─────────────────────────────────────────────────────────────
# 9. S2 vs S3 MATCH SPLIT
# ─────────────────────────────────────────────────────────────
banner("9. S2 vs S3 MATCH SPLIT")
t1 = time.time()

s2s3 = con.execute(f"""
    WITH gt AS (
        SELECT source1_entity_id, matched_entity_ids
        FROM read_csv_auto('{GT}', delim='\t', header=true)
        WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != ''
    ),
    exploded AS (
        SELECT source1_entity_id,
               TRIM(unnest(string_split(matched_entity_ids, ','))) AS mid
        FROM gt
    )
    SELECT
        SUM(CASE WHEN mid LIKE 'S2-%' THEN 1 ELSE 0 END) AS s2_matches,
        SUM(CASE WHEN mid LIKE 'S3-%' THEN 1 ELSE 0 END) AS s3_matches,
        COUNT(DISTINCT source1_entity_id) AS s1_with_any_match
    FROM exploded
""").df()
print(s2s3.to_string(index=False))
results['s2_s3_split'] = s2s3.iloc[0].to_dict()

combo = con.execute(f"""
    WITH all_s1 AS (
        SELECT source1_entity_id, matched_entity_ids
        FROM read_csv_auto('{GT}', delim='\t', header=true)
    ),
    exploded AS (
        SELECT a.source1_entity_id,
               TRIM(m.mid) AS mid
        FROM all_s1 a, unnest(string_split(COALESCE(matched_entity_ids,''), ',')) AS m(mid)
        WHERE TRIM(m.mid) != ''
    ),
    per_s1 AS (
        SELECT source1_entity_id,
               SUM(CASE WHEN mid LIKE 'S2-%' THEN 1 ELSE 0 END) AS s2_cnt,
               SUM(CASE WHEN mid LIKE 'S3-%' THEN 1 ELSE 0 END) AS s3_cnt
        FROM exploded
        GROUP BY source1_entity_id
    ),
    joined AS (
        SELECT a.source1_entity_id,
               COALESCE(p.s2_cnt,0) AS s2_cnt,
               COALESCE(p.s3_cnt,0) AS s3_cnt
        FROM all_s1 a LEFT JOIN per_s1 p USING(source1_entity_id)
    )
    SELECT
        CASE
            WHEN s2_cnt=0 AND s3_cnt=0 THEN 'no matches'
            WHEN s2_cnt>0 AND s3_cnt=0 THEN 'S2 only'
            WHEN s2_cnt=0 AND s3_cnt>0 THEN 'S3 only'
            ELSE 'both S2 and S3'
        END AS category,
        COUNT(*) AS s1_entity_count
    FROM joined GROUP BY 1 ORDER BY 2 DESC
""").df()
print(f"\n  S1 by match source [{time.time()-t1:.1f}s]:")
print(combo.to_string(index=False))
results['s1_match_source_combo'] = combo.to_dict(orient='records')

# ─────────────────────────────────────────────────────────────
# 10. CREATE VIEWS FOR FAST LOOKUPS
# ─────────────────────────────────────────────────────────────
banner("10. BUILDING VIEWS")
con.execute(f"CREATE OR REPLACE VIEW s1_view AS SELECT * FROM read_csv_auto('{S1}', delim='\t', header=true)")
con.execute(f"CREATE OR REPLACE VIEW s2_view AS SELECT * FROM read_csv_auto('{S2}', delim='\t', header=true)")
con.execute(f"CREATE OR REPLACE VIEW s3_view AS SELECT * FROM read_csv_auto('{S3}', delim='\t', header=true)")
con.execute(f"CREATE OR REPLACE VIEW gt_view  AS SELECT * FROM read_csv_auto('{GT}',  delim='\t', header=true)")
print("  Views created.")

# ─────────────────────────────────────────────────────────────
# 11. POSITIVE MATCH EXAMPLES (20 samples)
# ─────────────────────────────────────────────────────────────
banner("11. POSITIVE MATCH EXAMPLES (20 random samples)")

sample_gt = con.execute(f"""
    SELECT source1_entity_id, matched_entity_ids
    FROM gt_view
    WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != ''
    USING SAMPLE 20 (reservoir, seed=42)
""").df()

positive_examples = []
for _, row in sample_gt.iterrows():
    s1_id = row['source1_entity_id']
    match_ids = [m.strip() for m in row['matched_entity_ids'].split(',') if m.strip()]
    s1_rec = con.execute(f"SELECT * FROM s1_view WHERE entity_id='{s1_id}'").df()
    if s1_rec.empty: continue
    s1_info = s1_rec.iloc[0].to_dict()
    match_records = []
    for mid in match_ids:
        tbl = 's2_view' if mid.startswith('S2-') else 's3_view'
        mr = con.execute(f"SELECT * FROM {tbl} WHERE entity_id='{mid}'").df()
        if not mr.empty:
            match_records.append(mr.iloc[0].to_dict())
    positive_examples.append({'s1': s1_info, 'matches': match_records})
    print(f"\n  SOURCE 1: {s1_id}")
    print(f"    name   : {s1_info.get('business_name','')}")
    print(f"    address: {str(s1_info.get('business_address',''))[:100]}")
    print(f"    country: {s1_info.get('country','')}")
    for mr in match_records:
        print(f"    MATCH {mr.get('entity_id','')}")
        print(f"      name   : {mr.get('business_name','')}")
        print(f"      address: {str(mr.get('business_address',''))[:100]}")
results['positive_examples_count'] = len(positive_examples)

# ─────────────────────────────────────────────────────────────
# 12. HARD POSITIVE EXAMPLES
# ─────────────────────────────────────────────────────────────
banner("12. HARD POSITIVE EXAMPLES (lowest similarity)")

hard_sample = con.execute(f"""
    WITH gt_sample AS (
        SELECT source1_entity_id,
               TRIM(unnest(string_split(matched_entity_ids, ','))) AS match_id
        FROM gt_view
        WHERE matched_entity_ids IS NOT NULL AND TRIM(matched_entity_ids) != ''
        USING SAMPLE 3000 (reservoir, seed=42)
    ),
    joined AS (
        SELECT g.source1_entity_id, g.match_id,
               s1.business_name AS s1_name, s1.business_address AS s1_addr,
               COALESCE(s2.business_name, s3.business_name) AS m_name,
               COALESCE(s2.business_address, s3.business_address) AS m_addr
        FROM gt_sample g
        JOIN s1_view s1 ON s1.entity_id = g.source1_entity_id
        LEFT JOIN s2_view s2 ON s2.entity_id = g.match_id AND g.match_id LIKE 'S2-%'
        LEFT JOIN s3_view s3 ON s3.entity_id = g.match_id AND g.match_id LIKE 'S3-%'
    )
    SELECT * FROM joined
    WHERE s1_name IS NOT NULL AND m_name IS NOT NULL
    LIMIT 3000
""").df()

print(f"  Computing RapidFuzz similarity on {len(hard_sample)} pairs...")
t1 = time.time()
hard_sample['name_sim'] = hard_sample.apply(
    lambda r: fuzz.token_sort_ratio(str(r['s1_name']).lower(), str(r['m_name']).lower()), axis=1
)
hard_sample['addr_sim'] = hard_sample.apply(
    lambda r: fuzz.token_sort_ratio(str(r['s1_addr']).lower(), str(r['m_addr']).lower())
    if pd.notna(r['s1_addr']) and pd.notna(r['m_addr']) else 0, axis=1
)
hard_sample['avg_sim'] = (hard_sample['name_sim'] + hard_sample['addr_sim']) / 2
hard_positives = hard_sample.nsmallest(20, 'avg_sim')
print(f"  [fuzzy done: {time.time()-t1:.1f}s]")

print(f"\n  20 hardest positive matches:")
for _, r in hard_positives.iterrows():
    print(f"\n  S1 {r['source1_entity_id']} <-> {r['match_id']}")
    print(f"    S1 name  : {r['s1_name']}")
    print(f"    M  name  : {r['m_name']}  [name_sim={r['name_sim']:.0f}]")
    print(f"    S1 addr  : {str(r['s1_addr'])[:80]}")
    print(f"    M  addr  : {str(r['m_addr'])[:80]}  [addr_sim={r['addr_sim']:.0f}]")
results['hard_positives_sample'] = hard_positives[['source1_entity_id','match_id','name_sim','addr_sim','avg_sim']].to_dict(orient='records')

# ─────────────────────────────────────────────────────────────
# 13. POTENTIAL HARD NEGATIVES
# ─────────────────────────────────────────────────────────────
banner("13. POTENTIAL HARD NEGATIVES (same norm name, NOT matched)")

t1 = time.time()
hard_neg = con.execute(f"""
    WITH s1_sample AS (
        SELECT entity_id AS s1_id,
               LOWER(TRIM(regexp_replace(business_name,'\\s+',' '))) AS norm_name,
               business_name AS s1_name, business_address AS s1_addr, country AS s1_country
        FROM s1_view
        USING SAMPLE 50000 (reservoir, seed=42)
    ),
    s2_sample AS (
        SELECT entity_id AS s2_id,
               LOWER(TRIM(regexp_replace(business_name,'\\s+',' '))) AS norm_name,
               business_name AS s2_name, business_address AS s2_addr, country AS s2_country
        FROM s2_view
        USING SAMPLE 100000 (reservoir, seed=42)
    ),
    candidates AS (
        SELECT s1.s1_id, s2.s2_id, s1.s1_name, s2.s2_name,
               s1.s1_addr, s2.s2_addr, s1.s1_country, s2.s2_country
        FROM s1_sample s1 JOIN s2_sample s2
            ON s1.norm_name = s2.norm_name AND s1.s1_country = s2.s2_country
    ),
    matched_pairs AS (
        SELECT source1_entity_id AS s1_id, TRIM(m.mid) AS match_id
        FROM gt_view,
             unnest(string_split(COALESCE(matched_entity_ids,''), ',')) AS m(mid)
        WHERE TRIM(m.mid) LIKE 'S2-%'
    )
    SELECT c.s1_id, c.s2_id, c.s1_name, c.s2_name,
           c.s1_addr, c.s2_addr, c.s1_country
    FROM candidates c
    LEFT JOIN matched_pairs mp ON c.s1_id = mp.s1_id AND c.s2_id = mp.match_id
    WHERE mp.match_id IS NULL
    LIMIT 20
""").df()
print(f"  [{time.time()-t1:.1f}s] Found {len(hard_neg)} potential hard negatives (from sample)")
print(f"  NOTE: POTENTIAL only — verify against full GT before treating as negatives")
for _, r in hard_neg.iterrows():
    print(f"\n  {r['s1_id']} (S1) <-> {r['s2_id']} (S2)  country={r['s1_country']}")
    print(f"    S1 name: {r['s1_name']}")
    print(f"    S2 name: {r['s2_name']}")
    print(f"    S1 addr: {str(r['s1_addr'])[:80]}")
    print(f"    S2 addr: {str(r['s2_addr'])[:80]}")
results['hard_negatives_found'] = len(hard_neg)

# ─────────────────────────────────────────────────────────────
# 14. CROSS-SOURCE EXACT OVERLAP / BLOCKING STATS
# ─────────────────────────────────────────────────────────────
banner("14. CROSS-SOURCE BLOCKING STATS")
blocking_stats = {}

for pair_label, src_view in [("S1xS2","s2_view"),("S1xS3","s3_view")]:
    t1 = time.time()
    print(f"\n  --- {pair_label} ---")
    pair_stats = {}

    # A. Exact name
    r = con.execute(f"""
        SELECT COUNT(DISTINCT s1.entity_id) AS s1_covered,
               COUNT(*) AS candidate_pairs,
               ROUND(COUNT(*)*1.0/NULLIF(COUNT(DISTINCT s1.entity_id),0),1) AS avg_cands,
               MAX(sx.cnt) AS max_cands
        FROM s1_view s1
        JOIN (SELECT business_name, entity_id,
                     COUNT(*) OVER (PARTITION BY business_name) AS cnt
              FROM {src_view}) sx ON s1.business_name = sx.business_name
    """).df().iloc[0].to_dict()
    pair_stats['exact_name'] = r; print(f"  A. Exact name:        {r}")

    # B. Norm name
    r = con.execute(f"""
        SELECT COUNT(DISTINCT s1.entity_id) AS s1_covered,
               COUNT(*) AS candidate_pairs,
               ROUND(COUNT(*)*1.0/NULLIF(COUNT(DISTINCT s1.entity_id),0),1) AS avg_cands,
               MAX(sx.cnt) AS max_cands
        FROM (SELECT entity_id, LOWER(TRIM(regexp_replace(business_name,'\\s+',' '))) AS nn FROM s1_view) s1
        JOIN (SELECT entity_id, LOWER(TRIM(regexp_replace(business_name,'\\s+',' '))) AS nn,
                     COUNT(*) OVER (PARTITION BY LOWER(TRIM(regexp_replace(business_name,'\\s+',' ')))) AS cnt
              FROM {src_view}) sx ON s1.nn = sx.nn
    """).df().iloc[0].to_dict()
    pair_stats['norm_name'] = r; print(f"  B. Norm name:         {r}")

    # C. Exact address
    r = con.execute(f"""
        SELECT COUNT(DISTINCT s1.entity_id) AS s1_covered,
               COUNT(*) AS candidate_pairs,
               ROUND(COUNT(*)*1.0/NULLIF(COUNT(DISTINCT s1.entity_id),0),1) AS avg_cands,
               MAX(sx.cnt) AS max_cands
        FROM s1_view s1
        JOIN (SELECT business_address, entity_id,
                     COUNT(*) OVER (PARTITION BY business_address) AS cnt
              FROM {src_view}) sx ON s1.business_address = sx.business_address
        WHERE s1.business_address IS NOT NULL AND TRIM(s1.business_address)!=''
    """).df().iloc[0].to_dict()
    pair_stats['exact_address'] = r; print(f"  C. Exact address:     {r}")

    # D. Norm address
    r = con.execute(f"""
        SELECT COUNT(DISTINCT s1.entity_id) AS s1_covered,
               COUNT(*) AS candidate_pairs,
               ROUND(COUNT(*)*1.0/NULLIF(COUNT(DISTINCT s1.entity_id),0),1) AS avg_cands,
               MAX(sx.cnt) AS max_cands
        FROM (SELECT entity_id, LOWER(TRIM(regexp_replace(business_address,'\\s+',' '))) AS na
              FROM s1_view WHERE business_address IS NOT NULL AND TRIM(business_address)!='') s1
        JOIN (SELECT entity_id, LOWER(TRIM(regexp_replace(business_address,'\\s+',' '))) AS na,
                     COUNT(*) OVER (PARTITION BY LOWER(TRIM(regexp_replace(business_address,'\\s+',' ')))) AS cnt
              FROM {src_view} WHERE business_address IS NOT NULL AND TRIM(business_address)!='') sx
        ON s1.na = sx.na
    """).df().iloc[0].to_dict()
    pair_stats['norm_address'] = r; print(f"  D. Norm address:      {r}")

    # E. Norm name + country
    r = con.execute(f"""
        SELECT COUNT(DISTINCT s1.entity_id) AS s1_covered,
               COUNT(*) AS candidate_pairs,
               ROUND(COUNT(*)*1.0/NULLIF(COUNT(DISTINCT s1.entity_id),0),1) AS avg_cands,
               MAX(sx.cnt) AS max_cands
        FROM (SELECT entity_id, LOWER(TRIM(regexp_replace(business_name,'\\s+',' '))) AS nn, country FROM s1_view) s1
        JOIN (SELECT entity_id, LOWER(TRIM(regexp_replace(business_name,'\\s+',' '))) AS nn, country,
                     COUNT(*) OVER (PARTITION BY LOWER(TRIM(regexp_replace(business_name,'\\s+',' '))), country) AS cnt
              FROM {src_view}) sx ON s1.nn = sx.nn AND s1.country = sx.country
    """).df().iloc[0].to_dict()
    pair_stats['norm_name_country'] = r; print(f"  E. Norm name+country: {r}")

    # F. Norm address + country
    r = con.execute(f"""
        SELECT COUNT(DISTINCT s1.entity_id) AS s1_covered,
               COUNT(*) AS candidate_pairs,
               ROUND(COUNT(*)*1.0/NULLIF(COUNT(DISTINCT s1.entity_id),0),1) AS avg_cands,
               MAX(sx.cnt) AS max_cands
        FROM (SELECT entity_id, LOWER(TRIM(regexp_replace(business_address,'\\s+',' '))) AS na, country
              FROM s1_view WHERE business_address IS NOT NULL AND TRIM(business_address)!='') s1
        JOIN (SELECT entity_id, LOWER(TRIM(regexp_replace(business_address,'\\s+',' '))) AS na, country,
                     COUNT(*) OVER (PARTITION BY LOWER(TRIM(regexp_replace(business_address,'\\s+',' '))), country) AS cnt
              FROM {src_view} WHERE business_address IS NOT NULL AND TRIM(business_address)!='') sx
        ON s1.na = sx.na AND s1.country = sx.country
    """).df().iloc[0].to_dict()
    pair_stats['norm_addr_country'] = r; print(f"  F. Norm addr+country: {r}")

    blocking_stats[pair_label] = pair_stats
    print(f"  [{time.time()-t1:.1f}s]")

results['blocking_stats'] = blocking_stats

# ─────────────────────────────────────────────────────────────
# 15. BLOCKING RECALL
# ─────────────────────────────────────────────────────────────
banner("15. BLOCKING RECALL EVALUATION")

total_pairs = con.execute(f"""
    SELECT
        SUM(CASE WHEN TRIM(m.mid) LIKE 'S2-%' THEN 1 ELSE 0 END) AS total_s1s2,
        SUM(CASE WHEN TRIM(m.mid) LIKE 'S3-%' THEN 1 ELSE 0 END) AS total_s1s3
    FROM gt_view,
         unnest(string_split(COALESCE(matched_entity_ids,''), ',')) AS m(mid)
    WHERE TRIM(m.mid) != ''
""").df().iloc[0]
total_s1s2 = int(total_pairs['total_s1s2'])
total_s1s3 = int(total_pairs['total_s1s3'])
print(f"  True S1-S2 pairs: {total_s1s2:,}   True S1-S3 pairs: {total_s1s3:,}")

recall_stats = {}

for pair_label, src_view, total_true, src_prefix in [
    ("S1xS2","s2_view",total_s1s2,"S2-%"),
    ("S1xS3","s3_view",total_s1s3,"S3-%"),
]:
    t1 = time.time()
    print(f"\n  --- {pair_label} (total true: {total_true:,}) ---")
    pair_recall = {}

    tp_cte = f"""
        WITH true_pairs AS (
            SELECT source1_entity_id AS s1_id, TRIM(m.mid) AS match_id
            FROM gt_view,
                 unnest(string_split(COALESCE(matched_entity_ids,''), ',')) AS m(mid)
            WHERE TRIM(m.mid) LIKE '{src_prefix}'
        )
    """

    # 1. Exact name
    r = con.execute(f"""{tp_cte}, blocked AS (
        SELECT s1.entity_id AS s1_id, sx.entity_id AS match_id FROM s1_view s1
        JOIN {src_view} sx ON s1.business_name = sx.business_name)
        SELECT COUNT(*) FROM true_pairs tp JOIN blocked b ON tp.s1_id=b.s1_id AND tp.match_id=b.match_id
    """).fetchone()[0]
    recall = r / total_true if total_true > 0 else 0
    pair_recall['exact_name'] = {'retrieved': r, 'recall': round(recall,4)}
    print(f"  1. Exact name:         retrieved={r:>10,}  recall={recall:.4f}")

    # 2. Norm name
    r = con.execute(f"""{tp_cte}, blocked AS (
        SELECT s1.entity_id AS s1_id, sx.entity_id AS match_id
        FROM (SELECT entity_id, LOWER(TRIM(regexp_replace(business_name,'\\s+',' '))) AS nn FROM s1_view) s1
        JOIN (SELECT entity_id, LOWER(TRIM(regexp_replace(business_name,'\\s+',' '))) AS nn FROM {src_view}) sx
        ON s1.nn = sx.nn)
        SELECT COUNT(*) FROM true_pairs tp JOIN blocked b ON tp.s1_id=b.s1_id AND tp.match_id=b.match_id
    """).fetchone()[0]
    recall = r / total_true if total_true > 0 else 0
    pair_recall['norm_name'] = {'retrieved': r, 'recall': round(recall,4)}
    print(f"  2. Norm name:          retrieved={r:>10,}  recall={recall:.4f}")

    # 3. Exact address
    r = con.execute(f"""{tp_cte}, blocked AS (
        SELECT s1.entity_id AS s1_id, sx.entity_id AS match_id FROM s1_view s1
        JOIN {src_view} sx ON s1.business_address = sx.business_address
        WHERE s1.business_address IS NOT NULL AND TRIM(s1.business_address)!='')
        SELECT COUNT(*) FROM true_pairs tp JOIN blocked b ON tp.s1_id=b.s1_id AND tp.match_id=b.match_id
    """).fetchone()[0]
    recall = r / total_true if total_true > 0 else 0
    pair_recall['exact_address'] = {'retrieved': r, 'recall': round(recall,4)}
    print(f"  3. Exact address:      retrieved={r:>10,}  recall={recall:.4f}")

    # 4. Norm address
    r = con.execute(f"""{tp_cte}, blocked AS (
        SELECT s1.entity_id AS s1_id, sx.entity_id AS match_id
        FROM (SELECT entity_id, LOWER(TRIM(regexp_replace(business_address,'\\s+',' '))) AS na FROM s1_view WHERE business_address IS NOT NULL AND TRIM(business_address)!='') s1
        JOIN (SELECT entity_id, LOWER(TRIM(regexp_replace(business_address,'\\s+',' '))) AS na FROM {src_view} WHERE business_address IS NOT NULL AND TRIM(business_address)!='') sx
        ON s1.na = sx.na)
        SELECT COUNT(*) FROM true_pairs tp JOIN blocked b ON tp.s1_id=b.s1_id AND tp.match_id=b.match_id
    """).fetchone()[0]
    recall = r / total_true if total_true > 0 else 0
    pair_recall['norm_address'] = {'retrieved': r, 'recall': round(recall,4)}
    print(f"  4. Norm address:       retrieved={r:>10,}  recall={recall:.4f}")

    # 5. Norm name + country
    r = con.execute(f"""{tp_cte}, blocked AS (
        SELECT s1.entity_id AS s1_id, sx.entity_id AS match_id
        FROM (SELECT entity_id, LOWER(TRIM(regexp_replace(business_name,'\\s+',' '))) AS nn, country FROM s1_view) s1
        JOIN (SELECT entity_id, LOWER(TRIM(regexp_replace(business_name,'\\s+',' '))) AS nn, country FROM {src_view}) sx
        ON s1.nn = sx.nn AND s1.country = sx.country)
        SELECT COUNT(*) FROM true_pairs tp JOIN blocked b ON tp.s1_id=b.s1_id AND tp.match_id=b.match_id
    """).fetchone()[0]
    recall = r / total_true if total_true > 0 else 0
    pair_recall['norm_name_country'] = {'retrieved': r, 'recall': round(recall,4)}
    print(f"  5. Norm name+country:  retrieved={r:>10,}  recall={recall:.4f}")

    # 6. Norm address + country
    r = con.execute(f"""{tp_cte}, blocked AS (
        SELECT s1.entity_id AS s1_id, sx.entity_id AS match_id
        FROM (SELECT entity_id, LOWER(TRIM(regexp_replace(business_address,'\\s+',' '))) AS na, country FROM s1_view WHERE business_address IS NOT NULL AND TRIM(business_address)!='') s1
        JOIN (SELECT entity_id, LOWER(TRIM(regexp_replace(business_address,'\\s+',' '))) AS na, country FROM {src_view} WHERE business_address IS NOT NULL AND TRIM(business_address)!='') sx
        ON s1.na = sx.na AND s1.country = sx.country)
        SELECT COUNT(*) FROM true_pairs tp JOIN blocked b ON tp.s1_id=b.s1_id AND tp.match_id=b.match_id
    """).fetchone()[0]
    recall = r / total_true if total_true > 0 else 0
    pair_recall['norm_addr_country'] = {'retrieved': r, 'recall': round(recall,4)}
    print(f"  6. Norm addr+country:  retrieved={r:>10,}  recall={recall:.4f}")

    # 7. UNION name OR addr
    r = con.execute(f"""{tp_cte},
        by_name AS (
            SELECT s1.entity_id AS s1_id, sx.entity_id AS match_id
            FROM (SELECT entity_id, LOWER(TRIM(regexp_replace(business_name,'\\s+',' '))) AS nn FROM s1_view) s1
            JOIN (SELECT entity_id, LOWER(TRIM(regexp_replace(business_name,'\\s+',' '))) AS nn FROM {src_view}) sx ON s1.nn = sx.nn
        ),
        by_addr AS (
            SELECT s1.entity_id AS s1_id, sx.entity_id AS match_id
            FROM (SELECT entity_id, LOWER(TRIM(regexp_replace(business_address,'\\s+',' '))) AS na FROM s1_view WHERE business_address IS NOT NULL AND TRIM(business_address)!='') s1
            JOIN (SELECT entity_id, LOWER(TRIM(regexp_replace(business_address,'\\s+',' '))) AS na FROM {src_view} WHERE business_address IS NOT NULL AND TRIM(business_address)!='') sx ON s1.na = sx.na
        ),
        combined AS (SELECT * FROM by_name UNION SELECT * FROM by_addr)
        SELECT COUNT(*) FROM true_pairs tp JOIN combined b ON tp.s1_id=b.s1_id AND tp.match_id=b.match_id
    """).fetchone()[0]
    recall = r / total_true if total_true > 0 else 0
    pair_recall['union_name_or_addr'] = {'retrieved': r, 'recall': round(recall,4)}
    print(f"  7. UNION name|addr:    retrieved={r:>10,}  recall={recall:.4f}")

    recall_stats[pair_label] = pair_recall
    print(f"  [{time.time()-t1:.1f}s total]")

results['blocking_recall'] = recall_stats
results['total_true_pairs'] = {'S1S2': total_s1s2, 'S1S3': total_s1s3}

# ─────────────────────────────────────────────────────────────
# 16. SAVE
# ─────────────────────────────────────────────────────────────
elapsed = time.time() - t0
results['total_runtime_seconds'] = round(elapsed, 1)
print(f"\n\nTotal runtime: {elapsed:.1f}s")

def make_serializable(obj):
    if isinstance(obj, dict): return {k: make_serializable(v) for k,v in obj.items()}
    if isinstance(obj, list): return [make_serializable(i) for i in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, float) and (str(obj) in ('nan','inf','-inf')): return None
    return obj

with open(EXP / 'eda_summary.json', 'w') as f:
    json.dump(make_serializable(results), f, indent=2)
print(f"Saved: experiments/eda_summary.json")
print("\n=== EDA SCRIPT COMPLETE ===")
