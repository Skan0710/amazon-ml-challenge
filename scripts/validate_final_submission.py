#!/usr/bin/env python3
"""
Strict, memory-safe pre-submission validator (Phase 4). Stdlib only; streams both files.

Stricter than the official ``utils/validate_submission.py``: it FAILS (exit 1) instead of warning
when a matched ID is not among the candidates, and it also checks exact header case, stray
whitespace/quotes, Python-object artefacts (``nan``, ``None``, ``[`` ``]`` ``'``), empty list
elements, extra columns, CRLF, BOM, the S1 id format, and (optionally) that every S2/S3 id exists.

Memory: O(#S1 + #matched ids). Candidate lists are never held — each candidate row is checked
against that S1's matched ids and discarded (the official validator keeps every candidate id in
a Python set: ~16-24 GB for ~175M ids). Matched/target ids are stored as integers where exact.

    python3 scripts/validate_final_submission.py \
        --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
        --source1 <dir>/test_source1.tsv [--source2 ... --source3 ...]

Nothing is repaired: it only reports.
"""
import argparse
import re
import sys

MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]
S1_RE = re.compile(r"^S1-[0-9]+$")
T_RE = re.compile(r"^S[23]-[0-9]+$")
BAD_TOKENS = ("nan", "none", "null", "[", "]", "'", '"', " ")
MAX_EX = 5
_MSGS = {"dup_s1": "duplicate source1_entity_id rows", "bad_s1": "malformed S1 ids",
         "unexpected_s1": "S1 ids not in the source-1 file", "dup_in_list": "duplicate ids inside a list",
         "bad_id": "list ids that are not S2-/S3- ids", "artefact": "lists containing nan/None/quotes/brackets/spaces",
         "empty_element": "lists with an empty element (',,' or trailing comma)", "unknown_id": "ids not in source 2/3",
         "no_tab": "rows without a tab", "extra_cols": "rows with more than 2 columns", "crlf": "CRLF line endings",
         "not_candidate": "S1 rows whose matches are not all in their candidate list"}


def id_code(tid):
    """Compact exact key for an S2/S3 id: an int when the numeric part has no leading zero
    (``S2-12`` -> 2000000012), else the string itself (so ``S2-012`` never collides with ``S2-12``)."""
    src, _, num = tid.partition("-")
    if len(src) == 2 and src[1] in "23" and num.isdigit() and (num == "0" or num[0] != "0") and len(num) <= 9:
        return int(src[1]) * 1_000_000_000 + int(num)
    return tid


def read_first_column(path):
    ids = []
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            if line.strip():
                ids.append(line.split("\t", 1)[0].strip())
    return ids


def _scan(path, header, required, valid_targets, problems, label, on_row):
    """Stream one results-style file. ``on_row(s1, ids)`` receives each parsed row (ids list, may
    be empty) and may record extra issues. Returns stats or None on a fatal problem."""
    stats = {"rows": 0, "empty": 0, "ids": 0, "max_list": 0}
    seen = set()
    issues = {k: [] for k in _MSGS}
    try:
        f = open(path, encoding="utf-8", newline="")
    except OSError as e:
        problems.append(f"{label}: cannot open {path}: {e}")
        return None
    with f:
        try:
            first = f.readline()
            if first == "":
                problems.append(f"{label}: file is empty")
                return None
            if first.startswith("﻿"):
                problems.append(f"{label}: file starts with a UTF-8 BOM")
                first = first[1:]
            cols = first.rstrip("\n").rstrip("\r").split("\t")
            if cols != header:
                problems.append(f"{label}: header {cols!r} != required {header!r} (exact, tab-separated, in this order)")
                return None
            last_line_had_newline = first.endswith("\n")
            for n, line in enumerate(f, start=2):
                last_line_had_newline = line.endswith("\n")
                if line.endswith("\r\n"):
                    issues["crlf"].append(n)
                line = line.rstrip("\n").rstrip("\r")
                if line == "":
                    continue
                parts = line.split("\t")
                if len(parts) == 1:
                    issues["no_tab"].append(n)
                    continue
                if len(parts) > 2:
                    issues["extra_cols"].append(n)
                    continue
                s1, lst = parts
                stats["rows"] += 1
                if s1 in seen:
                    issues["dup_s1"].append(s1)
                seen.add(s1)
                if not S1_RE.match(s1):
                    issues["bad_s1"].append(s1)
                if required is not None and s1 not in required:
                    issues["unexpected_s1"].append(s1)
                if lst == "":
                    stats["empty"] += 1
                    on_row(s1, [], issues)
                    continue
                if any(tok in lst.lower() for tok in BAD_TOKENS):
                    issues["artefact"].append(s1)
                ids = lst.split(",")
                if any(i == "" for i in ids):
                    issues["empty_element"].append(s1)
                if len(ids) != len(set(ids)):
                    issues["dup_in_list"].append(s1)
                for i in ids:
                    if i and not T_RE.match(i):
                        issues["bad_id"].append(i)
                    elif valid_targets is not None and i and id_code(i) not in valid_targets:
                        issues["unknown_id"].append(i)
                stats["ids"] += len(ids)
                stats["max_list"] = max(stats["max_list"], len(ids))
                on_row(s1, ids, issues)
            if stats["rows"] and not last_line_had_newline:
                problems.append(f"{label}: last line has no trailing newline (possibly truncated file)")
        except UnicodeDecodeError:
            problems.append(f"{label}: not valid UTF-8")
            return None
    for k, v in issues.items():
        if v:
            problems.append(f"{label}: {len(v)} {_MSGS[k]}, e.g. {v[:MAX_EX]}")
    if required is not None:
        missing = required - seen
        if missing:
            problems.append(f"{label}: {len(missing)} required S1 ids missing, e.g. {sorted(missing)[:MAX_EX]}")
        stats["missing_s1"] = len(missing)
    stats["present_s1"] = len(seen)
    stats["duplicate_s1_rows"] = len(issues["dup_s1"])
    stats["lists_with_duplicate_ids"] = len(issues["dup_in_list"])
    return stats


def validate(matching, candidate, source1, source2=None, source3=None):
    problems = []
    req_list = read_first_column(source1)
    required = set(req_list)
    if len(required) != len(req_list):
        problems.append(f"source1 file has {len(req_list) - len(required)} duplicate ids")
    del req_list
    valid = None
    if source2 and source3:
        valid = {id_code(i) for p in (source2, source3) for i in read_first_column(p)}
    report = {"expected_s1": len(required)}

    matched = {}                                   # s1 -> tuple of id codes (non-empty rows only)
    def keep_matches(s1, ids, issues):
        if ids:
            matched[s1] = tuple(id_code(i) for i in ids if i)
    ms = _scan(matching, MATCHING_HEADER, required, valid, problems, "matching_results.tsv", keep_matches)
    report["matching"] = ms

    if candidate:
        checked = set()
        def check_subset(s1, ids, issues):
            m = matched.get(s1)
            if m is not None:
                checked.add(s1)
                if not set(m) <= {id_code(i) for i in ids if i}:
                    issues["not_candidate"].append(s1)
        cs = _scan(candidate, CANDIDATE_HEADER, required, valid, problems, "candidate_pairs.tsv", check_subset)
        report["candidate"] = cs
        if ms is not None and cs is not None:
            orphan = [s for s in matched if s not in checked]     # matched S1 with no candidate row at all
            if orphan:
                problems.append(f"{len(orphan)} S1 entities have matches but no candidate row, e.g. {orphan[:MAX_EX]}")
            report["matched_subset_of_candidates"] = not orphan and not any("not all in their candidate" in p for p in problems)
    else:
        problems.append("candidate_pairs.tsv not given (required in the final package)")
    if ms is not None:
        report["matching_present_s1"] = ms["present_s1"]
        report["matching_missing_s1"] = ms.get("missing_s1")
    return problems, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--source1", required=True, help="source-1 TSV defining the required S1 ids")
    ap.add_argument("--source2")
    ap.add_argument("--source3")
    a = ap.parse_args(argv)
    problems, report = validate(a.matching, a.candidate, a.source1, a.source2, a.source3)
    for k, v in report.items():
        print(f"  {k}: {v}")
    if problems:
        print(f"FAIL — {len(problems)} hard problem(s):")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        return 1
    print("PASS — all hard submission checks satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
