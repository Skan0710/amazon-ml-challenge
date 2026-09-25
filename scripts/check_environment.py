#!/usr/bin/env python3
"""
scripts/check_environment.py
-----------------------------
Amazon ML Challenge 2026 — environment verification script.

Checks:
  - Python version and executable
  - Platform and architecture
  - All required package versions
  - Dataset file existence and sizes (no data loaded)

Usage:
    .venv/bin/python scripts/check_environment.py
"""

import sys
import platform
import importlib
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ── 1. Python / Platform ─────────────────────────────────────────────────────

section("Python & Platform")
print(f"  Python version   : {sys.version}")
print(f"  Python executable: {sys.executable}")
print(f"  Platform         : {platform.platform()}")
print(f"  Architecture     : {platform.machine()}")
print(f"  Processor        : {platform.processor()}")

# Verify the executable belongs to the project .venv
project_root = Path(__file__).resolve().parent.parent
expected_prefix = project_root / ".venv"
exe = Path(sys.executable)
if str(exe).startswith(str(expected_prefix)):
    print(f"  OK  Executable is inside .venv: {exe}")
else:
    print(f"  WARN  Executable is NOT inside project .venv!")
    print(f"        Expected prefix : {expected_prefix}")
    print(f"        Actual executable: {exe}")

# ── 2. Package Versions ───────────────────────────────────────────────────────

section("Package Versions")

REQUIRED_PACKAGES = [
    ("duckdb",       "duckdb"),
    ("pandas",       "pandas"),
    ("numpy",        "numpy"),
    ("scipy",        "scipy"),
    ("sklearn",      "scikit-learn"),
    ("rapidfuzz",    "rapidfuzz"),
    ("tqdm",         "tqdm"),
    ("joblib",       "joblib"),
    ("matplotlib",   "matplotlib"),
    ("seaborn",      "seaborn"),
    ("jupyter_core", "jupyter-core"),
    ("ipykernel",    "ipykernel"),
]

all_ok = True
for import_name, display_name in REQUIRED_PACKAGES:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  OK  {display_name:<22} {version}")
    except ImportError as e:
        print(f"  MISS {display_name:<22} MISSING — {e}")
        all_ok = False

if all_ok:
    print("\n  All required packages are present.")
else:
    print("\n  WARN Some packages are missing — re-run pip install.")

# ── 3. Optional: XGBoost ─────────────────────────────────────────────────────

try:
    import xgboost as xgb
    print(f"\n  INFO xgboost (optional): {xgb.__version__}")
except ImportError:
    print("\n  INFO xgboost (optional): not installed — OK for now")

# ── 4. Dataset Files ──────────────────────────────────────────────────────────

section("Dataset Files")

# Training files in project data/raw/
TRAIN_DIR = project_root / "data" / "raw"
TRAIN_FILES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
]

# Test files in student_resource (outside project)
TEST_DIR = Path("/Users/omkardabholkar/student_resource/dataset/test")
TEST_FILES = [
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv",
]

print(f"\n  Training files → {TRAIN_DIR}")
for fname in TRAIN_FILES:
    p = TRAIN_DIR / fname
    if p.exists():
        size = p.stat().st_size
        print(f"    OK  {fname:<32} {fmt_bytes(size)}")
    else:
        print(f"    MISS {fname:<32} NOT FOUND")

print(f"\n  Test files → {TEST_DIR}")
for fname in TEST_FILES:
    p = TEST_DIR / fname
    if p.exists():
        size = p.stat().st_size
        print(f"    OK  {fname:<32} {fmt_bytes(size)}")
    else:
        print(f"    MISS {fname:<32} NOT FOUND")

# ── 5. DuckDB smoke-test ──────────────────────────────────────────────────────

section("DuckDB Smoke-test")
try:
    import duckdb
    conn = duckdb.connect()
    result = conn.execute("SELECT 42 AS answer").fetchone()
    assert result[0] == 42
    print(f"  OK  DuckDB in-memory query OK  (SELECT 42 -> {result[0]})")
    conn.close()
except Exception as e:
    print(f"  FAIL DuckDB smoke-test FAILED: {e}")

# ── Done ──────────────────────────────────────────────────────────────────────

section("Done")
print("  Environment check complete.\n")
