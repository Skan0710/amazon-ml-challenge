#!/usr/bin/env python3
"""
Assemble the final submission package exactly as the official README specifies:

    submission/<team>_submission/
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/business_entity_resolution/
    │   ├── src/            (all pipeline source)
    │   ├── models/         (trained model + decision rule used for the submission)
    │   ├── README.md
    │   └── requirements.txt
    └── Documentation_template.md   (filled-in methodology)

    python scripts/build_submission_package.py --team <team_name> [--zip]

Safety: refuses to run unless output/{matching_results,candidate_pairs}.tsv exist (i.e. were
published after validation) and no *.tmp outputs are lying around; output files are APFS clones
(``cp -c``: no extra disk) when possible; the package is checked for forbidden content
(datasets, *.tmp, *.parquet, *.duckdb, __pycache__, .DS_Store, .venv) and its source is import-tested.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_FILES = ["src/__init__.py", "src/normalization.py", "src/candidate_generation.py", "src/matching.py",
             "scripts/build_train_artifacts.py", "scripts/run_phase4.py", "scripts/preflight_phase4.py",
             "scripts/validate_final_submission.py", "scripts/run_official_validator_sharded.py",
             "scripts/audit_submission.py"]
FORBIDDEN_SUFFIXES = (".tmp", ".parquet", ".duckdb", ".wal", ".pyc", ".log")
FORBIDDEN_NAMES = {"__pycache__", ".DS_Store", ".venv", ".ipynb_checkpoints"}
DATASET_NAMES = {f"{p}_source{i}.tsv" for p in ("train", "test") for i in (1, 2, 3)} | {"train_ground_truth.tsv"}


def clone(src: Path, dst: Path) -> None:
    """APFS clone (no extra disk) with a plain-copy fallback."""
    r = subprocess.run(["cp", "-c", str(src), str(dst)], capture_output=True)
    if r.returncode != 0:
        shutil.copy2(src, dst)


def audit(pkg: Path) -> list:
    bad = []
    for p in pkg.rglob("*"):
        if p.name in FORBIDDEN_NAMES or p.suffix in FORBIDDEN_SUFFIXES or p.name in DATASET_NAMES:
            bad.append(str(p.relative_to(pkg)))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--doc", type=Path, default=ROOT / "packaging" / "Documentation_template.md")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    out = ROOT / "output"
    finals = [out / "matching_results.tsv", out / "candidate_pairs.tsv"]
    if not all(f.exists() for f in finals):
        raise SystemExit("final outputs missing — run the validated test inference first")
    if list(out.glob("*.tmp")):
        raise SystemExit(f"temporary outputs present in {out}: {list(out.glob('*.tmp'))} — refusing to package")
    pkg = ROOT / "submission" / f"{a.team}_submission"
    shutil.rmtree(pkg, ignore_errors=True)
    code = pkg / "code" / "business_entity_resolution"
    (pkg / "output").mkdir(parents=True)
    (code / "src").mkdir(parents=True)
    (code / "models").mkdir()
    for f in finals:
        clone(f, pkg / "output" / f.name)
    for rel in SRC_FILES:
        shutil.copy2(ROOT / rel, code / "src" / Path(rel).name)
    for m in ("phase4_model.joblib", "phase4_rule.json"):
        shutil.copy2(ROOT / "models" / m, code / "models" / m)
    shutil.copy2(ROOT / "packaging" / "CODE_README.md", code / "README.md")
    shutil.copy2(ROOT / "packaging" / "requirements.txt", code / "requirements.txt")
    if a.doc.exists():
        shutil.copy2(a.doc, pkg / "Documentation_template.md")
    else:
        print(f"WARNING: methodology document {a.doc} not found")
    bad = audit(pkg)
    if bad:
        raise SystemExit(f"forbidden content in package: {bad}")
    r = subprocess.run([sys.executable, "-c", "import src.normalization, src.candidate_generation, src.matching; print('imports ok')"],
                       cwd=code, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"packaged source does not import: {r.stderr}")
    for p in code.rglob("__pycache__"):
        shutil.rmtree(p)
    print(r.stdout.strip())
    sizes = {str(p.relative_to(pkg)): p.stat().st_size for p in sorted(pkg.rglob("*")) if p.is_file()}
    for k, v in sizes.items():
        print(f"  {v:>14,}  {k}")
    if a.zip:
        # official layout: output/, code/ and Documentation_template.md at the ZIP ROOT
        z = shutil.make_archive(str(pkg), "zip", root_dir=pkg)
        print(f"zip: {z} ({Path(z).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
