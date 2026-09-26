#!/usr/bin/env python3
"""Package the final submission zip per the problem statement's required layout:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

REPO = Path("/Users/uniteditservices/Desktop/ML-challenge")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-name", default="submission")
    parser.add_argument("--output-dir", type=Path, default=REPO / "output_final_v2")
    parser.add_argument("--staging-dir", type=Path, default=REPO / "_submission_staging")
    args = parser.parse_args()

    matching = args.output_dir / "matching_results.tsv"
    candidates = args.output_dir / "candidate_pairs.tsv"
    if not matching.exists() or not candidates.exists():
        raise SystemExit(f"Missing output files in {args.output_dir}; run the pipeline first.")

    staging = args.staging_dir
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    (staging / "output").mkdir()
    shutil.copy2(matching, staging / "output" / "matching_results.tsv")
    shutil.copy2(candidates, staging / "output" / "candidate_pairs.tsv")

    code_dir = staging / "code" / "business_entity_resolution"
    code_dir.mkdir(parents=True)
    shutil.copytree(
        REPO / "src", code_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(REPO / "README.md", code_dir / "README.md")
    shutil.copy2(REPO / "requirements.txt", code_dir / "requirements.txt")
    scripts_dir = code_dir / "scripts"
    scripts_dir.mkdir()
    for script in ("train_validate.py", "make_final_submission.py", "run_full_pipeline.py"):
        shutil.copy2(REPO / "scripts" / script, scripts_dir / script)
    docs_dir = code_dir / "docs"
    docs_dir.mkdir()
    shutil.copy2(REPO / "docs" / "memory_postmortem.md", docs_dir / "memory_postmortem.md")

    doc_src = REPO / "student_resource" / "Documentation_template.md"
    shutil.copy2(doc_src, staging / "Documentation_template.md")

    zip_path = REPO / f"{args.team_name}_submission.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in staging.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(staging))

    shutil.rmtree(staging)
    print(f"Wrote {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
