#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a valid all-empty first submission for the Amazon ML Challenge."
    )
    parser.add_argument("--resource-dir", type=Path, default=Path("/Users/tanishakhanna/Downloads/student_resource"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_source1 = args.resource_dir / "dataset" / "test" / "test_source1.tsv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source1 = pd.read_csv(test_source1, sep="\t", keep_default_na=False, usecols=["entity_id"])
    matching_path = args.output_dir / "matching_results.tsv"
    candidate_path = args.output_dir / "candidate_pairs.tsv"

    # Write raw TSV lines so empty match/candidate lists are represented exactly
    # as a trailing tab followed by newline, matching the challenge examples.
    with matching_path.open("w", encoding="utf-8", newline="") as matching_file:
        matching_file.write("source1_entity_id\tmatched_entity_ids\n")
        for source1_id in source1["entity_id"]:
            matching_file.write(f"{source1_id}\t\n")

    with candidate_path.open("w", encoding="utf-8", newline="") as candidate_file:
        candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")
        for source1_id in source1["entity_id"]:
            candidate_file.write(f"{source1_id}\t\n")

    print(f"Wrote {len(source1):,} rows to {matching_path}")
    print(f"Wrote {len(source1):,} rows to {candidate_path}")


if __name__ == "__main__":
    main()
