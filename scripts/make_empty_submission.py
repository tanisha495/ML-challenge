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
    matching = pd.DataFrame(
        {
            "source1_entity_id": source1["entity_id"],
            "matched_entity_ids": "",
        }
    )
    candidates = pd.DataFrame(
        {
            "source1_entity_id": source1["entity_id"],
            "candidate_entity_ids": "",
        }
    )

    matching_path = args.output_dir / "matching_results.tsv"
    candidate_path = args.output_dir / "candidate_pairs.tsv"
    matching.to_csv(matching_path, sep="\t", index=False)
    candidates.to_csv(candidate_path, sep="\t", index=False)
    print(f"Wrote {len(source1):,} rows to {matching_path}")
    print(f"Wrote {len(source1):,} rows to {candidate_path}")


if __name__ == "__main__":
    main()
