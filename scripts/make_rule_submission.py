#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ber.normalize import normalize_frame, token_set


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a conservative rule-based mapped test submission."
    )
    parser.add_argument("--resource-dir", type=Path, default=Path("/Users/tanishakhanna/Downloads/student_resource"))
    parser.add_argument("--output-dir", type=Path, default=Path("output_rule"))
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--max-ids-per-list", type=int, default=8)
    return parser.parse_args()


def number_tokens(value: str) -> set[str]:
    if not value:
        return set()
    return {item for item in str(value).split("|") if item}


def useful_numbers(value: str) -> set[str]:
    # Prefer postal/PIN/ZIP-like numbers, but keep medium address numbers too.
    return {num for num in number_tokens(value) if len(num) >= 3}


def build_source1_indexes(source1: pd.DataFrame) -> tuple[dict[tuple[str, str, str], list[str]], dict[tuple[str, str, str], list[str]], dict[str, set[str]]]:
    exact_index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    name_number_index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    address_tokens_by_s1: dict[str, set[str]] = {}

    for row in source1.itertuples(index=False):
        source1_id = row.entity_id.strip()
        if not row.name_norm:
            continue
        address_tokens_by_s1[source1_id] = token_set(row.address_norm)

        if row.address_norm:
            exact_index[(row.country_norm, row.name_norm, row.address_norm)].append(source1_id)

        for num in useful_numbers(row.address_numbers):
            name_number_index[(row.country_norm, row.name_norm, num)].append(source1_id)

    # Ambiguous keys are dangerous under F0.5, so remove them.
    exact_index = {key: ids for key, ids in exact_index.items() if len(ids) == 1}
    name_number_index = {key: ids for key, ids in name_number_index.items() if len(ids) <= 3}
    return exact_index, name_number_index, address_tokens_by_s1


def add_match(
    matches: dict[str, set[str]],
    candidates: dict[str, set[str]],
    source1_id: str,
    target_id: str,
    max_ids_per_list: int,
) -> None:
    source1_id = source1_id.strip()
    target_id = target_id.strip()
    if not source1_id or not target_id:
        return
    if len(matches[source1_id]) >= max_ids_per_list:
        return
    matches[source1_id].add(target_id)
    candidates[source1_id].add(target_id)


def address_overlap_ok(source_tokens: set[str], target_tokens: set[str]) -> bool:
    if not source_tokens or not target_tokens:
        return False
    shared = len(source_tokens & target_tokens)
    smaller = min(len(source_tokens), len(target_tokens))
    return shared >= 2 and shared / smaller >= 0.35


def process_target_file(
    path: Path,
    exact_index: dict[tuple[str, str, str], list[str]],
    name_number_index: dict[tuple[str, str, str], list[str]],
    address_tokens_by_s1: dict[str, set[str]],
    matches: dict[str, set[str]],
    candidates: dict[str, set[str]],
    chunk_size: int,
    max_ids_per_list: int,
) -> int:
    added = 0
    for chunk in pd.read_csv(path, sep="\t", keep_default_na=False, usecols=SOURCE_COLUMNS, chunksize=chunk_size):
        norm = normalize_frame(chunk)
        for row in norm.itertuples(index=False):
            if not row.name_norm:
                continue
            target_id = row.entity_id.strip()
            target_tokens = token_set(row.address_norm)

            exact_key = (row.country_norm, row.name_norm, row.address_norm)
            for source1_id in exact_index.get(exact_key, ()):
                before = len(matches[source1_id])
                add_match(matches, candidates, source1_id, target_id, max_ids_per_list)
                added += int(len(matches[source1_id]) > before)

            for num in useful_numbers(row.address_numbers):
                key = (row.country_norm, row.name_norm, num)
                for source1_id in name_number_index.get(key, ()):
                    if not address_overlap_ok(address_tokens_by_s1.get(source1_id, set()), target_tokens):
                        continue
                    before = len(matches[source1_id])
                    add_match(matches, candidates, source1_id, target_id, max_ids_per_list)
                    added += int(len(matches[source1_id]) > before)
    return added


def write_output(path: Path, header: str, column_name: str, source1_ids: list[str], values: dict[str, set[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{column_name}\n")
        for source1_id in source1_ids:
            source1_id = source1_id.strip()
            ids = sorted(item.strip() for item in values.get(source1_id, set()) if item.strip())
            f.write(f"{source1_id}\t{','.join(ids)}\n")


def main() -> None:
    args = parse_args()
    test_dir = args.resource_dir / "dataset" / "test"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and normalizing test Source 1...")
    source1 = pd.read_csv(test_dir / "test_source1.tsv", sep="\t", keep_default_na=False, usecols=SOURCE_COLUMNS)
    source1["entity_id"] = source1["entity_id"].str.strip()
    source1 = normalize_frame(source1)
    source1_ids = source1["entity_id"].tolist()

    print("Building conservative Source 1 indexes...")
    exact_index, name_number_index, address_tokens_by_s1 = build_source1_indexes(source1)
    print(f"  exact name+address keys: {len(exact_index):,}")
    print(f"  name+number keys: {len(name_number_index):,}")

    matches: dict[str, set[str]] = defaultdict(set)
    candidates: dict[str, set[str]] = defaultdict(set)

    for filename in ("test_source2.tsv", "test_source3.tsv"):
        print(f"Scanning {filename}...")
        added = process_target_file(
            test_dir / filename,
            exact_index,
            name_number_index,
            address_tokens_by_s1,
            matches,
            candidates,
            args.chunk_size,
            args.max_ids_per_list,
        )
        print(f"  added {added:,} matched links")

    matching_path = args.output_dir / "matching_results.tsv"
    candidate_path = args.output_dir / "candidate_pairs.tsv"
    write_output(matching_path, "matching", "matched_entity_ids", source1_ids, matches)
    write_output(candidate_path, "candidate", "candidate_entity_ids", source1_ids, candidates)
    print(f"Wrote {matching_path}")
    print(f"Wrote {candidate_path}")
    print(f"Rows with matches: {sum(bool(matches.get(source1_id)) for source1_id in source1_ids):,}")
    print(f"Total matched links: {sum(len(v) for v in matches.values()):,}")


if __name__ == "__main__":
    main()
