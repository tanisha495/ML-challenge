#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from ber.normalize import normalize_frame, token_set


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a noisy-data-aware rule submission using token overlap and address cues."
    )
    parser.add_argument("--resource-dir", type=Path, default=Path("/Users/tanishakhanna/Downloads/student_resource"))
    parser.add_argument("--output-dir", type=Path, default=Path("output_fuzzy"))
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--max-s1-token-frequency", type=int, default=2500)
    parser.add_argument("--max-candidates-per-target", type=int, default=25)
    parser.add_argument("--max-ids-per-s1", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.64)
    return parser.parse_args()


def clean_name_tokens(value: str) -> set[str]:
    return {tok for tok in token_set(value) if len(tok) >= 3 and not tok.isdigit()}


def clean_address_tokens(value: str) -> set[str]:
    return {tok for tok in token_set(value) if len(tok) >= 3}


def useful_numbers(value: str) -> set[str]:
    if not value:
        return set()
    return {num for num in str(value).split("|") if len(num) >= 3}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def build_source1_indexes(
    source1: pd.DataFrame,
    max_token_frequency: int,
) -> tuple[
    dict[tuple[str, str, str], list[int]],
    dict[tuple[str, str, str], list[int]],
    dict[tuple[str, str], list[int]],
    list[set[str]],
    list[set[str]],
    list[set[str]],
]:
    exact_index: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    name_number_index: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    name_tokens_by_row: list[set[str]] = []
    address_tokens_by_row: list[set[str]] = []
    numbers_by_row: list[set[str]] = []
    token_frequency: Counter[tuple[str, str]] = Counter()

    for idx, row in enumerate(source1.itertuples(index=False)):
        name_tokens = clean_name_tokens(row.name_norm)
        address_tokens = clean_address_tokens(row.address_norm)
        numbers = useful_numbers(row.address_numbers)
        name_tokens_by_row.append(name_tokens)
        address_tokens_by_row.append(address_tokens)
        numbers_by_row.append(numbers)

        if row.name_norm and row.address_norm:
            exact_index[(row.country_norm, row.name_norm, row.address_norm)].append(idx)
        for num in numbers:
            if row.name_norm:
                name_number_index[(row.country_norm, row.name_norm, num)].append(idx)
        for token in name_tokens:
            token_frequency[(row.country_norm, token)] += 1

    token_index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(source1.itertuples(index=False)):
        for token in name_tokens_by_row[idx]:
            freq = token_frequency[(row.country_norm, token)]
            if 1 <= freq <= max_token_frequency:
                token_index[(row.country_norm, token)].append(idx)

    exact_index = {key: rows for key, rows in exact_index.items() if len(rows) == 1}
    name_number_index = {key: rows for key, rows in name_number_index.items() if len(rows) <= 4}
    return (
        exact_index,
        name_number_index,
        token_index,
        name_tokens_by_row,
        address_tokens_by_row,
        numbers_by_row,
    )


def score_pair(
    s1_idx: int,
    target_name_tokens: set[str],
    target_address_tokens: set[str],
    target_numbers: set[str],
    name_tokens_by_row: list[set[str]],
    address_tokens_by_row: list[set[str]],
    numbers_by_row: list[set[str]],
) -> tuple[float, float, float, bool]:
    s1_name = name_tokens_by_row[s1_idx]
    s1_addr = address_tokens_by_row[s1_idx]
    s1_nums = numbers_by_row[s1_idx]
    name_score = jaccard(s1_name, target_name_tokens)
    address_score = jaccard(s1_addr, target_address_tokens)
    number_overlap = bool(s1_nums and target_numbers and (s1_nums & target_numbers))
    score = 0.68 * name_score + 0.24 * address_score + (0.08 if number_overlap else 0.0)
    return score, name_score, address_score, number_overlap


def add_link(
    links: dict[int, set[str]],
    s1_idx: int,
    target_id: str,
    max_ids_per_s1: int,
) -> bool:
    target_id = target_id.strip()
    if not target_id:
        return False
    if len(links[s1_idx]) >= max_ids_per_s1:
        return False
    before = len(links[s1_idx])
    links[s1_idx].add(target_id)
    return len(links[s1_idx]) > before


def process_target_file(
    path: Path,
    exact_index: dict[tuple[str, str, str], list[int]],
    name_number_index: dict[tuple[str, str, str], list[int]],
    token_index: dict[tuple[str, str], list[int]],
    name_tokens_by_row: list[set[str]],
    address_tokens_by_row: list[set[str]],
    numbers_by_row: list[set[str]],
    links: dict[int, set[str]],
    chunk_size: int,
    max_candidates_per_target: int,
    max_ids_per_s1: int,
    score_threshold: float,
) -> int:
    added = 0
    for chunk_no, chunk in enumerate(
        pd.read_csv(path, sep="\t", keep_default_na=False, usecols=SOURCE_COLUMNS, chunksize=chunk_size),
        start=1,
    ):
        norm = normalize_frame(chunk)
        for row in norm.itertuples(index=False):
            if not row.name_norm:
                continue
            target_id = row.entity_id.strip()
            target_name_tokens = clean_name_tokens(row.name_norm)
            target_address_tokens = clean_address_tokens(row.address_norm)
            target_numbers = useful_numbers(row.address_numbers)

            candidate_counts: Counter[int] = Counter()
            exact_key = (row.country_norm, row.name_norm, row.address_norm)
            for s1_idx in exact_index.get(exact_key, ()):
                candidate_counts[s1_idx] += 6

            for num in target_numbers:
                key = (row.country_norm, row.name_norm, num)
                for s1_idx in name_number_index.get(key, ()):
                    candidate_counts[s1_idx] += 4

            for token in target_name_tokens:
                for s1_idx in token_index.get((row.country_norm, token), ()):
                    candidate_counts[s1_idx] += 1

            for s1_idx, shared_count in candidate_counts.most_common(max_candidates_per_target):
                score, name_score, address_score, number_overlap = score_pair(
                    s1_idx,
                    target_name_tokens,
                    target_address_tokens,
                    target_numbers,
                    name_tokens_by_row,
                    address_tokens_by_row,
                    numbers_by_row,
                )
                strong_name = name_score >= 0.55 and shared_count >= 2
                address_evidence = address_score >= 0.16 or number_overlap
                exact_like = score >= 0.82 and name_score >= 0.70
                fuzzy_ok = score >= score_threshold and strong_name and address_evidence
                if not (exact_like or fuzzy_ok):
                    continue
                added += int(add_link(links, s1_idx, target_id, max_ids_per_s1))

        print(f"    processed chunk {chunk_no}, links so far: {sum(len(v) for v in links.values()):,}")
    return added


def write_output(path: Path, column_name: str, source1_ids: list[str], links: dict[int, set[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{column_name}\n")
        for idx, source1_id in enumerate(source1_ids):
            ids = sorted(item.strip() for item in links.get(idx, set()) if item.strip())
            f.write(f"{source1_id.strip()}\t{','.join(ids)}\n")


def main() -> None:
    args = parse_args()
    test_dir = args.resource_dir / "dataset" / "test"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and normalizing test Source 1...")
    source1 = pd.read_csv(test_dir / "test_source1.tsv", sep="\t", keep_default_na=False, usecols=SOURCE_COLUMNS)
    source1["entity_id"] = source1["entity_id"].str.strip()
    source1 = normalize_frame(source1)
    source1_ids = source1["entity_id"].tolist()

    print("Building Source 1 indexes for noisy matching...")
    (
        exact_index,
        name_number_index,
        token_index,
        name_tokens_by_row,
        address_tokens_by_row,
        numbers_by_row,
    ) = build_source1_indexes(source1, args.max_s1_token_frequency)
    print(f"  exact name+address keys: {len(exact_index):,}")
    print(f"  name+number keys: {len(name_number_index):,}")
    print(f"  name-token index keys: {len(token_index):,}")

    links: dict[int, set[str]] = defaultdict(set)
    for filename in ("test_source2.tsv", "test_source3.tsv"):
        print(f"Scanning {filename}...")
        added = process_target_file(
            test_dir / filename,
            exact_index,
            name_number_index,
            token_index,
            name_tokens_by_row,
            address_tokens_by_row,
            numbers_by_row,
            links,
            args.chunk_size,
            args.max_candidates_per_target,
            args.max_ids_per_s1,
            args.score_threshold,
        )
        print(f"  added {added:,} links from {filename}")

    matching_path = args.output_dir / "matching_results.tsv"
    candidate_path = args.output_dir / "candidate_pairs.tsv"
    write_output(matching_path, "matched_entity_ids", source1_ids, links)
    write_output(candidate_path, "candidate_entity_ids", source1_ids, links)
    print(f"Wrote {matching_path}")
    print(f"Wrote {candidate_path}")
    print(f"Rows with matches: {sum(bool(links.get(i)) for i in range(len(source1_ids))):,}")
    print(f"Total matched links: {sum(len(v) for v in links.values()):,}")


if __name__ == "__main__":
    main()
