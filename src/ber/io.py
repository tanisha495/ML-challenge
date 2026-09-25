from __future__ import annotations

from pathlib import Path

import pandas as pd


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def assert_train_only_path(path: Path) -> None:
    """Protect this implementation from accidentally reading challenge test data."""

    parts = {part.lower() for part in path.resolve().parts}
    if "test" in parts:
        raise ValueError(f"Train-only workflow refused to read test path: {path}")


def read_tsv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    assert_train_only_path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", keep_default_na=False, usecols=columns)
    return df


def read_train_source1(train_dir: Path) -> pd.DataFrame:
    """Read only Source 1 training records."""

    assert_train_only_path(train_dir)
    return read_tsv(train_dir / "train_source1.tsv", SOURCE_COLUMNS)


def read_ground_truth(train_dir: Path) -> pd.DataFrame:
    assert_train_only_path(train_dir)
    truth = read_tsv(train_dir / "train_ground_truth.tsv", TRUTH_COLUMNS)
    truth["matched_list"] = truth["matched_entity_ids"].map(parse_id_list)
    truth["n_matches"] = truth["matched_list"].map(len)
    return truth


def parse_id_list(value: str) -> list[str]:
    if not value:
        return []
    return [item for item in str(value).split(",") if item]


def truth_dict(truth: pd.DataFrame) -> dict[str, set[str]]:
    return {
        row.source1_entity_id: set(row.matched_list)
        for row in truth[["source1_entity_id", "matched_list"]].itertuples(index=False)
    }


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
