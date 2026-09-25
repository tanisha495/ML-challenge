from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Filesystem paths for the train-only workflow."""

    resource_dir: Path
    artifact_dir: Path

    @property
    def train_dir(self) -> Path:
        return self.resource_dir / "dataset" / "train"

    @property
    def output_dir(self) -> Path:
        return self.artifact_dir / "output"

    @property
    def model_dir(self) -> Path:
        return self.artifact_dir / "models"

    @property
    def cache_dir(self) -> Path:
        return self.artifact_dir / "cache"


@dataclass(frozen=True)
class BlockingConfig:
    """Controls high-recall candidate generation."""

    name_top_k: int = 35
    address_top_k: int = 35
    combined_top_k: int = 45
    token_top_k: int = 30
    max_candidates_per_entity: int = 80
    query_batch_size: int = 4096
    min_df: int = 2
    max_features: int | None = 600_000
    ngram_min: int = 3
    ngram_max: int = 5


@dataclass(frozen=True)
class TrainingConfig:
    """Controls train-only validation and modeling."""

    random_seed: int = 42
    max_train_s1: int | None = None
    target_sample_modulo: int = 500
    target_chunk_size: int = 500_000
    negative_downsample_ratio: float = 6.0
    decision_thresholds: tuple[float, ...] = tuple(x / 100 for x in range(20, 91, 2))
    max_match_caps: tuple[int | None, ...] = (None, 1, 2, 3, 5, 8, 12)


DEFAULT_RESOURCE_DIR = Path("/Users/tanishakhanna/Downloads/student_resource")
