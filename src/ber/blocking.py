from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ber.config import BlockingConfig


def _require_sklearn() -> tuple[object, object]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for sparse blocking. Install requirements.txt first."
        ) from exc
    return TfidfVectorizer, normalize


def _topk_from_sparse_row(row, right_ids: np.ndarray, top_k: int) -> list[tuple[str, float, int]]:
    if row.nnz == 0:
        return []
    data = row.data
    indices = row.indices
    if len(data) > top_k:
        chosen = np.argpartition(data, -top_k)[-top_k:]
        data = data[chosen]
        indices = indices[chosen]
    order = np.argsort(-data)
    return [(str(right_ids[indices[i]]), float(data[i]), rank + 1) for rank, i in enumerate(order)]


def sparse_tfidf_block(
    left: pd.DataFrame,
    right: pd.DataFrame,
    text_col: str,
    top_k: int,
    blocker_name: str,
    config: BlockingConfig,
) -> pd.DataFrame:
    """Generate same-country top-k candidates with char n-gram TF-IDF cosine."""

    TfidfVectorizer, normalize = _require_sklearn()
    records: list[dict[str, object]] = []
    for country, left_country in left.groupby("country_norm", sort=False):
        right_country = right[right["country_norm"].eq(country)]
        if left_country.empty or right_country.empty:
            continue

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(config.ngram_min, config.ngram_max),
            min_df=config.min_df,
            max_features=config.max_features,
            dtype=np.float32,
        )
        right_matrix = vectorizer.fit_transform(right_country[text_col].fillna(""))
        right_matrix = normalize(right_matrix, norm="l2", copy=False)
        right_ids = right_country["entity_id"].to_numpy()

        for start in range(0, len(left_country), config.query_batch_size):
            left_batch = left_country.iloc[start : start + config.query_batch_size]
            left_matrix = vectorizer.transform(left_batch[text_col].fillna(""))
            left_matrix = normalize(left_matrix, norm="l2", copy=False)
            sims = left_matrix @ right_matrix.T
            for row_idx, source1_id in enumerate(left_batch["entity_id"].to_numpy()):
                for candidate_id, score, rank in _topk_from_sparse_row(sims.getrow(row_idx), right_ids, top_k):
                    records.append(
                        {
                            "source1_entity_id": source1_id,
                            "candidate_entity_id": candidate_id,
                            f"{blocker_name}_score": score,
                            f"{blocker_name}_rank": rank,
                            "blocker": blocker_name,
                        }
                    )

    return pd.DataFrame.from_records(records)


def token_block(
    left: pd.DataFrame,
    right: pd.DataFrame,
    token_cols: Iterable[str],
    top_k: int,
    blocker_name: str,
    max_token_frequency: int = 20_000,
) -> pd.DataFrame:
    """Cheap high-recall blocker based on shared rare tokens and numbers."""

    token_cols = list(token_cols)
    token_frequency: Counter[tuple[str, str]] = Counter()
    right_tokens: list[set[str]] = []
    for row in right[token_cols + ["country_norm"]].itertuples(index=False):
        country = getattr(row, "country_norm")
        tokens = set()
        for col in token_cols:
            tokens.update(str(getattr(row, col) or "").split())
        right_tokens.append(tokens)
        for token in tokens:
            token_frequency[(country, token)] += 1

    index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, row in enumerate(right[token_cols + ["country_norm"]].itertuples(index=False)):
        country = getattr(row, "country_norm")
        for token in right_tokens[idx]:
            if 0 < token_frequency[(country, token)] <= max_token_frequency:
                index[(country, token)].append(idx)

    right_ids = right["entity_id"].to_numpy()
    records: list[dict[str, object]] = []
    for row in left[["entity_id", "country_norm"] + token_cols].itertuples(index=False):
        source1_id = getattr(row, "entity_id")
        country = getattr(row, "country_norm")
        left_tokens = set()
        for col in token_cols:
            left_tokens.update(str(getattr(row, col) or "").split())
        counts: Counter[int] = Counter()
        for token in left_tokens:
            for idx in index.get((country, token), ()):
                counts[idx] += 1
        scored: list[tuple[float, int]] = []
        for idx, count in counts.most_common(top_k * 20):
            denom = len(left_tokens | right_tokens[idx])
            if denom:
                scored.append((count / denom, idx))
        scored.sort(reverse=True)
        for rank, (score, idx) in enumerate(scored[:top_k], start=1):
            records.append(
                {
                    "source1_entity_id": source1_id,
                    "candidate_entity_id": str(right_ids[idx]),
                    f"{blocker_name}_score": float(score),
                    f"{blocker_name}_rank": rank,
                    "blocker": blocker_name,
                }
            )

    return pd.DataFrame.from_records(records)


def combine_candidate_frames(frames: list[pd.DataFrame], config: BlockingConfig) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])

    all_candidates = pd.concat(frames, ignore_index=True, sort=False)
    score_cols = [col for col in all_candidates.columns if col.endswith("_score")]
    rank_cols = [col for col in all_candidates.columns if col.endswith("_rank")]
    agg_spec = {col: "max" for col in score_cols}
    agg_spec.update({col: "min" for col in rank_cols})
    agg_spec["blocker"] = lambda values: "|".join(sorted(set(values)))

    combined = (
        all_candidates.groupby(["source1_entity_id", "candidate_entity_id"], as_index=False)
        .agg(agg_spec)
        .fillna(0)
    )
    combined["candidate_strength"] = combined[score_cols].max(axis=1) if score_cols else 0.0
    combined = combined.sort_values(
        ["source1_entity_id", "candidate_strength"], ascending=[True, False], kind="mergesort"
    )
    combined["candidate_rank"] = combined.groupby("source1_entity_id").cumcount() + 1
    combined = combined[combined["candidate_rank"] <= config.max_candidates_per_entity].copy()
    return combined.reset_index(drop=True)


def build_union_candidates(
    source1: pd.DataFrame,
    targets: pd.DataFrame,
    config: BlockingConfig,
) -> pd.DataFrame:
    """Union of name, address, combined text, and token blockers."""

    frames = [
        sparse_tfidf_block(
            source1,
            targets,
            "name_norm",
            config.name_top_k,
            "name_tfidf",
            config,
        ),
        sparse_tfidf_block(
            source1,
            targets,
            "address_norm",
            config.address_top_k,
            "address_tfidf",
            config,
        ),
        sparse_tfidf_block(
            source1,
            targets,
            "combined_norm",
            config.combined_top_k,
            "combined_tfidf",
            config,
        ),
        token_block(
            source1,
            targets,
            ["name_norm", "address_norm", "address_numbers"],
            config.token_top_k,
            "token",
        ),
    ]
    return combine_candidate_frames(frames, config)


def candidate_mapping(candidates: pd.DataFrame) -> dict[str, list[str]]:
    if candidates.empty:
        return {}
    ordered = candidates.sort_values(["source1_entity_id", "candidate_rank"])
    return (
        ordered.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda values: list(dict.fromkeys(values)))
        .to_dict()
    )
