from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from ber.normalize import pipe_set, token_set


def _ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, left or "", right or "").ratio()


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _overlap_count(left: set[str], right: set[str]) -> int:
    return len(left & right)


def build_pair_features(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    targets: pd.DataFrame,
    truth: dict[str, set[str]] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Build pair features for candidate rows."""

    if candidates.empty:
        return pd.DataFrame(), None

    left_cols = [
        "entity_id",
        "business_name",
        "business_address",
        "country_norm",
        "name_norm",
        "address_norm",
        "combined_norm",
        "legal_form",
        "address_numbers",
        "landmark_tokens",
    ]
    right_cols = left_cols + ["target_source"]
    left = source1[left_cols].add_prefix("s1_")
    right = targets[right_cols].add_prefix("cand_")

    pairs = candidates.merge(
        left,
        left_on="source1_entity_id",
        right_on="s1_entity_id",
        how="left",
        validate="many_to_one",
    ).merge(
        right,
        left_on="candidate_entity_id",
        right_on="cand_entity_id",
        how="left",
        validate="many_to_one",
    )

    rows: list[dict[str, float]] = []
    for row in pairs.itertuples(index=False):
        s1_name = str(row.s1_name_norm or "")
        c_name = str(row.cand_name_norm or "")
        s1_addr = str(row.s1_address_norm or "")
        c_addr = str(row.cand_address_norm or "")
        s1_comb = str(row.s1_combined_norm or "")
        c_comb = str(row.cand_combined_norm or "")
        s1_name_tokens = token_set(s1_name)
        c_name_tokens = token_set(c_name)
        s1_addr_tokens = token_set(s1_addr)
        c_addr_tokens = token_set(c_addr)
        s1_num = pipe_set(row.s1_address_numbers)
        c_num = pipe_set(row.cand_address_numbers)
        s1_legal = pipe_set(row.s1_legal_form)
        c_legal = pipe_set(row.cand_legal_form)
        s1_landmark = pipe_set(row.s1_landmark_tokens)
        c_landmark = pipe_set(row.cand_landmark_tokens)

        feature = {
            "candidate_rank": float(getattr(row, "candidate_rank", 9999)),
            "candidate_strength": float(getattr(row, "candidate_strength", 0.0)),
            "name_tfidf_score": float(getattr(row, "name_tfidf_score", 0.0)),
            "address_tfidf_score": float(getattr(row, "address_tfidf_score", 0.0)),
            "combined_tfidf_score": float(getattr(row, "combined_tfidf_score", 0.0)),
            "token_score": float(getattr(row, "token_score", 0.0)),
            "keyed_score": float(getattr(row, "keyed_score", 0.0)),
            "keyed_rank": float(getattr(row, "keyed_rank", 9999)),
            "name_jaccard": _jaccard(s1_name_tokens, c_name_tokens),
            "address_jaccard": _jaccard(s1_addr_tokens, c_addr_tokens),
            "combined_jaccard": _jaccard(token_set(s1_comb), token_set(c_comb)),
            "name_sequence_ratio": _ratio(s1_name, c_name),
            "address_sequence_ratio": _ratio(s1_addr, c_addr),
            "shared_name_tokens": float(_overlap_count(s1_name_tokens, c_name_tokens)),
            "shared_address_tokens": float(_overlap_count(s1_addr_tokens, c_addr_tokens)),
            "shared_numbers": float(_overlap_count(s1_num, c_num)),
            "has_number_overlap": float(bool(s1_num and c_num and (s1_num & c_num))),
            "number_conflict": float(bool(s1_num and c_num and not (s1_num & c_num))),
            "legal_form_overlap": float(bool(s1_legal and c_legal and (s1_legal & c_legal))),
            "legal_form_conflict": float(bool(s1_legal and c_legal and not (s1_legal & c_legal))),
            "landmark_overlap": float(bool(s1_landmark and c_landmark and (s1_landmark & c_landmark))),
            "same_country": float(row.s1_country_norm == row.cand_country_norm),
            "candidate_is_s2": float(str(row.cand_target_source) == "S2"),
            "candidate_is_s3": float(str(row.cand_target_source) == "S3"),
            "name_length_diff": float(abs(len(s1_name) - len(c_name))),
            "address_length_diff": float(abs(len(s1_addr) - len(c_addr))),
            "s1_name_token_count": float(len(s1_name_tokens)),
            "candidate_name_token_count": float(len(c_name_tokens)),
            "s1_address_token_count": float(len(s1_addr_tokens)),
            "candidate_address_token_count": float(len(c_addr_tokens)),
        }
        rows.append(feature)

    feature_frame = pd.DataFrame(rows).replace([np.inf, -np.inf], 0).fillna(0)
    labels = None
    if truth is not None:
        labels = pairs.apply(
            lambda row: int(row["candidate_entity_id"] in truth.get(row["source1_entity_id"], set())),
            axis=1,
        )
    return feature_frame, labels

