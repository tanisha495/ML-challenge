from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ber.metrics import macro_fbeta


def predictions_from_scores(
    scored: pd.DataFrame,
    source1_ids: Iterable[str],
    threshold: float,
    cap: int | None,
) -> dict[str, list[str]]:
    predictions = {source1_id: [] for source1_id in source1_ids}
    if scored.empty:
        return predictions
    eligible = scored[scored["probability"].ge(threshold)].copy()
    if cap is not None:
        eligible = eligible[eligible["prob_rank"].le(cap)]
    for source1_id, group in eligible.groupby("source1_entity_id", sort=False):
        predictions[source1_id] = group["candidate_entity_id"].tolist()
    return predictions


def tune_decision_rule(
    scored: pd.DataFrame,
    truths: dict[str, set[str]],
    source1_ids: list[str],
    thresholds: Iterable[float],
    caps: Iterable[int | None],
) -> tuple[dict[str, object], dict[str, list[str]]]:
    best_score = -1.0
    best_params: dict[str, object] = {"threshold": 0.5, "cap": None, "macro_f05": 0.0}
    best_predictions: dict[str, list[str]] = {}
    for threshold in thresholds:
        for cap in caps:
            predictions = predictions_from_scores(scored, source1_ids, threshold, cap)
            score = macro_fbeta(predictions, truths, source1_ids, beta=0.5)
            if score > best_score:
                best_score = score
                best_params = {"threshold": float(threshold), "cap": cap, "macro_f05": float(score)}
                best_predictions = predictions
    return best_params, best_predictions
