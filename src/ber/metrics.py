from __future__ import annotations

from collections.abc import Iterable, Mapping


def fbeta_score_for_entity(predicted: Iterable[str], truth: Iterable[str], beta: float = 0.5) -> float:
    predicted_set = set(predicted)
    truth_set = set(truth)
    if not truth_set:
        return 1.0 if not predicted_set else 0.0
    if not predicted_set:
        return 0.0

    tp = len(predicted_set & truth_set)
    if tp == 0:
        return 0.0

    precision = tp / len(predicted_set)
    recall = tp / len(truth_set)
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_fbeta(
    predictions: Mapping[str, Iterable[str]],
    truths: Mapping[str, Iterable[str]],
    source1_ids: Iterable[str],
    beta: float = 0.5,
) -> float:
    ids = list(source1_ids)
    if not ids:
        return 0.0
    total = 0.0
    for source1_id in ids:
        total += fbeta_score_for_entity(predictions.get(source1_id, ()), truths.get(source1_id, ()), beta)
    return total / len(ids)


def blocking_audit(
    candidates: Mapping[str, Iterable[str]],
    truths: Mapping[str, Iterable[str]],
    source1_ids: Iterable[str],
) -> dict[str, float]:
    total_true = 0
    found_true = 0
    matched_entities = 0
    fully_recovered = 0
    total_candidates = 0
    ids = list(source1_ids)

    for source1_id in ids:
        cand_set = set(candidates.get(source1_id, ()))
        true_set = set(truths.get(source1_id, ()))
        total_candidates += len(cand_set)
        if not true_set:
            continue
        matched_entities += 1
        hits = len(cand_set & true_set)
        found_true += hits
        total_true += len(true_set)
        fully_recovered += int(hits == len(true_set))

    oracle_predictions = {
        source1_id: sorted(set(candidates.get(source1_id, ())) & set(truths.get(source1_id, ())))
        for source1_id in ids
    }

    return {
        "pair_recall": found_true / total_true if total_true else 0.0,
        "full_entity_recall": fully_recovered / matched_entities if matched_entities else 0.0,
        "avg_candidates_per_s1": total_candidates / len(ids) if ids else 0.0,
        "oracle_macro_f05": macro_fbeta(oracle_predictions, truths, ids, beta=0.5),
    }
