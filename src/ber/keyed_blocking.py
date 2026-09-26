from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd
from rapidfuzz import fuzz

from ber.blocking import combine_candidate_frames
from ber.config import BlockingConfig

# NOTE: multiprocessing (both "spawn" and "fork" start methods) was tried
# here to parallelize row-scoring across CPU cores and measured SLOWER than
# single-process in both cases at real (4M+ row) corpus scale -- "spawn"
# pays a large one-time pickling cost for the per-country index broadcast to
# each worker, and "fork" still triggers copy-on-write page faults across a
# large reference-counted Python dict as workers merely read it (refcount
# increments are writes). Neither paid for itself here, so this stays
# single-process; a real speedup would need a non-Python-object index
# (e.g. numpy/arrow-backed) shared via true shared memory.

# Deterministic bucket keys used to shrink the search space before any pairwise
# scoring happens. Each row contributes to several buckets (one per key type);
# two records are only ever compared if they share at least one bucket. This
# turns an O(left x right) or O(left x corpus) problem into O(left x avg bucket
# size), which is what makes blocking tractable at multi-million-row scale --
# the plain full-corpus TF-IDF/token blockers in blocking.py do not scale past
# a few thousand left rows against a multi-million row corpus (see benchmarks:
# ~13h+ extrapolated for a single country/blocker at full test-set size).

NAME_PREFIX_LEN = 5
MAX_BUCKET_SIZE = 5000  # keys with a posting list bigger than this are dropped outright (pathological/degenerate keys); candidate_budget is what actually bounds per-row scoring cost
MIN_TOKEN_LEN = 3
N_RARE_NAME_TOKENS = 3
N_RARE_ADDRESS_TOKENS = 2


def _alnum_prefix(value: str, length: int) -> str:
    compact = "".join(ch for ch in str(value) if ch.isalnum())
    return compact[:length]


def _row_keys(
    name_norm: str,
    address_norm: str,
    address_numbers: str,
    rare_name_tokens: set[str],
    rare_address_tokens: set[str],
) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    name_prefix = _alnum_prefix(name_norm, NAME_PREFIX_LEN)
    if len(name_prefix) >= 3:
        keys.append(("name_prefix", name_prefix))
    # sorted() for deterministic key order across process runs (set
    # iteration order is not stable across runs -- see _rare_tokens_per_row)
    for token in sorted(rare_name_tokens):
        keys.append(("name_token", token))
    for token in sorted(rare_address_tokens):
        keys.append(("addr_token", token))
    if address_numbers:
        for num in address_numbers.split("|"):
            if len(num) >= MIN_TOKEN_LEN:
                keys.append(("addr_num", num))
    return keys


def _rare_tokens_per_row(text_series: pd.Series, n: int) -> list[set[str]]:
    """Pick each row's N rarest tokens (by corpus frequency) as blocking keys.

    Using only the rarest tokens (rather than every token, as the old
    token_block did) keeps each row's key list short and keeps posting lists
    for common words out of the index entirely.
    """
    freq: Counter[str] = Counter()
    token_lists: list[list[str]] = []
    for text in text_series:
        tokens = [t for t in str(text or "").split() if len(t) >= MIN_TOKEN_LEN]
        token_lists.append(tokens)
        freq.update(set(tokens))

    result: list[set[str]] = []
    for tokens in token_lists:
        # Tie-break on the token string itself (not just frequency) so this
        # is deterministic across process runs -- Python randomizes string
        # hash values per process by default, so breaking ties purely via
        # set() iteration order made two runs of the same code pick
        # different "rarest" tokens on frequency ties, producing different
        # (though equally valid) candidate sets between runs.
        ranked = sorted(set(tokens), key=lambda t: (freq[t], t))
        result.append(set(ranked[:n]))
    return result


def _build_right_index(right: pd.DataFrame) -> dict[tuple[str, str], list[int]]:
    rare_name = _rare_tokens_per_row(right["name_norm"], N_RARE_NAME_TOKENS)
    rare_addr = _rare_tokens_per_row(right["address_norm"], N_RARE_ADDRESS_TOKENS)
    index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for pos, row in enumerate(right.itertuples(index=False)):
        for key in _row_keys(row.name_norm, row.address_norm, row.address_numbers, rare_name[pos], rare_addr[pos]):
            index[key].append(pos)
    return {key: postings for key, postings in index.items() if len(postings) <= MAX_BUCKET_SIZE}


def _score(s1_name: str, cand_name: str, s1_addr: str, cand_addr: str) -> float:
    name_score = fuzz.token_sort_ratio(s1_name or "", cand_name or "")
    addr_score = fuzz.token_sort_ratio(s1_addr or "", cand_addr or "")
    return 0.7 * name_score + 0.3 * addr_score


def _score_rows(
    rows: list[tuple],
    index: dict[tuple[str, str], list[int]],
    right_ids,
    right_names,
    right_addrs,
    top_k: int,
    candidate_budget: int,
    blocker_name: str,
) -> list[dict[str, object]]:
    """Score a batch of left rows against an already-built right-side index.

    A row's keys are processed smallest-bucket-first (its most specific,
    highest-precision signals: a shared postal code or rare token beats a
    generic 5-char name prefix), accumulating scored candidates until
    ``candidate_budget`` unique right-side rows have been gathered, then
    stopping. This bounds the per-row scoring cost to O(budget) regardless
    of how large a bucket a row happens to touch -- capping the bucket size
    itself (MAX_BUCKET_SIZE) does not bound this, since a row can still
    belong to several large buckets whose union is unbounded.
    """
    records: list[dict[str, object]] = []
    for entity_id, name_norm, address_norm, address_numbers, rare_name, rare_addr in rows:
        keys = _row_keys(name_norm, address_norm, address_numbers, rare_name, rare_addr)
        if not keys:
            continue

        keyed_postings = [(key, index[key]) for key in keys if key in index]
        keyed_postings.sort(key=lambda item: len(item[1]))

        selected: dict[int, float] = {}
        for _key, postings in keyed_postings:
            for right_pos in postings:
                if right_pos in selected:
                    continue
                selected[right_pos] = _score(name_norm, right_names[right_pos], address_norm, right_addrs[right_pos])
            if len(selected) >= candidate_budget:
                break

        if not selected:
            continue

        ranked = sorted(selected.items(), key=lambda item: -item[1])[:top_k]
        for rank, (right_pos, score) in enumerate(ranked, start=1):
            records.append(
                {
                    "source1_entity_id": entity_id,
                    "candidate_entity_id": str(right_ids[right_pos]),
                    f"{blocker_name}_score": float(score) / 100.0,
                    f"{blocker_name}_rank": rank,
                    "blocker": blocker_name,
                }
            )
    return records


class CountryIndex:
    """A pre-built blocking index for one country's target pool, reusable
    across many batches of left (S1) rows without rebuilding it each time.

    Building this once per country (instead of once per batch, or once per
    the whole left side as the old code did) is what makes batched,
    memory-bounded processing cheap: the index build is O(target pool size)
    and is the only step that needs the full target pool's text in memory
    at once -- everything downstream (per-batch scoring) only touches
    small slices.
    """

    __slots__ = ("index", "right_ids", "right_names", "right_addrs")

    def __init__(self, right_country: pd.DataFrame) -> None:
        right_country = right_country.reset_index(drop=True)
        self.right_ids = right_country["entity_id"].to_numpy()
        self.right_names = right_country["name_norm"].to_numpy()
        self.right_addrs = right_country["address_norm"].to_numpy()
        self.index = _build_right_index(right_country)


def build_country_index(right_country: pd.DataFrame) -> CountryIndex:
    """Build a reusable blocking index for a single country's target pool.

    ``right_country`` must already be filtered to one country (the caller
    processes countries separately to bound memory to one country's target
    pool at a time, not the whole multi-country corpus).
    """
    return CountryIndex(right_country)


def score_left_batch(
    left_batch: pd.DataFrame,
    country_index: CountryIndex,
    top_k: int = 60,
    candidate_budget: int = 150,
    blocker_name: str = "keyed",
) -> pd.DataFrame:
    """Score one batch of left (S1) rows against an already-built CountryIndex.

    This is the memory-bounded entry point: peak memory scales with
    ``len(left_batch)``, not with the country's total S1 count or target
    pool size (those were already paid for once, in ``country_index``).
    """
    left_rare_name = _rare_tokens_per_row(left_batch["name_norm"], N_RARE_NAME_TOKENS)
    left_rare_addr = _rare_tokens_per_row(left_batch["address_norm"], N_RARE_ADDRESS_TOKENS)
    rows = [
        (row.entity_id, row.name_norm, row.address_norm, row.address_numbers, left_rare_name[pos], left_rare_addr[pos])
        for pos, row in enumerate(left_batch.itertuples(index=False))
    ]
    records = _score_rows(
        rows, country_index.index, country_index.right_ids, country_index.right_names, country_index.right_addrs,
        top_k, candidate_budget, blocker_name,
    )
    return pd.DataFrame.from_records(records)


def build_keyed_candidates(
    left: pd.DataFrame,
    right: pd.DataFrame,
    config: BlockingConfig,
    top_k: int = 60,
    candidate_budget: int = 150,
    blocker_name: str = "keyed",
) -> pd.DataFrame:
    """Fast blocking: cheap deterministic bucket keys, then RapidFuzz scoring
    only within each S1 row's (small) candidate set -- not against the full
    per-country corpus. Grouped by country like the other blockers.

    Convenience wrapper for small/moderate data (train-time validation,
    where the whole left side comfortably fits in memory at once). For
    large-scale test inference, build a CountryIndex once per country via
    ``build_country_index`` and call ``score_left_batch`` per batch instead
    -- see scripts/make_final_submission.py.
    """
    all_records: list[dict[str, object]] = []

    for country, left_country in left.groupby("country_norm", sort=False):
        right_country = right[right["country_norm"].eq(country)]
        if left_country.empty or right_country.empty:
            continue

        country_index = build_country_index(right_country)
        scored = score_left_batch(left_country, country_index, top_k, candidate_budget, blocker_name)
        if not scored.empty:
            all_records.extend(scored.to_dict("records"))

    return pd.DataFrame.from_records(all_records)


def build_keyed_union_candidates(
    source1: pd.DataFrame,
    targets: pd.DataFrame,
    config: BlockingConfig,
    top_k: int = 60,
    candidate_budget: int = 150,
) -> pd.DataFrame:
    """Drop-in replacement for blocking.build_union_candidates.

    Scales to the competition's real multi-million-row target pools (the
    full TF-IDF/token union does not -- see benchmarks in the diagnosis).
    Used for both train-time candidate generation and final test inference
    so the classifier is trained on the same candidate distribution it will
    see at inference time.
    """
    candidates = build_keyed_candidates(source1, targets, config, top_k=top_k, candidate_budget=candidate_budget)
    return combine_candidate_frames([candidates], config)
