# Business Entity Resolution

Full pipeline for the Amazon ML Challenge 2026 business entity resolution problem: train-time
validation plus final test-set submission generation.

## Approach

1. Normalize business names and addresses (Unicode NFKD accent stripping, lowercasing,
   English + French stopword filtering, legal-form extraction, address abbreviation
   canonicalization, landmark-phrase extraction, number extraction).
2. Generate candidates with a scalable keyed blocker (`src/ber/keyed_blocking.py`): cheap
   deterministic bucket keys (rarest name/address tokens, name prefix, address numbers) build a
   per-country inverted index once, then RapidFuzz-score each Source 1 row only against its own
   buckets' contents. This is what makes the real ~5-10M-row-per-source test corpus tractable; an
   earlier full-corpus TF-IDF blocker (still present in `src/ber/blocking.py` for reference)
   extrapolated to 10+ hours per blocker per country at that scale and is not used by the shipped
   pipeline.
3. Audit blocking recall and oracle macro F0.5 against ground truth (train-only).
4. Build ~28 pairwise features per (Source 1, candidate) pair: RapidFuzz ratios, token Jaccard,
   shared numbers/legal-form/landmark cues, blocking-stage score/rank, candidate source.
5. Train a LightGBM pairwise matcher (GroupKFold-style split by Source 1 entity).
6. Tune the decision rule (threshold + max-match cap) directly against macro F0.5 on a train-only
   tune split.
7. Run memory-bounded, checkpointed test inference (`scripts/make_final_submission.py` /
   `scripts/run_full_pipeline.py`) to produce `output/matching_results.tsv` and
   `output/candidate_pairs.tsv`.

See `Documentation_template.md` (or `student_resource/Documentation_template.md`) for the full
methodology write-up, including the diagnosis of why an earlier submission scored 0.51 and the
memory-safety postmortem in `docs/memory_postmortem.md`.

## Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

LightGBM on macOS additionally needs the OpenMP runtime: `brew install libomp`.

## Step 1: Train-only validation

```bash
PYTHONPATH=src python scripts/train_validate.py \
  --resource-dir <path-to-student_resource> \
  --artifact-dir artifacts_final \
  --max-s1 30000 \
  --target-sample-modulo 300
```

Outputs:

```text
artifacts_final/output/validation_metrics.json
artifacts_final/output/validation_predictions.tsv
artifacts_final/models/match_model.joblib     # trained model, used by step 2
artifacts_final/models/decision_rule.json     # tuned {threshold, cap}, used by step 2
```

`validation_predictions.tsv` is not a leaderboard submission -- it is a held-out training
validation artifact. Check `blocking.pair_recall`, `blocking.oracle_macro_f05`, and held-out
`macro_f05` when comparing runs.

For sampled train validation, the script does **not** load all 10M+ Source 2/3 records into
memory. It streams those files and keeps every true positive target for the selected Source 1
rows plus a deterministic sample of negative target rows (`--target-sample-modulo`; larger values
use less memory).

## Step 2: Final test submission

```bash
PYTHONPATH=src python scripts/run_full_pipeline.py
```

This runs France, India, and US sequentially through `scripts/make_final_submission.py`
(France uses a separately-tuned, more conservative decision rule -- see the methodology doc for
why), running correctness checks after each country before proceeding, then the official
`utils/validate_submission.py`. Progress is logged to `output_final_v2/RUN_LOG.md`.

To run a single country by hand (e.g. to resume one that was interrupted):

```bash
PYTHONPATH=src python scripts/make_final_submission.py \
  --resource-dir <path-to-student_resource> \
  --model-dir artifacts_final/models \
  --output-dir output_final_v2 \
  --country india
```

**Memory safety:** an earlier version of this script OOM-crashed the machine (see
`docs/memory_postmortem.md`) because feature-building denormalized every text column across every
candidate pair for a whole country at once -- tens of millions of rows at India/US scale. The
current version processes S1 rows in small batches (`--batch-size`, default 12,000) against a
target-pool index built once per country; peak memory is bounded by target-pool size (~6GB for the
largest country, India) and does not grow with batch count or candidate-pair volume. It also:

- checkpoints progress per batch (`_progress.json` in the output dir), so a crash loses at most
  one batch's work, not a whole country;
- refuses to start a second run against the same output directory (`.lock` file) while one is in
  progress;
- aborts cleanly if peak RSS exceeds an 11GB safety ceiling, rather than risking another OOM.

**Never run two large jobs (training or inference) concurrently** -- that compounded the original
crash. Run one country, one process, at a time. (This was learned the hard way: killing an
orchestrator process with a plain `kill` does not kill its already-spawned `subprocess.run` child.
The orphaned child kept running invisibly and wrote a second, full copy of France's results into
the same output files, producing 259,452 duplicate rows that the first version of the correctness
checker -- which built a dict keyed by S1 ID -- couldn't detect, since a dict silently collapses
duplicate keys instead of flagging them. Both are fixed: `run_full_pipeline.py`'s checker now
counts raw row occurrences, not just dict membership.)

**Determinism:** `keyed_blocking.py`'s tie-breaking used raw `set()` iteration order in two places
(rarest-token selection on frequency ties, and blocking-key ordering), which Python randomizes
per-process by default. Two runs of the identical code could therefore select slightly different
"rarest tokens" on ties and produce different (though equally valid) candidate sets for a small
fraction of rows -- this is exactly what surfaced when the France duplicate-row bug above was
being diagnosed. Fixed by sorting on the token string as an explicit tie-breaker; verified
byte-identical output across two runs with different `PYTHONHASHSEED` values.

## Final Results

The full test set (1,732,544 Source 1 entities) was run end to end through the memory-safe,
checkpointed pipeline. `utils/validate_submission.py --check-ids` passes cleanly (every matched/
candidate ID verified to exist among the 9,969,589 real Source 2/3 test IDs; zero duplicate rows;
every required S1 entity present exactly once).

| Country | S1 rows | Non-empty | Avg predicted set size |
|---|---|---|---|
| France | 259,452 | 258,689 (99.7%) | 6.733 |
| India | 809,986 | 774,957 (95.7%) | 4.256 |
| US | 663,106 | 649,368 (97.9%) | 4.723 |
| **Total** | **1,732,544** | **1,683,014 (97.1%)** | **4.806** |

Train-only holdout validation (30,000 Source 1 entities, US+India, grouped by entity -- never
split by pair): **0.945 macro F0.5**. There is no ground truth for the actual test set, so the
real leaderboard score is unknown until scored; expect the India/US portion to track closer to
that holdout number and the France portion to be more uncertain (see `Documentation_template.md`
Section 5 for the full story on why, and what was and wasn't fixed).

## Current Scope

Implemented: train-only data loading guard, normalization (with French-aware stopwords),
scalable keyed blocking, blocking recall/oracle score audit, feature generation, LightGBM model
wrapper, macro F0.5 decision tuning, held-out train validation artifacts, memory-bounded
checkpointed final test inference, submission validation.

Known limitations (see `Documentation_template.md` Section 5 for detail): blocking recall at true
full-corpus scale is lower than at moderate validation-sample scale; no out-of-fold probability
calibration (isotonic/Platt) is implemented, which is the most likely explanation for the model's
overconfident, bimodal probabilities on France (unseen in training) -- mitigated with a bounded,
country-specific decision-rule override rather than fixed at the root.
