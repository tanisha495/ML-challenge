# OOM crash postmortem (2026-09-26)

## What happened

Two copies of `scripts/make_final_submission.py` were run concurrently:
one processing the main test set (France done, India in progress), and a
second reprocessing France alone with a corrected decision rule. Both were
in the "building features" stage when the machine ran out of memory; VS
Code (hosting the terminal) reached ~79GB RSS on a 16GB machine and was
force-killed, requiring a restart.

## Root cause

`build_pair_features()` in `src/ber/features.py` builds a fully denormalized
merge of the candidates table against every text column on both the
Source-1 side and the candidate side (`business_name`, `business_address`,
`name_norm`, `address_norm`, `combined_norm`, `legal_form`,
`landmark_tokens`, etc. -- roughly 19 text-bearing columns), then iterates
it row by row into a list of Python dicts. This was only ever exercised at
up to ~1M candidate-pair scale (the 30K-S1 training run). At India's real
test-set scale -- 47,432,389 candidate pairs -- that merge alone plausibly
needed 40-70+GB, since pandas `object`-dtype string columns carry real
per-string Python object overhead rather than compact storage. Running the
France job (15.4M pairs, proportionally ~15-25GB) at the same time pushed
the combined peak into OOM territory.

Two contributing factors, in order of impact:

1. **Nothing bounded memory to less than a whole country's candidate-pair
   count.** `build_pair_features` was called once per country with the
   full candidate set (tens of millions of rows for India), not in
   batches.
2. **Two large jobs ran concurrently**, each independently vulnerable to
   (1), which halved the safety margin each had.

The blocking step itself (`keyed_blocking.py`) was not the cause -- it does
not hold the full corpus in a dense matrix, and the old TF-IDF blocker
(`sparse_tfidf_block`) is unused in the current pipeline entirely.

## Fix

`scripts/make_final_submission.py` was restructured:

- Per country: the target pool (S2+S3) is loaded and normalized once, and
  a `CountryIndex` (the blocking index) is built once via
  `ber.keyed_blocking.build_country_index` -- both scale with target-pool
  size only (a few GB at most per country), not with candidate-pair count.
- S1 rows are processed in small batches (`--batch-size`, default 12,000).
  Each batch's candidates, features, scoring and decision all happen
  independently via `score_left_batch`, and results are written to disk
  immediately -- peak memory now scales with `batch_size`, not with a
  country's total candidate-pair count.
- Progress is checkpointed per batch (`_progress.json`), not just per
  country, so a crash mid-country loses at most one batch's work.
- Peak RSS (`resource.getrusage(...).ru_maxrss`) is logged after every
  batch, so runaway growth is visible immediately instead of silently
  compounding until the OS kills something.
- A lockfile (`.lock` in the output dir) refuses to start a second run
  against the same output directory. It cannot prevent two *different*
  output-dir runs from overlapping -- that remains an operator discipline
  rule: never launch a second large run while one is already in progress.

## Verification before any large rerun

Before relaunching anything at country scale, the batched script was
smoke-tested at small scale with `--limit-s1` and the logged peak RSS was
checked to sanity-check the batch-size choice against the actual (not just
estimated) memory footprint. See the run log for the measured numbers.
