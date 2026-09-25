# Business Entity Resolution

Train-only implementation for the Amazon ML Challenge 2026 business entity resolution problem.

This code intentionally reads only:

```text
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
```

The I/O layer refuses paths containing `test`, so this stage cannot accidentally use test data.

## Approach

The implemented baseline follows the recommended competition strategy:

1. Normalize business names and addresses.
2. Generate high-recall candidates with a union of blockers:
   - name character TF-IDF
   - address character TF-IDF
   - combined name/address/country character TF-IDF
   - rare-token overlap blocker
3. Audit blocking recall and oracle macro F0.5.
4. Build pairwise features:
   - TF-IDF blocker scores and ranks
   - token Jaccard features
   - string sequence ratios
   - shared number, legal-form, and landmark cues
   - candidate source indicators
5. Train a LightGBM pairwise matcher.
6. Tune the final decision rule for macro F0.5 on a train-only tune split.
7. Report held-out train-only macro F0.5.

## Setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Recommended Train-Only Validation Run

```bash
PYTHONPATH=src python scripts/train_validate.py \
  --resource-dir /Users/tanishakhanna/Downloads/student_resource \
  --artifact-dir artifacts_5000 \
  --max-s1 5000 \
  --target-sample-modulo 1000
```

Outputs:

```text
artifacts_5000/output/validation_metrics.json
artifacts_5000/output/validation_predictions.tsv
artifacts_5000/output/tune_scored_pairs.parquet
artifacts_5000/output/holdout_scored_pairs.parquet
artifacts_5000/cache/*_candidates.parquet
```

`validation_predictions.tsv` is not a leaderboard submission. It is a held-out training validation artifact.

## Scaling Notes

If memory gets tight, reduce `--max-s1` or increase `--target-sample-modulo`.

For example:

```bash
PYTHONPATH=src python scripts/train_validate.py \
  --resource-dir /Users/tanishakhanna/Downloads/student_resource \
  --artifact-dir artifacts_1000 \
  --max-s1 1000 \
  --target-sample-modulo 2000
```

When comparing runs, check:

- `blocking.pair_recall`
- `blocking.full_entity_recall`
- `blocking.oracle_macro_f05`
- held-out `macro_f05`
- average candidates per Source 1 entity

Avoid omitting `--max-s1` on a laptop. The full training files are very large.

For sampled train validation, the script does **not** load all 10M+ Source 2/3 records into memory. It streams those files and keeps:

- every true positive target for the selected Source 1 rows
- a deterministic sample of negative target rows

The negative sample size is controlled by:

```bash
--target-sample-modulo 500
```

Larger values use less memory, for example `--target-sample-modulo 1000`. Smaller values add more negative distractors, for example `--target-sample-modulo 250`.

## Current Scope

Implemented:

- train-only data loading guard
- normalization
- sparse union blocking
- blocking recall/oracle score audit
- feature generation
- LightGBM/scikit-learn model wrapper
- macro F0.5 decision tuning
- held-out train validation artifacts

Not yet implemented:

- dense multilingual embedding blocker
- cross-encoder reranker
- final test inference/submission generation
- conflict-resolution post-processing

Those should be added after the sparse pipeline produces a reliable blocking ceiling and a strong train-only validation score.
