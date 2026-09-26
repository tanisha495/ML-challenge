# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We resolve Source-1 business entities to Source-2/Source-3 records with a
blocking-plus-classifier pipeline: cheap deterministic bucket keys
(rare name/address tokens, shared address numbers, name prefix) generate a
candidate set that scales to the competition's real multi-million-row
corpora, RapidFuzz-scored within each bucket; a LightGBM classifier trained
on ~40 pairwise features scores each candidate; and a decision layer tuned
directly against macro F0.5 (not accuracy or log-loss) picks the final
match set per entity. On a grouped-by-S1 train holdout, this reaches
**0.945 macro F0.5** (30,000-entity holdout; see Section 5). The main
technical effort was making every stage of this correct at the
competition's actual data volume (multi-million-row target pools per
country) rather than only at small-sample validation scale, which is where
an earlier iteration of this pipeline silently broke down.

---

## 2. Methodology

### 2.1 Problem Analysis

Key patterns observed in the training data that shaped the design:

- **Legal-form and abbreviation noise** dominates name variation: `Pvt`
  vs `Private`, `Ltd` vs `Limited`, `Corp`/`Corporation`, and French
  equivalents (`SARL`, `SAS`, `SASU`, `EURL`, `SNC`, `SCI`) needed to be
  recognized and *separated* from the core business name rather than
  either left in place (where they add noise) or blindly stripped (where
  they lose a genuine matching signal) -- they are extracted into a
  dedicated `legal_form` feature instead.
- **Address noise**: abbreviation variants (`St`/`Street`, `Rd`/`Road`,
  `Ave`/`Avenue`, French `Rue`/`Chemin`/`Impasse`), missing components (no
  postal code, no state), landmark-based references (`near`, `opp`,
  `behind`), and municipal numbering/component reordering.
- **France is unseen in training and must be treated as an open country
  set.** No feature or code path hard-codes `{US, India}`; `country` is
  used only as a generic same/different/missing signal. Critically, an
  **English-only stopword list silently let French function words
  (`de`, `la`, `le`, `du`, `des`, ...) survive normalization as if they
  were meaningful content tokens**, inflating spurious token-overlap
  similarity between unrelated French businesses. This was only caught by
  inspecting predicted match sizes on France specifically (average
  predicted matches per entity was far above the training distribution's
  true average) and is now fixed with a generic multilingual stopword list
  (see Section 5 for the fuller story, including a second, harder-to-fix
  generalization gap this did **not** fully explain).
- **Ground truth shape**: 94.42% of Source-1 entities have at least one
  match (average 3.46 matches, max 11); 5.58% are true singletons. Both
  tails matter under F0.5 -- a false positive on a singleton costs a full
  point, and F0.5 weights precision 2x over recall, so the decision layer
  cannot simply threshold at 0.5.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier + tuned decision rule.

**Core Innovation:** The candidate-generation stage went through two
architecturally different designs before landing on one that is actually
correct at the competition's real scale:

1. A full-corpus TF-IDF/character-n-gram blocker (adapted from a common
   entity-resolution recipe) reached ~99% blocking recall, but only at toy
   scale (tens of thousands of rows). Benchmarked against the real test
   corpus (millions of rows per country per source), it extrapolated to
   **10+ hours per blocker per country** -- not viable.
2. The design that shipped: cheap deterministic bucket keys (a row's
   rarest 2-3 name tokens by corpus frequency, its rarest 1-2 address
   tokens, a 5-character name prefix, and any address numbers of length
   >= 3) build an inverted index over the target pool once per country;
   each Source-1 row then only scores against the union of its own
   buckets' contents (processed smallest-bucket-first, so a row's most
   specific signal is never drowned out by a generic one), via RapidFuzz
   `token_sort_ratio` on name and address. This reduces blocking cost from
   O(S1 x target-corpus-size) to O(S1 x average-bucket-size) and made the
   real test set tractable within hours instead of days, at a validated
   ~95% pair recall / 0.98 oracle macro F0.5 on a moderate train sample
   (recall drops somewhat at the true multi-million-row corpus scale --
   see Section 3).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** rarest 2-3 name tokens (by country-scoped corpus
  frequency), rarest 1-2 address tokens, a 5-character alphanumeric name
  prefix, and address numbers (postal/PIN/house numbers) of length >= 3.
  Keys with a posting list larger than 5,000 are dropped outright as
  non-discriminative; per Source-1 row, keys are processed
  smallest-bucket-first and candidate gathering stops once 150 unique
  candidates have been collected, bounding worst-case per-row cost
  regardless of corpus size.
- **Within-bucket scoring:** RapidFuzz `token_sort_ratio` on normalized
  name (weight 0.7) and address (weight 0.3); top 60 candidates per
  Source-1 entity are kept.
- **Candidate pairs generated:** 102,888,155 total (Source 1, candidate)
  pairs across the full test set (1,732,544 Source 1 entities), averaging
  59.4 candidates per entity (top-60 cap).
- **How true matches were not lost:** validated with a grouped-by-S1 train
  holdout (never split by pair -- see Section 5), comparing candidate
  coverage against ground truth (`blocking_audit` in `src/ber/metrics.py`)
  before ever training on top of it. At moderate scale (tens of thousands
  of Source-1 rows against a proportionally sampled target pool) this
  reached ~95% pair recall / 0.98 oracle macro F0.5 (the score an
  otherwise-perfect classifier would get, given only what blocking found).
  At the competition's true full-corpus scale (millions of target rows
  per country) recall is lower (~74-90% depending on country target-pool
  density) -- corpus density at real scale defeats cheap key-based
  blocking more than a moderate-scale validation sample suggested, and
  this is the main remaining ceiling on overall score. `candidate_pairs.tsv`
  is exactly the candidate set the model scored (via
  `ber.blocking.candidate_mapping` on the real blocking output) --
  never a copy of the final matches.

---

## 4. Matching Model

**Features used** (`src/ber/features.py`, ~28 features):

- **Name features:** RapidFuzz sequence ratio, token Jaccard, shared token
  count, name length difference, token counts (both sides).
- **Address features:** the same ratio/Jaccard/shared-token/length-diff
  family on normalized addresses; shared address-number overlap
  (exact/conflict), landmark-token overlap.
- **Blocking-derived:** the blocking-stage score and rank for this
  candidate (`keyed_score`, `keyed_rank`, `candidate_strength`,
  `candidate_rank`).
- **Structural:** legal-form overlap/conflict, same-country (generic
  same/different/missing, never one-hot to specific countries), which
  source (S2/S3) the candidate came from.

**Model type:** LightGBM binary classifier (MIT-licensed), `scale_pos_weight`
balanced for the natural class imbalance, trained on a GroupKFold-style
split by Source-1 entity (fit/tune/holdout, never split by pair, so no
leakage between a Source-1 entity's own candidates across splits).

**Threshold selection method:** the decision layer does not use a single
flat 0.5 cutoff. The (threshold, max-matches cap) pair that maximizes true
macro F0.5 is grid-searched directly on a held-out tune split
(`ber.decision.tune_decision_rule`). The result was threshold=0.90,
uncapped, on the US+India training distribution -- but this rule
over-triggered badly when applied to France (unseen in training; see
Section 5), so a separately-chosen, more conservative rule (threshold=0.95,
cap=8, matching the training data's true observed maximum of 11) is used
specifically for France in the final submission, as a bounded,
defensible compromise given no France ground truth is available to
properly re-tune against.

---

## 5. Results & Error Analysis

**The starting point of this work was diagnosing a 0.51 leaderboard score**
that was far below a train-only proxy's oracle ceiling (>0.99) and
realistic target (0.90-0.96). The diagnosis found the actual submission
had bypassed the trained pipeline entirely: it came from a hand-rolled
heuristic script with hardcoded, never-tuned thresholds and no grouped-by-S1
validation, and its `candidate_pairs.tsv` was a byte-identical copy of
`matching_results.tsv` rather than a real candidate set. Once the actual
blocking -> features -> model -> tuned-decision pipeline was built and run
end to end for the first time, it reached:

- **F_0.5 Score (macro), grouped train holdout (30,000 Source-1 entities,
  never split by pair):** 0.945
- **Blocking pair recall (same holdout, moderate-scale target pool):**
  ~95%; oracle macro F0.5 (given only what blocking found): ~0.98
- **F_0.5 Score (macro), test set (leaderboard):** no ground truth is
  provided for the test set, so this is unknown until scored; see
  Section 3 for the caveat that real full-corpus-scale blocking recall is
  lower than the holdout figures above (denser real corpora defeat cheap
  key-based blocking more than a moderate-scale validation sample
  suggested).

**Common false positives (wrong merges):** at true test-set scale, the
model showed a specific, diagnosed failure mode on France (the country
absent from training): probabilities were **bimodal and overconfident**
rather than smoothly calibrated -- many candidates saturated near
probability 1.0 regardless of correctness, and the model correctly
identified almost no true singletons in France (~0.2-0.5% of France
entities predicted empty, versus a training-distribution base rate of
5.58%). Manually inspecting the highest-confidence France predictions
showed they were genuinely correct matches (real near-duplicate records
with typical noise -- abbreviation, accent, and word-order variation), so
the model is not hallucinating outright; but the elevated match
multiplicity relative to the training distribution could not be fully
resolved without France ground truth, so a bounded decision-rule override
(Section 4) was applied as a precision-oriented safety measure rather than
trusting the in-distribution-tuned rule unconditionally out of
distribution. An English-only stopword list was also found and fixed (it
let French function words inflate token-overlap similarity), but this only
partially explained the over-triggering -- the residual is attributed to
genuine train/test distribution shift affecting probability calibration
(no out-of-fold isotonic calibration is currently implemented; this is the
most promising remaining improvement).

**Common false negatives (missed matches):** bounded primarily by blocking
recall at true corpus scale (Section 3) -- a true match that never becomes
a blocking candidate cannot be recovered downstream regardless of
classifier quality. Secondarily, very short or generic business names with
no genuinely rare token (e.g., single-word names dominated by common
industry terms) are the hardest case for the rare-token blocking key, since
"rare" is relative to what tokens a short name actually contains.

---

## 6. Conclusion

The largest score improvement in this project came not from a better
model but from discovering that the trained model was never actually
being used for the real submission, and from making the candidate
generation stage correct at the competition's true multi-million-row
scale rather than only at small validation samples -- both the original
TF-IDF blocker and an early keyed-blocking implementation looked
successful at toy scale and were not, in different ways, once run against
real data volume. Two further engineering lessons came from running the
final pipeline itself at scale: a naive full-merge feature-building step
that duplicates every text column per candidate pair is not memory-safe
at tens of millions of rows (it caused an out-of-memory crash on the
first full run), fixed by batching all downstream processing per Source-1
chunk with per-batch checkpointing; and a model trained only on US/India
data does not calibrate safely to the unseen France country without
either isotonic probability calibration or a country-aware decision-rule
safety margin, of which only the latter was implemented under time
constraints. Both are documented above as the clearest next steps.

---

## Appendix

### A. Code Artefacts

All source lives under `code/business_entity_resolution/src/ber/`:

- `normalize.py` -- Unicode NFKD accent stripping, lowercasing,
  multilingual (English + French) stopword filtering, legal-form
  extraction, address abbreviation canonicalization, landmark-phrase
  extraction, number extraction.
- `keyed_blocking.py` -- the scalable candidate-generation stage (bucket
  keys, per-country reusable index, batch-friendly scoring entry point).
- `blocking.py` -- an earlier full-corpus TF-IDF/token blocker, retained
  for reference/comparison but not used by the shipped pipeline (does not
  scale to the real test corpus; see Section 2.2).
- `features.py` -- pairwise feature engineering.
- `modeling.py` -- LightGBM training wrapper, negative downsampling.
- `decision.py` -- F0.5-optimal threshold/cap grid search.
- `metrics.py` -- macro F0.5 and blocking-recall/oracle audit.
- `train_validate.py` -- the train-only validation entry point
  (`scripts/train_validate.py`): produces the trained model, the tuned
  decision rule, and the holdout diagnostics reported in Section 5.

Entry points to reproduce the two output files from the training/test
data:

```bash
pip install -r requirements.txt

# 1. Train the model and tune the decision rule (writes
#    artifacts/models/match_model.joblib and decision_rule.json)
PYTHONPATH=src python scripts/train_validate.py \
  --resource-dir <path-to-student_resource> \
  --artifact-dir artifacts_final \
  --max-s1 30000 --target-sample-modulo 300

# 2. Run the memory-bounded, checkpointed test inference for all three
#    countries (France uses a separately-tuned, more conservative decision
#    rule -- see Section 4)
PYTHONPATH=src python scripts/run_full_pipeline.py
# (equivalent to three sequential invocations of
#  scripts/make_final_submission.py --country {france,india,us},
#  with correctness checks gating each step -- see that script for the
#  per-country CLI if you want to run/resume one country by hand)
```

Full run instructions, including the memory-safety design and its
postmortem, are in `README.md` and `docs/memory_postmortem.md`.

### B. Additional Results

Final test-set submission (`output_final_v2/`, full 1,732,544-entity test
set, validated PASS by `utils/validate_submission.py` including the
optional `--check-ids` existence check):

| Country | S1 rows | Non-empty | Avg predicted set size |
|---|---|---|---|
| France | 259,452 | 258,689 (99.7%) | 6.733 |
| India | 809,986 | 774,957 (95.7%) | 4.256 |
| US | 663,106 | 649,368 (97.9%) | 4.723 |
| **Total** | **1,732,544** | **1,683,014 (97.1%)** | **4.806** |

India and US (in-distribution) land close to the training ground truth's
true non-empty rate (94.42%) and average matched-entity size (3.46,
though that figure excludes singletons while the table above includes
them, so they are not directly comparable -- India/US's non-empty rates
are the more informative comparison point). France's higher non-empty
rate and average size are consistent with the out-of-distribution
over-triggering documented in Section 5, bounded by the country-specific
decision-rule override rather than fully resolved.

Two engineering incidents worth recording for anyone reproducing this
work: (1) the first full-scale run OOM-crashed the machine, root-caused
and fixed by batching feature-building per S1-row chunk instead of per
whole country (`docs/memory_postmortem.md`); (2) the first successful run
had 259,452 duplicate France rows, caused by an orphaned subprocess from
an earlier `kill` that did not reach its child process, compounded by a
genuine non-determinism bug (Python's per-process string-hash
randomization affecting tie-breaking in blocking-key selection, fixed by
sorting on token string as a tie-breaker) that made two runs of the same
code produce slightly different candidate sets for ~0.3% of France
entities. France was cleanly regenerated after both fixes; the numbers
above are from that clean run.

---

**Note:** Teams can modify sections according to their approach while
maintaining clarity and technical depth.
