# Changelog

## Unreleased

### Added

- Multi-category counterfactual perturbation engine
  (`whymatched.perturbations`): negation, antonym, numeric, temporal,
  comparative, quantifier, and modal perturbation kinds, plus an optional
  `entity` kind behind the new `entity` extra (`pip install whymatched[entity]`
  + `python -m spacy download en_core_web_sm`).
- `whymatched.cache.EmbeddingCache`: content-addressed, batched embedding
  cache shared across a `Debugger.analyze()` call so collapse detection no
  longer re-embeds the same query per chunk.
- `detect_collapse()` and `Debugger` gained `kinds`, `legacy_rules`,
  `seed`, `cache`, and `calibrate`/`n_null`/`quantile`/`correction`/`alpha`/
  `profile` parameters.
- Span arbitration between perturbation kinds that could both claim the
  same substring (e.g. a bare year like "2020" is claimable by both the
  numeric and temporal rules; temporal wins).
- **Calibrated collapse detection** (`whymatched.calibration`):
  `detect_collapse(..., calibrate=True)` replaces the fixed `threshold`
  rule with a calibrated quantile rule -- a candidate's `relative_delta` is
  compared against the distribution of deltas from meaning-preserving edits
  ("nulls") of the same text, rather than one global number. Two null
  families are computed and reported separately, never averaged:
  `SynonymNull` (primary, curated `data/synonym_pairs.json`) drives the
  decision; `OrthographicNull` (whitespace/quote/percent-spellout/`and`-`&`
  edits) is the noise floor. `DeletionReference` (a genuine content-word
  deletion) is reported alongside as an upper reference, not a null. Also
  reports a bootstrap CI on the quantile, a ratio and z-score against the
  null distribution, and a descriptive (never decision-driving by default)
  p-value. Texts too short to produce `min_null = max(20, ceil(1/alpha)-1)`
  null samples fall back to the plain `threshold` rule
  (`calibration_status="insufficient_nulls"`) rather than synthesizing
  statistics from a handful of samples. Optional Benjamini-Hochberg
  correction (`correction="bh"`) applies within-kind; raises `ValueError`
  naming the required `n_null` if the configuration can't reject at the
  requested `alpha`. Power users who want the full statistical detail
  (not just which candidates got flagged) should call
  `whymatched.calibration.evaluate_calibrated()` directly --
  `detect_collapse()` itself still always returns `List[CollapseFlag]`.
- `whymatched.calibration.CalibrationProfile`/`fit_profile()`: pre-compute
  null-delta quantiles once against a corpus and reuse them across calls
  (`detect_collapse(..., profile=profile)`) instead of resampling nulls
  per call. Exposed via `whymatched calibrate`.
- **Corpus/batch-mode scanning + CI assertions** (`whymatched.batch`,
  `whymatched.testing`): `scan(model, cases)` runs collapse detection
  across many `EvalCase(query, chunks)` pairs at once (one shared
  `EmbeddingCache` for the whole scan; a case whose evaluation raises never
  aborts the scan, it's caught into `CaseReport(error=...)` and counted in
  `n_errored`), producing a `BatchReport` with `collapse_rate()`/
  `candidate_collapse_rate()`/`rate_by_kind()`/`worst()`. `load_cases()`
  reads JSONL (preferred) or a JSON array, raising with the offending line
  number/index on malformed input. `whymatched.testing.assert_collapse_rate`/
  `assert_no_collapse`/`assert_not_worse_than` turn a `BatchReport` into a
  CI gate; the assertion message always names the threshold, actual rate,
  denominator (with skipped/errored called out), decision rule, and the
  top-3 offenders. New `whymatched scan --input cases.jsonl
  [--max-collapse-rate ...] [--baseline ... --fail-on-regression]` CLI
  command with a CI-friendly exit-code contract: `0` thresholds met, `1`
  threshold exceeded, `2` usage/IO error, `3` completed but some cases
  errored; `--json -` writes clean JSON to stdout (summary table always
  goes to stderr) so `whymatched scan ... --json - | jq` works.
  `CaseReport` gains one field beyond the original design doc's dataclass
  (`calibration: dict[str, NullCalibration] | None`, the top-ranked
  chunk's orthographic/synonym/deletion triple) since `PerturbationResult`
  alone can't carry per-generator delta arrays -- needed by
  `render_batch_html()`'s worst-case detail view.

### Behavior changes (opt-in via `legacy_rules=False`; becomes the default
in the next minor version)

- **B1 fix**: contracted negations (`don't`, `won't`, `isn't`, `can't`,
  `didn't`, `wouldn't`, ...) are now detected. Previously `\bn't\b` could
  never match inside a contraction (no word boundary between `o` and `n`
  in `don't`), so all contracted negations were silently missed.
- **B2 fix**: `never`/`always` no longer double-fire as both a negation cue
  and an antonym pair. Both now live in the quantifier rule set
  (`always -> sometimes`, `never -> always`); negation still separately
  proposes deleting `never` as a cue, with span arbitration (quantifier
  priority 75 < negation priority 85) resolving the overlap in
  quantifier's favor.
- **B3 fix**: eight comparative/temporal pairs (`increase/decrease`,
  `increase/reduce`, `increased/decreased`, `more/less`, `higher/lower`,
  `above/below`, `increase/diminish`, `before/after`) that were already
  double-firing from `antonym_pairs.json` moved to the dedicated
  `comparative.py`/`temporal.py` rule sets.
- `must/may` and `mandatory/optional` moved from the antonym file to the
  new `modal` kind (they're deontic/modal contrasts, not plain antonyms).
- `can/cannot` stays in `antonym_pairs.json`: `cannot`/`can not` are also
  negation cues, but this is the same sanctioned, arbitration-resolved
  overlap as `required`/`optional` (negation priority 85 < antonym
  priority 90 resolves it in negation's favor) rather than a double-count,
  now that span arbitration exists.

`detect_collapse(..., legacy_rules=True)` (the default) reproduces
`0.2.2`'s exact output, including these bugs, via a private
`_LEGACY_ANTONYM_SUPPLEMENT` list in `collapse.py` that merges the migrated
pairs back in for that code path only. The shipped
`data/antonym_pairs.json` no longer contains them.

### Fixed

- `temporal.py`'s `DD/MM/YYYY` date-shift rule no longer raises when a
  matched date isn't actually valid as `DD/MM/YYYY` (e.g. a US-style
  `12/25/2023`); it's skipped instead of crashing the whole `analyze()`/
  `detect_collapse()` call.
- `max_checks_per_side` now actually limits candidate counts for every
  perturbation kind, not just negation/antonym -- numeric, temporal,
  comparative, quantifier, modal, and entity previously always used their
  default cap of 8 regardless of the caller's value.
- `increase/reduce` and `increase/diminish` -- part of the original B3
  migration but missed when `comparative.py` was first written -- are now
  present as comparative pairs.
- `render_batch_html()` now shows a distinct error banner for cases where
  `scan()` caught an exception (`CaseReport.error`), instead of rendering
  them identically to a case that was cleanly evaluated and simply had no
  collapses.
- `Debugger.analyze()` now derives a distinct seed per chunk
  (`seed + rank`) instead of reusing one seed across every chunk in a
  query, matching the fix `batch.py`'s `scan()` already had for cases --
  avoids correlated null-sampling across chunks under calibrated
  detection.
- `whymatched scan --max-collapse-rate-kind` no longer silently accepts a
  malformed `=RATE` pair (empty kind name) as a CI gate that can never
  fail; it's now a usage error (exit code 2).
- `scan(..., max_workers>1)` now gives each worker thread one persistent
  `EmbeddingCache` it reuses across every case that thread processes,
  instead of a fresh cache per case, which had silently defeated
  cross-case cache reuse the moment concurrency was enabled.
- `evaluate_calibrated()`'s default null generators are now built once at
  module load instead of re-constructing (and re-compiling the synonym
  regex) on every call.

### Known follow-up

- `data/synonym_pairs.json` (the primary null generator's vocabulary for
  calibrated collapse detection) is a starter list and has **not** yet had
  a second-pair-of-eyes curation review. A synonym pair that isn't actually
  substitutable in retrieval prose inflates the null distribution and
  *hides* true collapses -- review this list before relying on it for
  anything load-bearing.
