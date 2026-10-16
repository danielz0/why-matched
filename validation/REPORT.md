# Validation Report: Collapse Detector vs. NevIR

WS-A's gate document. Answers the question the rest of the roadmap depends
on: how often does whymatched's collapse detector agree with reality?

**Verdict: mixed. 1 of 3 models clears the ≥0.8 precision gate; the other
2 land in the guide's 0.6–0.8 "tune" band.** Per the guide's own rule (§1),
this means WS-B (benchmark corpus) and WS-C (leaderboard) should **not**
ship anything public yet — the gate isn't cleanly met. See
[Recommendation](#recommendation).

## Method

- **Dataset**: [NevIR](https://huggingface.co/datasets/orionweller/NevIR)
  (Weller et al., EACL 2024), MIT-licensed, `test` split (1,383 pairs,
  empirically confirmed — some secondary sources cite 1,386).
- **Sample**: 200 pairs, fixed-seed random sample (`seed=0`, plain
  `random.Random(seed).shuffle()` over indices — NevIR has no
  `PerturbationKind` axis to stratify on, since every pair is
  negation-only by construction; see `validation/README.md`'s Known Gaps).
  `sample_sha256 = 56aa68ab0459c54bdd7088d64e43122b19644b2427d7fc9dd94df8c1f92aa53c`
  (hash of the sorted sampled `pair_id`s), identical across all three runs
  below, confirming the same 200 pairs were used throughout.
- **Models**: `sentence-transformers/all-MiniLM-L6-v2`,
  `sentence-transformers/all-mpnet-base-v2`, `BAAI/bge-large-en-v1.5`, run
  via `whymatched.LocalModel.from_sentence_transformers`.
- **Detector config** (identical across all three): `calibrate=True`,
  `n_null=50`, `quantile=0.10`, `correction="none"`, `alpha=0.05`,
  `threshold=0.10`, `kinds=None` (all 7 `DEFAULT_KINDS`), `legacy_rules=True`,
  `seed=0`. Ground-truth noise band `noise_eps=0.01` (a design choice made
  in this session, not specified by the guide — see
  [Limitations](#limitations)).
- **whymatched version**: `0.3.0`. **Commit**: `b8db32c9fdab95b188226e768cdac53474c007bd`.
- **Measurement** (`validation/metrics.py::evaluate_detector`): each pair
  yields two directions, `(query_a, doc_a vs. doc_b)` and
  `(query_b, doc_b vs. doc_a)`. Ground truth per direction: `true_failure =
  (score(q, other_doc) > score(q, doc)) or (|score(q,doc) - score(q,other_doc)| <= noise_eps)`.
  Detector's call: `flagged = bool(detect_collapse(model, q, doc, ...))` —
  presence of *any* collapse flag counts, matching the guide's TP/FP
  definition literally (§A2).

## Results

400 directions per model (200 pairs × 2).

| Model | tp | fp | fn | tn | Precision | Recall | F1 | Gate verdict |
|---|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 304 | 88 | 6 | 2 | 0.776 | 0.981 | 0.866 | tune (0.6–0.8) |
| all-mpnet-base-v2 | 311 | 81 | 7 | 1 | 0.793 | 0.978 | 0.876 | tune (0.6–0.8) |
| BAAI/bge-large-en-v1.5 | 319 | 73 | 6 | 2 | **0.814** | 0.982 | 0.890 | **proceed (≥0.8)** |

Per-category breakdown is identical to overall for all three models — NevIR
is negation-only by construction, so `by_category` has exactly one
populated entry (`"negation"`). This makes the per-category reporting the
guide asks for honest-but-thin from this dataset alone; see
`validation/README.md`'s Known Gaps.

Raw result files (full config + confusion matrix, JSON): `validation/results/all-MiniLM-L6-v2.json`, `validation/results/all-mpnet-base-v2.json`, `validation/results/bge-large-en-v1.5.json`.

## Base rate check
On the 200-pair sample, the share of directions where the model ranks the wrong document higher is 0.775 (MiniLM), 0.795 (mpnet) and 0.8125 (BGE). A detector that flags every direction scores exactly those precisions. v0.3.0 flagged 392 of 400 directions in each of the three main runs, so its precision (0.776, 0.793, 0.814) matches the base rate: on NevIR it shows no lift. The "proceed (>=0.8)" verdict for BGE above reflects BGE's higher failure rate, not detector performance. The construct-mismatch discussion below explains why a single-pair sensitivity score may not predict a two-document ranking label.

## Diagnostic: is low precision caused by unrelated-kind crosstalk?

**Hypothesis** (raised before this run, and confirmed as a real bug in this
session's own unit tests — see `tests/test_validation_metrics.py`):
`evaluate_detector`'s default `kinds=None` means `detect_collapse` probes
all 7 `DEFAULT_KINDS` (negation, antonym, numeric, temporal, comparative,
quantifier, modal) per pair, not just negation. Since "flagged" means *any*
flag counts, a flag from an entirely unrelated kind (e.g. an incidental
antonym pair like "open"/"closed" elsewhere in the sentence) would count
against a pair whose ground truth is purely about negation sensitivity —
inflating false positives independent of the model's actual negation
handling.

**Test**: re-ran `all-MiniLM-L6-v2` on the identical 200-pair sample
(`sample_sha256` matches) with `--kinds negation`.

| Run | tp | fp | fn | tn | Precision | Recall |
|---|---|---|---|---|---|---|
| all 7 kinds (main run) | 304 | 88 | 6 | 2 | 0.776 | 0.981 |
| `kinds=["negation"]` only | 228 | 60 | 82 | 30 | 0.792 | 0.735 |

**Result: hypothesis not confirmed as the primary driver.** Precision barely
moved (0.776 → 0.792) — restricting to the ground-truth-relevant kind
removed roughly proportional numbers of true and false positives (tp −76,
fp −28), not a disproportionate share of false ones. The ~20–25%
false-positive rate is a property of the *negation* rule itself, even in
isolation, not primarily crosstalk from unrelated kinds — and per the
[Interpretation](#interpretation) section below, "the negation rule" here
turned out to mean the plain uncalibrated threshold rule, not a calibrated
one; `calibrate=True` was not actually engaging calibration for this
dataset at all.

**What the restriction did reveal**: recall collapsed from 0.981 to 0.735
(fn 6 → 82). This *does* confirm a real structural effect found earlier in
this session's unit tests: one direction per NevIR pair (the "original,
unedited" side, `doc1`/`q1`) frequently carries no negation cue in the
specific `(query, chunk)` text being tested, so negation-only detection
structurally can't flag it there — the other 6 kinds were catching a
meaningful share of those pairs incidentally, propping up the unrestricted
run's recall.

Diagnostic result file (not committed — scratch, per this session's
`outputs/` convention): `outputs/diagnostic-minilm-negation-only.json`.

## Interpretation

**Root cause confirmed, not speculative.** The initial write-up here
guessed at a `SynonymNull`-scale mismatch. A direct diagnostic (calling
`whymatched.calibration.evaluate_calibrated` on the sample and inspecting
`NullCalibration.calibration_status`) instead found something more basic:

> **`SynonymNull` never had enough candidates to calibrate at all.**
> `_min_null(alpha=0.05)` requires **≥20** distinct synonym-substitutable
> occurrences in a single `(query, chunk)` text before calibration can run
> (`calibration.py:253-257`, a hardcoded `max(20, ...)` floor). Across 190
> evaluated directions in this sample, the mean was **0.89** candidates
> per text and **0 of 190** reached "ok" status — every single one fell
> back to `calibration_status="insufficient_nulls"`, which uses the plain
> fixed `threshold=0.10` rule (`calibration.py:460-468`), not a calibrated
> one. **The entire gate run above, despite `--calibrate`, was in practice
> running the uncalibrated legacy threshold rule end to end.**

This also explains why `quantile` (the calibration's own tuning knob) was
never actually in play, and reframes the guide's "tune the quantile"
prescription as inapplicable here — there's nothing to tune if calibration
never engages.

**Correction (2026-08-29):** this section originally attributed the
insufficient-nulls fallback to "NevIR's short single-sentence pairs" —
that was wrong, and asserted without checking actual document lengths.
Real NevIR documents are **not short**: 200/200/400-direction sample,
median **113 words**, mean 117, up to 192 (queries: median 10 words).
The actual mechanism, confirmed directly: `SynonymNull`'s curated
dictionary (`data/synonym_pairs.json`, 138 surface forms / ~50 base word
pairs — deliberately narrow, business/procedural vocabulary like
`purchase/buy`, `require/need`, `obtain/get`) has almost no overlap with
NevIR's actual vocabulary (general narrative/encyclopedic Wikipedia text —
video games, physics, crystallography, figure skating, Shakespeare). Direct
count: `SynonymNull().propose()` on the 400 real documents in this sample
averaged **1.68 candidates per document** (median 1), with **94/400 (23.5%)
producing zero candidates outright** — a vocabulary-coverage gap, not a
document-length one. `fit_pair_profile` "fixed" the sample-count symptom
by pooling across many documents, but that pools the *same* narrow,
sparse vocabulary hits across a bigger corpus rather than fixing the
underlying coverage gap — which is a more mundane, more tractable problem
than the original (wrong) diagnosis suggested. See
[Update: vocabulary coverage](#update-2026-08-29-part-2-the-real-mechanism-is-vocabulary-coverage-not-document-length)
below.

**Follow-up: is `threshold` itself the fix, then?** No — tested directly.
Sweeping `threshold` from 0.03 to 0.30 (kinds=negation, same 200-pair
sample) left precision essentially flat:

| threshold | 0.03 | 0.05 | 0.08 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|---|
| precision | 0.799 | 0.790 | 0.790 | 0.792 | 0.793 | 0.794 | 0.794 |
| recall | 0.690 | 0.716 | 0.729 | 0.735 | 0.742 | 0.745 | 0.745 |
| fp | 54 | 59 | 60 | 60 | 60 | 60 | 60 |

`fp` barely moves across an order-of-magnitude threshold range. That means
the false positives aren't near-boundary cases a stricter cutoff would
clear — a consistent ~60/400 directions produce a `relative_delta` near
zero (the embedding barely shifts under the negation-cue edit) while still
being ground-truth "correct" by the pairwise doc_a-vs-doc_b ranking
comparison. This points at something more fundamental than a tunable
constant: `relative_delta` (how much a *single* document's embedding shifts
under a small edit) and this report's ground truth (whether the model
ranks *two different* documents correctly) are related but distinct
signals, and a nontrivial base rate of disagreement between them may be
inherent to combining them this way — not obviously fixable by retuning
either `threshold` or `quantile`.

**Does the existing `fit_profile()`/`CalibrationProfile` path fix the
coverage gap?** Tested directly: `fit_profile(model, corpus_texts, n_null=200)`
over all 800 texts in this sample produces `profile.n_samples=725` (≫ the
20-sample floor), so a profile-based run would reach `calibration_status=
"ok"` instead of falling back. **But its resulting quantiles are on a
wildly different scale** than the live per-pair metric: `p10 ≈ 4.37e-5`,
vs. the `relative_delta` values from the threshold sweep above, which
range ~0.03–0.3 — four orders of magnitude apart. This is exactly what
`fit_profile`'s own docstring warns about: it computes corpus-level
*self-similarity drop* (`1 - cosine(embed(text), embed(null_edit(text)))`),
"an APPROXIMATION of the live per-pair relative_delta metric... not proven
numerically equivalent." Passing this profile in as-is would almost
certainly crater recall (the threshold becomes near-impossible to clear),
not fix precision. **Not a safe drop-in fix without first reconciling the
two statistics.**

- BGE-large's higher precision (0.814 vs. ~0.78–0.79) tracks with it being
  the largest, most capable model of the three (335M params vs. 110M/22M),
  though n=1 per model doesn't support a strong claim beyond "worth
  re-checking at scale" — and given the above, BGE-large's calibration is
  almost certainly *also* falling back to the plain threshold rule, so this
  isn't a calibration-quality difference between models either.

### Fix options, ranked by confidence

1. **[Needs a design decision, not a code change] Reconcile the ground-truth
   definition with what `relative_delta` actually measures.** The ~60
   threshold-invariant false positives suggest the guide's TP/FP definition
   (pairwise doc-vs-doc ranking) and the detector's own signal (single-doc
   local sensitivity to an edit) are legitimately different constructs that
   won't converge through calibration tuning alone. This is the honest
   first question, and it's a `calibration.py`-adjacent methodology call —
   per `GOVERNANCE.md`, exactly the kind of change this project now says
   needs a two-reviewer design review before it ships.
2. **[Moderate effort, most promising *if* pursued] Build a corpus-level
   null distribution on the *live* `relative_delta` statistic**, not
   `fit_profile`'s self-similarity-drop approximation — i.e. a new
   profile-fitting function that runs real `(query, chunk)` pairs from a
   representative corpus through the same delta computation
   `evaluate_calibrated` uses live, then lets sparse-vocabulary-coverage
   pairs borrow that pooled, correctly-scaled distribution instead of
   requiring 20 candidates from their own single text. Fixes the coverage
   gap without the scale mismatch `fit_profile` currently has — but see
   the update below: implemented, and it doesn't fix precision.
3. **[Cheap, narrower] A negation-specific null family**, scale-matched to
   short cue-word edits (e.g. removing/inserting a short, meaning-neutral
   filler) instead of reusing `SynonymNull`'s content-word substitutions as
   the only reference for negation specifically. Directly tests my original
   (now-superseded) scale-mismatch guess, cheaply, without touching the
   core calibration machinery.
4. **[Not recommended] Lower the hardcoded `min_null=20` floor.** Technically
   the smallest diff, but it's a deliberate statistical-power guard
   (`_min_null`, `calibration.py:253`) against unreliable quantile estimates
   from too few samples — weakening it trades away the exact rigor this
   validation effort exists to protect.

None of these were implemented this session — `calibration.py` is
explicitly flagged in `CODEOWNERS`/`GOVERNANCE.md` as needing two-reviewer
sign-off, and a fix here changes every downstream collapse-rate number in
the project.

## Limitations

- **NevIR-only.** `load_capretrieval()` is a documented stub (schema
  mismatch — see `validation/README.md`); every pair here is `category=
  "negation"`. No evidence in this report about the other 6 perturbation
  kinds' precision/recall.
- **`noise_eps=0.01`** (the ground-truth tie threshold) is a judgment call
  made in this session, not specified by the guide. A tighter or looser
  band would shift which pairs count as "model got it right," and
  therefore shift precision/recall by an unmeasured amount. Worth a
  sensitivity sweep before treating these numbers as final.
- **200-pair sample, not the full 1,383-pair test split** — scoped for a
  single-session compute budget (~12–100+ min per model depending on size).
  A full-split run is a natural follow-up once the calibration tuning
  below is done, so it isn't run twice.
- **Single sample draw** (one seed) per model — no confidence interval on
  precision/recall themselves (as opposed to the calibration's own
  bootstrap CI on individual quantile estimates, which is a different
  thing).

## Recommendation

Per the guide's own gate table (§A3): **0.6 ≤ precision < 0.8 → tune the
null generator and/or quantile, re-run, timebox to one week** — this
applies to 2 of 3 models (MiniLM, mpnet); the third (BGE-large) already
clears ≥0.8, though its calibration is almost certainly falling back the
same way (see Interpretation), so treat "BGE-large proceeds" cautiously
until re-verified with calibration actually engaged. The guide's own
prescription ("tune the quantile") doesn't directly apply here — the
confirmed root cause is that calibration wasn't running at all for this
dataset (0/190 directions reached `calibration_status="ok"`, due to
`SynonymNull`'s narrow vocabulary rarely matching NevIR's text — see the
correction in Interpretation, not document length), not that its quantile
was poorly chosen. See
[Interpretation](#interpretation) for the four ranked fix options; all
require a `calibration.py` design decision before implementation, per
`GOVERNANCE.md`'s two-reviewer rule for that file.

**Do not start WS-B/WS-C public-facing work from this report alone** — the
gate is not cleanly met across all three models. This is a "tune," not a
"stop the roadmap" (no model came close to precision < 0.6), and not
outright "proceed" either.

## Update 2026-08-29: `fit_pair_profile()` fixes the coverage bug, not the precision problem

Following the [Fix options](#fix-options-ranked-by-confidence) above, option 2
was implemented: `whymatched.calibration.fit_pair_profile()`, a new
corpus-level profile-fitting function using the *live* `relative_delta`
formula (pooled across real `(query, chunk)` pairs) instead of
`fit_profile()`'s mismatched self-similarity-drop statistic. Wired into
`validation/runner.py` via `--fit-profile-split {train,validation}`
(`test` excluded from choices to prevent fitting and gating on the same
split). Fit on NevIR's `validation` split (225 pairs → 450 real
`(query, chunk)` pairs), then re-ran `all-MiniLM-L6-v2` on the identical
200-pair `test` sample (`sample_sha256` matches every prior run).

**The profile itself is correctly scaled** — `n_samples=821` (well past the
20-sample floor) with `p10=0.0005`, `p50=0.0032`: non-vanishing, unlike
`fit_profile`'s `p10≈4.37e-5`. This confirms the coverage-and-scale bug is
fixed as designed.

**But precision barely moved, and recall collapsed:**

| Run | tp | fp | fn | tn | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Flat fallback rule (main run) | 304 | 88 | 6 | 2 | 0.776 | 0.981 | 0.866 |
| `fit_pair_profile` (validation split) | 128 | 39 | 182 | 51 | 0.766 | **0.413** | **0.537** |

Precision is unchanged within noise (0.776 → 0.766). Recall craters
(0.981 → 0.413) and F1 drops by more than a third. `fp` did drop
substantially (88 → 39) and `tn` improved (2 → 51) — the profile-driven
quantile (`p10=0.0005`) is *far* stricter than the flat rule's
`threshold=0.10` it replaced, so it correctly clears many of the pairs the
flat rule was wrongly flagging. But that same strictness now also misses
176 additional real failures (`fn` 6 → 182): most real negation-cue-removal
edits move `all-MiniLM-L6-v2`'s embedding by far more than 0.0005 (recall
the threshold sweep above: real deltas cluster ~0.03–0.3), so a properly-
calibrated null threshold judges the *embedding itself* as clearly reacting
to the edit — "not a collapse" by the `relative_delta` definition — even on
pairs where the model still ranks the wrong document higher by this
report's separate pairwise ground truth.

**This is exactly [fix option 1](#fix-options-ranked-by-confidence)
confirmed empirically, not just theorized.** Making the null threshold
statistically correct didn't help, because the deeper issue survives a
correct calibration: `relative_delta` (does *this document's* embedding
noticeably move under a small edit) and the ground truth used here (does
the model rank *two different* documents correctly) are different
constructs, and a good chunk of NevIR's failures are cases where MiniLM's
embedding visibly reacts to the negation edit yet still ranks the wrong
document higher. No amount of null-distribution tuning fixes that — it's
a definitional question, not a calibration one.

**Recommendation, updated**: don't pursue further calibration tuning
(quantile, `n_null`, generator choice) on this metric definition — it's
very unlikely to move precision meaningfully, per the threshold-sweep
and profile results above, both of which changed the decision boundary
substantially with almost no precision movement. The open question is
fix option 1: whether "collapse" should be measured/defined differently
for this validation methodology (e.g. against the pairwise score gap
directly, rather than through a single-document `relative_delta`
proxy) — a design question, not a tuning one.

`mpnet`/`bge-large` were not re-run with `--fit-profile-split` — the
`all-MiniLM-L6-v2` result already answers the question this fix was meant
to test (does scale-correct calibration improve precision?) with a clear
"no," so spending ~30–90 more minutes per model to confirm the same
mechanism on two more models wasn't judged worth it before checking in.
Full result file: `validation/results/all-MiniLM-L6-v2-profile.json`.

## Update 2026-08-29, part 2: the real mechanism is vocabulary coverage, not document length

The "Interpretation" section's original root-cause claim — that NevIR's
pairs are too short to produce 20 `SynonymNull` candidates — was checked
directly and was **wrong**. Real document lengths in this sample: median
**113 words**, mean 117, up to 192 (queries: median 10 words). Not short.

The actual mechanism: `data/synonym_pairs.json` is a deliberately narrow,
curated list (138 surface forms / ~50 base word pairs, business/procedural
vocabulary — `purchase/buy`, `require/need`, `obtain/get`, `assist/help`,
etc.), built for this project's own procedural-text test fixtures. It has
almost no overlap with NevIR's actual content (Wikipedia narrative and
encyclopedic text — video games, physics, crystallography, figure skating,
Shakespeare characters). Direct measurement: `SynonymNull().propose()`
against the 400 real documents in this sample averaged **1.68 candidates
per document** (median 1), with **94/400 (23.5%) producing zero candidates
outright** — a vocabulary-coverage gap. `fit_pair_profile` "fixed" the
sample-count symptom (it can always reach 20+ by pooling across enough
documents) without fixing this underlying gap — it's still drawing from
the same narrow ~50-word vocabulary, just scattered across more text.

This reframes what a real fix looks like, and it's more tractable than the
original (wrong) diagnosis implied:

1. **[Most promising, directly targets the confirmed cause] Broaden
   `SynonymNull`'s vocabulary coverage** — either expand
   `data/synonym_pairs.json` well beyond its current ~50 business-process
   pairs toward general-purpose vocabulary, or add a broader-coverage
   generator (WordNet-based synonym substitution is a natural fit — the
   codebase already has exactly this pattern for the *antonym* perturbation,
   `AntonymPerturbation(use_wordnet=True)` → `_wordnet_antonym()` in
   `perturbations/antonym.py`, lazy `nltk` import, not a new dependency
   pattern). This would let **per-document** calibration engage on real
   text like NevIR's, using nulls drawn from that specific document's own
   vocabulary — the scientifically-intended design of `evaluate_calibrated`
   — rather than either falling back to a flat threshold or borrowing a
   cross-document pooled distribution (`fit_pair_profile`) that isn't
   necessarily representative of any single document's own embedding
   sensitivity. Untested this session — the open question is whether
   *properly per-document-calibrated* thresholds behave differently from
   both things already tried (flat threshold, pooled profile), since
   neither of those actually exercised genuine per-document calibration on
   real text. Needs real curation care: this project's own CHANGELOG
   already flags synonym-list quality as "the most damaging failure mode
   for this tool's credibility" — a rushed broad wordlist (especially a
   raw WordNet dump) risks including near-synonyms that shift meaning,
   which would inflate the null and hide real collapses. Any expansion
   needs the same second-pair-of-eyes review the existing list is already
   flagged as needing.
2. **Redefine the ground truth to match what `relative_delta` actually
   measures** (carried over from the original fix option 1, now on firmer
   footing after the `fit_pair_profile` result): pairwise doc-vs-doc
   ranking and single-document edit-sensitivity are still plausibly
   different constructs regardless of the vocabulary fix, and the
   `fit_pair_profile` result (precision flat, recall collapsed under a
   *statistically correct* threshold) is evidence for this independent of
   the vocabulary-coverage bug. Worth revisiting once (1) is tried, to see
   how much of the remaining gap it explains.
3. **Do both (1) then (2))** — fix the vocabulary-coverage bug first (cheap
   to test, directly targets a confirmed mechanism), re-measure, and only
   then decide whether the definitional question in (2) still needs
   addressing on top.

None of these were implemented this session — flagging back to the user
per the same `calibration.py`/`GOVERNANCE.md` reasoning as before.

## Reproduce

```bash
pip install -e ".[local,viz,validation,dev]"

python -m validation.runner --dataset nevir --split test --sample-size 200 --seed 0 \
    --model sentence-transformers/all-MiniLM-L6-v2 --calibrate --markdown \
    --out validation/results/all-MiniLM-L6-v2.json

python -m validation.runner --dataset nevir --split test --sample-size 200 --seed 0 \
    --model sentence-transformers/all-mpnet-base-v2 --calibrate --markdown \
    --out validation/results/all-mpnet-base-v2.json

python -m validation.runner --dataset nevir --split test --sample-size 200 --seed 0 \
    --model BAAI/bge-large-en-v1.5 --calibrate --markdown \
    --out validation/results/bge-large-en-v1.5.json

# Diagnostic (kind-crosstalk check):
python -m validation.runner --dataset nevir --split test --sample-size 200 --seed 0 \
    --model sentence-transformers/all-MiniLM-L6-v2 --calibrate --kinds negation --markdown
```

**Diagnostic (calibration coverage — confirms `calibration_status`)**:

```python
import random
from validation.datasets import load_nevir
from whymatched.cache import EmbeddingCache
from whymatched.collapse import _build_engine_perturbations
from whymatched.perturbations import propose_all
from whymatched.calibration import evaluate_calibrated
from whymatched.models.local import LocalModel

pairs = load_nevir(split="test")
rng = random.Random(0)
idx = list(range(len(pairs)))
rng.shuffle(idx)
sample = [pairs[i] for i in sorted(idx[:200])]

model = LocalModel.from_sentence_transformers("sentence-transformers/all-MiniLM-L6-v2")
cache = EmbeddingCache(model)
cache.prime([t for p in sample for t in (p.query_a, p.query_b, p.doc_a, p.doc_b)])
perturbations = _build_engine_perturbations(["negation"], False, True, 6)

ok = insufficient = 0
for p in sample:
    for query, chunk in [(p.query_a, p.doc_a), (p.query_b, p.doc_b)]:
        proposals = propose_all(query, chunk, perturbations, max_per_side_per_kind=6, rng=random.Random(0))
        if not proposals:
            continue
        _, calib_by_side = evaluate_calibrated(cache, query, chunk, proposals, n_null=50, quantile=0.10, seed=0)
        for nc in calib_by_side.values():
            ok += nc.calibration_status == "ok"
            insufficient += nc.calibration_status == "insufficient_nulls"
print("ok:", ok, "insufficient_nulls:", insufficient)
```

**Diagnostic (`fit_profile` scale check)**:

```python
from whymatched.calibration import fit_profile

profile = fit_profile(model, corpus_texts, n_null=200, seed=0)  # corpus_texts = every text in `sample`
print(profile.n_samples, profile.null_delta_quantiles)
```
