"""Tests for validation/metrics.py against tests/fakes.py's deterministic
models -- no real datasets or real models needed. Placed under root
tests/ (already a package, `from .fakes import ...` matches
test_calibration.py/test_cache.py's own pattern) so it's discovered by
CI's plain `pytest -q` unconditionally; it needs no `datasets`/network
dependency."""

from validation.datasets import LabelledPair
from validation.metrics import ConfusionCounts, evaluate_detector

from .fakes import CountingModel, NegationAwareModel, NegationBlindModel

# Cue words only ("not", "never", "no") -- NegationBlindModel's documented
# scope is _NEG_RE cue matches, not contraction expansion.
NEGATION_PAIRS = [
    LabelledPair(
        pair_id="synthetic-1",
        query_a="shop is staffed",
        query_b="shop is not staffed",
        doc_a="shop is staffed",
        doc_b="shop is not staffed",
        source="custom",
        category="negation",
    ),
    LabelledPair(
        pair_id="synthetic-2",
        query_a="office closes early",
        query_b="office never closes early",
        doc_a="office closes early",
        doc_b="office never closes early",
        source="custom",
        category="negation",
    ),
    LabelledPair(
        pair_id="synthetic-3",
        query_a="seats remain available",
        query_b="no seats remain available",
        doc_a="seats remain available",
        doc_b="no seats remain available",
        source="custom",
        category="negation",
    ),
]


def test_confusion_counts_precision_recall_f1():
    c = ConfusionCounts(tp=2, fp=1, fn=1, tn=0)
    assert c.n == 4
    assert c.precision == 2 / 3
    assert c.recall == 2 / 3
    assert c.f1 == 2 / 3


def test_confusion_counts_zero_division_guards():
    c = ConfusionCounts()
    assert c.precision == 0.0
    assert c.recall == 0.0
    assert c.f1 == 0.0


def test_negation_blind_model_true_positive_on_genuine_failure():
    # NegationBlindModel strips every negation cue before hashing, so
    # doc_a/doc_b (which differ only by a negation cue) embed identically
    # -- it genuinely can't rank query_a's own doc above the negated
    # counterpart in *either* direction. But detect_collapse can only test
    # a pair by removing a cue that's actually present in that (query,
    # chunk) call: the (query_a, doc_a) direction mirrors NevIR's own
    # "original, unedited" side and carries no negation cue to remove, so
    # it structurally can't be flagged even though the ground truth is a
    # real failure there too -- an honest recall ceiling, not a test bug.
    # Only the (query_b, doc_b) "edited" direction, which does carry the
    # cue, is catchable -- giving exactly 3 TP (one per pair) and 3 FN.
    model = NegationBlindModel()
    metrics = evaluate_detector(model, NEGATION_PAIRS, kinds=["negation"])
    assert metrics.overall.tp == 3
    assert metrics.overall.fn == 3
    assert metrics.overall.fp == 0
    assert metrics.by_category["negation"].tp == metrics.overall.tp


def test_negation_aware_model_true_negative():
    # Mirror image: NegationAwareModel makes negated vs. non-negated
    # variants maximally distinguishable, so it should both rank correctly
    # (no genuine failure) and not get flagged (no collapse) in either
    # direction. kinds=["negation"] keeps this fixture free of incidental
    # crosstalk from unrelated perturbation kinds matching fixture
    # vocabulary by coincidence.
    model = NegationAwareModel()
    metrics = evaluate_detector(model, NEGATION_PAIRS, kinds=["negation"])
    assert metrics.overall.tn == 6
    assert metrics.overall.fp == 0
    assert metrics.overall.tp == 0
    assert metrics.overall.fn == 0


def test_evaluate_detector_calibrated_with_both_fake_models():
    # Definition-of-Done requirement (CONTRIBUTING.md): anything touching
    # calibration.py needs a test using both NegationBlindModel and
    # NegationAwareModel -- a test proving you can detect collapse says
    # nothing about your false-positive rate.
    blind_metrics = evaluate_detector(
        NegationBlindModel(), NEGATION_PAIRS, calibrate=True, n_null=10, seed=0, kinds=["negation"]
    )
    aware_metrics = evaluate_detector(
        NegationAwareModel(), NEGATION_PAIRS, calibrate=True, n_null=10, seed=0, kinds=["negation"]
    )
    assert blind_metrics.overall.recall > aware_metrics.overall.recall
    assert aware_metrics.overall.fp == 0


def test_evaluate_detector_goes_through_cache_not_per_pair_embed():
    inner = NegationBlindModel()
    counting = CountingModel(inner)
    evaluate_detector(counting, NEGATION_PAIRS)
    # 3 pairs -> 12 distinct texts primed in one batched call, plus however
    # many batched counterfactual-embedding calls detect_collapse makes
    # internally -- what matters is it stays small and batched, not one
    # embed() call per individual text (would be 12+ for priming alone).
    assert counting.calls < 12


def test_evaluate_detector_by_category_matches_pair_categories():
    metrics = evaluate_detector(NegationBlindModel(), NEGATION_PAIRS)
    assert set(metrics.by_category.keys()) == {"negation"}
    bucket = metrics.by_category["negation"]
    assert bucket.n == metrics.overall.n
    assert metrics.n_pairs == len(NEGATION_PAIRS)
    assert metrics.n_directions == 2 * len(NEGATION_PAIRS)
    assert metrics.source == "custom"
