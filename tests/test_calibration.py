import dataclasses
import random

import numpy as np
import pytest

from whymatched.cache import EmbeddingCache
from whymatched.calibration import (
    CalibrationProfile,
    DeletionReference,
    OrthographicNull,
    SynonymNull,
    evaluate_calibrated,
    fit_profile,
    _benjamini_hochberg,
    _bh_required_n_null,
    _bootstrap_quantile_ci,
    _interp_profile_quantile,
    _min_null,
    _p_value,
    _quantile,
    _ratio,
    _stopwords,
    _z_score,
)
from whymatched.perturbations import get_perturbations, propose_all
from whymatched.perturbations.negation import NegationPerturbation

from .fakes import CountingModel, FakeBagOfWordsModel

LONG_FIXTURE = (
    "The vendor must obtain approval before it can purchase additional equipment. "
    "The team will request approval and then purchase the equipment once approval is obtained. "
    "Approval must be verified and confirmed by the finance department before any purchase is completed. "
    "If approval is not obtained, the purchase cannot proceed, and the vendor must request approval again. "
    "The finance department will examine each purchase request and verify that the required approval was "
    "properly obtained before the purchase is finalized. "
    "Once verified, the team can complete the purchase and begin using the equipment immediately. "
    "The vendor should also retain a copy of every approval and provide it to finance when requested. "
    "Additionally, the finance department will assist with any purchase that needs modification once it has begun."
)


def test_synonym_null_occurrence_based():
    text = "use it and use it again and use it once more"
    candidates = SynonymNull().propose(text, n=50, rng=random.Random(0))
    text2 = "purchase it and purchase it again and purchase it once more"
    candidates2 = SynonymNull().propose(text2, n=50, rng=random.Random(0))
    assert len(candidates2) == 3
    assert all(c.meaning_preserving for c in candidates2)


def test_synonym_null_reaches_min_null_on_long_fixture():
    candidates = SynonymNull().propose(LONG_FIXTURE, n=50, rng=random.Random(0))
    assert len(candidates) >= 20


def test_synonym_null_sampling_deterministic_and_capped():
    first = SynonymNull().propose(LONG_FIXTURE, n=10, rng=random.Random(0))
    second = SynonymNull().propose(LONG_FIXTURE, n=10, rng=random.Random(0))
    assert len(first) == 10
    assert [c.trigger for c in first] == [c.trigger for c in second]
    assert [c.span.start for c in first] == sorted(c.span.start for c in first)


def test_synonym_null_sampling_varies_with_seed():
    a = SynonymNull().propose(LONG_FIXTURE, n=10, rng=random.Random(0))
    b = SynonymNull().propose(LONG_FIXTURE, n=10, rng=random.Random(1))
    assert [c.trigger for c in a] != [c.trigger for c in b]


def test_orthographic_null_covers_all_five_rules():
    text = 'Revenue grew  20%. It was "great" and surprising. Costs and margins improved.'
    candidates = OrthographicNull().propose(text, n=50, rng=random.Random(0))
    triggers = {c.trigger for c in candidates}
    assert any("double-space" in t for t in triggers)
    assert any("single-space" in t for t in triggers)
    assert any("\"" in t for t in triggers)
    assert any("percent" in t for t in triggers)
    assert any("&" in t for t in triggers)
    assert all(c.meaning_preserving for c in candidates)


def test_orthographic_null_trailing_period_toggle():
    with_period = OrthographicNull().propose("no trailing period issue.", n=50, rng=random.Random(0))
    assert any(c.trigger == "remove trailing period" for c in with_period)

    without_period = OrthographicNull().propose("no trailing period here", n=50, rng=random.Random(0))
    assert any(c.trigger == "add trailing period" for c in without_period)


def test_deletion_reference_skips_stopwords_and_punctuation():
    stop = _stopwords()
    candidates = DeletionReference().propose(LONG_FIXTURE, n=50, rng=random.Random(0))
    assert candidates
    for c in candidates:
        assert c.span.text.lower() not in stop
        assert c.span.text.isalpha()


def test_deletion_reference_not_meaning_preserving():
    candidates = DeletionReference().propose(LONG_FIXTURE, n=50, rng=random.Random(0))
    assert candidates
    assert all(c.meaning_preserving is False for c in candidates)


def test_deletion_reference_occurrence_based_and_capped():
    candidates = DeletionReference().propose(LONG_FIXTURE, n=5, rng=random.Random(0))
    assert len(candidates) == 5


def test_quantile_matches_numpy():
    deltas = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    assert _quantile(deltas, 0.10) == pytest.approx(np.quantile(deltas, 0.10))


def test_p_value_lower_bound_both_tails():
    deltas = np.linspace(0.05, 0.15, 50)
    p_low = _p_value(-1.0, deltas)
    assert p_low == pytest.approx(1 / 51)
    assert p_low >= 1 / 51
    p_high = _p_value(1.0, deltas)
    assert p_high == pytest.approx(51 / 51)


def test_p_value_never_zero():
    deltas = np.array([0.5] * 50)
    assert _p_value(-100.0, deltas) > 0


def test_min_null_formula():
    assert _min_null(0.05) == 20
    assert _min_null(0.01) == 99
    assert _min_null(0.5) == 20


def test_ratio_basic():
    deltas = np.array([0.1, 0.2, 0.3])
    assert _ratio(0.1, deltas) == pytest.approx(0.1 / 0.2)


def test_ratio_guards_zero_median():
    deltas = np.array([0.0, 0.0, 0.0])
    assert _ratio(0.05, deltas) == pytest.approx(0.05 / 1e-9)


def test_z_score_guards_zero_std():
    deltas = np.array([0.2, 0.2, 0.2])
    z = _z_score(0.3, deltas)
    assert np.isfinite(z)
    assert z == pytest.approx(0.1 / 1e-9)


def test_bootstrap_ci_deterministic_under_seed():
    deltas = np.random.default_rng(0).normal(0.2, 0.05, size=50)
    ci_a = _bootstrap_quantile_ci(deltas, 0.10, rng=np.random.default_rng(7))
    ci_b = _bootstrap_quantile_ci(deltas, 0.10, rng=np.random.default_rng(7))
    assert ci_a == ci_b


def test_bootstrap_ci_brackets_the_quantile_reasonably():
    deltas = np.random.default_rng(1).normal(0.2, 0.05, size=200)
    q = _quantile(deltas, 0.10)
    lo, hi = _bootstrap_quantile_ci(deltas, 0.10, rng=np.random.default_rng(1))
    assert lo <= q <= hi


def test_bh_required_n_null_matches_spec_worked_examples():
    assert _bh_required_n_null(1, 0.05) == 19
    assert _bh_required_n_null(4, 0.05) == 79
    assert _bh_required_n_null(8, 0.05) == 159
    assert _bh_required_n_null(20, 0.05) == 399


def test_bh_hand_example_from_spec():
    p = np.array([.001, .008, .039, .041, .042, .60, .99])
    q = _benjamini_hochberg(p)
    rejected = q <= 0.05
    assert rejected.tolist() == [True, True, False, False, False, False, False]


def test_bh_monotone_in_p_order():
    p = np.array([.001, .008, .039, .041, .042, .60, .99])
    q = _benjamini_hochberg(p)
    order = np.argsort(p)
    q_in_p_order = q[order]
    assert np.all(np.diff(q_in_p_order) >= -1e-12)


def test_bh_clipped_to_unit_interval():
    p = np.array([0.9, 0.95, 0.99, 1.0])
    q = _benjamini_hochberg(p)
    assert np.all(q <= 1.0)
    assert np.all(q >= 0.0)


SHORT_QUERY = "is approval required for a purchase"


def _long_vs_short_proposals(kinds=("negation", "antonym", "numeric"), max_per_side_per_kind=4, seed=0):
    perturbations = get_perturbations(kinds=kinds)
    rng = random.Random(seed)
    proposals = propose_all(SHORT_QUERY, LONG_FIXTURE, perturbations, max_per_side_per_kind=max_per_side_per_kind, rng=rng)
    return proposals


def test_evaluate_calibrated_one_embed_call():
    model = CountingModel(FakeBagOfWordsModel())
    cache = EmbeddingCache(model)
    proposals = _long_vs_short_proposals()
    assert proposals
    results, calibration = evaluate_calibrated(cache, SHORT_QUERY, LONG_FIXTURE, proposals, seed=0)
    assert results
    assert model.calls == 1


def test_evaluate_calibrated_insufficient_nulls_on_short_side():
    model = FakeBagOfWordsModel()
    cache = EmbeddingCache(model)
    proposals = _long_vs_short_proposals()
    results, calibration = evaluate_calibrated(cache, SHORT_QUERY, LONG_FIXTURE, proposals, seed=0, threshold=0.10)
    assert calibration["query"].calibration_status == "insufficient_nulls"
    assert calibration["chunk"].calibration_status == "ok"

    query_side_results = [r for r in results if r.side == "query"]
    assert query_side_results
    for r in query_side_results:
        assert r.decision_rule == "threshold"
        assert r.calibration_status == "insufficient_nulls"
        assert r.p_value is None
        assert r.null_quantile is None
        assert r.null_quantile_ci is None
        assert r.is_collapse == (r.relative_delta < 0.10)


def test_evaluate_calibrated_quantile_rule_on_long_side():
    model = FakeBagOfWordsModel()
    cache = EmbeddingCache(model)
    proposals = _long_vs_short_proposals()
    results, calibration = evaluate_calibrated(cache, SHORT_QUERY, LONG_FIXTURE, proposals, seed=0)
    chunk_side_results = [r for r in results if r.side == "chunk"]
    assert chunk_side_results
    Q = calibration["chunk"].null_quantile
    n_synonym = calibration["chunk"].n_null_effective["synonym"]
    for r in chunk_side_results:
        assert r.decision_rule == "quantile"
        assert r.calibration_status == "ok"
        assert r.null_quantile == pytest.approx(Q)
        assert r.p_value is not None and r.p_value >= 1 / (n_synonym + 1)
        assert r.is_collapse == (r.relative_delta < Q)


def test_evaluate_calibrated_reports_triple_separately_never_averaged():
    model = FakeBagOfWordsModel()
    cache = EmbeddingCache(model)
    proposals = _long_vs_short_proposals()
    results, calibration = evaluate_calibrated(cache, SHORT_QUERY, LONG_FIXTURE, proposals, seed=0)
    chunk_calibration = calibration["chunk"]
    assert set(chunk_calibration.generator_deltas.keys()) == {"synonym", "orthographic", "deletion_reference"}
    assert chunk_calibration.generator_deltas["synonym"] != chunk_calibration.generator_deltas["orthographic"]


def test_evaluate_calibrated_bh_only_touches_ok_status_candidates():
    short_negation_query = "is approval not required for this purchase"
    model = FakeBagOfWordsModel()
    cache = EmbeddingCache(model)
    perturbations = get_perturbations(kinds=("negation",))
    rng = random.Random(0)
    proposals = propose_all(short_negation_query, NEGATION_RICH_FIXTURE, perturbations, max_per_side_per_kind=8, rng=rng)
    results, _ = evaluate_calibrated(
        cache, short_negation_query, NEGATION_RICH_FIXTURE, proposals, seed=0, n_null=200, correction="bh"
    )
    ok_results = [r for r in results if r.calibration_status == "ok"]
    insufficient_results = [r for r in results if r.calibration_status == "insufficient_nulls"]
    assert ok_results and insufficient_results
    assert all(r.decision_rule == "bh" and r.q_value is not None for r in ok_results)
    assert all(r.decision_rule == "threshold" and r.q_value is None for r in insufficient_results)


NEGATION_RICH_FIXTURE = (
    "The vendor must obtain approval before it can purchase additional equipment, and it cannot "
    "purchase equipment without prior approval. The team will never request approval without "
    "confirming the budget first, and it will not purchase equipment unless approval is granted. "
    "The finance department will examine each purchase request and verify that the required "
    "approval was properly obtained before the purchase is finalized, and it will not complete "
    "the purchase unless every requirement is confirmed. The vendor should also retain a copy of "
    "every approval and provide it to finance when requested, and it should never begin using the "
    "equipment without that approval on file. Additionally, the finance department will assist "
    "with any purchase that needs modification once it has begun, but it will not assist unless "
    "the request has been verified and confirmed."
)


def test_evaluate_calibrated_bh_infeasible_raises_naming_required_n():
    model = FakeBagOfWordsModel()
    cache = EmbeddingCache(model)
    perturbations = get_perturbations(kinds=("negation",))
    rng = random.Random(0)
    proposals = propose_all(
        NEGATION_RICH_FIXTURE, NEGATION_RICH_FIXTURE, perturbations, max_per_side_per_kind=4, rng=rng
    )
    assert len(proposals) == 8
    with pytest.raises(ValueError, match="159"):
        evaluate_calibrated(cache, NEGATION_RICH_FIXTURE, NEGATION_RICH_FIXTURE, proposals, seed=0, n_null=50, correction="bh")


def test_evaluate_calibrated_generators_must_include_synonym():
    model = FakeBagOfWordsModel()
    cache = EmbeddingCache(model)
    proposals = _long_vs_short_proposals()
    with pytest.raises(ValueError, match="synonym"):
        evaluate_calibrated(cache, SHORT_QUERY, LONG_FIXTURE, proposals, generators=[OrthographicNull()])


def test_evaluate_calibrated_deletion_reference_always_runs():
    model = FakeBagOfWordsModel()
    cache = EmbeddingCache(model)
    proposals = _long_vs_short_proposals()
    _, calibration = evaluate_calibrated(cache, SHORT_QUERY, LONG_FIXTURE, proposals, generators=[SynonymNull()])
    assert "deletion_reference" in calibration["chunk"].generator_deltas


def test_evaluate_calibrated_deterministic_under_seed():
    model = FakeBagOfWordsModel()
    cache_a = EmbeddingCache(model)
    cache_b = EmbeddingCache(model)
    proposals_a = _long_vs_short_proposals(seed=0)
    proposals_b = _long_vs_short_proposals(seed=0)

    results_a, calibration_a = evaluate_calibrated(cache_a, SHORT_QUERY, LONG_FIXTURE, proposals_a, seed=7)
    results_b, calibration_b = evaluate_calibrated(cache_b, SHORT_QUERY, LONG_FIXTURE, proposals_b, seed=7)

    assert [dataclasses.asdict(r) for r in results_a] == [dataclasses.asdict(r) for r in results_b]
    assert calibration_a["chunk"].null_quantile_ci == calibration_b["chunk"].null_quantile_ci


def test_evaluate_calibrated_empty_proposals_returns_empty():
    model = CountingModel(FakeBagOfWordsModel())
    cache = EmbeddingCache(model)
    results, calibration = evaluate_calibrated(cache, "the cat sat", "the mat sat", [])
    assert results == []
    assert calibration == {}
    assert model.calls == 0


CORPUS = [
    "The vendor must obtain approval before it can purchase additional equipment.",
    "The team will request approval and then purchase the equipment once approval is obtained.",
    "Approval must be verified and confirmed by the finance department before any purchase is completed.",
    "The finance department will examine each purchase request and verify that approval was obtained.",
    "The vendor should retain a copy of every approval and provide it to finance when requested.",
    "Additionally, the finance department will assist with any purchase that needs modification.",
]


def test_fit_profile_basic_shape():
    model = FakeBagOfWordsModel()
    profile = fit_profile(model, CORPUS, n_null=50, seed=0)
    assert profile.model_name == model.name
    assert profile.generator == "synonym"
    assert profile.n_samples > 0
    assert set(profile.null_delta_quantiles.keys()) == {"p05", "p10", "p25", "p50", "p75", "p95"}


def test_profile_round_trip(tmp_path):
    model = FakeBagOfWordsModel()
    profile = fit_profile(model, CORPUS, n_null=50, seed=0)
    path = tmp_path / "profile.json"
    profile.save(path)
    loaded = CalibrationProfile.load(path)
    assert loaded == profile


def test_profile_version_mismatch_raises(tmp_path):
    model = FakeBagOfWordsModel()
    profile = fit_profile(model, CORPUS, n_null=50, seed=0)
    stale = dataclasses.replace(profile, whymatched_version="0.0.0-stale")
    path = tmp_path / "profile.json"
    stale.save(path)
    with pytest.raises(ValueError, match="0.0.0-stale"):
        CalibrationProfile.load(path)


def test_profile_model_name_mismatch_raises_at_use_time():
    model = FakeBagOfWordsModel()
    profile = fit_profile(model, CORPUS, n_null=50, seed=0)

    class OtherModel(FakeBagOfWordsModel):
        name = "other-model"

    other_cache = EmbeddingCache(OtherModel())
    proposals = _long_vs_short_proposals()
    with pytest.raises(ValueError, match="other-model"):
        evaluate_calibrated(other_cache, SHORT_QUERY, LONG_FIXTURE, proposals, profile=profile)


def test_profile_quantile_interpolation_exact_and_between_points():
    profile = CalibrationProfile(
        model_name="m", n_samples=100,
        null_delta_quantiles={"p05": 0.01, "p10": 0.02, "p25": 0.04, "p50": 0.08, "p75": 0.12, "p95": 0.20},
        generator="synonym", seed=0, whymatched_version="0.0.0", created_at="now",
    )
    assert _interp_profile_quantile(profile, 0.10) == pytest.approx(0.02)
    assert _interp_profile_quantile(profile, 0.175) == pytest.approx(0.03)


def test_profile_quantile_interpolation_flat_extrapolates_outside_range():
    profile = CalibrationProfile(
        model_name="m", n_samples=100,
        null_delta_quantiles={"p05": 0.01, "p10": 0.02, "p25": 0.04, "p50": 0.08, "p75": 0.12, "p95": 0.20},
        generator="synonym", seed=0, whymatched_version="0.0.0", created_at="now",
    )
    assert _interp_profile_quantile(profile, 0.01) == pytest.approx(0.01)
    assert _interp_profile_quantile(profile, 0.99) == pytest.approx(0.20)


def test_evaluate_calibrated_bh_and_profile_incompatible_raises_before_embedding():
    model = CountingModel(FakeBagOfWordsModel())
    cache = EmbeddingCache(model)
    profile = fit_profile(FakeBagOfWordsModel(), CORPUS, n_null=50, seed=0)
    proposals = _long_vs_short_proposals()
    with pytest.raises(ValueError, match="bh"):
        evaluate_calibrated(cache, SHORT_QUERY, LONG_FIXTURE, proposals, correction="bh", profile=profile)
    assert model.calls == 0


def test_evaluate_calibrated_profile_driven_same_quantile_both_sides():
    model = FakeBagOfWordsModel()
    cache = EmbeddingCache(model)
    profile = fit_profile(model, CORPUS * 4, n_null=50, seed=0)
    proposals = _long_vs_short_proposals()
    results, calibration = evaluate_calibrated(cache, SHORT_QUERY, LONG_FIXTURE, proposals, profile=profile)
    if calibration["query"].calibration_status == "ok" and calibration["chunk"].calibration_status == "ok":
        assert calibration["query"].null_quantile == pytest.approx(calibration["chunk"].null_quantile)
    for r in results:
        assert r.z_score is None
        assert r.p_value is None
