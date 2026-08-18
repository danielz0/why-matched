"""Frozen fixture pinning detect_collapse(..., legacy_rules=True) to the
exact 0.2.2 output on tests/test_collapse.py's fixtures (that file is the
gate and must not be edited). This test is the explicit "same seed -> same
output" backward-compatibility guarantee the perturbation-engine refactor
promises; if this test needs to change, the refactor broke something.
"""
import pytest

from whymatched.collapse import detect_collapse

from .fakes import FakeBagOfWordsModel


def test_legacy_rules_matches_0_2_2_output_negation_case():
    model = FakeBagOfWordsModel()
    query = "is remote work allowed for contractors"
    chunk = "remote work is not allowed for contractors under this policy document"
    flags = detect_collapse(model, query, chunk, threshold=0.15, legacy_rules=True)

    assert len(flags) == 3

    negation = next(f for f in flags if f.kind == "negation_collapse")
    assert negation.side == "chunk"
    assert negation.trigger == "not"
    assert negation.counterfactual_snippet == (
        "remote work is allowed for contractors under this policy document"
    )
    assert negation.base_score == pytest.approx(0.7925939559936523)
    assert negation.counterfactual_score == pytest.approx(0.8249578475952148)
    assert negation.relative_delta == pytest.approx(0.04083287710791185)

    antonym_query = next(f for f in flags if f.kind == "antonym_collapse" and f.side == "query")
    assert antonym_query.trigger == "allowed -> prohibited"
    assert antonym_query.counterfactual_snippet == "is remote work prohibited for contractors"
    assert antonym_query.relative_delta == pytest.approx(0.14285716434342643)

    antonym_chunk = next(f for f in flags if f.kind == "antonym_collapse" and f.side == "chunk")
    assert antonym_chunk.trigger == "allowed -> prohibited"
    assert antonym_chunk.counterfactual_snippet == (
        "remote work is not prohibited for contractors under this policy document"
    )
    assert antonym_chunk.relative_delta == pytest.approx(0.14285716434342643)


def test_legacy_rules_matches_0_2_2_output_antonym_case():
    model = FakeBagOfWordsModel()
    query = "is this particular business action allowed under our rules today"
    chunk = "this particular business action is allowed under company policy rules today now"
    flags = detect_collapse(model, query, chunk, threshold=0.3, legacy_rules=True)

    assert len(flags) == 2
    for f in flags:
        assert f.kind == "antonym_collapse"
        assert f.trigger == "allowed -> prohibited"

    query_flag = next(f for f in flags if f.side == "query")
    assert query_flag.relative_delta == pytest.approx(0.009886881304992119)

    chunk_flag = next(f for f in flags if f.side == "chunk")
    assert chunk_flag.relative_delta == pytest.approx(0.16666666666666666)


def test_legacy_rules_matches_0_2_2_output_no_candidates():
    model = FakeBagOfWordsModel()
    flags = detect_collapse(model, "the cat sat", "the mat sat", threshold=0.1, legacy_rules=True)
    assert flags == []


def test_legacy_rules_is_the_default():
    model = FakeBagOfWordsModel()
    query = "is remote work allowed for contractors"
    chunk = "remote work is not allowed for contractors under this policy document"
    default_flags = detect_collapse(model, query, chunk, threshold=0.15)
    explicit_flags = detect_collapse(model, query, chunk, threshold=0.15, legacy_rules=True)
    assert default_flags == explicit_flags
