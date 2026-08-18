import random

import pytest

from whymatched.perturbations.negation import NegationPerturbation


@pytest.mark.parametrize(
    "text,expected_trigger_substr",
    [
        ("this is not allowed", "not"),
        ("nobody is allowed here", "nobody"),
        ("access without a badge", "without"),
        ("unable to comply", "unable to"),
        ("cannot proceed", "cannot"),
    ],
)
def test_plain_negation_cues_detected(text, expected_trigger_substr):
    p = NegationPerturbation(legacy_rules=False)
    candidates = p.propose(text, rng=random.Random(0))
    assert any(expected_trigger_substr in c.trigger for c in candidates)


@pytest.mark.parametrize(
    "text,expected_replacement",
    [
        ("don't use the portal", "do"),
        ("won't be accepted", "will"),
        ("isn't valid", "is"),
        ("can't proceed", "can"),
        ("didn't finish", "did"),
        ("wouldn't work", "would"),
    ],
)
def test_b1_contractions_produce_negation_candidates(text, expected_replacement):
    p = NegationPerturbation(legacy_rules=False)
    candidates = p.propose(text, rng=random.Random(0))
    assert candidates, f"expected a candidate for {text!r}"
    assert any(c.kind == "negation" and c.replacement == expected_replacement for c in candidates)


def test_b1_dead_under_legacy_rules():
    p = NegationPerturbation(legacy_rules=True)
    candidates = p.propose("don't use the portal", rng=random.Random(0))
    assert candidates == []


def test_negation_no_false_positive_inside_nonetheless():
    p = NegationPerturbation(legacy_rules=False)
    candidates = p.propose("nonetheless we proceeded", rng=random.Random(0))
    assert not any(c.trigger == "none" for c in candidates)


def test_negation_no_candidates_on_plain_text():
    p = NegationPerturbation(legacy_rules=False)
    candidates = p.propose("the cat sat on the mat", rng=random.Random(0))
    assert candidates == []


def test_negation_respects_max_per_text():
    text = " ".join(["not"] * 20)
    p = NegationPerturbation(legacy_rules=False, max_per_text=3)
    candidates = p.propose(text, rng=random.Random(0))
    assert len(candidates) == 3
