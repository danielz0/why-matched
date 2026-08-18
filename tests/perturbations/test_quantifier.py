import random

import pytest

from whymatched.perturbations.quantifier import QuantifierPerturbation


@pytest.mark.parametrize(
    "text,expected_triggers",
    [
        ("all users are affected", {"all -> some", "all -> none"}),
        ("none of the tests failed", {"none -> all"}),
        ("every request is logged", {"every -> some"}),
        ("this always happens", {"always -> sometimes"}),
        ("most reviewers agreed", {"most -> few"}),
        ("both options are valid", {"both -> neither"}),
    ],
)
def test_quantifier_fanout_positive_cases(text, expected_triggers):
    p = QuantifierPerturbation()
    candidates = p.propose(text, rng=random.Random(0))
    triggers = {c.trigger for c in candidates}
    assert expected_triggers <= triggers


@pytest.mark.parametrize(
    "text",
    [
        "the cat sat on the mat",
        "revenue grew steadily",
        "no quantifiers present here",
    ],
)
def test_quantifier_negative_cases(text):
    p = QuantifierPerturbation()
    candidates = p.propose(text, rng=random.Random(0))
    assert candidates == []


def test_all_fans_out_to_two_candidates_same_span():
    p = QuantifierPerturbation()
    candidates = p.propose("all users are affected", rng=random.Random(0))
    all_candidates = [c for c in candidates if c.span.text == "all"]
    assert len(all_candidates) == 2
    assert {c.replacement for c in all_candidates} == {"some", "none"}


def test_max_per_text_respected_across_fanout():
    p = QuantifierPerturbation(max_per_text=1)
    candidates = p.propose("all users are affected", rng=random.Random(0))
    assert len(candidates) == 1
