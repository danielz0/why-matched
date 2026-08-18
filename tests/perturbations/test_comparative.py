import random

import pytest

from whymatched.perturbations.comparative import ComparativePerturbation


@pytest.mark.parametrize(
    "text,expected_trigger",
    [
        ("revenue was higher this year", "higher -> lower"),
        ("costs were more than expected", "more -> less"),
        ("throughput exceeds the baseline", "exceeds -> falls below"),
        ("at least five items required", "at least -> at most"),
        ("results were faster than before", "faster -> slower"),
    ],
)
def test_comparative_positive_cases(text, expected_trigger):
    p = ComparativePerturbation()
    candidates = p.propose(text, rng=random.Random(0))
    assert any(c.trigger == expected_trigger for c in candidates)


@pytest.mark.parametrize(
    "text",
    [
        "the cat sat on the mat",
        "please review the document",
        "no comparison words at all",
    ],
)
def test_comparative_negative_cases(text):
    p = ComparativePerturbation()
    candidates = p.propose(text, rng=random.Random(0))
    assert candidates == []


def test_migrated_from_antonym_file_b3():
    p = ComparativePerturbation()
    candidates = p.propose("increased costs were above the maximum", rng=random.Random(0))
    triggers = {c.trigger for c in candidates}
    assert "increased -> decreased" in triggers
    assert "above -> below" in triggers
    assert "maximum -> minimum" in triggers
