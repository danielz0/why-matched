import random

import pytest

from whymatched.perturbations.antonym import AntonymPerturbation


@pytest.mark.parametrize(
    "text,expected_trigger",
    [
        ("this is a good outcome", "good -> bad"),
        ("access is allowed here", "allowed -> prohibited"),
        ("the field is required", "required -> optional"),
        ("the statement is true", "true -> false"),
        ("the connection is secure", "secure -> insecure"),
    ],
)
def test_known_antonym_pairs_detected(text, expected_trigger):
    p = AntonymPerturbation()
    candidates = p.propose(text, rng=random.Random(0))
    assert any(c.trigger == expected_trigger for c in candidates)


@pytest.mark.parametrize(
    "text",
    [
        "the cat sat on the mat",
        "revenue grew steadily",
        "please review the document",
    ],
)
def test_no_antonym_candidates_for_unrelated_text(text):
    p = AntonymPerturbation()
    candidates = p.propose(text, rng=random.Random(0))
    assert candidates == []


def test_migrated_pairs_no_longer_in_antonym_file():
    p = AntonymPerturbation()
    text = "revenue was higher before the change and more than expected"
    candidates = p.propose(text, rng=random.Random(0))
    triggers = {c.trigger for c in candidates}
    assert not any("higher" in t or "before" in t or "more" in t for t in triggers)
