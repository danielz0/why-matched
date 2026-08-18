import random

import pytest

from whymatched.perturbations.antonym import AntonymPerturbation
from whymatched.perturbations.engine import _arbitrate
from whymatched.perturbations.modal import ModalPerturbation


@pytest.mark.parametrize(
    "text,expected_trigger",
    [
        ("contractors must comply", "must -> may"),
        ("employees shall attend", "shall -> should"),
        ("approval is required", "required -> permitted"),
        ("attendance is mandatory", "mandatory -> optional"),
    ],
)
def test_modal_positive_cases(text, expected_trigger):
    p = ModalPerturbation()
    candidates = p.propose(text, rng=random.Random(0))
    assert any(c.trigger == expected_trigger for c in candidates)


@pytest.mark.parametrize(
    "text",
    [
        "the cat sat on the mat",
        "revenue grew steadily",
        "no modal words at all",
    ],
)
def test_modal_negative_cases(text):
    p = ModalPerturbation()
    candidates = p.propose(text, rng=random.Random(0))
    assert candidates == []


def test_migrated_from_antonym_file_e6():
    p = ModalPerturbation()
    candidates = p.propose("this is mandatory and must be done", rng=random.Random(0))
    triggers = {c.trigger for c in candidates}
    assert "mandatory -> optional" in triggers
    assert "must -> may" in triggers


def test_required_permitted_modal_wins_over_antonym_optional():
    text = "approval is required before shipping"
    rng = random.Random(0)
    modal_candidates = ModalPerturbation().propose(text, rng=rng)
    antonym_candidates = AntonymPerturbation().propose(text, rng=rng)

    resolved = _arbitrate(modal_candidates + antonym_candidates)
    required_span_candidates = [c for c in resolved if c.span.text == "required"]

    assert len(required_span_candidates) == 1
    assert required_span_candidates[0].kind == "modal"
    assert required_span_candidates[0].replacement == "permitted"
