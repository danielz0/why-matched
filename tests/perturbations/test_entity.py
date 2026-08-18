import random
import sys
from unittest import mock

import pytest

from whymatched.perturbations.entity import EntityPerturbation


def test_entity_skips_cleanly_without_spacy():
    with mock.patch.dict(sys.modules, {"spacy": None}):
        import whymatched.perturbations.entity as entity_module

        entity_module._NLP = None
        p = EntityPerturbation()
        assert p.available() is False
        assert p.propose("Alice works at Acme Corp", rng=random.Random(0)) == []
        entity_module._NLP = None


def test_entity_requires_spacy_dependency_declared():
    assert EntityPerturbation.requires == ("spacy",)


@pytest.mark.optional_deps
def test_entity_swaps_person():
    pytest.importorskip("spacy")
    p = EntityPerturbation()
    if not p.available():
        pytest.skip("en_core_web_sm not installed")
    candidates = p.propose("Alice Johnson approved the request.", rng=random.Random(0))
    assert any(c.kind == "entity" for c in candidates)


@pytest.mark.optional_deps
def test_entity_surrogate_differs_from_original_and_not_in_text():
    pytest.importorskip("spacy")
    p = EntityPerturbation()
    if not p.available():
        pytest.skip("en_core_web_sm not installed")
    text = "Acme Corporation signed the contract with Acme Corporation's subsidiary."
    candidates = p.propose(text, rng=random.Random(0))
    for c in candidates:
        assert c.replacement.lower() != c.span.text.lower()
        assert c.replacement.lower() not in text.lower()
