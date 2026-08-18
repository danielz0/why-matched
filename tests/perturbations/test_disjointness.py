"""Guards against B2/B3-class regressions: no surface form should trigger
two different perturbation kinds on the same occurrence, except the
deliberate, arbitration-resolved overlaps documented below."""
import random

from whymatched.perturbations.antonym import _load_antonyms
from whymatched.perturbations.comparative import ComparativePerturbation
from whymatched.perturbations.modal import ModalPerturbation
from whymatched.perturbations.negation import NEGATION_CUES
from whymatched.perturbations.quantifier import QuantifierPerturbation

_ALLOWED_OVERLAP = {"required", "optional", "none", "never", "cannot"}


def _rule_set_keys():
    sets = {
        "negation_cues": {c.lower() for c in NEGATION_CUES},
        "antonym": set(_load_antonyms().keys()),
        "comparative": set(ComparativePerturbation._lookup_keys())
        if hasattr(ComparativePerturbation, "_lookup_keys")
        else set(),
        "quantifier": set(QuantifierPerturbation._lookup_keys())
        if hasattr(QuantifierPerturbation, "_lookup_keys")
        else set(),
        "modal": set(ModalPerturbation._lookup_keys()) if hasattr(ModalPerturbation, "_lookup_keys") else set(),
    }
    return sets


def test_no_surface_form_in_multiple_rule_sets():
    sets = _rule_set_keys()
    names = list(sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            overlap = sets[a_name] & sets[b_name]
            unexpected = overlap - _ALLOWED_OVERLAP
            assert not unexpected, f"{a_name} and {b_name} both claim {unexpected}"


def test_b2_never_not_double_counted():
    from whymatched.perturbations.antonym import AntonymPerturbation
    from whymatched.perturbations.negation import NegationPerturbation

    text = "Contractors may never use the portal."
    rng = random.Random(0)
    negation_candidates = NegationPerturbation(legacy_rules=False).propose(text, rng=rng)
    antonym_candidates = AntonymPerturbation().propose(text, rng=rng)

    never_negation = [c for c in negation_candidates if c.span.text.lower() == "never"]
    never_antonym = [c for c in antonym_candidates if c.span.text.lower() == "never"]

    assert len(never_negation) == 1
    assert len(never_antonym) == 0


def test_cannot_negation_wins_over_antonym_can():
    from whymatched.perturbations.antonym import AntonymPerturbation
    from whymatched.perturbations.engine import _arbitrate
    from whymatched.perturbations.negation import NegationPerturbation

    text = "contractors cannot use the portal"
    rng = random.Random(0)
    negation_candidates = NegationPerturbation(legacy_rules=False).propose(text, rng=rng)
    antonym_candidates = AntonymPerturbation().propose(text, rng=rng)

    resolved = _arbitrate(negation_candidates + antonym_candidates)
    cannot_candidates = [c for c in resolved if c.span.text == "cannot"]

    assert len(cannot_candidates) == 1
    assert cannot_candidates[0].kind == "negation"
