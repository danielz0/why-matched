from whymatched.perturbations import DEFAULT_KINDS, get_perturbations, register
from whymatched.perturbations.negation import NegationPerturbation


def test_default_kinds_excludes_entity():
    assert "entity" not in DEFAULT_KINDS
    assert set(DEFAULT_KINDS) == {
        "negation", "antonym", "numeric", "temporal",
        "comparative", "quantifier", "modal",
    }


def test_get_perturbations_filters_by_kind():
    result = get_perturbations(kinds=["negation"])
    assert all(p.kind == "negation" for p in result)
    assert len(result) == 1


def test_get_perturbations_excludes_unavailable_by_default():
    result = get_perturbations(kinds=["entity"])
    assert result == []


def test_get_perturbations_include_unavailable():
    result = get_perturbations(kinds=["entity"], include_unavailable=True)
    assert len(result) == 1


def test_register_replaces_existing_name():
    class ReplacementNegation(NegationPerturbation):
        pass

    replacement = ReplacementNegation()
    register(replacement)
    try:
        result = get_perturbations(kinds=["negation"])
        assert any(p is replacement for p in result)
    finally:
        register(NegationPerturbation())
