import random
from decimal import Decimal

import pytest

from whymatched.perturbations.numeric import NumericPerturbation, _reformat


@pytest.mark.parametrize(
    "text,expected_trigger",
    [
        ("the price is $50 today", "$50 -> $500"),
        ("total was €1,200.00 exactly", "€1,200.00 -> €12,000.00"),
        ("grew by 15% this quarter", "15% -> 150%"),
        ("running v1 now", "v1 -> v2"),
        ("dosage 500mg twice daily", "500mg -> 5000mg"),
    ],
)
def test_numeric_positive_cases(text, expected_trigger):
    p = NumericPerturbation()
    candidates = p.propose(text, rng=random.Random(0))
    assert any(c.trigger == expected_trigger for c in candidates)


@pytest.mark.parametrize(
    "text",
    [
        "hash SHA256 abcdef is stored",
        "id ID4521X was created",
        "no numbers here at all",
    ],
)
def test_numeric_negative_cases(text):
    p = NumericPerturbation()
    candidates = p.propose(text, rng=random.Random(0))
    assert candidates == []


def test_division_uses_decimal_exact():
    assert Decimal("1.15") / 10 == Decimal("0.115")
    assert Decimal("0.07") * 10 == Decimal("0.70")


def test_reformat_thousands_separator():
    assert _reformat("1,200", Decimal(12000)) == "12,000"


def test_reformat_two_decimal_currency():
    assert _reformat("1,200.00", Decimal(120)) == "120.00"


def test_reformat_leading_zeros():
    assert _reformat("0050", Decimal(500)) == "0500"


def test_reformat_unit_suffix_style():
    assert _reformat("500", Decimal(5000)) == "5000"
    assert _reformat("8", Decimal(Decimal(8) / 10)) == "0.8"


def test_reformat_percent():
    assert _reformat("15", Decimal(150)) == "150"


def test_reformat_semver_component():
    assert _reformat("3", Decimal(4)) == "4"


def test_percentage_has_no_divide_candidate():
    p = NumericPerturbation()
    candidates = p.propose("grew by 15% this quarter", rng=random.Random(0))
    triggers = [c.trigger for c in candidates]
    assert "15% -> 150%" in triggers
    assert "15% -> 51%" in triggers
    assert not any(t.startswith("15% -> 1.5%") or t.startswith("15% -> 2%") for t in triggers)


def test_version_bump_all_components():
    p = NumericPerturbation()
    candidates = p.propose("running v2.3.1 now", rng=random.Random(0))
    triggers = {c.trigger for c in candidates}
    assert triggers == {"v2.3.1 -> v2.3.2", "v2.3.1 -> v2.4.1", "v2.3.1 -> v3.3.1"}


def test_bare_decimal_scaled_exactly():
    p = NumericPerturbation()
    candidates = p.propose("value is 4.5 exactly", rng=random.Random(0))
    triggers = {c.trigger for c in candidates}
    assert "4.5 -> 45.0" in triggers
    assert "4.5 -> 0.45" in triggers


def test_unit_bearing_exact_division():
    p = NumericPerturbation()
    candidates = p.propose("transfer 8GB of data", rng=random.Random(0))
    triggers = {c.trigger for c in candidates}
    assert "8GB -> 80GB" in triggers
    assert "8GB -> 0.8GB" in triggers


def test_max_per_text_respected():
    text = "$1 $2 $3 $4 $5 $6 $7 $8 $9 $10"
    p = NumericPerturbation(max_per_text=4)
    candidates = p.propose(text, rng=random.Random(0))
    assert len(candidates) == 4
