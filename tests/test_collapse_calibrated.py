"""Integration tests for detect_collapse(calibrate=True), kept separate from
the frozen tests/test_collapse.py and tests/test_collapse_legacy_fixture.py
files, which must stay untouched."""
from whymatched.collapse import detect_collapse

from .fakes import FakeBagOfWordsModel, MagnitudeBlindModel, NegationAwareModel, NegationBlindModel

LONG_CHUNK = (
    "The vendor must obtain approval before it can purchase additional equipment. "
    "The team will request approval and then purchase the equipment once approval is obtained. "
    "Approval must be verified and confirmed by the finance department before any purchase is completed. "
    "If approval is not obtained, the purchase cannot proceed, and the vendor must request approval again. "
    "The finance department will examine each purchase request and verify that the required approval was "
    "properly obtained before the purchase is finalized. "
    "Once verified, the team can complete the purchase and begin using the equipment immediately. "
    "The vendor should also retain a copy of every approval and provide it to finance when requested. "
    "Additionally, the finance department will assist with any purchase that needs modification once it has begun."
)

SINGLE_NEGATION_CHUNK = (
    "The vendor must obtain approval before it can purchase additional equipment. "
    "The team will request approval and then purchase the equipment once approval is obtained. "
    "Approval must be verified and confirmed by the finance department before any purchase is completed. "
    "The finance department will examine each purchase request and verify that the required approval was "
    "properly obtained before the purchase is finalized, but it is not yet fully processed. "
    "Once verified, the team can complete the purchase and begin using the equipment immediately. "
    "The vendor should also retain a copy of every approval and provide it to finance when requested. "
    "Additionally, the finance department will assist with any purchase that needs modification once it has begun."
)


def test_detect_collapse_calibrate_true_uses_is_collapse_not_threshold():
    model = FakeBagOfWordsModel()
    query = "is approval required for a purchase"
    threshold_flags = detect_collapse(model, query, LONG_CHUNK, threshold=0.10, kinds=("negation", "antonym"), legacy_rules=False)
    calibrated_flags = detect_collapse(
        model, query, LONG_CHUNK, threshold=0.10, kinds=("negation", "antonym"), legacy_rules=False, calibrate=True,
    )
    for f in calibrated_flags:
        assert hasattr(f, "kind") and hasattr(f, "relative_delta")
        assert not hasattr(f, "is_collapse")


def test_detect_collapse_calibrate_false_unchanged():
    model = FakeBagOfWordsModel()
    query = "is remote work allowed for contractors"
    chunk = "remote work is not allowed for contractors under this policy document"
    flags = detect_collapse(model, query, chunk, threshold=0.15)
    assert len(flags) == 3
    kinds = {f.kind for f in flags}
    assert "negation_collapse" in kinds
    assert "antonym_collapse" in kinds


def test_negation_blind_model_flagged_via_detect_collapse_calibrate_true():
    model = NegationBlindModel()
    query = "is approval required for this purchase"
    flags = detect_collapse(
        model, query, SINGLE_NEGATION_CHUNK, kinds=("negation",), legacy_rules=False, calibrate=True, seed=0,
    )
    assert any(f.kind == "negation_collapse" for f in flags)


def test_negation_aware_model_not_flagged_via_detect_collapse_calibrate_true():
    model = NegationAwareModel()
    query = "is approval required for this purchase"
    flags = detect_collapse(
        model, query, SINGLE_NEGATION_CHUNK, kinds=("negation",), legacy_rules=False, calibrate=True, seed=0,
    )
    assert not any(f.kind == "negation_collapse" for f in flags)


def test_magnitude_blind_model_flagged_via_detect_collapse_calibrate_true():
    model = MagnitudeBlindModel()
    query = "the price is $50 for this purchase"
    flags = detect_collapse(model, query, LONG_CHUNK, kinds=("numeric",), legacy_rules=False, calibrate=True, seed=0)
    assert any(f.kind == "numeric_collapse" for f in flags)
