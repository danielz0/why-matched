"""Proves the F2 probe models' documented blindness/awareness in isolation,
before any calibration wiring exists. A broken fake here would otherwise be
indistinguishable from a calibration-layer bug."""
import numpy as np

from .fakes import MagnitudeBlindModel, NegationAwareModel, NegationBlindModel


def test_negation_blind_model_ignores_negation_cue():
    model = NegationBlindModel()
    a, b = model.embed(["contractors may use the portal", "contractors may not use the portal"])
    assert np.array_equal(a, b)


def test_negation_blind_model_still_distinguishes_other_words():
    model = NegationBlindModel()
    a, b = model.embed(["contractors may use the portal", "employees may use the gateway"])
    assert not np.array_equal(a, b)


def test_negation_aware_model_distinguishes_negation_cue():
    model = NegationAwareModel()
    a, b = model.embed(["contractors may use the portal", "contractors may not use the portal"])
    assert not np.array_equal(a, b)


def test_negation_aware_model_identical_without_negation():
    model = NegationAwareModel()
    a, b = model.embed(["contractors may use the portal", "contractors may use the portal"])
    assert np.array_equal(a, b)


def test_magnitude_blind_model_ignores_scale_change():
    model = MagnitudeBlindModel()
    a, b = model.embed(["the price is $50 today", "the price is $500 today"])
    assert np.array_equal(a, b)


def test_magnitude_blind_model_still_distinguishes_other_words():
    model = MagnitudeBlindModel()
    a, b = model.embed(["the price is $50 today", "the cost is $50 tomorrow"])
    assert not np.array_equal(a, b)
