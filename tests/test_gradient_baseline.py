"""Unit tests for the Integrated Gradients baseline-selection logic in
gradient.py. These don't need torch: _baseline_token_id is plain Python
operating on a duck-typed tokenizer stand-in."""
import pytest

from whymatched.attribution.gradient import _baseline_token_id


class _FakeTokenizer:
    def __init__(self, pad_token_id=None, eos_token_id=None, mask_token_id=None):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id


def test_pad_baseline_uses_pad_token_id():
    tok = _FakeTokenizer(pad_token_id=0, eos_token_id=2, mask_token_id=4)
    assert _baseline_token_id(tok, "pad") == 0


def test_pad_baseline_falls_back_to_eos_token_id():
    tok = _FakeTokenizer(pad_token_id=None, eos_token_id=2, mask_token_id=4)
    assert _baseline_token_id(tok, "pad") == 2


def test_pad_baseline_raises_without_pad_or_eos():
    tok = _FakeTokenizer(pad_token_id=None, eos_token_id=None, mask_token_id=4)
    with pytest.raises(ValueError):
        _baseline_token_id(tok, "pad")


def test_mask_baseline_uses_mask_token_id():
    tok = _FakeTokenizer(pad_token_id=0, mask_token_id=103)
    assert _baseline_token_id(tok, "mask") == 103


def test_mask_baseline_raises_without_mask_token():
    tok = _FakeTokenizer(pad_token_id=0, mask_token_id=None)
    with pytest.raises(ValueError):
        _baseline_token_id(tok, "mask")


def test_unknown_baseline_raises():
    tok = _FakeTokenizer(pad_token_id=0)
    with pytest.raises(ValueError):
        _baseline_token_id(tok, "bogus")
