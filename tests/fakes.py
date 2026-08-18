"""A tiny deterministic bag-of-words embedding model used to unit test the
attribution/collapse/projection *logic* without depending on a real
transformer model (kept in tests/ separately for a slow integration test)."""
from __future__ import annotations

import hashlib
import re

import numpy as np

from whymatched.models.api import simple_tokenize
from whymatched.models.base import EmbeddingModel
from whymatched.perturbations.negation import _NEG_RE


class FakeBagOfWordsModel(EmbeddingModel):
    supports_gradients = False
    supports_token_embeddings = False
    name = "fake-bow"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _word_vec(self, word: str) -> np.ndarray:
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dim
        v = np.zeros(self.dim, dtype=np.float32)
        v[h] = 1.0
        return v

    def embed(self, texts):
        vecs = []
        for t in texts:
            words = simple_tokenize(t.lower())
            if not words:
                vecs.append(np.zeros(self.dim, dtype=np.float32))
                continue
            v = np.sum([self._word_vec(w) for w in words], axis=0)
            vecs.append(v.astype(np.float32))
        return np.stack(vecs)

    def tokenize(self, text):
        return simple_tokenize(text)


class CountingModel(EmbeddingModel):
    """Wraps another EmbeddingModel; records .calls (number of embed() invocations)."""

    supports_gradients = False
    supports_token_embeddings = False

    def __init__(self, inner: EmbeddingModel):
        self._inner = inner
        self.calls = 0
        self.name = f"counting({inner.name})"
        self.supports_gradients = inner.supports_gradients
        self.supports_token_embeddings = inner.supports_token_embeddings

    def embed(self, texts):
        self.calls += 1
        return self._inner.embed(texts)

    def tokenize(self, text):
        return self._inner.tokenize(text)


class NegationBlindModel(EmbeddingModel):
    """Strips every negation cue matched by the real negation perturbation's
    own regex before hashing -- structurally unable to represent polarity.
    Scope: cue words only (matches _NEG_RE); does not additionally handle
    contractions ("don't" -> "do"), so test fixtures for this model should
    use plain cue words like "not", not contractions."""

    supports_gradients = False
    supports_token_embeddings = False
    name = "fake-negation-blind"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _word_vec(self, word: str) -> np.ndarray:
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dim
        v = np.zeros(self.dim, dtype=np.float32)
        v[h] = 1.0
        return v

    def embed(self, texts):
        vecs = []
        for t in texts:
            stripped = _NEG_RE.sub(" ", t)
            words = simple_tokenize(stripped.lower())
            if not words:
                vecs.append(np.zeros(self.dim, dtype=np.float32))
                continue
            v = np.sum([self._word_vec(w) for w in words], axis=0)
            vecs.append(v.astype(np.float32))
        return np.stack(vecs)

    def tokenize(self, text):
        return simple_tokenize(text)


class NegationAwareModel(EmbeddingModel):
    """Appends a synthetic '__negated__' token to the bag when any negation
    cue is present, in addition to keeping the cue word itself -- the mirror
    image of NegationBlindModel, making negated vs. non-negated variants
    maximally distinguishable."""

    supports_gradients = False
    supports_token_embeddings = False
    name = "fake-negation-aware"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _word_vec(self, word: str) -> np.ndarray:
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dim
        v = np.zeros(self.dim, dtype=np.float32)
        v[h] = 1.0
        return v

    def embed(self, texts):
        vecs = []
        for t in texts:
            words = simple_tokenize(t.lower())
            if _NEG_RE.search(t):
                words = words + ["__negated__"]
            if not words:
                vecs.append(np.zeros(self.dim, dtype=np.float32))
                continue
            v = np.sum([self._word_vec(w) for w in words], axis=0)
            vecs.append(v.astype(np.float32))
        return np.stack(vecs)

    def tokenize(self, text):
        return simple_tokenize(text)


class MagnitudeBlindModel(EmbeddingModel):
    """Normalizes every digit run to '0' before hashing -- "$50" and "$500"
    both become the token "$0", making a magnitude-blind model
    indistinguishable regardless of the actual scale change."""

    supports_gradients = False
    supports_token_embeddings = False
    name = "fake-magnitude-blind"
    _DIGITS_RE = re.compile(r"\d+")

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _word_vec(self, word: str) -> np.ndarray:
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % self.dim
        v = np.zeros(self.dim, dtype=np.float32)
        v[h] = 1.0
        return v

    def embed(self, texts):
        vecs = []
        for t in texts:
            normalized = self._DIGITS_RE.sub("0", t.lower())
            words = simple_tokenize(normalized)
            if not words:
                vecs.append(np.zeros(self.dim, dtype=np.float32))
                continue
            v = np.sum([self._word_vec(w) for w in words], axis=0)
            vecs.append(v.astype(np.float32))
        return np.stack(vecs)

    def tokenize(self, text):
        return simple_tokenize(text)
