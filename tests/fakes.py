"""A tiny deterministic bag-of-words embedding model used to unit test the
attribution/collapse/projection *logic* without depending on a real
transformer model (kept in tests/ separately for a slow integration test)."""
from __future__ import annotations

import hashlib

import numpy as np

from whymatched.models.api import simple_tokenize
from whymatched.models.base import EmbeddingModel


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
