"""Content-addressed embedding cache. Perturbation-based collapse detection
re-embeds many counterfactual variants of the same query/chunk; without a
cache, callers end up issuing an embed() call per candidate, which is
unaffordable against an API-backed model. Callers collect every text they
need up front, prime() once (batched), then score from cache."""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .utils import cosine_similarity


class EmbeddingCache:
    """Content-addressed embedding cache with batched flush. NOT thread-safe."""

    def __init__(self, model, max_batch: int = 256) -> None:
        self._model = model
        self._max_batch = max_batch
        self._store: dict = {}
        self._calls = 0

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def prime(self, texts: Sequence[str]) -> None:
        """Queue unseen texts; embed all in one batched call (chunked by max_batch)."""
        seen_in_call: set = set()
        to_embed_keys = []
        to_embed_texts = []
        for t in texts:
            k = self._key(t)
            if k in self._store or k in seen_in_call:
                continue
            seen_in_call.add(k)
            to_embed_keys.append(k)
            to_embed_texts.append(t)

        for start in range(0, len(to_embed_texts), self._max_batch):
            batch_texts = to_embed_texts[start : start + self._max_batch]
            batch_keys = to_embed_keys[start : start + self._max_batch]
            vecs = self._model.embed(batch_texts)
            self._calls += 1
            for key, vec in zip(batch_keys, vecs):
                self._store[key] = vec

    def get(self, text: str) -> np.ndarray:
        k = self._key(text)
        if k not in self._store:
            raise KeyError(f"text not primed: {text!r}")
        return self._store[k]

    def score(self, a: str, b: str) -> float:
        return float(cosine_similarity(self.get(a), self.get(b))[0, 0])

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def model_name(self) -> str:
        return getattr(self._model, "name", "unknown")
