from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


class EmbeddingModel:
    """Interface every embedding backend implements.

    ``embed`` is the only method required to use the library at all
    (occlusion attribution, collapse detection, and sentence-level
    projection only need this). ``supports_gradients`` /
    ``supports_token_embeddings`` tell :class:`whymatched.Debugger` which
    additional attribution methods (Integrated Gradients, MaxSim) and
    token-level projection are available for this backend.
    """

    name: str = "unnamed-model"
    supports_gradients: bool = False
    supports_token_embeddings: bool = False

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) float32 array of embeddings for ``texts``."""
        raise NotImplementedError

    def tokenize(self, text: str) -> List[str]:
        """Human-readable token split (used for display, not required for occlusion)."""
        raise NotImplementedError

    def token_embeddings(self, text: str) -> Tuple[List[str], np.ndarray]:
        """Return (tokens, per-token embeddings) — local models only."""
        raise NotImplementedError(f"{self.name} does not expose per-token embeddings")
