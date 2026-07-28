from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

import numpy as np

from .base import EmbeddingModel

_WORD_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)


class APIEmbeddingModel(EmbeddingModel):
    """Wraps any hosted embedding API (`texts -> (n, d) array`).

    Only occlusion-based attribution and sentence-level projection are
    available for these models: hosted providers don't expose gradients or
    per-token vectors, so ``supports_gradients`` / ``supports_token_embeddings``
    stay False and ``Debugger`` falls back to occlusion automatically.
    """

    supports_gradients = False
    supports_token_embeddings = False

    def __init__(
        self,
        embed_fn: Callable[[Sequence[str]], np.ndarray],
        name: str = "api-model",
        normalize: bool = True,
    ):
        self._embed_fn = embed_fn
        self.name = name
        self.normalize = normalize

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vecs = np.asarray(self._embed_fn(texts), dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs

    def tokenize(self, text: str) -> List[str]:
        return simple_tokenize(text)

    # -- convenience constructors for common providers --------------------

    @classmethod
    def openai(
        cls,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        **client_kwargs,
    ) -> "APIEmbeddingModel":
        from openai import OpenAI

        client = OpenAI(api_key=api_key, **client_kwargs) if api_key else OpenAI(**client_kwargs)

        def embed_fn(texts: Sequence[str]):
            resp = client.embeddings.create(model=model, input=list(texts))
            return [d.embedding for d in resp.data]

        return cls(embed_fn, name=f"openai:{model}")

    @classmethod
    def cohere(
        cls,
        model: str = "embed-english-v3.0",
        api_key: Optional[str] = None,
        input_type: str = "search_document",
        **client_kwargs,
    ) -> "APIEmbeddingModel":
        import cohere

        client = cohere.Client(api_key, **client_kwargs) if api_key else cohere.Client(**client_kwargs)

        def embed_fn(texts: Sequence[str]):
            resp = client.embed(texts=list(texts), model=model, input_type=input_type)
            return resp.embeddings

        return cls(embed_fn, name=f"cohere:{model}")

    @classmethod
    def voyage(
        cls,
        model: str = "voyage-3",
        api_key: Optional[str] = None,
        input_type: Optional[str] = None,
        **client_kwargs,
    ) -> "APIEmbeddingModel":
        import voyageai

        client = voyageai.Client(api_key=api_key, **client_kwargs) if api_key else voyageai.Client(**client_kwargs)

        def embed_fn(texts: Sequence[str]):
            resp = client.embed(texts=list(texts), model=model, input_type=input_type)
            return resp.embeddings

        return cls(embed_fn, name=f"voyage:{model}")

    @classmethod
    def custom(
        cls,
        embed_fn: Callable[[Sequence[str]], np.ndarray],
        name: str = "custom-api-model",
    ) -> "APIEmbeddingModel":
        """Wrap any callable `texts -> vectors`, for providers without a
        built-in constructor (Azure OpenAI, self-hosted TEI servers, etc.)."""
        return cls(embed_fn, name=name)
