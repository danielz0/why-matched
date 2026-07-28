from __future__ import annotations

import numpy as np

from ..text import remove_span, split_words
from ..utils import cosine_similarity
from .base import AttributionResult, TokenScore


def occlusion_attribution(model, query: str, chunk: str) -> AttributionResult:
    """Leave-one-word-out attribution: works with *any* embedding backend
    (local or hosted API) since it only needs ``model.embed()``.

    For each word, re-embed the text with that word removed and measure how
    much the similarity drops. ``weight = base_score - score_without_token``,
    so a positive weight means the word was pulling the score up; negative
    means it was dragging the score down (e.g. a mismatched or negated word).
    """
    base_vecs = model.embed([query, chunk])
    query_vec, chunk_vec = base_vecs[0], base_vecs[1]
    base_score = float(cosine_similarity(query_vec, chunk_vec)[0, 0])

    q_words = split_words(query)
    c_words = split_words(chunk)

    ablated_queries = [remove_span(query, w) or query for w in q_words]
    ablated_chunks = [remove_span(chunk, w) or chunk for w in c_words]

    dim = base_vecs.shape[1]
    q_vecs = model.embed(ablated_queries) if ablated_queries else np.zeros((0, dim), dtype=np.float32)
    c_vecs = model.embed(ablated_chunks) if ablated_chunks else np.zeros((0, dim), dtype=np.float32)

    query_tokens = []
    for word, qv in zip(q_words, q_vecs):
        score_without = float(cosine_similarity(qv, chunk_vec)[0, 0])
        query_tokens.append(TokenScore(token=word.text, weight=base_score - score_without))

    chunk_tokens = []
    for word, cv in zip(c_words, c_vecs):
        score_without = float(cosine_similarity(query_vec, cv)[0, 0])
        chunk_tokens.append(TokenScore(token=word.text, weight=base_score - score_without))

    return AttributionResult(
        method="occlusion",
        query_tokens=query_tokens,
        chunk_tokens=chunk_tokens,
        base_score=base_score,
    )
