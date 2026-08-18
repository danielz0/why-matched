from __future__ import annotations

import numpy as np

from ..utils import cosine_similarity
from .base import AttributionResult, TokenScore


def maxsim_attribution(model, query: str, chunk: str) -> AttributionResult:
    """ColBERT-style late-interaction diagnostic: compute the full query-token
    x chunk-token cosine similarity matrix from pre-pooling hidden states,
    then score each token by its best match on the other side (MaxSim).

    This does not require gradients, only per-token hidden states, so it
    works for any :class:`~whymatched.models.local.LocalModel`. It answers a
    different question than occlusion/gradient attribution: not "how much
    did this token move the pooled score" but "which specific token pair is
    responsible for the match" — useful for spotting a single chunk token
    (e.g. a near-synonym) silently satisfying a query token.
    """
    if not getattr(model, "supports_token_embeddings", False):
        raise ValueError(
            f"{model.name} does not expose per-token embeddings; use method='occlusion' instead"
        )

    q_tokens, q_vecs = model.token_embeddings(query)
    c_tokens, c_vecs = model.token_embeddings(chunk)

    special = set(getattr(model.tokenizer, "all_special_tokens", []))
    q_keep = [i for i, t in enumerate(q_tokens) if t not in special]
    c_keep = [i for i, t in enumerate(c_tokens) if t not in special]
    q_tokens_f = [q_tokens[i] for i in q_keep]
    c_tokens_f = [c_tokens[i] for i in c_keep]
    q_vecs_f = q_vecs[q_keep] if q_keep else q_vecs[:0]
    c_vecs_f = c_vecs[c_keep] if c_keep else c_vecs[:0]

    sim_matrix = cosine_similarity(q_vecs_f, c_vecs_f) if q_vecs_f.size and c_vecs_f.size else np.zeros(
        (len(q_tokens_f), len(c_tokens_f)), dtype=np.float32
    )

    base_vecs = model.embed([query, chunk])
    base_score = float(cosine_similarity(base_vecs[0], base_vecs[1])[0, 0])

    q_maxsim = sim_matrix.max(axis=1) if sim_matrix.size else np.zeros(len(q_tokens_f))
    c_maxsim = sim_matrix.max(axis=0) if sim_matrix.size else np.zeros(len(c_tokens_f))
    q_argmax = sim_matrix.argmax(axis=1) if sim_matrix.size else np.zeros(len(q_tokens_f), dtype=int)
    c_argmax = sim_matrix.argmax(axis=0) if sim_matrix.size else np.zeros(len(c_tokens_f), dtype=int)

    query_tokens = [TokenScore(token=t, weight=float(w)) for t, w in zip(q_tokens_f, q_maxsim)]
    chunk_tokens = [TokenScore(token=t, weight=float(w)) for t, w in zip(c_tokens_f, c_maxsim)]

    return AttributionResult(
        method="maxsim",
        query_tokens=query_tokens,
        chunk_tokens=chunk_tokens,
        base_score=base_score,
        extra={
            "sim_matrix": sim_matrix.tolist(),
            "query_tokens": q_tokens_f,
            "chunk_tokens": c_tokens_f,
            "query_best_match": [c_tokens_f[i] if len(c_tokens_f) else None for i in q_argmax],
            "chunk_best_match": [q_tokens_f[i] if len(q_tokens_f) else None for i in c_argmax],
        },
    )
