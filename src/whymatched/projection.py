from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np


@dataclass
class ProjectedPoint:
    label: str
    kind: Literal["query", "chunk", "query_token", "chunk_token"]
    x: float
    y: float
    chunk_index: Optional[int] = None


def _reduce(vecs: np.ndarray, method: str = "pca", random_state: int = 0) -> np.ndarray:
    n = vecs.shape[0]
    if n <= 2:
        out = np.zeros((n, 2), dtype=np.float32)
        d = vecs.shape[1]
        out[:, 0] = vecs[:, 0] if d >= 1 else 0.0
        out[:, 1] = vecs[:, 1] if d >= 2 else 0.0
        return out

    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError("UMAP projection requires: pip install whymatched[viz]") from e
        reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=min(15, n - 1))
        return np.asarray(reducer.fit_transform(vecs))

    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = max(2, min(30, (n - 1) // 3))
        return TSNE(n_components=2, random_state=random_state, perplexity=perplexity, init="pca").fit_transform(vecs)

    from sklearn.decomposition import PCA

    n_components = min(2, vecs.shape[1], n)
    reduced = PCA(n_components=n_components, random_state=random_state).fit_transform(vecs)
    if reduced.shape[1] < 2:
        reduced = np.pad(reduced, ((0, 0), (0, 2 - reduced.shape[1])))
    return reduced


def project_sentence_level(
    model, query: str, chunks: List[str], method: str = "pca", random_state: int = 0
) -> List[ProjectedPoint]:
    """Project the whole-text (pooled) embeddings of query + chunks into 2D.
    Works for any backend, local or API."""
    vecs = model.embed([query] + list(chunks))
    coords = _reduce(vecs, method=method, random_state=random_state)
    points = [ProjectedPoint(label=query, kind="query", x=float(coords[0, 0]), y=float(coords[0, 1]))]
    for i, chunk in enumerate(chunks):
        points.append(
            ProjectedPoint(
                label=chunk, kind="chunk", x=float(coords[i + 1, 0]), y=float(coords[i + 1, 1]), chunk_index=i
            )
        )
    return points


def project_token_level(
    model, query: str, chunks: List[str], method: str = "pca", random_state: int = 0
) -> List[ProjectedPoint]:
    """Project every individual query/chunk token's hidden-state embedding
    into 2D. Requires a local model (per-token embeddings)."""
    if not getattr(model, "supports_token_embeddings", False):
        raise ValueError(
            f"{model.name} does not expose per-token embeddings; token-level projection requires a local model"
        )

    special = set(getattr(model.tokenizer, "all_special_tokens", []))
    all_vecs = []
    meta = []  # (label, kind, chunk_index)

    q_tokens, q_vecs = model.token_embeddings(query)
    for t, v in zip(q_tokens, q_vecs):
        if t in special:
            continue
        all_vecs.append(v)
        meta.append((t, "query_token", None))

    for i, chunk in enumerate(chunks):
        c_tokens, c_vecs = model.token_embeddings(chunk)
        for t, v in zip(c_tokens, c_vecs):
            if t in special:
                continue
            all_vecs.append(v)
            meta.append((t, "chunk_token", i))

    vecs = np.stack(all_vecs)
    coords = _reduce(vecs, method=method, random_state=random_state)
    return [
        ProjectedPoint(label=label, kind=kind, x=float(x), y=float(y), chunk_index=idx)
        for (label, kind, idx), (x, y) in zip(meta, coords)
    ]
