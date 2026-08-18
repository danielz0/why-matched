from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence

import numpy as np

from .attribution.base import AttributionResult
from .attribution.gradient import gradient_attribution
from .attribution.maxsim import maxsim_attribution
from .attribution.occlusion import occlusion_attribution
from .cache import EmbeddingCache
from .collapse import CollapseFlag, detect_collapse
from .projection import ProjectedPoint, project_sentence_level, project_token_level
from .utils import cosine_similarity

AttributionMethod = Literal["occlusion", "integrated_gradients", "maxsim", "auto"]


@dataclass
class ChunkAnalysis:
    chunk: str
    score: float
    rank: int
    attribution: AttributionResult
    collapse_flags: List[CollapseFlag] = field(default_factory=list)

    @property
    def top_query_tokens(self) -> List:
        return sorted(self.attribution.query_tokens, key=lambda t: t.weight, reverse=True)

    @property
    def top_chunk_tokens(self) -> List:
        return sorted(self.attribution.chunk_tokens, key=lambda t: t.weight, reverse=True)

    @property
    def has_collapse(self) -> bool:
        return len(self.collapse_flags) > 0


@dataclass
class AnalysisResult:
    query: str
    model_name: str
    method: str
    chunks: List[ChunkAnalysis]
    projection: Optional[List[ProjectedPoint]] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class Debugger:
    """The main library entry point.

    ``Debugger(model).analyze(query, chunks)`` returns a fully populated
    :class:`AnalysisResult` — per-chunk similarity, token-level attribution,
    negation/antonym collapse flags, and a 2D projection — as plain
    dataclasses your own application can consume directly (no plotting or
    server dependency required).
    """

    def __init__(
        self,
        model,
        method: AttributionMethod = "auto",
        collapse_threshold: float = 0.10,
        use_wordnet: bool = False,
        gradient_baseline: str = "pad",
        perturbation_kinds: Optional[Sequence[str]] = None,
        legacy_rules: bool = True,
        calibrate_collapse: bool = False,
        n_null: int = 50,
        quantile: float = 0.10,
        correction: Literal["none", "bh"] = "none",
        alpha: float = 0.05,
        seed: int = 0,
    ):
        self.model = model
        self.method = method
        self.collapse_threshold = collapse_threshold
        self.use_wordnet = use_wordnet
        self.gradient_baseline = gradient_baseline
        self.perturbation_kinds = perturbation_kinds
        self.legacy_rules = legacy_rules
        self.calibrate_collapse = calibrate_collapse
        self.n_null = n_null
        self.quantile = quantile
        self.correction = correction
        self.alpha = alpha
        self.seed = seed

    def _resolve_method(self) -> str:
        if self.method != "auto":
            return self.method
        if getattr(self.model, "supports_gradients", False):
            return "integrated_gradients"
        return "occlusion"

    def _attribute(self, method: str, query: str, chunk: str) -> AttributionResult:
        if method == "occlusion":
            return occlusion_attribution(self.model, query, chunk)
        if method == "integrated_gradients":
            return gradient_attribution(self.model, query, chunk, baseline=self.gradient_baseline)
        if method == "maxsim":
            return maxsim_attribution(self.model, query, chunk)
        raise ValueError(f"unknown attribution method: {method!r}")

    def analyze(
        self,
        query: str,
        chunks: List[str],
        project: bool = True,
        projection_level: Literal["sentence", "token"] = "sentence",
        projection_method: str = "pca",
        detect_collapse_flags: bool = True,
    ) -> AnalysisResult:
        if not chunks:
            raise ValueError("chunks must be a non-empty list of retrieved texts")

        method = self._resolve_method()
        vecs = self.model.embed([query] + list(chunks))
        query_vec, chunk_vecs = vecs[0], vecs[1:]
        scores = cosine_similarity(query_vec, chunk_vecs)[0]
        order = np.argsort(-scores)

        collapse_cache = EmbeddingCache(self.model) if detect_collapse_flags else None

        chunk_analyses: List[ChunkAnalysis] = []
        for rank, idx in enumerate(order):
            chunk = chunks[idx]
            attribution = self._attribute(method, query, chunk)
            flags = (
                detect_collapse(
                    self.model,
                    query,
                    chunk,
                    threshold=self.collapse_threshold,
                    use_wordnet=self.use_wordnet,
                    kinds=self.perturbation_kinds,
                    legacy_rules=self.legacy_rules,
                    seed=self.seed + rank,
                    cache=collapse_cache,
                    calibrate=self.calibrate_collapse,
                    n_null=self.n_null,
                    quantile=self.quantile,
                    correction=self.correction,
                    alpha=self.alpha,
                )
                if detect_collapse_flags
                else []
            )
            chunk_analyses.append(
                ChunkAnalysis(chunk=chunk, score=float(scores[idx]), rank=rank, attribution=attribution, collapse_flags=flags)
            )

        projection = None
        if project:
            if projection_level == "token":
                projection = project_token_level(self.model, query, chunks, method=projection_method)
            else:
                projection = project_sentence_level(self.model, query, chunks, method=projection_method)

        return AnalysisResult(
            query=query,
            model_name=getattr(self.model, "name", "unknown"),
            method=method,
            chunks=chunk_analyses,
            projection=projection,
        )
