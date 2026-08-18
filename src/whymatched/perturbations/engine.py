"""Owns the embed-and-compare loop and resolves span conflicts between
perturbation kinds that both want to edit the same substring (e.g. "2020" is
claimable by both the numeric bare-integer rule and the temporal year rule)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from ..cache import EmbeddingCache
from ..text import Span
from .base import Candidate, Perturbation, PerturbationKind

PRIORITY: Dict[str, int] = {
    "temporal": 40,
    "numeric": 60,
    "modal": 70,
    "quantifier": 75,
    "comparative": 80,
    "negation": 85,
    "antonym": 90,
    "entity": 95,
}


@dataclass
class PerturbationResult:
    kind: PerturbationKind
    side: Literal["query", "chunk"]
    trigger: str
    counterfactual_snippet: str
    base_score: float
    counterfactual_score: float
    relative_delta: float
    null_quantile: Optional[float] = None
    null_quantile_ci: Optional[Tuple[float, float]] = None
    ratio: Optional[float] = None
    z_score: Optional[float] = None
    p_value: Optional[float] = None
    q_value: Optional[float] = None
    decision_rule: str = "threshold"
    calibration_status: str = "ok"
    is_collapse: bool = False


def _spans_overlap(a: Span, b: Span) -> bool:
    return a.start < b.end and b.start < a.end


def _arbitrate(all_candidates: List[Candidate]) -> List[Candidate]:
    """Sort by (priority, span.start, kind); drop a candidate whose span
    overlaps an already-claimed span of a DIFFERENT kind. Same-kind
    candidates on the same span are all kept (alternatives, not a conflict)."""
    ordered = sorted(all_candidates, key=lambda c: (PRIORITY.get(c.kind, 100), c.span.start, c.kind))
    claimed: List[Tuple[Span, str]] = []
    accepted: List[Candidate] = []
    for cand in ordered:
        conflict = any(
            claimed_kind != cand.kind and _spans_overlap(claimed_span, cand.span)
            for claimed_span, claimed_kind in claimed
        )
        if conflict:
            continue
        accepted.append(cand)
        claimed.append((cand.span, cand.kind))
    return accepted


def propose_all(
    query: str,
    chunk: str,
    perturbations: Sequence[Perturbation],
    *,
    max_per_side_per_kind: int = 4,
    rng: random.Random,
) -> List[Tuple[str, Candidate]]:
    """Returns [(side, candidate), ...]. Drops candidates whose apply() would
    yield empty/whitespace-only text. Arbitrates span conflicts per side
    independently, then caps per (side, kind) *after* arbitration so a
    lower-priority kind that survived isn't starved by one that lost its
    span elsewhere."""
    out: List[Tuple[str, Candidate]] = []
    for side, text in (("query", query), ("chunk", chunk)):
        raw: List[Candidate] = []
        for p in perturbations:
            if not p.available():
                continue
            raw.extend(p.propose(text, rng=rng))
        raw = [c for c in raw if c.apply(text).strip()]
        arbitrated = _arbitrate(raw)
        per_kind_count: Dict[str, int] = {}
        for c in arbitrated:
            n = per_kind_count.get(c.kind, 0)
            if n >= max_per_side_per_kind:
                continue
            per_kind_count[c.kind] = n + 1
            out.append((side, c))
    return out


def evaluate(
    cache: EmbeddingCache,
    query: str,
    chunk: str,
    proposals: List[Tuple[str, Candidate]],
) -> List[PerturbationResult]:
    texts_to_prime = [query, chunk]
    applied: List[Tuple[str, Candidate, str, str]] = []
    for side, cand in proposals:
        cf_query = cand.apply(query) if side == "query" else query
        cf_chunk = cand.apply(chunk) if side == "chunk" else chunk
        applied.append((side, cand, cf_query, cf_chunk))
        texts_to_prime.append(cf_query if side == "query" else cf_chunk)

    cache.prime(texts_to_prime)
    base_score = cache.score(query, chunk)

    results: List[PerturbationResult] = []
    for side, cand, cf_query, cf_chunk in applied:
        cf_score = cache.score(cf_query, cf_chunk)
        relative_delta = abs(base_score - cf_score) / max(abs(base_score), 1e-6)
        snippet = (cf_query if side == "query" else cf_chunk)[:160]
        results.append(
            PerturbationResult(
                kind=cand.kind,
                side=side,
                trigger=cand.trigger,
                counterfactual_snippet=snippet,
                base_score=base_score,
                counterfactual_score=cf_score,
                relative_delta=relative_delta,
            )
        )
    return results
