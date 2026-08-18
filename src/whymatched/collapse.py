"""Negation / near-synonym ("antonym") collapse detection, generalized on top
of the multi-category perturbation engine in :mod:`whymatched.perturbations`.

The core idea: build a counterfactual variant of the query or chunk that
*flips its meaning* (remove a negation cue, swap a word for its antonym,
scale a number, shift a date, ...), re-embed, and check whether the
similarity score barely moves. If flipping the meaning doesn't change the
score, the embedding model can't distinguish the two -- a "collapse" that
directly explains why a retriever surfaces a chunk that says something
different from what the query asked.

``detect_collapse()`` always returns ``List[CollapseFlag]`` (the pre-F1
shape) regardless of how many perturbation kinds are in play; ``kind``
just widens from 2 possible values to 8. By default (``legacy_rules=True``,
``kinds=None``) it reproduces ``0.2.2`` output byte-for-byte, bugs and all --
see ``_LEGACY_ANTONYM_SUPPLEMENT`` below and CHANGELOG.md.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

from .cache import EmbeddingCache
from .calibration import CalibrationProfile, evaluate_calibrated
from .perturbations import DEFAULT_KINDS, propose_all
from .perturbations import evaluate as _engine_evaluate
from .perturbations.antonym import AntonymPerturbation, _load_antonyms, _wordnet_antonym
from .perturbations.comparative import ComparativePerturbation
from .perturbations.entity import EntityPerturbation
from .perturbations.modal import ModalPerturbation
from .perturbations.negation import NegationPerturbation, _find_negation_spans
from .perturbations.numeric import NumericPerturbation
from .perturbations.quantifier import QuantifierPerturbation
from .perturbations.temporal import TemporalPerturbation
from .text import remove_span, replace_span, split_words

_ENGINE_KIND_CLASSES = {
    "numeric": NumericPerturbation,
    "temporal": TemporalPerturbation,
    "comparative": ComparativePerturbation,
    "quantifier": QuantifierPerturbation,
    "modal": ModalPerturbation,
    "entity": EntityPerturbation,
}

CollapseKind = Literal[
    "negation_collapse", "antonym_collapse", "numeric_collapse",
    "temporal_collapse", "comparative_collapse", "quantifier_collapse",
    "modal_collapse", "entity_collapse",
]

_LEGACY_ANTONYM_SUPPLEMENT = [
    ("increase", "decrease"), ("increase", "reduce"), ("increased", "decreased"),
    ("more", "less"), ("higher", "lower"), ("above", "below"),
    ("increase", "diminish"), ("before", "after"), ("always", "never"),
    ("must", "may"), ("mandatory", "optional"),
]


@dataclass
class CollapseFlag:
    kind: CollapseKind
    side: Literal["query", "chunk"]
    trigger: str
    counterfactual_snippet: str
    base_score: float
    counterfactual_score: float
    relative_delta: float


def _legacy_antonym_lookup() -> dict:
    lookup = dict(_load_antonyms())
    for a, b in _LEGACY_ANTONYM_SUPPLEMENT:
        lookup[a.lower()] = b
        lookup[b.lower()] = a
    return lookup


def _legacy_find_antonym_words(text: str, lookup: dict, use_wordnet: bool):
    found = []
    for word in split_words(text):
        lw = word.text.lower()
        antonym = lookup.get(lw)
        if antonym is None and use_wordnet:
            antonym = _wordnet_antonym(lw)
        if antonym:
            found.append((word, antonym))
    return found


def _detect_collapse_legacy(
    model, query: str, chunk: str, threshold: float, use_wordnet: bool,
    max_checks_per_side: int, cache: Optional[EmbeddingCache],
) -> List[CollapseFlag]:
    if cache is None:
        cache = EmbeddingCache(model)

    candidates = []

    for span in _find_negation_spans(query, legacy_rules=True)[:max_checks_per_side]:
        cf_query = remove_span(query, span) or query
        candidates.append(("query", "negation_collapse", span.text, cf_query, chunk))
    for span in _find_negation_spans(chunk, legacy_rules=True)[:max_checks_per_side]:
        cf_chunk = remove_span(chunk, span) or chunk
        candidates.append(("chunk", "negation_collapse", span.text, query, cf_chunk))

    lookup = _legacy_antonym_lookup()
    for word, antonym in _legacy_find_antonym_words(query, lookup, use_wordnet)[:max_checks_per_side]:
        cf_query = replace_span(query, word, antonym)
        candidates.append(("query", "antonym_collapse", f"{word.text} -> {antonym}", cf_query, chunk))
    for word, antonym in _legacy_find_antonym_words(chunk, lookup, use_wordnet)[:max_checks_per_side]:
        cf_chunk = replace_span(chunk, word, antonym)
        candidates.append(("chunk", "antonym_collapse", f"{word.text} -> {antonym}", query, cf_chunk))

    if not candidates:
        return []

    texts = [query, chunk] + [c[3] for c in candidates] + [c[4] for c in candidates]
    cache.prime(texts)
    base_score = cache.score(query, chunk)

    flags: List[CollapseFlag] = []
    for side, kind, trigger, cf_query, cf_chunk in candidates:
        cf_score = cache.score(cf_query, cf_chunk)
        relative_delta = abs(base_score - cf_score) / max(abs(base_score), 1e-6)
        if relative_delta < threshold:
            snippet = cf_query if side == "query" else cf_chunk
            flags.append(
                CollapseFlag(
                    kind=kind,
                    side=side,
                    trigger=trigger,
                    counterfactual_snippet=snippet[:160],
                    base_score=base_score,
                    counterfactual_score=cf_score,
                    relative_delta=relative_delta,
                )
            )
    return flags


def _build_engine_perturbations(
    resolved_kinds: List[str], use_wordnet: bool, legacy_rules: bool, max_checks_per_side: int
) -> list:
    perturbations = []
    for k in resolved_kinds:
        if k == "negation":
            perturbations.append(NegationPerturbation(legacy_rules=legacy_rules, max_per_text=max_checks_per_side))
        elif k == "antonym":
            perturbations.append(AntonymPerturbation(use_wordnet=use_wordnet, max_per_text=max_checks_per_side))
        elif k in _ENGINE_KIND_CLASSES:
            instance = _ENGINE_KIND_CLASSES[k](max_per_text=max_checks_per_side)
            if instance.available():
                perturbations.append(instance)
    return perturbations


def _detect_collapse_calibrated(
    model, query: str, chunk: str, threshold: float, use_wordnet: bool,
    max_checks_per_side: int, kinds: Optional[Sequence[str]], legacy_rules: bool,
    n_null: int, quantile: float, correction: Literal["none", "bh"], alpha: float,
    seed: int, cache: Optional[EmbeddingCache], profile: Optional[CalibrationProfile],
) -> List[CollapseFlag]:
    resolved_kinds = list(kinds) if kinds is not None else list(DEFAULT_KINDS)
    perturbations = _build_engine_perturbations(resolved_kinds, use_wordnet, legacy_rules, max_checks_per_side)

    if cache is None:
        cache = EmbeddingCache(model)
    rng = random.Random(seed)
    proposals = propose_all(query, chunk, perturbations, max_per_side_per_kind=max_checks_per_side, rng=rng)
    results, _calibration_by_side = evaluate_calibrated(
        cache, query, chunk, proposals, threshold=threshold, n_null=n_null,
        quantile=quantile, correction=correction, alpha=alpha, seed=seed, profile=profile,
    )
    return [
        CollapseFlag(
            kind=f"{r.kind}_collapse",  # type: ignore[arg-type]
            side=r.side,
            trigger=r.trigger,
            counterfactual_snippet=r.counterfactual_snippet,
            base_score=r.base_score,
            counterfactual_score=r.counterfactual_score,
            relative_delta=r.relative_delta,
        )
        for r in results
        if r.is_collapse
    ]


def detect_collapse(
    model,
    query: str,
    chunk: str,
    threshold: float = 0.10,
    use_wordnet: bool = False,
    max_checks_per_side: int = 6,
    kinds: Optional[Sequence[str]] = None,
    legacy_rules: bool = True,
    calibrate: bool = False,
    n_null: int = 50,
    quantile: float = 0.10,
    correction: Literal["none", "bh"] = "none",
    alpha: float = 0.05,
    seed: int = 0,
    cache: Optional[EmbeddingCache] = None,
    profile: Optional[CalibrationProfile] = None,
) -> List[CollapseFlag]:
    """Flag counterfactuals that barely move the similarity score (relative
    change below ``threshold``) despite a meaning-changing edit.

    With ``kinds=None, legacy_rules=True`` (the default), reproduces
    ``0.2.2``'s negation/antonym-only output exactly. Set ``legacy_rules=
    False`` and/or pass ``kinds`` to use the full multi-category engine
    (negation, antonym, numeric, temporal, comparative, quantifier, modal).

    ``calibrate=True`` replaces the fixed ``threshold`` rule with a
    calibrated quantile rule (see :mod:`whymatched.calibration`):
    ``n_null``/``quantile``/``correction``/``alpha``/``profile`` configure
    it. Power users who want the full statistical detail (Q, its CI, ratio,
    z-score, the orthographic/synonym/deletion triple) should call
    :func:`whymatched.calibration.evaluate_calibrated` directly -- this
    function always returns ``List[CollapseFlag]``, never the richer type.
    """
    if calibrate:
        return _detect_collapse_calibrated(
            model, query, chunk, threshold, use_wordnet, max_checks_per_side,
            kinds, legacy_rules, n_null, quantile, correction, alpha, seed, cache, profile,
        )

    if kinds is None and legacy_rules:
        return _detect_collapse_legacy(model, query, chunk, threshold, use_wordnet, max_checks_per_side, cache)

    resolved_kinds = list(kinds) if kinds is not None else list(DEFAULT_KINDS)
    perturbations = _build_engine_perturbations(resolved_kinds, use_wordnet, legacy_rules, max_checks_per_side)

    if cache is None:
        cache = EmbeddingCache(model)
    rng = random.Random(seed)
    proposals = propose_all(query, chunk, perturbations, max_per_side_per_kind=max_checks_per_side, rng=rng)
    results = _engine_evaluate(cache, query, chunk, proposals)

    flags: List[CollapseFlag] = []
    for r in results:
        if r.relative_delta < threshold:
            flags.append(
                CollapseFlag(
                    kind=f"{r.kind}_collapse",  # type: ignore[arg-type]
                    side=r.side,
                    trigger=r.trigger,
                    counterfactual_snippet=r.counterfactual_snippet,
                    base_score=r.base_score,
                    counterfactual_score=r.counterfactual_score,
                    relative_delta=r.relative_delta,
                )
            )
    return flags
