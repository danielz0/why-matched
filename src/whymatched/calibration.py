"""Statistical calibration layer (F2). Decide whether a perturbation
candidate's relative_delta is a real collapse by comparing it against a
distribution of relative_deltas from meaning-preserving edits ("nulls") of
the SAME text, rather than one global fixed threshold.

The null generator is the scientific crux: if the nulls aren't genuinely
meaning-preserving, every number downstream is wrong. SynonymNull is
primary and drives the decision rule; OrthographicNull and DeletionReference
are computed and reported for context but never blended into the decision.

Note on data/synonym_pairs.json: this starter list has NOT yet had the
second-pair-of-eyes curation review the spec asks for -- a bad entry
inflates the null and hides true collapses, the most damaging failure mode
for this tool's credibility. Treat it as a starting point, not a final list.
"""
from __future__ import annotations

import importlib.metadata
import importlib.resources
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .cache import EmbeddingCache
from .perturbations._dictionary import build_lookup, compile_phrase_regex
from .perturbations.base import Candidate
from .perturbations.engine import PerturbationResult
from .text import Span, split_words


class NullGenerator:
    """Protocol every null/reference generator implements. Not a Protocol
    class (duck-typed, like whymatched.perturbations.base.Perturbation) so
    concrete generators stay plain classes."""

    name: str

    def propose(self, text: str, *, n: int, rng: random.Random) -> List[Candidate]:
        raise NotImplementedError


def _sample_candidates(candidates: List[Candidate], n: int, rng: random.Random) -> List[Candidate]:
    """Occurrence-based candidates, capped at n. Fewer than n existing is
    expected (not a bug) -- it's what drives the insufficient-nulls
    fallback for short texts. More than n existing draws a seeded random
    subset, re-sorted by span position for readable, deterministic output."""
    if len(candidates) <= n:
        return candidates
    chosen = rng.sample(candidates, n)
    return sorted(chosen, key=lambda c: c.span.start)


_SYNONYM_LOOKUP: Optional[Dict[str, str]] = None


def _load_synonyms() -> Dict[str, str]:
    global _SYNONYM_LOOKUP
    if _SYNONYM_LOOKUP is None:
        raw = importlib.resources.files("whymatched.data").joinpath("synonym_pairs.json").read_text()
        _SYNONYM_LOOKUP = build_lookup(json.loads(raw))
    return _SYNONYM_LOOKUP


class SynonymNull:
    """Curated data/synonym_pairs.json swaps -- the primary null. Both
    inflected forms required per entry, or skip (see the JSON file's own
    curation note). Meaning-preserving by construction, assuming the
    curation held. Candidates are occurrence-based: a word repeated 10x in
    a paragraph yields 10 distinct candidates, which is how test fixtures
    (and real corpora) reach n_null=20-50 from ordinary prose."""

    name = "synonym"

    def __init__(self) -> None:
        self._lookup = _load_synonyms()
        self._regex = compile_phrase_regex(self._lookup.keys())

    def propose(self, text: str, *, n: int, rng: random.Random) -> List[Candidate]:
        out: List[Candidate] = []
        for m in self._regex.finditer(text):
            surface = m.group(0)
            repl = self._lookup.get(surface.lower())
            if repl is None:
                continue
            span = Span(surface, m.start(), m.end())
            out.append(
                Candidate(
                    kind="synonym_null",
                    span=span,
                    replacement=repl,
                    trigger=f"{surface} -> {repl}",
                    meaning_preserving=True,
                    rationale="curated synonym substitution",
                )
            )
        return _sample_candidates(out, n, rng)


_DOUBLE_SPACE_RE = re.compile(r" {2,}")
_SINGLE_SPACE_RE = re.compile(r"(?<=\S) (?=\S)")
_QUOTE_RE = re.compile(r'"')
_SENTENCE_END_RE = re.compile(r"\.(?= [A-Z])")
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s?%")
_AND_RE = re.compile(r"\band\b", re.IGNORECASE)


def _ortho(m: "re.Match", replacement: str, trigger: str) -> Candidate:
    return Candidate(
        kind="orthographic_null",
        span=Span(m.group(0), m.start(), m.end()),
        replacement=replacement,
        trigger=trigger,
        meaning_preserving=True,
    )


class OrthographicNull:
    """Semantically inert surface transformations -- the noise floor.
    Whitespace, quote style, sentence-boundary spacing, %-spellout, and
    and/&. Reported for context, never used to drive the decision rule."""

    name = "orthographic"

    def propose(self, text: str, *, n: int, rng: random.Random) -> List[Candidate]:
        out: List[Candidate] = []

        for m in _DOUBLE_SPACE_RE.finditer(text):
            out.append(_ortho(m, " ", "double-space -> single"))
        for m in _SINGLE_SPACE_RE.finditer(text):
            out.append(_ortho(m, "  ", "single-space -> double"))

        for m in _QUOTE_RE.finditer(text):
            out.append(_ortho(m, "'", "\" -> '"))

        for m in _SENTENCE_END_RE.finditer(text):
            out.append(_ortho(m, ".  ", "sentence boundary spacing"))

        stripped = text.rstrip()
        if stripped:
            end = len(stripped)
            if stripped.endswith("."):
                out.append(
                    Candidate(
                        kind="orthographic_null", span=Span(".", end - 1, end), replacement="",
                        trigger="remove trailing period", meaning_preserving=True,
                    )
                )
            else:
                out.append(
                    Candidate(
                        kind="orthographic_null", span=Span("", end, end), replacement=".",
                        trigger="add trailing period", meaning_preserving=True,
                    )
                )

        for m in _PERCENT_RE.finditer(text):
            surface = m.group(0)
            spelled = surface.rstrip("% ").rstrip() + " percent"
            out.append(
                Candidate(
                    kind="orthographic_null", span=Span(surface, m.start(), m.end()), replacement=spelled,
                    trigger=f"{surface} -> {spelled}", meaning_preserving=True,
                )
            )

        for m in _AND_RE.finditer(text):
            out.append(_ortho(m, "&", "and -> &"))

        return _sample_candidates(out, n, rng)


_STOPWORDS: Optional[frozenset] = None


def _stopwords() -> frozenset:
    global _STOPWORDS
    if _STOPWORDS is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        _STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
    return _STOPWORDS


class DeletionReference:
    """Delete one content word -- NOT a null, an upper reference showing
    what a genuine content change looks like for this model/text. n content-
    word occurrences -> up to n distinct single-word-deletion candidates,
    one per eligible occurrence, capped and seed-sampled like the nulls."""

    name = "deletion_reference"

    def propose(self, text: str, *, n: int, rng: random.Random) -> List[Candidate]:
        stop = _stopwords()
        out: List[Candidate] = []
        for word in split_words(text):
            if not word.text.isalpha():
                continue
            if len(word.text) < 2:
                continue
            if word.text.lower() in stop:
                continue
            out.append(
                Candidate(
                    kind="deletion_reference", span=word, replacement="",
                    trigger=f"delete: {word.text}", meaning_preserving=False,
                    rationale="reference: genuine content deletion, not a null",
                )
            )
        return _sample_candidates(out, n, rng)


def _quantile(deltas: np.ndarray, q: float) -> float:
    return float(np.quantile(deltas, q, method="linear"))


def _bootstrap_quantile_ci(
    deltas: np.ndarray, q: float, *, rng: np.random.Generator,
    n_resamples: int = 1000, ci: float = 0.90,
) -> Tuple[float, float]:
    """Deterministic under a given numpy Generator. 1000 resamples, 90% CI
    by default."""
    m = len(deltas)
    idx = rng.integers(0, m, size=(n_resamples, m))
    resample_q = np.quantile(deltas[idx], q, axis=1, method="linear")
    lo = float(np.quantile(resample_q, (1 - ci) / 2, method="linear"))
    hi = float(np.quantile(resample_q, 1 - (1 - ci) / 2, method="linear"))
    return (lo, hi)


def _ratio(d_obs: float, deltas: np.ndarray) -> float:
    return d_obs / max(float(np.median(deltas)), 1e-9)


def _z_score(d_obs: float, deltas: np.ndarray) -> float:
    return (d_obs - float(np.mean(deltas))) / max(float(np.std(deltas, ddof=0)), 1e-9)


def _p_value(d_obs: float, deltas: np.ndarray) -> float:
    """Add-one-corrected, one-sided LEFT tail: a candidate that moves the
    score a lot is never a collapse. p >= 1/(n+1); never zero."""
    n = len(deltas)
    hits = int(np.sum(deltas <= d_obs))
    return (1 + hits) / (n + 1)


def _min_null(alpha: float) -> int:
    """max(20, ceil(1/alpha) - 1). The -1e-9 epsilon guards against float
    division overshoot nudging an exact integer result up by one before
    ceil (same float hazard class as spec's E4 for /10 division)."""
    return max(20, math.ceil(1 / alpha - 1e-9) - 1)


def _bh_required_n_null(k: int, alpha: float) -> int:
    """min_p = 1/(n+1); a single true collapse in a family of k candidates
    needs min_p * k <= alpha, i.e. n_null >= k/alpha - 1."""
    return math.ceil(k / alpha - 1e-9) - 1


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Sort ascending -> p*n/rank -> enforce monotonicity via a
    reverse-cummin -> clip to [0,1]. Returns q-values in the ORIGINAL
    (unsorted) order of p_values."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    raw_q = ranked * n / np.arange(1, n + 1)
    monotone_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    q = np.clip(monotone_q, 0.0, 1.0)
    out = np.empty(n)
    out[order] = q
    return out


@dataclass
class NullCalibration:
    """Per-side calibration summary. Not part of PerturbationResult (F1-
    frozen) -- this is the architectural answer to "report the [orthographic
    / synonym / deletion] triple separately, never averaged." F3's future
    batch HTML report is the expected consumer of this type."""

    side: Literal["query", "chunk"]
    calibration_status: Literal["ok", "insufficient_nulls"]
    quantile: float
    null_quantile: Optional[float]
    null_quantile_ci: Optional[Tuple[float, float]]
    n_null_requested: int
    min_null: int
    generator_deltas: Dict[str, List[float]] = field(default_factory=dict)
    n_null_effective: Dict[str, int] = field(default_factory=dict)


_DEFAULT_SYNONYM_NULL: Optional["SynonymNull"] = None
_DEFAULT_ORTHOGRAPHIC_NULL: Optional["OrthographicNull"] = None
_DEFAULT_DELETION_REFERENCE: Optional["DeletionReference"] = None


def _default_generators() -> List[object]:
    global _DEFAULT_SYNONYM_NULL, _DEFAULT_ORTHOGRAPHIC_NULL
    if _DEFAULT_SYNONYM_NULL is None:
        _DEFAULT_SYNONYM_NULL = SynonymNull()
    if _DEFAULT_ORTHOGRAPHIC_NULL is None:
        _DEFAULT_ORTHOGRAPHIC_NULL = OrthographicNull()
    return [_DEFAULT_SYNONYM_NULL, _DEFAULT_ORTHOGRAPHIC_NULL]


def _default_deletion_reference() -> "DeletionReference":
    global _DEFAULT_DELETION_REFERENCE
    if _DEFAULT_DELETION_REFERENCE is None:
        _DEFAULT_DELETION_REFERENCE = DeletionReference()
    return _DEFAULT_DELETION_REFERENCE


def evaluate_calibrated(
    cache: EmbeddingCache,
    query: str,
    chunk: str,
    proposals: List[Tuple[str, Candidate]],
    *,
    threshold: float = 0.10,
    generators: Optional[Sequence[object]] = None,
    n_null: int = 50,
    quantile: float = 0.10,
    correction: Literal["none", "bh"] = "none",
    alpha: float = 0.05,
    seed: int = 0,
    min_null: Optional[int] = None,
    profile: Optional["CalibrationProfile"] = None,
) -> Tuple[List[PerturbationResult], Dict[str, NullCalibration]]:
    """Calibrated collapse decision: compare each real candidate's
    relative_delta against a per-side distribution of relative_deltas from
    meaning-preserving edits, instead of one fixed global threshold.

    Returns (results, calibration_by_side). `results` has the same shape as
    `perturbations.engine.evaluate()`'s output, with the F2 statistical
    fields populated. `calibration_by_side` carries the full per-generator
    delta breakdown (the orthographic/synonym/deletion triple) for callers
    that want it -- detect_collapse() discards it to keep its
    List[CollapseFlag] contract stable.
    """
    if correction == "bh" and profile is not None:
        raise ValueError(
            "correction='bh' requires per-call p-values, which profile-driven "
            "calibration cannot produce; pass profile=None or correction='none'"
        )

    gens: List[object] = list(generators) if generators is not None else _default_generators()
    if not any(getattr(g, "name", None) == "synonym" for g in gens):
        raise ValueError(
            "generators must include a NullGenerator named 'synonym' (SynonymNull) "
            "-- it is the primary null that drives the decision rule"
        )
    if not any(getattr(g, "name", None) == "deletion_reference" for g in gens):
        gens = gens + [_default_deletion_reference()]

    min_null_eff = min_null if min_null is not None else _min_null(alpha)

    if profile is not None and profile.model_name != cache.model_name:
        raise ValueError(
            f"profile was fit for model_name={profile.model_name!r}, "
            f"but this cache's model is {cache.model_name!r}"
        )

    if not proposals:
        return [], {}

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    sides = {"query": query, "chunk": chunk}
    null_candidates: Dict[str, Dict[str, List[Candidate]]] = {"query": {}, "chunk": {}}

    to_prime: List[str] = [query, chunk]

    if profile is None:
        for side, text in sides.items():
            for gen in gens:
                cands = gen.propose(text, n=n_null, rng=py_rng)
                cands = [c for c in cands if c.apply(text).strip()]
                null_candidates[side][gen.name] = cands
                to_prime.extend(c.apply(text) for c in cands)

    for side, cand in proposals:
        cf_query = cand.apply(query) if side == "query" else query
        cf_chunk = cand.apply(chunk) if side == "chunk" else chunk
        to_prime.append(cf_query if side == "query" else cf_chunk)

    cache.prime(to_prime)
    base_score = cache.score(query, chunk)

    calibration_by_side: Dict[str, NullCalibration] = {}
    synonym_deltas_by_side: Dict[str, np.ndarray] = {}

    for side, text in sides.items():
        other_text = chunk if side == "query" else query

        def _score_for(cf_text: str, side=side, other_text=other_text) -> float:
            return cache.score(cf_text, other_text) if side == "query" else cache.score(other_text, cf_text)

        if profile is None:
            generator_deltas: Dict[str, List[float]] = {}
            n_null_effective: Dict[str, int] = {}
            for gen_name, cands in null_candidates[side].items():
                deltas = [
                    abs(base_score - _score_for(c.apply(text))) / max(abs(base_score), 1e-6) for c in cands
                ]
                generator_deltas[gen_name] = deltas
                n_null_effective[gen_name] = len(deltas)

            synonym_deltas = np.array(generator_deltas.get("synonym", []))
            if len(synonym_deltas) >= min_null_eff:
                Q = _quantile(synonym_deltas, quantile)
                CI = _bootstrap_quantile_ci(synonym_deltas, quantile, rng=np_rng)
                calibration_by_side[side] = NullCalibration(
                    side=side, calibration_status="ok", quantile=quantile,
                    null_quantile=Q, null_quantile_ci=CI, n_null_requested=n_null,
                    min_null=min_null_eff, generator_deltas=generator_deltas,
                    n_null_effective=n_null_effective,
                )
                synonym_deltas_by_side[side] = synonym_deltas
            else:
                calibration_by_side[side] = NullCalibration(
                    side=side, calibration_status="insufficient_nulls", quantile=quantile,
                    null_quantile=None, null_quantile_ci=None, n_null_requested=n_null,
                    min_null=min_null_eff, generator_deltas=generator_deltas,
                    n_null_effective=n_null_effective,
                )
        else:
            if profile.n_samples >= min_null_eff:
                Q = _interp_profile_quantile(profile, quantile)
                calibration_by_side[side] = NullCalibration(
                    side=side, calibration_status="ok", quantile=quantile,
                    null_quantile=Q, null_quantile_ci=None, n_null_requested=n_null,
                    min_null=min_null_eff, generator_deltas={},
                    n_null_effective={"synonym": profile.n_samples},
                )
            else:
                calibration_by_side[side] = NullCalibration(
                    side=side, calibration_status="insufficient_nulls", quantile=quantile,
                    null_quantile=None, null_quantile_ci=None, n_null_requested=n_null,
                    min_null=min_null_eff, generator_deltas={},
                    n_null_effective={"synonym": profile.n_samples},
                )

    results: List[PerturbationResult] = []
    for side, cand in proposals:
        cf_query = cand.apply(query) if side == "query" else query
        cf_chunk = cand.apply(chunk) if side == "chunk" else chunk
        cf_score = cache.score(cf_query, cf_chunk)
        relative_delta = abs(base_score - cf_score) / max(abs(base_score), 1e-6)
        snippet = (cf_query if side == "query" else cf_chunk)[:160]

        side_calibration = calibration_by_side[side]
        if side_calibration.calibration_status == "insufficient_nulls":
            results.append(
                PerturbationResult(
                    kind=cand.kind, side=side, trigger=cand.trigger,
                    counterfactual_snippet=snippet, base_score=base_score,
                    counterfactual_score=cf_score, relative_delta=relative_delta,
                    decision_rule="threshold", calibration_status="insufficient_nulls",
                    is_collapse=relative_delta < threshold,
                )
            )
            continue

        Q = side_calibration.null_quantile
        if profile is None:
            synonym_deltas = synonym_deltas_by_side[side]
            ratio = _ratio(relative_delta, synonym_deltas)
            z_score = _z_score(relative_delta, synonym_deltas)
            p_value = _p_value(relative_delta, synonym_deltas)
        else:
            p50 = profile.null_delta_quantiles["p50"]
            ratio = relative_delta / max(p50, 1e-9)
            z_score = None
            p_value = None

        results.append(
            PerturbationResult(
                kind=cand.kind, side=side, trigger=cand.trigger,
                counterfactual_snippet=snippet, base_score=base_score,
                counterfactual_score=cf_score, relative_delta=relative_delta,
                null_quantile=Q, null_quantile_ci=side_calibration.null_quantile_ci,
                ratio=ratio, z_score=z_score, p_value=p_value,
                decision_rule="quantile", calibration_status="ok",
                is_collapse=relative_delta < Q,
            )
        )

    if correction == "bh":
        by_kind: Dict[str, List[PerturbationResult]] = {}
        for r in results:
            if r.calibration_status == "ok":
                by_kind.setdefault(r.kind, []).append(r)
        for kind, group in by_kind.items():
            k = len(group)
            required = _bh_required_n_null(k, alpha)
            if n_null < required:
                raise ValueError(
                    f"BH correction infeasible for kind={kind!r}: {k} candidates at "
                    f"alpha={alpha} requires n_null >= {required} (got n_null={n_null}). "
                    "Increase n_null or use correction='none'."
                )
            p_values = np.array([r.p_value for r in group])
            q_values = _benjamini_hochberg(p_values)
            for r, q in zip(group, q_values):
                r.q_value = float(q)
                r.decision_rule = "bh"
                r.is_collapse = bool(q <= alpha)

    return results, calibration_by_side


_PROFILE_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.95)
_PROFILE_QUANTILE_KEYS = ("p05", "p10", "p25", "p50", "p75", "p95")


def _current_version() -> str:
    try:
        return importlib.metadata.version("whymatched")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@dataclass
class CalibrationProfile:
    """Pre-computed null-delta quantiles for a model, fit once against a
    corpus via fit_profile() and reused across evaluate_calibrated(profile=
    ...) calls instead of resampling nulls per call. Applies to both query
    and chunk sides equally -- it characterizes the model's null-delta
    behavior, not a query/chunk role.
    """

    model_name: str
    n_samples: int
    null_delta_quantiles: Dict[str, float]
    generator: str
    seed: int
    whymatched_version: str
    created_at: str

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path) -> "CalibrationProfile":
        data = json.loads(Path(path).read_text())
        profile = cls(**data)
        current = _current_version()
        if profile.whymatched_version != current:
            raise ValueError(
                f"profile was fit with whymatched=={profile.whymatched_version}, "
                f"current install is =={current}; refit to avoid a stale null generator"
            )
        return profile


def _interp_profile_quantile(profile: CalibrationProfile, q: float) -> float:
    xp = list(_PROFILE_QUANTILES)
    fp = [profile.null_delta_quantiles[k] for k in _PROFILE_QUANTILE_KEYS]
    return float(np.interp(q, xp, fp))


def fit_profile(
    model,
    corpus_texts: Sequence[str],
    *,
    generator: Optional[object] = None,
    n_null: int = 200,
    seed: int = 0,
    cache: Optional[EmbeddingCache] = None,
) -> CalibrationProfile:
    """Corpus-level null calibration. Statistic: self-similarity drop under
    one null edit -- 1 - cosine(embed(text), embed(null_edit(text))) -- an
    APPROXIMATION of the live per-pair relative_delta metric used elsewhere
    in this module (there is no query/chunk pair structure here to compute
    a true relative_delta against). Good enough for the structural round-
    trip contract CalibrationProfile promises; not proven numerically
    equivalent to evaluate_calibrated()'s live per-call deltas.
    """
    gen = generator or SynonymNull()
    cache = cache or EmbeddingCache(model)
    rng = random.Random(seed)

    all_texts = list(corpus_texts)
    per_text_candidates = {t: gen.propose(t, n=n_null, rng=rng) for t in all_texts}

    to_prime = list(all_texts)
    for t, cands in per_text_candidates.items():
        to_prime.extend(c.apply(t) for c in cands)
    cache.prime(to_prime)

    deltas: List[float] = []
    for t, cands in per_text_candidates.items():
        base = cache.get(t)
        base_norm = float(np.linalg.norm(base))
        for c in cands:
            cf = cache.get(c.apply(t))
            cf_norm = float(np.linalg.norm(cf))
            cos = float(np.dot(base, cf)) / (base_norm * cf_norm + 1e-12)
            deltas.append(1.0 - cos)

    arr = np.array(deltas) if deltas else np.array([0.0])
    quantiles = {k: _quantile(arr, q) for k, q in zip(_PROFILE_QUANTILE_KEYS, _PROFILE_QUANTILES)}
    return CalibrationProfile(
        model_name=getattr(model, "name", "unknown"),
        n_samples=len(deltas),
        null_delta_quantiles=quantiles,
        generator=gen.name,
        seed=seed,
        whymatched_version=_current_version(),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
