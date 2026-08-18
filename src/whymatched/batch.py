"""Corpus/batch-mode scanning (F3): run collapse detection across many
(query, chunks) pairs at once and gate a build on a collapse-rate
threshold, the way you'd gate on a test-coverage threshold."""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.metadata
import json
import random
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from .cache import EmbeddingCache
from .calibration import CalibrationProfile, NullCalibration, evaluate_calibrated
from .collapse import _build_engine_perturbations
from .perturbations import DEFAULT_KINDS, PerturbationResult, propose_all
from .perturbations import evaluate as _engine_evaluate
from .utils import cosine_similarity


def _current_version() -> str:
    try:
        return importlib.metadata.version("whymatched")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@dataclass
class EvalCase:
    query: str
    chunks: List[str]
    case_id: Optional[str] = None
    expected_chunk_ids: Optional[List[int]] = None
    tags: List[str] = field(default_factory=list)

    def resolved_id(self) -> str:
        """Default sha1(query)[:8]. Two cases sharing the same query text
        collide on this id -- both are still scanned and both appear in
        BatchReport.cases; case_id is a display label, not an enforced
        unique key. Set case_id explicitly to avoid the collision."""
        return self.case_id or hashlib.sha1(self.query.encode("utf-8")).hexdigest()[:8]


@dataclass
class CaseReport:
    case_id: str
    query: str
    n_chunks: int
    top_score: float
    results: List[PerturbationResult]
    error: Optional[str] = None
    calibration: Optional[Dict[str, NullCalibration]] = None

    @property
    def collapses(self) -> List[PerturbationResult]:
        return [r for r in self.results if r.is_collapse]

    @property
    def severity(self) -> Tuple[int, float, float]:
        """Descending sort key (use sorted(..., reverse=True)). E5 fix:
        typed as a 3-tuple, not float; empty-collapse sentinel avoids the
        min()-on-empty-list ValueError."""
        c = self.collapses
        if not c:
            return (0, 1.0, float("inf"))
        best_sig = min((r.q_value if r.q_value is not None else r.p_value) or 1.0 for r in c)
        best_ratio = min(r.ratio if r.ratio is not None else float("inf") for r in c)
        return (len(c), -best_sig, -best_ratio)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "n_chunks": self.n_chunks,
            "top_score": self.top_score,
            "results": [asdict(r) for r in self.results],
            "error": self.error,
            "calibration": (
                {side: asdict(nc) for side, nc in self.calibration.items()}
                if self.calibration is not None
                else None
            ),
        }


@dataclass
class BatchReport:
    model_name: str
    whymatched_version: str
    created_at: str
    config: dict
    cases: List[CaseReport]
    n_skipped: int
    n_errored: int

    def _applicable_cases(self) -> List[CaseReport]:
        """A case is applicable iff it has no error AND >=1 candidate."""
        return [c for c in self.cases if c.error is None and c.results]

    def _collapse_counts(self, kind: Optional[str] = None) -> Tuple[int, int]:
        """(n_cases_with_collapse, n_applicable_cases) as exact integers --
        shared by collapse_rate() and testing.py's message builder so the
        printed counts are never float-rounding-derived."""
        applicable = self._applicable_cases()
        if kind is not None:
            applicable = [c for c in applicable if any(r.kind == kind for r in c.results)]

        def _has_collapse(c: CaseReport) -> bool:
            cs = c.collapses
            if kind is not None:
                cs = [r for r in cs if r.kind == kind]
            return len(cs) > 0

        n_collapsed = sum(1 for c in applicable if _has_collapse(c))
        return n_collapsed, len(applicable)

    def collapse_rate(self, kind: Optional[str] = None) -> float:
        n, d = self._collapse_counts(kind)
        return (n / d) if d else 0.0

    def candidate_collapse_rate(self, kind: Optional[str] = None) -> float:
        total = 0
        collapsed = 0
        for c in self.cases:
            if c.error is not None:
                continue
            for r in c.results:
                if kind is not None and r.kind != kind:
                    continue
                total += 1
                if r.is_collapse:
                    collapsed += 1
        return (collapsed / total) if total else 0.0

    def rate_by_kind(self) -> Dict[str, float]:
        kinds = sorted({r.kind for c in self.cases if c.error is None for r in c.results})
        return {k: self.collapse_rate(kind=k) for k in kinds}

    def worst(self, n: int = 20) -> List[CaseReport]:
        return sorted(self.cases, key=lambda c: c.severity, reverse=True)[:n]

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "whymatched_version": self.whymatched_version,
            "created_at": self.created_at,
            "config": self.config,
            "n_cases": len(self.cases),
            "n_skipped": self.n_skipped,
            "n_errored": self.n_errored,
            "collapse_rate": self.collapse_rate(),
            "candidate_collapse_rate": self.candidate_collapse_rate(),
            "rate_by_kind": self.rate_by_kind(),
            "cases": [c.to_dict() for c in self.cases],
        }


_SCAN_THRESHOLD = 0.10
_SCAN_USE_WORDNET = False
_SCAN_LEGACY_RULES = False
_SCAN_MAX_CHECKS_PER_SIDE = 6

_SEED_CASE_STRIDE = 1000


def _evaluate_case(
    case: EvalCase,
    cache: EmbeddingCache,
    perturbations: list,
    *,
    calibrate: bool,
    n_null: int,
    quantile: float,
    correction: Literal["none", "bh"],
    alpha: float,
    profile: Optional[CalibrationProfile],
    top_k_chunks: Optional[int],
    seed: int,
    case_index: int,
) -> CaseReport:
    case_id = case.resolved_id()
    try:
        if not case.chunks:
            raise ValueError("EvalCase.chunks must be non-empty")

        cache.prime([case.query] + list(case.chunks))
        query_vec = cache.get(case.query)
        chunk_vecs = np.stack([cache.get(c) for c in case.chunks])
        scores = cosine_similarity(query_vec, chunk_vecs)[0]
        order = np.argsort(-scores)
        top_score = float(scores[order[0]])

        k = len(order) if top_k_chunks is None else min(top_k_chunks, len(order))
        selected = order[:k]

        all_results: List[PerturbationResult] = []
        calibration_for_case: Optional[Dict[str, NullCalibration]] = None

        for rank_pos, idx in enumerate(selected):
            chunk = case.chunks[int(idx)]
            chunk_seed = seed + case_index * _SEED_CASE_STRIDE + rank_pos
            rng = random.Random(chunk_seed)
            proposals = propose_all(
                case.query, chunk, perturbations,
                max_per_side_per_kind=_SCAN_MAX_CHECKS_PER_SIDE, rng=rng,
            )
            if not proposals:
                continue
            if calibrate:
                results, calib = evaluate_calibrated(
                    cache, case.query, chunk, proposals, threshold=_SCAN_THRESHOLD,
                    n_null=n_null, quantile=quantile, correction=correction,
                    alpha=alpha, seed=chunk_seed, profile=profile,
                )
            else:
                results = _engine_evaluate(cache, case.query, chunk, proposals)
                for r in results:
                    r.is_collapse = r.relative_delta < _SCAN_THRESHOLD
                    r.decision_rule = "threshold"
                calib = None
            all_results.extend(results)
            if rank_pos == 0:
                calibration_for_case = calib

        return CaseReport(
            case_id=case_id, query=case.query, n_chunks=len(case.chunks),
            top_score=top_score, results=all_results, error=None,
            calibration=calibration_for_case,
        )
    except Exception as exc:
        return CaseReport(
            case_id=case_id, query=case.query, n_chunks=len(case.chunks),
            top_score=float("nan"), results=[], error=str(exc), calibration=None,
        )


def scan(
    model,
    cases: Sequence[EvalCase],
    *,
    kinds: Optional[Sequence[str]] = None,
    calibrate: bool = True,
    n_null: int = 50,
    quantile: float = 0.10,
    correction: Literal["none", "bh"] = "none",
    alpha: float = 0.05,
    profile: Optional[CalibrationProfile] = None,
    top_k_chunks: Optional[int] = 3,
    max_workers: int = 1,
    seed: int = 0,
    progress: Optional[Callable[[int, int], None]] = None,
) -> BatchReport:
    """Corpus/batch-mode scan (F3). One EmbeddingCache for the whole scan
    when max_workers<=1 (proves cross-case cache reuse -- boilerplate
    queries/chunks repeat across cases). max_workers>1 gives each worker
    its OWN cache (EmbeddingCache is documented not thread-safe; per spec,
    intended for network-bound API models -- keep max_workers=1 for local
    torch models to avoid GPU oversubscription). A case raising anywhere
    in its evaluation never aborts the scan -- caught into
    CaseReport(error=...), counted in n_errored; a case with zero
    applicable candidates (no exception) is counted in n_skipped instead.
    """
    resolved_kinds = list(kinds) if kinds is not None else list(DEFAULT_KINDS)
    perturbations = _build_engine_perturbations(
        resolved_kinds, _SCAN_USE_WORDNET, _SCAN_LEGACY_RULES, _SCAN_MAX_CHECKS_PER_SIDE
    )

    def _run(case_index: int, case: EvalCase, cache: EmbeddingCache) -> CaseReport:
        return _evaluate_case(
            case, cache, perturbations, calibrate=calibrate, n_null=n_null,
            quantile=quantile, correction=correction, alpha=alpha, profile=profile,
            top_k_chunks=top_k_chunks, seed=seed, case_index=case_index,
        )

    reports: List[Optional[CaseReport]] = [None] * len(cases)

    if max_workers <= 1:
        cache = EmbeddingCache(model)
        for i, case in enumerate(cases):
            reports[i] = _run(i, case, cache)
            if progress is not None:
                progress(i + 1, len(cases))
    else:
        thread_local = threading.local()

        def _thread_cache() -> EmbeddingCache:
            cache = getattr(thread_local, "cache", None)
            if cache is None:
                cache = EmbeddingCache(model)
                thread_local.cache = cache
            return cache

        def _worker(item):
            i, case = item
            return i, _run(i, case, _thread_cache())

        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, report in ex.map(_worker, enumerate(cases)):
                reports[i] = report
                done += 1
                if progress is not None:
                    progress(done, len(cases))

    final_reports: List[CaseReport] = [r for r in reports if r is not None]
    n_errored = sum(1 for r in final_reports if r.error is not None)
    n_skipped = sum(1 for r in final_reports if r.error is None and not r.results)

    config = {
        "kinds": resolved_kinds, "calibrate": calibrate, "n_null": n_null,
        "quantile": quantile, "correction": correction, "alpha": alpha,
        "seed": seed, "top_k_chunks": top_k_chunks, "max_workers": max_workers,
        "threshold": _SCAN_THRESHOLD, "legacy_rules": _SCAN_LEGACY_RULES,
    }

    return BatchReport(
        model_name=getattr(model, "name", "unknown"),
        whymatched_version=_current_version(),
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config, cases=final_reports, n_skipped=n_skipped, n_errored=n_errored,
    )


class CaseLoadError(ValueError):
    """Raised by load_cases() on malformed input. Message always begins
    with '{path}:{lineno}:' (JSONL) or '{path}[{index}]:' (JSON array) so
    the offending case is locatable without re-parsing by hand."""


_REQUIRED_KEYS = {"query", "chunks"}
_ALLOWED_KEYS = {"query", "chunks", "case_id", "expected_chunk_ids", "tags"}


def _case_from_dict(d, where: str) -> EvalCase:
    if not isinstance(d, dict):
        raise CaseLoadError(f"{where}: expected a JSON object, got {type(d).__name__}")
    missing = _REQUIRED_KEYS - d.keys()
    if missing:
        raise CaseLoadError(f"{where}: missing required key(s): {sorted(missing)}")
    extra = d.keys() - _ALLOWED_KEYS
    if extra:
        raise CaseLoadError(f"{where}: unknown key(s): {sorted(extra)}")
    if not isinstance(d["query"], str) or not d["query"]:
        raise CaseLoadError(f"{where}: 'query' must be a non-empty string")
    if not isinstance(d["chunks"], list) or not all(isinstance(c, str) for c in d["chunks"]):
        raise CaseLoadError(f"{where}: 'chunks' must be a list of strings")
    return EvalCase(
        query=d["query"], chunks=list(d["chunks"]), case_id=d.get("case_id"),
        expected_chunk_ids=d.get("expected_chunk_ids"), tags=list(d.get("tags", [])),
    )


def load_cases(path: Union[str, "Path"]) -> List[EvalCase]:
    """JSONL (preferred, one EvalCase JSON object per non-blank line) or a
    top-level JSON array, auto-detected. `chunks` may be an empty list --
    non-emptiness is a scan()-time concern (surfaced as a per-case error,
    exit code 3), not a load-time schema failure (exit code 2)."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()

    if stripped.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise CaseLoadError(f"{path}: invalid JSON array: {e}") from e
        if not isinstance(data, list):
            raise CaseLoadError(f"{path}: top-level JSON must be an array of case objects")
        if not data:
            raise CaseLoadError(f"{path}: no cases found (empty array)")
        return [_case_from_dict(item, f"{path}[{i}]") for i, item in enumerate(data)]

    cases: List[EvalCase] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            raise CaseLoadError(f"{path}:{lineno}: invalid JSON: {e}") from e
        cases.append(_case_from_dict(d, f"{path}:{lineno}"))
    if not cases:
        raise CaseLoadError(f"{path}: no cases found (empty file)")
    return cases
