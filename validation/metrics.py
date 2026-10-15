"""The measurement (WS-A / A2): precision/recall/F1 of whymatched's
collapse detector against externally-labelled ground truth.

Definition (from instructions/TEAM_IMPLEMENTATION_GUIDE.md, §A2):

    A pair is a true positive when the model genuinely fails on it (it
    ranks doc_b above doc_a for query_a, or the scores are within noise)
    AND whymatched raises a collapse flag on that pair.

    A false positive is a flag raised on a pair the model handles
    correctly.

No precision/recall/confusion-matrix logic exists anywhere in
src/whymatched/ -- this module is new, built entirely on top of the
existing detect_collapse()/EmbeddingCache contracts rather than
reimplementing anything.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import whymatched
from whymatched.cache import EmbeddingCache
from whymatched.calibration import CalibrationProfile
from whymatched.collapse import detect_collapse

from .datasets import LabelledPair


@dataclass
class ConfusionCounts:
    """Exact integer counts, never float-derived -- mirrors BatchReport's
    own rate properties (src/whymatched/batch.py), zero-division-guarded
    the same way."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return (self.tp / d) if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return (self.tp / d) if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


@dataclass
class DetectorMetrics:
    """One DetectorMetrics per model. Evaluating multiple models means
    calling evaluate_detector() once per model and collecting the results
    (mirrors benchmarks/negation_benchmark.py's per-`--model` invocation)."""

    model_name: str
    source: str  # e.g. "nevir", or "mixed" if pairs came from more than one source
    overall: ConfusionCounts
    by_category: Dict[str, ConfusionCounts] = field(default_factory=dict)
    n_pairs: int = 0
    n_directions: int = 0
    config: dict = field(default_factory=dict)
    whymatched_version: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    def summary(self) -> str:
        """Formatted text table: model / category / tp,fp,fn,tn / P / R / F1."""
        lines = [
            f"model: {self.model_name}  source: {self.source}  "
            f"n_pairs: {self.n_pairs}  n_directions: {self.n_directions}",
            "",
            f"{'category':14} {'tp':>5} {'fp':>5} {'fn':>5} {'tn':>5} {'precision':>10} {'recall':>10} {'f1':>10}",
        ]

        def _row(label: str, c: ConfusionCounts) -> str:
            return (
                f"{label:14} {c.tp:>5} {c.fp:>5} {c.fn:>5} {c.tn:>5} "
                f"{c.precision:>10.3f} {c.recall:>10.3f} {c.f1:>10.3f}"
            )

        lines.append(_row("overall", self.overall))
        for category in sorted(self.by_category):
            lines.append(_row(category, self.by_category[category]))
        return "\n".join(lines)


def evaluate_detector(
    model,
    pairs: Sequence[LabelledPair],
    *,
    noise_eps: float = 0.01,
    threshold: float = 0.10,
    use_wordnet: bool = False,
    max_checks_per_side: int = 6,
    kinds: Optional[Sequence[str]] = None,
    legacy_rules: bool = True,
    calibrate: bool = False,
    n_null: int = 50,
    quantile: float = 0.10,
    correction: str = "none",
    alpha: float = 0.05,
    seed: int = 0,
    profile: Optional[CalibrationProfile] = None,
) -> DetectorMetrics:
    """precision/recall/F1, plus the 2x2 confusion matrix, of
    whymatched's collapse detector against `pairs`.

    Each LabelledPair yields two evaluation *directions*:
      - (query_a, doc_a) with alternative doc_b
      - (query_b, doc_b) with alternative doc_a

    Ground truth per direction (a genuine model failure):
        s_correct = cache.score(query, doc)
        s_other   = cache.score(query, other_doc)
        true_failure = (s_other > s_correct) or (abs(s_correct - s_other) <= noise_eps)

    Detector's call: flagged = bool(detect_collapse(model, query, doc, ...))
    -- i.e. whymatched raised *a* collapse flag on that pair, matching the
    guide's TP/FP definition literally (not restricted to a same-kind flag).

    TP = true_failure & flagged   FP = !true_failure & flagged
    FN = true_failure & !flagged  TN = !true_failure & !flagged

    Every embedding goes through one shared EmbeddingCache for the whole
    call -- primes all four texts per pair up front for the ground-truth
    scoring, then passes that same cache into every detect_collapse() call
    so its counterfactual embeds land in the same batch-dedup store. Never
    calls model.embed() directly.

    `profile`: an optional pre-fit `CalibrationProfile` (see
    `whymatched.calibration.fit_pair_profile`), passed straight through to
    every `detect_collapse(..., profile=profile)` call when `calibrate=
    True`. Lets short single-sentence pairs (which often can't produce
    enough per-call SynonymNull candidates to calibrate on their own --
    see `validation/REPORT.md`) borrow a corpus-level null distribution
    instead of silently falling back to the uncalibrated threshold rule.
    Fit it on a split disjoint from `pairs` to avoid leaking the evaluated
    pairs into their own null reference.
    """
    cache = EmbeddingCache(model)
    all_texts: List[str] = []
    for p in pairs:
        all_texts.extend([p.query_a, p.query_b, p.doc_a, p.doc_b])
    cache.prime(all_texts)

    overall = ConfusionCounts()
    by_category: Dict[str, ConfusionCounts] = {}

    directions = []
    for p in pairs:
        directions.append((p, p.query_a, p.doc_a, p.doc_b))
        directions.append((p, p.query_b, p.doc_b, p.doc_a))

    detect_kwargs = dict(
        threshold=threshold,
        use_wordnet=use_wordnet,
        max_checks_per_side=max_checks_per_side,
        kinds=kinds,
        legacy_rules=legacy_rules,
        calibrate=calibrate,
        n_null=n_null,
        quantile=quantile,
        correction=correction,
        alpha=alpha,
        seed=seed,
        cache=cache,
        profile=profile,
    )

    for pair, query, doc, other_doc in directions:
        s_correct = cache.score(query, doc)
        s_other = cache.score(query, other_doc)
        true_failure = (s_other > s_correct) or (abs(s_correct - s_other) <= noise_eps)

        flags = detect_collapse(model, query, doc, **detect_kwargs)
        flagged = bool(flags)

        bucket = by_category.setdefault(pair.category, ConfusionCounts())
        for counts in (overall, bucket):
            if true_failure and flagged:
                counts.tp += 1
            elif not true_failure and flagged:
                counts.fp += 1
            elif true_failure and not flagged:
                counts.fn += 1
            else:
                counts.tn += 1

    config = dict(
        noise_eps=noise_eps,
        threshold=threshold,
        use_wordnet=use_wordnet,
        max_checks_per_side=max_checks_per_side,
        kinds=list(kinds) if kinds is not None else None,
        legacy_rules=legacy_rules,
        calibrate=calibrate,
        n_null=n_null,
        quantile=quantile,
        correction=correction,
        alpha=alpha,
        seed=seed,
        profile_model_name=profile.model_name if profile is not None else None,
        profile_n_samples=profile.n_samples if profile is not None else None,
        profile_generator=profile.generator if profile is not None else None,
        profile_seed=profile.seed if profile is not None else None,
    )
    sources = {p.source for p in pairs}
    return DetectorMetrics(
        model_name=getattr(model, "name", "unknown"),
        source=next(iter(sources)) if len(sources) == 1 else "mixed",
        overall=overall,
        by_category=by_category,
        n_pairs=len(pairs),
        n_directions=len(directions),
        config=config,
        whymatched_version=whymatched.__version__,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
