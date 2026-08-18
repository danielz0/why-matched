"""CI-style assertions on a BatchReport, meant to gate a build the way
you'd gate on a test-coverage threshold. The assertion message IS the
product: it always names the threshold, the actual rate, the denominator
(with skipped/errored called out separately), the decision rule, and the
top-3 offenders, so a CI failure is legible without re-running the scan.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .batch import BatchReport, CaseReport


def _rule_summary(report: BatchReport) -> str:
    cfg = report.config
    if cfg.get("calibrate"):
        return f"rule=quantile q={cfg.get('quantile')} n_null={cfg.get('n_null')}"
    return f"rule=threshold threshold={cfg.get('threshold')}"


def _best_collapse(case: CaseReport):
    """The single most-severe collapse in a case (lowest q_value-or-
    p_value), matching severity()'s own "best_sig" logic, so the offender
    table shows the actual worst trigger, not an arbitrary first one."""
    return min(
        case.collapses,
        key=lambda r: (r.q_value if r.q_value is not None else r.p_value) or 1.0,
    )


def _offender_rows(cases: Sequence[CaseReport], n: int = 3) -> list:
    worst = sorted([c for c in cases if c.collapses], key=lambda c: c.severity, reverse=True)[:n]
    rows = []
    for c in worst:
        top = _best_collapse(c)
        q = f"{top.null_quantile:.3f}" if top.null_quantile is not None else "n/a"
        ratio = f"{top.ratio:.2f}" if top.ratio is not None else "n/a"
        rows.append(
            f'    [{c.case_id}] {top.kind:<10} "{top.trigger}"  '
            f"{top.base_score:.3f} -> {top.counterfactual_score:.3f}  Q={q} ratio={ratio}"
        )
    return rows


def _denominator_line(report: BatchReport, kind: Optional[str]) -> str:
    n, d = report._collapse_counts(kind)
    return (
        f"  ({n} of {d} applicable cases; {report.n_skipped} skipped, "
        f"{report.n_errored} errored; {_rule_summary(report)})"
    )


def assert_collapse_rate(report: BatchReport, *, max_rate: float, kind: Optional[str] = None) -> None:
    """Raise AssertionError if report.collapse_rate(kind) exceeds max_rate."""
    actual = report.collapse_rate(kind=kind)
    if actual <= max_rate:
        return
    label = f"collapse_rate(kind={kind})" if kind else "collapse_rate"
    lines = [
        f"{label}={actual:.2f} exceeds max_rate={max_rate:.2f}",
        _denominator_line(report, kind),
        "  worst offenders:",
    ]
    lines.extend(_offender_rows(report.cases))
    raise AssertionError("\n".join(lines))


def assert_no_collapse(report: BatchReport, *, kinds: Optional[Sequence[str]] = None) -> None:
    """Raise AssertionError if any case collapses (whole report, or
    restricted to `kinds` when given)."""
    if kinds is None:
        assert_collapse_rate(report, max_rate=0.0)
        return
    for k in kinds:
        assert_collapse_rate(report, max_rate=0.0, kind=k)


def assert_not_worse_than(report: BatchReport, baseline: BatchReport, *, tolerance: float = 0.0) -> None:
    """Raise AssertionError if report.collapse_rate() regresses beyond
    baseline.collapse_rate() + tolerance. This is what turns a batch scan
    into a guardrail rather than a one-off report -- commit `baseline` and
    compare future scans against it."""
    actual = report.collapse_rate()
    base = baseline.collapse_rate()
    if actual <= base + tolerance:
        return
    lines = [
        f"collapse_rate={actual:.2f} regressed beyond baseline={base:.2f} (tolerance={tolerance:.2f})",
        _denominator_line(report, None),
        "  worst offenders:",
    ]
    lines.extend(_offender_rows(report.cases))
    raise AssertionError("\n".join(lines))
