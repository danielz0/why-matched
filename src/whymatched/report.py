"""Self-contained HTML report renderer. Pure stdlib — no plotting library
required — so `pip install whymatched` alone is enough to produce a report
once you have an AnalysisResult from Debugger.analyze()."""
from __future__ import annotations

import html
from typing import List, Sequence

from .batch import BatchReport, CaseReport
from .core import AnalysisResult, ChunkAnalysis
from .attribution.base import TokenScore
from .projection import ProjectedPoint

_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5;
       background: #ffffff; color: #111827; }
@media (prefers-color-scheme: dark) {
  body { background: #0f1115; color: #e5e7eb; }
  .chunk { background: #171a21 !important; border-color: #2a2e37 !important; }
  .flag { background: #3a2a1a !important; border-color: #7a5a2a !important; }
  svg { background: #1a1d24 !important; }
}
h1 { font-size: 1.3rem; }
h2 { font-size: 1.05rem; margin-bottom: 0.25rem; }
.meta { color: #6b7280; font-size: 0.85rem; margin-bottom: 1.5rem; }
.chunk { border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.25rem; background: #fafafa; }
.score { font-weight: 600; }
.tok { display: inline-block; padding: 0.08rem 0.15rem; border-radius: 4px; margin: 0.03rem; }
.flag { background: #fff7ed; border: 1px solid #fdba74; border-radius: 8px; padding: 0.5rem 0.75rem; margin: 0.4rem 0; font-size: 0.88rem; }
.flag-kind { font-weight: 600; }
.section-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: #6b7280; margin: 0.6rem 0 0.2rem; }
.legend { font-size: 0.8rem; color: #6b7280; margin: 0.3rem 0 1rem; }
"""


def _color_for_weight(weight: float, max_abs: float) -> str:
    if max_abs <= 0:
        return "rgba(128,128,128,0.15)"
    ratio = max(-1.0, min(1.0, weight / max_abs))
    if ratio >= 0:
        alpha = 0.12 + 0.68 * ratio
        return f"rgba(34,139,87,{alpha:.2f})"
    alpha = 0.12 + 0.68 * (-ratio)
    return f"rgba(200,60,60,{alpha:.2f})"


def _render_tokens(tokens: List[TokenScore]) -> str:
    if not tokens:
        return "<em>(no tokens)</em>"
    max_abs = max((abs(t.weight) for t in tokens), default=1.0) or 1.0
    spans = []
    for t in tokens:
        color = _color_for_weight(t.weight, max_abs)
        label = html.escape(t.token)
        title = f"{label}: {t.weight:+.3f}"
        spans.append(f'<span class="tok" style="background:{color}" title="{title}">{label}</span>')
    return " ".join(spans)


def _svg_projection(points: Sequence[ProjectedPoint], width: int = 640, height: int = 420, pad: int = 30) -> str:
    if not points:
        return ""
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    spanx = (maxx - minx) or 1.0
    spany = (maxy - miny) or 1.0

    def sx(x: float) -> float:
        return pad + (x - minx) / spanx * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - miny) / spany * (height - 2 * pad)

    color_map = {"query": "#2563eb", "chunk": "#16a34a", "query_token": "#93c5fd", "chunk_token": "#86efac"}
    elems = []
    for p in points:
        cx, cy = sx(p.x), sy(p.y)
        color = color_map.get(p.kind, "#888888")
        r = 8 if p.kind in ("query", "chunk") else 3.5
        label = html.escape(p.label[:60])
        elems.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" fill-opacity="0.85" stroke="#00000022"><title>{label}</title></circle>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="border-radius:8px;">'
        + "".join(elems)
        + "</svg>"
    )


def _render_chunk(chunk_analysis: ChunkAnalysis) -> str:
    c = chunk_analysis
    flags_html = ""
    if c.collapse_flags:
        rows = []
        for f in c.collapse_flags:
            rows.append(
                f'<div class="flag"><span class="flag-kind">{html.escape(f.kind)}</span> '
                f'({html.escape(f.side)}) trigger=<code>{html.escape(f.trigger)}</code> — '
                f"score {f.base_score:.3f} → {f.counterfactual_score:.3f} "
                f"({f.relative_delta*100:.1f}% relative change)<br>"
                f'<span style="opacity:0.8">counterfactual: "{html.escape(f.counterfactual_snippet)}"</span></div>'
            )
        flags_html = "".join(rows)

    return f"""
<div class="chunk">
  <h2>#{c.rank + 1} &middot; <span class="score">{c.score:.3f}</span></h2>
  <div class="section-label">Chunk (highlighted by {html.escape(c.attribution.method)} attribution)</div>
  <div>{_render_tokens(c.attribution.chunk_tokens)}</div>
  <div class="section-label">Query tokens, attributed against this chunk</div>
  <div>{_render_tokens(c.attribution.query_tokens)}</div>
  {flags_html}
</div>
"""


def render_html(result: AnalysisResult, title: str = "why-matched report") -> str:
    chunks_html = "".join(_render_chunk(c) for c in result.chunks)
    projection_html = ""
    if result.projection:
        projection_html = f"""
<div class="section-label">Query vs. chunks (reduced-space projection)</div>
<div class="legend">blue = query &middot; green = chunk (hover for text)</div>
{_svg_projection(result.projection)}
"""
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>why-matched</h1>
<div class="meta">model: <code>{html.escape(result.model_name)}</code> &middot; attribution method: <code>{html.escape(result.method)}</code></div>
<div class="section-label">Query</div>
<p>{html.escape(result.query)}</p>
{projection_html}
{chunks_html}
</body>
</html>
"""


def write_html(result: AnalysisResult, path: str, title: str = "why-matched report") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(result, title=title))


_BATCH_CSS_EXTRA = """
.batch-summary { border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem 1.25rem;
                  margin-bottom: 1.5rem; background: #fafafa; }
.rate { font-size: 1.4rem; font-weight: 700; }
table.kind-table { border-collapse: collapse; font-size: 0.85rem; margin: 0.5rem 0; }
table.kind-table td, table.kind-table th { padding: 0.2rem 0.6rem; text-align: left; border-bottom: 1px solid #e5e7eb; }
.triple { font-size: 0.8rem; color: #6b7280; margin-top: 0.3rem; }
.error { background: #fef2f2; border: 1px solid #fca5a5; border-radius: 8px; padding: 0.5rem 0.75rem; margin: 0.4rem 0; font-size: 0.88rem; color: #991b1b; }
@media (prefers-color-scheme: dark) {
  .batch-summary { background: #171a21 !important; border-color: #2a2e37 !important; }
  table.kind-table td, table.kind-table th { border-color: #2a2e37 !important; }
  .error { background: #3a1a1a !important; border-color: #7a2a2a !important; color: #fca5a5 !important; }
}
"""


def _render_kind_table(report: BatchReport) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v*100:.1f}%</td></tr>"
        for k, v in sorted(report.rate_by_kind().items())
    )
    if not rows:
        return ""
    return f'<table class="kind-table"><tr><th>kind</th><th>collapse rate</th></tr>{rows}</table>'


def _render_triple(calibration) -> str:
    if not calibration:
        return ""
    parts = []
    for side, nc in sorted(calibration.items()):
        means = {
            name: (sum(vals) / len(vals) if vals else float("nan"))
            for name, vals in nc.generator_deltas.items()
        }
        summary = ", ".join(f"{name}={v:.3f}" for name, v in sorted(means.items()))
        parts.append(f"{side}: {summary}")
    return f'<div class="triple">null deltas (mean): {html.escape(" | ".join(parts))}</div>'


def _render_case_report(case: CaseReport) -> str:
    if case.error is not None:
        return (
            f'<div class="chunk"><h2>[{html.escape(case.case_id)}] '
            f'<span class="score">error</span></h2>'
            f"<p>{html.escape(case.query)}</p>"
            f'<div class="error">case errored during evaluation: {html.escape(case.error)}</div>'
            f"</div>"
        )

    rows = []
    for r in case.collapses:
        q = f"{r.null_quantile:.3f}" if r.null_quantile is not None else "n/a"
        ratio = f"{r.ratio:.2f}" if r.ratio is not None else "n/a"
        rows.append(
            f'<div class="flag"><span class="flag-kind">{html.escape(r.kind)}</span> '
            f'({html.escape(r.side)}) trigger=<code>{html.escape(r.trigger)}</code> — '
            f"score {r.base_score:.3f} → {r.counterfactual_score:.3f}, Q={q}, ratio={ratio}<br>"
            f'<span style="opacity:0.8">counterfactual: "{html.escape(r.counterfactual_snippet)}"</span></div>'
        )
    return (
        f'<div class="chunk"><h2>[{html.escape(case.case_id)}] '
        f'<span class="score">{case.top_score:.3f}</span></h2>'
        f"<p>{html.escape(case.query)}</p>"
        f"{''.join(rows)}"
        f"{_render_triple(case.calibration)}"
        f"</div>"
    )


def render_batch_html(report: BatchReport, path: str, worst_n: int = 20) -> None:
    """Write a self-contained batch HTML report: summary header (overall
    rate, per-kind table, skipped/errored counts), then the worst-N cases
    expanded with trigger, base vs. counterfactual score, Q, ratio, and the
    orthographic/synonym/deletion triple. Reuses `_color_for_weight`-style
    CSS conventions from the single-analysis report."""
    rate = report.collapse_rate()
    cand_rate = report.candidate_collapse_rate()
    worst_html = "".join(_render_case_report(c) for c in report.worst(worst_n))
    html_out = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>whymatched batch report</title>
<style>{_CSS}{_BATCH_CSS_EXTRA}</style>
</head>
<body>
<h1>why-matched — batch scan</h1>
<div class="meta">model: <code>{html.escape(report.model_name)}</code> &middot;
whymatched {html.escape(report.whymatched_version)} &middot; {html.escape(report.created_at)}</div>
<div class="batch-summary">
  <div class="rate">collapse_rate = {rate*100:.1f}%</div>
  <div>candidate_collapse_rate = {cand_rate*100:.1f}%</div>
  <div>{len(report.cases)} cases scanned &middot; {report.n_skipped} skipped &middot; {report.n_errored} errored</div>
  {_render_kind_table(report)}
</div>
<div class="section-label">Worst {min(worst_n, len(report.cases))} cases</div>
{worst_html}
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_out)
