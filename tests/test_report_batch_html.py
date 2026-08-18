import html

from whymatched.batch import EvalCase, scan
from whymatched.report import render_batch_html

from .fakes import FakeBagOfWordsModel

LONG_CHUNK = (
    "The vendor must obtain approval before it can purchase additional equipment. "
    "The team will request approval and then purchase the equipment once approval is obtained. "
    "Approval must be verified and confirmed by the finance department before any purchase is completed. "
    "The finance department will examine each purchase request and verify that approval was obtained."
)


def _scanned_report():
    cases = [
        EvalCase(
            query="is remote work allowed for contractors",
            chunks=["remote work is not allowed for contractors under this policy"],
        ),
        EvalCase(query="is approval required for this purchase", chunks=[LONG_CHUNK]),
    ]
    model = FakeBagOfWordsModel()
    return scan(model, cases, seed=0)


def test_render_batch_html_writes_expected_sections(tmp_path):
    report = _scanned_report()
    path = tmp_path / "batch.html"
    render_batch_html(report, str(path))
    content = path.read_text()

    assert report.model_name in content
    assert "collapse_rate" in content
    assert "candidate_collapse_rate" in content
    assert "kind-table" in content
    assert "skipped" in content and "errored" in content
    worst = report.worst(1)[0]
    if worst.collapses:
        assert html.escape(worst.collapses[0].trigger) in content
    assert "null deltas" in content


def test_render_batch_html_escapes_query_text(tmp_path):
    from whymatched.batch import BatchReport, CaseReport

    case = CaseReport(case_id="a", query="<script>alert(1)</script>", n_chunks=1, top_score=0.5, results=[])
    report = BatchReport(
        model_name="m", whymatched_version="0.1", created_at="now",
        config={"calibrate": True}, cases=[case], n_skipped=1, n_errored=0,
    )
    path = tmp_path / "batch.html"
    render_batch_html(report, str(path))
    content = path.read_text()
    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;" in content


def test_render_batch_html_worst_n_limits_expanded_cases(tmp_path):
    report = _scanned_report()
    path = tmp_path / "batch.html"
    render_batch_html(report, str(path), worst_n=1)
    content = path.read_text()
    assert "Worst 1 cases" in content


def test_render_batch_html_shows_errored_case_distinctly(tmp_path):
    from whymatched.batch import BatchReport, CaseReport

    case = CaseReport(
        case_id="errcase", query="bad query", n_chunks=1, top_score=float("nan"),
        results=[], error="boom: something broke",
    )
    report = BatchReport(
        model_name="m", whymatched_version="0.1", created_at="now",
        config={}, cases=[case], n_skipped=0, n_errored=1,
    )
    path = tmp_path / "batch.html"
    render_batch_html(report, str(path))
    content = path.read_text()
    assert "boom: something broke" in content
    assert 'class="error"' in content
