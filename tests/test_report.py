"""HTML report rendering tests. Uses FakeBagOfWordsModel so this runs
without torch/transformers -- report.py is pure stdlib and should be
exercised on every Python version in CI, not just when torch is installed."""
from whymatched import Debugger
from whymatched.report import render_html, write_html

from .fakes import FakeBagOfWordsModel


def _analyze():
    model = FakeBagOfWordsModel()
    dbg = Debugger(model)
    return dbg.analyze(
        "is remote work allowed for contractors",
        [
            "remote work is not allowed for contractors under this policy",
            "totally unrelated cooking recipe",
        ],
    )


def test_render_html_contains_query_and_chunks():
    result = _analyze()
    out = render_html(result)
    assert "<html>" in out
    # the query is rendered verbatim as a whole string
    assert "is remote work allowed for contractors" in out
    # chunk text is rendered as individual per-token spans, not the raw
    # sentence, so check for a couple of its words instead of the phrase
    assert ">remote<" in out
    assert ">allowed<" in out
    assert result.model_name in out
    assert result.method in out


def test_render_html_includes_collapse_flags():
    result = _analyze()
    out = render_html(result)
    assert any(c.collapse_flags for c in result.chunks), "expected at least one collapse flag for this fixture"
    assert "negation_collapse" in out or "antonym_collapse" in out


def test_render_html_includes_projection_svg_by_default():
    result = _analyze()
    out = render_html(result)
    assert "<svg" in out


def test_render_html_omits_projection_svg_when_not_requested():
    model = FakeBagOfWordsModel()
    dbg = Debugger(model)
    result = dbg.analyze("a b c", ["a b c"], project=False)
    out = render_html(result)
    assert "<svg" not in out


def test_render_html_escapes_html_in_chunk_text():
    model = FakeBagOfWordsModel()
    dbg = Debugger(model)
    result = dbg.analyze("<script>alert(1)</script>", ["<b>bold chunk</b>"], project=False)
    out = render_html(result)
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out
    assert "<b>bold chunk</b>" not in out


def test_write_html_writes_file(tmp_path):
    result = _analyze()
    out_path = tmp_path / "report.html"
    write_html(result, str(out_path))
    content = out_path.read_text(encoding="utf-8")
    assert content.startswith("<!doctype html>")
    assert "is remote work allowed for contractors" in content
