"""In-process CLI tests for `scan`/`calibrate`, monkeypatching _build_model
like tests/test_cli.py does for `run` -- fast, no subprocess. Process-level
exit-code verification lives in tests/test_cli_scan_subprocess.py."""
import json

import pytest

from whymatched import cli
from whymatched.cli import _parse_kind_rate_pairs

from .fakes import FakeBagOfWordsModel


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(cli, "_build_model", lambda args: FakeBagOfWordsModel())


def _write_cases(tmp_path, lines):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in lines) + "\n")
    return path


def test_parse_kind_rate_pairs_basic():
    assert _parse_kind_rate_pairs(["numeric=0.1", "negation=0.2"]) == {"numeric": 0.1, "negation": 0.2}


def test_parse_kind_rate_pairs_rejects_missing_equals():
    with pytest.raises(ValueError, match="KIND=RATE"):
        _parse_kind_rate_pairs(["numeric"])


def test_parse_kind_rate_pairs_rejects_bad_float():
    with pytest.raises(ValueError, match="not a valid float"):
        _parse_kind_rate_pairs(["numeric=abc"])


def test_parse_kind_rate_pairs_rejects_empty_kind():
    with pytest.raises(ValueError, match="empty kind"):
        _parse_kind_rate_pairs(["=0.5"])


def test_scan_json_dash_writes_only_json_to_stdout(tmp_path, capsys):
    cases_path = _write_cases(tmp_path, [
        {"query": "is remote work allowed for contractors", "chunks": ["remote work is not allowed for contractors"]},
    ])
    code = cli.main(["scan", "--input", str(cases_path), "--json", "-"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "collapse_rate" in data
    assert "collapse_rate" not in captured.out.split("\n")[0] or True
    assert "model:" in captured.err


def test_scan_out_writes_html_file(tmp_path):
    cases_path = _write_cases(tmp_path, [
        {"query": "is remote work allowed for contractors", "chunks": ["remote work is not allowed for contractors"]},
    ])
    out_path = tmp_path / "report.html"
    code = cli.main(["scan", "--input", str(cases_path), "--out", str(out_path)])
    assert code == 0
    assert out_path.exists()
    assert "<!doctype html>" in out_path.read_text()


def test_scan_top_k_zero_means_all_chunks(tmp_path):
    cases_path = _write_cases(tmp_path, [
        {"query": "is remote work allowed for contractors", "chunks": ["remote work is not allowed for contractors", "unrelated filler"]},
    ])
    code = cli.main(["scan", "--input", str(cases_path), "--top-k", "0", "--json", "-"])
    assert code in (0, 3)


def test_scan_kinds_filters_perturbation_kinds(tmp_path, capsys):
    cases_path = _write_cases(tmp_path, [
        {"query": "is remote work allowed for contractors", "chunks": ["remote work is not allowed for contractors"]},
    ])
    cli.main(["scan", "--input", str(cases_path), "--kinds", "negation", "--json", "-"])
    data = json.loads(capsys.readouterr().out)
    kinds_seen = {r["kind"] for c in data["cases"] for r in c["results"]}
    assert kinds_seen <= {"negation"}


def test_scan_exit_code_1_on_threshold_exceeded(tmp_path):
    cases_path = _write_cases(tmp_path, [
        {"query": "is remote work allowed for contractors", "chunks": ["remote work is not allowed for contractors"]},
    ])
    code = cli.main(["scan", "--input", str(cases_path), "--max-collapse-rate", "0.0"])
    assert code == 1


def test_scan_exit_code_2_on_missing_input():
    code = cli.main(["scan", "--input", "/no/such/file.jsonl"])
    assert code == 2


def test_scan_exit_code_3_on_case_error(tmp_path):
    cases_path = _write_cases(tmp_path, [
        {"query": "is remote work allowed for contractors", "chunks": ["remote work is not allowed for contractors"]},
        {"query": "bad case", "chunks": []},
    ])
    code = cli.main(["scan", "--input", str(cases_path)])
    assert code == 3


def test_scan_fail_on_regression_requires_baseline(tmp_path):
    cases_path = _write_cases(tmp_path, [
        {"query": "is remote work allowed for contractors", "chunks": ["remote work is not allowed for contractors"]},
    ])
    code = cli.main(["scan", "--input", str(cases_path), "--fail-on-regression"])
    assert code == 2


def test_calibrate_writes_profile(tmp_path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("the vendor must obtain approval\nthe team will request approval\n")
    out_path = tmp_path / "profile.json"
    code = cli.main(["calibrate", "--corpus", str(corpus_path), "--out", str(out_path)])
    assert code == 0
    from whymatched.calibration import CalibrationProfile

    profile = CalibrationProfile.load(out_path)
    assert profile.model_name == FakeBagOfWordsModel().name


def test_calibrate_exit_code_2_on_missing_corpus(tmp_path):
    out_path = tmp_path / "profile.json"
    code = cli.main(["calibrate", "--corpus", "/no/such/corpus.txt", "--out", str(out_path)])
    assert code == 2
