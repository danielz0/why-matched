"""Process-level exit-code verification for `whymatched scan`, per the F3
acceptance criterion's literal wording ("subprocess tests"). Uses
WHYMATCHED_TEST_FAKE_MODEL so these stay fully offline and fast, in the
same no-torch CI job as everything else."""
import json
import os
import subprocess
import sys

_ENV = {**os.environ, "WHYMATCHED_TEST_FAKE_MODEL": "1"}


def _run_cli(args):
    return subprocess.run(
        [sys.executable, "-m", "whymatched.cli", *args],
        env=_ENV, capture_output=True, text=True, timeout=60,
    )


def _write_cases(path, lines):
    path.write_text("\n".join(json.dumps(c) for c in lines) + "\n")


def test_scan_exit_0_thresholds_met(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    _write_cases(cases_path, [
        {"query": "is remote work allowed for contractors", "chunks": ["remote work is not allowed for contractors"]},
    ])
    result = _run_cli(["scan", "--input", str(cases_path), "--max-collapse-rate", "1.0"])
    assert result.returncode == 0, result.stderr


def test_scan_exit_1_threshold_exceeded(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    _write_cases(cases_path, [
        {"query": "is remote work allowed for contractors", "chunks": ["remote work is not allowed for contractors"]},
    ])
    result = _run_cli(["scan", "--input", str(cases_path), "--max-collapse-rate", "0.0"])
    assert result.returncode == 1
    assert "exceeds max_rate" in result.stderr


def test_scan_exit_2_usage_error_bad_input_path():
    result = _run_cli(["scan", "--input", "/no/such/file.jsonl"])
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_scan_exit_2_malformed_jsonl(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text('{"query": "q1"}\n')
    result = _run_cli(["scan", "--input", str(cases_path)])
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_scan_exit_3_case_errored(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    _write_cases(cases_path, [
        {"query": "is remote work allowed for contractors", "chunks": ["remote work is not allowed for contractors"]},
        {"query": "bad case", "chunks": []},
    ])
    result = _run_cli(["scan", "--input", str(cases_path)])
    assert result.returncode == 3


def test_scan_json_stdout_deterministic_across_subprocess_runs(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    _write_cases(cases_path, [
        {"query": "is remote work allowed for contractors", "chunks": ["remote work is not allowed for contractors"]},
    ])
    result_a = _run_cli(["scan", "--input", str(cases_path), "--json", "-", "--seed", "0"])
    result_b = _run_cli(["scan", "--input", str(cases_path), "--json", "-", "--seed", "0"])
    assert result_a.returncode == result_b.returncode
    data_a = json.loads(result_a.stdout)
    data_b = json.loads(result_b.stdout)
    data_a.pop("created_at")
    data_b.pop("created_at")
    assert data_a == data_b


def test_scan_json_stdout_is_clean_json_only(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    _write_cases(cases_path, [
        {"query": "is remote work allowed for contractors", "chunks": ["remote work is not allowed for contractors"]},
    ])
    result = _run_cli(["scan", "--input", str(cases_path), "--json", "-"])
    json.loads(result.stdout)
    assert "collapse_rate" in result.stderr
