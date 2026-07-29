"""CLI tests. _build_model is monkeypatched to return FakeBagOfWordsModel so
these run end-to-end without torch, network access, or an API key -- the
one thing they don't cover is the real local/API model construction paths,
which is exercised by test_local_model_integration.py and manual API-key
testing instead."""
import json

import pytest

from whymatched import cli

from .fakes import FakeBagOfWordsModel


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(cli, "_build_model", lambda args: FakeBagOfWordsModel())


def test_run_json_output(capsys):
    cli.main(
        [
            "run",
            "--query", "is remote work allowed for contractors",
            "--chunk", "remote work is not allowed for contractors",
            "--chunk", "totally unrelated cooking recipe",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["query"] == "is remote work allowed for contractors"
    assert len(data["chunks"]) == 2
    assert data["chunks"][0]["score"] >= data["chunks"][1]["score"]


def test_run_writes_html_report(tmp_path, capsys):
    out_path = tmp_path / "report.html"
    cli.main(
        [
            "run",
            "--query", "is remote work allowed for contractors",
            "--chunk", "remote work is not allowed for contractors",
            "--out", str(out_path),
        ]
    )
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "<!doctype html>" in content
    printed = capsys.readouterr().out
    assert str(out_path) in printed


def test_run_no_collapse_and_no_projection(capsys):
    cli.main(
        [
            "run",
            "--query", "a b c",
            "--chunk", "a b c",
            "--no-collapse",
            "--no-projection",
            "--json",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    assert data["projection"] is None
    assert data["chunks"][0]["collapse_flags"] == []


def test_run_input_json_file(tmp_path, capsys):
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"query": "a b c", "chunks": ["a b c", "x y z"]}), encoding="utf-8"
    )
    cli.main(["run", "--input", str(input_path), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["query"] == "a b c"
    assert len(data["chunks"]) == 2


def test_run_chunks_file_as_plain_lines(tmp_path, capsys):
    chunks_path = tmp_path / "chunks.txt"
    chunks_path.write_text("a b c\nx y z\n", encoding="utf-8")
    cli.main(["run", "--query", "a b c", "--chunks-file", str(chunks_path), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert len(data["chunks"]) == 2


def test_run_missing_query_and_input_raises_systemexit():
    with pytest.raises(SystemExit):
        cli.main(["run"])


def test_run_missing_chunks_raises_systemexit():
    with pytest.raises(SystemExit):
        cli.main(["run", "--query", "a b c"])
