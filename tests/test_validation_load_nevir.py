"""Offline test for validation.datasets.load_nevir(): monkeypatches
datasets.load_dataset with a tiny fixture reproducing NevIR's real schema
(id, WorkerId, q1, q2, doc1, doc2). The one real network call this project
makes against NevIR is the manually-executed gate run recorded in
validation/REPORT.md -- not anything baked into automated CI.

Marked optional_deps (skipped by `make test-fast` / `-m "not
optional_deps"`) and importorskip'd (skipped, not failed, when the
`validation` extra isn't installed) since it exercises the `datasets`-
backed code path."""

import pytest

pytest.importorskip("datasets")

pytestmark = pytest.mark.optional_deps

from validation.datasets import load_capretrieval, load_nevir  # noqa: E402


class _FakeNevirRow(dict):
    pass


class _FakeNevirDataset(list):
    def select(self, indices):
        return _FakeNevirDataset(self[i] for i in indices)


_FIXTURE_ROWS = [
    {
        "id": "doc1-edit1",
        "WorkerId": 7,
        "q1": "is the pool open",
        "q2": "is the pool closed",
        "doc1": "the pool is open",
        "doc2": "the pool is not open",
    },
    {
        "id": "doc2-edit1",
        "WorkerId": 3,
        "q1": "are seats left",
        "q2": "are no seats left",
        "doc1": "seats are left",
        "doc2": "no seats are left",
    },
]


def test_load_nevir_maps_schema_correctly(monkeypatch):
    def _fake_load_dataset(name, split=None):
        assert name == "orionweller/NevIR"
        assert split == "test"
        return _FakeNevirDataset(_FIXTURE_ROWS)

    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    pairs = load_nevir(split="test")

    assert len(pairs) == 2
    p = pairs[0]
    assert p.source == "nevir"
    assert p.category == "negation"
    assert p.pair_id == "nevir-test-doc1-edit1"
    assert p.query_a == "is the pool open"
    assert p.query_b == "is the pool closed"
    assert p.doc_a == "the pool is open"
    assert p.doc_b == "the pool is not open"


def test_load_nevir_respects_limit(monkeypatch):
    def _fake_load_dataset(name, split=None):
        return _FakeNevirDataset(_FIXTURE_ROWS)

    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)

    pairs = load_nevir(split="test", limit=1)
    assert len(pairs) == 1


def test_load_capretrieval_raises_not_implemented_with_explanation():
    with pytest.raises(NotImplementedError, match="CapRetrieval"):
        load_capretrieval()
