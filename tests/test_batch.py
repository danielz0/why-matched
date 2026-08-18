import dataclasses

import pytest

from whymatched.batch import BatchReport, CaseLoadError, CaseReport, EvalCase, load_cases, scan
from whymatched.perturbations import PerturbationResult

from .fakes import CountingModel, FakeBagOfWordsModel

LONG_CHUNK = (
    "The vendor must obtain approval before it can purchase additional equipment. "
    "The team will request approval and then purchase the equipment once approval is obtained. "
    "Approval must be verified and confirmed by the finance department before any purchase is completed. "
    "The finance department will examine each purchase request and verify that approval was obtained."
)


def _result(kind="negation", is_collapse=True, q_value=None, p_value=None, ratio=None):
    return PerturbationResult(
        kind=kind, side="chunk", trigger="not", counterfactual_snippet="x",
        base_score=0.8, counterfactual_score=0.79, relative_delta=0.01,
        null_quantile=0.05, ratio=ratio, q_value=q_value, p_value=p_value,
        decision_rule="quantile", calibration_status="ok", is_collapse=is_collapse,
    )


def test_evalcase_resolved_id_default_is_sha1_prefix():
    case = EvalCase(query="is remote work allowed", chunks=["a"])
    assert case.resolved_id() == case.resolved_id()
    import hashlib

    assert case.resolved_id() == hashlib.sha1(b"is remote work allowed").hexdigest()[:8]


def test_evalcase_resolved_id_override():
    case = EvalCase(query="q", chunks=["a"], case_id="my-case")
    assert case.resolved_id() == "my-case"


def test_severity_zero_collapse_case_returns_sentinel_and_sorts_last():
    zero_collapse = CaseReport(case_id="z", query="q", n_chunks=1, top_score=0.9, results=[])
    assert zero_collapse.severity == (0, 1.0, float("inf"))

    has_collapse = CaseReport(
        case_id="h", query="q", n_chunks=1, top_score=0.9,
        results=[_result(q_value=0.02, ratio=0.2)],
    )
    ordered = sorted([zero_collapse, has_collapse], key=lambda c: c.severity, reverse=True)
    assert ordered == [has_collapse, zero_collapse]


def test_severity_prefers_more_collapses_then_stronger_significance():
    one_collapse = CaseReport(
        case_id="a", query="q", n_chunks=1, top_score=0.9, results=[_result(q_value=0.04, ratio=0.5)],
    )
    two_collapses = CaseReport(
        case_id="b", query="q", n_chunks=1, top_score=0.9,
        results=[_result(q_value=0.04, ratio=0.5), _result(q_value=0.01, ratio=0.1)],
    )
    ordered = sorted([one_collapse, two_collapses], key=lambda c: c.severity, reverse=True)
    assert ordered == [two_collapses, one_collapse]


def test_severity_falls_back_to_p_value_when_q_value_missing():
    c = CaseReport(
        case_id="a", query="q", n_chunks=1, top_score=0.9,
        results=[_result(q_value=None, p_value=0.03, ratio=0.2)],
    )
    assert c.severity == (1, -0.03, -0.2)


def _make_case(case_id, results, error=None):
    return CaseReport(case_id=case_id, query=f"q-{case_id}", n_chunks=1, top_score=0.9, results=results, error=error)


def test_collapse_rate_denominator_definition():
    cases = []
    for i in range(4):
        cases.append(_make_case(f"skip{i}", []))
    for i in range(17):
        cases.append(_make_case(f"hit{i}", [_result(is_collapse=True)]))
    for i in range(33):
        cases.append(_make_case(f"miss{i}", [_result(is_collapse=False)]))

    report = BatchReport(
        model_name="m", whymatched_version="0.0.0", created_at="now", config={},
        cases=cases, n_skipped=4, n_errored=0,
    )
    assert report.collapse_rate() == 17 / 50


def test_collapse_rate_excludes_skipped_and_errored_cases():
    cases = [
        _make_case("a", [_result(is_collapse=True)]),
        _make_case("skipped", []),
        _make_case("errored", [], error="boom"),
    ]
    report = BatchReport(
        model_name="m", whymatched_version="0.0.0", created_at="now", config={},
        cases=cases, n_skipped=1, n_errored=1,
    )
    assert report.collapse_rate() == 1 / 1


def test_collapse_rate_per_kind_restricts_denominator():
    cases = [
        _make_case("a", [_result(kind="numeric", is_collapse=True)]),
        _make_case("b", [_result(kind="numeric", is_collapse=False)]),
        _make_case("c", [_result(kind="negation", is_collapse=True)]),
    ]
    report = BatchReport(
        model_name="m", whymatched_version="0.0.0", created_at="now", config={},
        cases=cases, n_skipped=0, n_errored=0,
    )
    assert report.collapse_rate(kind="numeric") == 1 / 2


def test_candidate_collapse_rate_definition():
    cases = [
        _make_case("a", [_result(is_collapse=True), _result(is_collapse=False)]),
        _make_case("b", [_result(is_collapse=True)]),
    ]
    report = BatchReport(
        model_name="m", whymatched_version="0.0.0", created_at="now", config={},
        cases=cases, n_skipped=0, n_errored=0,
    )
    assert report.candidate_collapse_rate() == 2 / 3


def test_candidate_collapse_rate_excludes_errored_cases_entirely():
    cases = [
        _make_case("a", [_result(is_collapse=True)]),
        _make_case("errored", [_result(is_collapse=True)], error="boom"),
    ]
    report = BatchReport(
        model_name="m", whymatched_version="0.0.0", created_at="now", config={},
        cases=cases, n_skipped=0, n_errored=1,
    )
    assert report.candidate_collapse_rate() == 1 / 1


def test_rate_by_kind():
    cases = [
        _make_case("a", [_result(kind="numeric", is_collapse=True)]),
        _make_case("b", [_result(kind="negation", is_collapse=False)]),
    ]
    report = BatchReport(
        model_name="m", whymatched_version="0.0.0", created_at="now", config={},
        cases=cases, n_skipped=0, n_errored=0,
    )
    rates = report.rate_by_kind()
    assert rates == {"numeric": 1.0, "negation": 0.0}


def test_worst_ordering_uses_severity():
    low = _make_case("low", [_result(is_collapse=True, q_value=0.04, ratio=0.5)])
    high = _make_case(
        "high",
        [_result(is_collapse=True, q_value=0.04, ratio=0.5), _result(is_collapse=True, q_value=0.001, ratio=0.05)],
    )
    none = _make_case("none", [])
    report = BatchReport(
        model_name="m", whymatched_version="0.0.0", created_at="now", config={},
        cases=[low, none, high], n_skipped=1, n_errored=0,
    )
    assert report.worst(2) == [high, low]


def test_to_dict_shape():
    cases = [_make_case("a", [_result()])]
    report = BatchReport(
        model_name="m", whymatched_version="0.0.0", created_at="now",
        config={"calibrate": True}, cases=cases, n_skipped=0, n_errored=0,
    )
    d = report.to_dict()
    assert d["model_name"] == "m"
    assert d["n_cases"] == 1
    assert d["collapse_rate"] == 1.0
    assert "rate_by_kind" in d
    assert d["cases"][0]["case_id"] == "a"
    assert d["cases"][0]["calibration"] is None


def _make_50_cases():
    pool = [
        "remote work is not allowed for contractors under this policy",
        "the price is $50 for this purchase today",
        LONG_CHUNK,
        "revenue grew before the merger by 15%",
        "the cat sat quietly on the mat",
    ]
    return [EvalCase(query=f"case number {i}", chunks=[pool[i % len(pool)]]) for i in range(50)]


def test_scan_embed_calls_sublinear_in_candidates():
    cases = _make_50_cases()
    model = CountingModel(FakeBagOfWordsModel())
    report = scan(model, cases, seed=0)
    assert len(report.cases) == 50
    total_candidates = sum(len(c.results) for c in report.cases)
    assert total_candidates > 0
    assert model.calls < total_candidates


def test_scan_counts_n_skipped_for_zero_candidate_cases():
    cases = [
        EvalCase(query="the cat sat", chunks=["the mat sat"]),
        EvalCase(query="is remote work allowed", chunks=["remote work is not allowed here"]),
    ]
    model = FakeBagOfWordsModel()
    report = scan(model, cases, seed=0)
    assert report.n_skipped == 1
    assert report.n_errored == 0


def test_scan_case_exception_does_not_abort_scan():
    cases = [
        EvalCase(query="is remote work allowed", chunks=[]),
        EvalCase(query="is remote work allowed for contractors", chunks=["remote work is not allowed for contractors"]),
    ]
    model = FakeBagOfWordsModel()
    report = scan(model, cases, seed=0)
    assert len(report.cases) == 2
    assert report.n_errored == 1
    errored = [c for c in report.cases if c.error is not None][0]
    assert "chunks must be non-empty" in errored.error
    ok = [c for c in report.cases if c.error is None][0]
    assert ok.results


def test_scan_calibrate_false_sets_is_collapse_via_threshold():
    cases = [EvalCase(query="is remote work allowed for contractors", chunks=["remote work is not allowed for contractors under this policy"])]
    model = FakeBagOfWordsModel()
    report = scan(model, cases, calibrate=False, seed=0)
    case = report.cases[0]
    assert case.results
    assert all(r.decision_rule == "threshold" for r in case.results)
    for r in case.results:
        assert r.is_collapse == (r.relative_delta < 0.10)
    assert any(r.is_collapse for r in case.results)


def test_scan_same_seed_identical_report():
    cases = [EvalCase(query=f"is approval required case {i}", chunks=[LONG_CHUNK]) for i in range(3)]
    model = FakeBagOfWordsModel()
    r1 = scan(model, cases, seed=0)
    r2 = scan(model, cases, seed=0)
    assert [c.to_dict() for c in r1.cases] == [c.to_dict() for c in r2.cases]


def test_scan_case_calibration_keeps_top_chunk_only():
    cases = [EvalCase(query="is approval required for this purchase", chunks=[LONG_CHUNK, "short unrelated text"])]
    model = FakeBagOfWordsModel()
    report = scan(model, cases, top_k_chunks=2, seed=0)
    case = report.cases[0]
    assert case.calibration is not None
    assert set(case.calibration.keys()) == {"query", "chunk"}


def test_scan_top_k_chunks_none_evaluates_all_chunks():
    cases = [EvalCase(query="is remote work allowed for contractors", chunks=["remote work is not allowed for contractors here", "unrelated filler text about cooking"])]
    model = FakeBagOfWordsModel()
    report_top1 = scan(model, cases, top_k_chunks=1, seed=0)
    report_all = scan(model, cases, top_k_chunks=None, seed=0)
    assert len(report_all.cases[0].results) >= len(report_top1.cases[0].results)


def test_scan_max_workers_gives_same_result_set_as_single_threaded():
    cases = _make_50_cases()[:10]
    model_seq = FakeBagOfWordsModel()
    model_par = FakeBagOfWordsModel()
    seq = scan(model_seq, cases, seed=0, max_workers=1)
    par = scan(model_par, cases, seed=0, max_workers=4)
    seq_by_id = {c.case_id: c.to_dict() for c in seq.cases}
    par_by_id = {c.case_id: c.to_dict() for c in par.cases}
    assert seq_by_id == par_by_id


def test_scan_max_workers_reuses_cache_per_thread_not_per_case():
    cases = _make_50_cases()
    model_seq = CountingModel(FakeBagOfWordsModel())
    scan(model_seq, cases, seed=0, max_workers=1)

    model_par = CountingModel(FakeBagOfWordsModel())
    scan(model_par, cases, seed=0, max_workers=2)

    assert model_par.calls <= model_seq.calls * 2


def test_load_cases_jsonl_reports_line_number(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"query": "q1", "chunks": ["a"]}\n{"query": "q2"}\n')
    with pytest.raises(CaseLoadError, match=r"cases\.jsonl:2:"):
        load_cases(path)


def test_load_cases_json_array_reports_index(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text('[{"query": "q1", "chunks": ["a"]}, {"query": "q2"}]')
    with pytest.raises(CaseLoadError, match=r"\[1\]:"):
        load_cases(path)


def test_load_cases_rejects_missing_required_key():
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write('{"query": "q1"}\n')
        path = f.name
    try:
        with pytest.raises(CaseLoadError, match="missing required key"):
            load_cases(path)
    finally:
        os.unlink(path)


def test_load_cases_rejects_unknown_key(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"query": "q1", "chunks": ["a"], "bogus": 1}\n')
    with pytest.raises(CaseLoadError, match="unknown key"):
        load_cases(path)


def test_load_cases_allows_empty_chunks_list(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"query": "q1", "chunks": []}\n')
    cases = load_cases(path)
    assert cases[0].chunks == []


def test_load_cases_jsonl_roundtrip(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"query": "q1", "chunks": ["a", "b"]}\n'
        '{"query": "q2", "chunks": ["c"], "case_id": "custom", "tags": ["x"]}\n'
    )
    cases = load_cases(path)
    assert len(cases) == 2
    assert cases[0].query == "q1"
    assert cases[0].chunks == ["a", "b"]
    assert cases[1].case_id == "custom"
    assert cases[1].tags == ["x"]


def test_load_cases_json_array_roundtrip(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text('[{"query": "q1", "chunks": ["a"]}]')
    cases = load_cases(path)
    assert len(cases) == 1
    assert cases[0].query == "q1"
