import pytest

from whymatched.batch import BatchReport, CaseReport
from whymatched.perturbations import PerturbationResult
from whymatched.testing import assert_collapse_rate, assert_no_collapse, assert_not_worse_than


def _collapse_result(kind="numeric", trigger="$50 -> $500", base=0.812, cf=0.809, q=0.021, ratio=0.07):
    return PerturbationResult(
        kind=kind, side="chunk", trigger=trigger, counterfactual_snippet="x",
        base_score=base, counterfactual_score=cf, relative_delta=abs(base - cf),
        null_quantile=q, ratio=ratio, q_value=0.01, p_value=0.01,
        decision_rule="quantile", calibration_status="ok", is_collapse=True,
    )


def _no_collapse_result():
    return PerturbationResult(
        kind="numeric", side="chunk", trigger="x", counterfactual_snippet="y",
        base_score=0.5, counterfactual_score=0.1, relative_delta=0.4, is_collapse=False,
        decision_rule="quantile", calibration_status="ok",
    )


def _case(case_id, results, error=None):
    return CaseReport(case_id=case_id, query="q", n_chunks=1, top_score=0.8, results=results, error=error)


def _report_with_rate(n_collapsed, n_ok, n_skipped=0, n_errored=0, calibrate=True):
    cases = [_case(f"hit{i}", [_collapse_result()]) for i in range(n_collapsed)]
    cases += [_case(f"ok{i}", [_no_collapse_result()]) for i in range(n_ok)]
    cases += [_case(f"skip{i}", []) for i in range(n_skipped)]
    cases += [_case(f"err{i}", [], error="boom") for i in range(n_errored)]
    return BatchReport(
        model_name="m", whymatched_version="0.1", created_at="now",
        config={"calibrate": calibrate, "quantile": 0.10, "n_null": 50, "threshold": 0.10},
        cases=cases, n_skipped=n_skipped, n_errored=n_errored,
    )


def test_assert_collapse_rate_passes_when_under_threshold():
    report = _report_with_rate(1, 99)
    assert_collapse_rate(report, max_rate=0.20)


def test_assert_collapse_rate_raises_when_over_threshold():
    report = _report_with_rate(17, 33, n_skipped=4)
    with pytest.raises(AssertionError) as excinfo:
        assert_collapse_rate(report, max_rate=0.20)
    msg = str(excinfo.value)
    assert "collapse_rate=0.34" in msg
    assert "max_rate=0.20" in msg


def test_assert_collapse_rate_message_format_contains_required_fields():
    report = _report_with_rate(17, 33, n_skipped=4, n_errored=0)
    with pytest.raises(AssertionError) as excinfo:
        assert_collapse_rate(report, max_rate=0.20)
    msg = str(excinfo.value)
    assert "0.34" in msg
    assert "0.20" in msg
    assert "17 of 50 applicable cases" in msg
    assert "4 skipped" in msg
    assert "0 errored" in msg
    assert "rule=quantile" in msg
    assert "q=0.1" in msg
    assert "n_null=50" in msg
    assert "worst offenders" in msg
    assert "[hit0]" in msg or "[hit1]" in msg


def test_assert_collapse_rate_with_kind_labels_the_metric():
    report = _report_with_rate(17, 33, n_skipped=4)
    with pytest.raises(AssertionError) as excinfo:
        assert_collapse_rate(report, max_rate=0.0, kind="numeric")
    assert "collapse_rate(kind=numeric)" in str(excinfo.value)


def test_assert_no_collapse_passes_when_zero_collapses():
    report = _report_with_rate(0, 10)
    assert_no_collapse(report)


def test_assert_no_collapse_raises_when_any_collapse():
    report = _report_with_rate(1, 10)
    with pytest.raises(AssertionError):
        assert_no_collapse(report)


def test_assert_no_collapse_restricts_to_given_kinds():
    negation_collapse = CaseReport(
        case_id="n1", query="q", n_chunks=1, top_score=0.8,
        results=[_collapse_result(kind="negation")],
    )
    report = BatchReport(
        model_name="m", whymatched_version="0.1", created_at="now",
        config={"calibrate": True}, cases=[negation_collapse], n_skipped=0, n_errored=0,
    )
    assert_no_collapse(report, kinds=["numeric"])
    with pytest.raises(AssertionError):
        assert_no_collapse(report, kinds=["negation"])


def test_assert_not_worse_than_passes_on_identical_report():
    report = _report_with_rate(5, 95)
    baseline = _report_with_rate(5, 95)
    assert_not_worse_than(report, baseline)


def test_assert_not_worse_than_fails_on_regressed_report():
    report = _report_with_rate(20, 80)
    baseline = _report_with_rate(5, 95)
    with pytest.raises(AssertionError) as excinfo:
        assert_not_worse_than(report, baseline)
    msg = str(excinfo.value)
    assert "regressed beyond baseline" in msg
    assert "worst offenders" in msg


def test_assert_not_worse_than_respects_tolerance():
    report = _report_with_rate(10, 90)
    baseline = _report_with_rate(9, 91)
    with pytest.raises(AssertionError):
        assert_not_worse_than(report, baseline, tolerance=0.0)
    assert_not_worse_than(report, baseline, tolerance=0.02)
