def test_batch_types_importable_from_top_level():
    from whymatched import BatchReport, CaseReport, EvalCase, PerturbationResult, load_cases, scan

    assert EvalCase and CaseReport and BatchReport and scan and load_cases and PerturbationResult


def test_testing_assertions_importable_from_submodule():
    from whymatched.testing import assert_collapse_rate, assert_no_collapse, assert_not_worse_than

    assert assert_collapse_rate and assert_no_collapse and assert_not_worse_than
