from whymatched.collapse import detect_collapse

from .fakes import FakeBagOfWordsModel


def test_negation_collapse_flagged_when_score_barely_moves():
    model = FakeBagOfWordsModel()
    query = "is remote work allowed for contractors"
    chunk = "remote work is not allowed for contractors under this policy document"
    flags = detect_collapse(model, query, chunk, threshold=0.15)
    kinds = {f.kind for f in flags}
    assert "negation_collapse" in kinds
    neg_flags = [f for f in flags if f.kind == "negation_collapse"]
    assert any(f.trigger == "not" for f in neg_flags)


def test_antonym_collapse_detected_for_known_pairs():
    model = FakeBagOfWordsModel()
    query = "is this particular business action allowed under our rules today"
    chunk = "this particular business action is allowed under company policy rules today now"
    flags = detect_collapse(model, query, chunk, threshold=0.3)
    kinds = {f.kind for f in flags}
    assert "antonym_collapse" in kinds


def test_no_candidates_returns_empty_list():
    model = FakeBagOfWordsModel()
    flags = detect_collapse(model, "the cat sat", "the mat sat", threshold=0.1)
    assert flags == []
