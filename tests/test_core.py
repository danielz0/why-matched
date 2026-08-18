from whymatched import Debugger

from .fakes import FakeBagOfWordsModel


def test_debugger_ranks_chunks_by_score_descending():
    model = FakeBagOfWordsModel()
    dbg = Debugger(model)
    result = dbg.analyze(
        "remote work policy for contractors",
        [
            "totally unrelated cooking recipe",
            "remote work policy applies to contractors and vendors",
        ],
    )
    assert result.method == "occlusion"
    scores = [c.score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)
    assert result.chunks[0].chunk.startswith("remote work policy")
    assert [c.rank for c in result.chunks] == [0, 1]


def test_debugger_can_skip_projection_and_collapse():
    model = FakeBagOfWordsModel()
    dbg = Debugger(model)
    result = dbg.analyze("a b c", ["a b c"], project=False, detect_collapse_flags=False)
    assert result.projection is None
    assert result.chunks[0].collapse_flags == []


def test_debugger_rejects_empty_chunks():
    model = FakeBagOfWordsModel()
    dbg = Debugger(model)
    try:
        dbg.analyze("query", [])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_debugger_calibrate_collapse_no_longer_raises():
    model = FakeBagOfWordsModel()
    dbg = Debugger(model, calibrate_collapse=True, legacy_rules=False)
    result = dbg.analyze(
        "remote work policy for contractors",
        ["remote work policy applies to contractors and vendors"],
        project=False,
    )
    assert isinstance(result.chunks[0].collapse_flags, list)
