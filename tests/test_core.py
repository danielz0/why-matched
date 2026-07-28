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
    assert result.method == "occlusion"  # fake model has no gradients -> auto-fallback
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
