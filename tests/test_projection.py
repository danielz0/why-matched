from whymatched.projection import project_sentence_level, project_token_level

from .fakes import FakeBagOfWordsModel


def test_project_sentence_level_shapes():
    model = FakeBagOfWordsModel()
    points = project_sentence_level(model, "hello world", ["hello there", "goodbye world", "unrelated text"])
    assert len(points) == 4
    assert points[0].kind == "query"
    assert [p.kind for p in points[1:]] == ["chunk", "chunk", "chunk"]
    assert points[1].chunk_index == 0


def test_project_sentence_level_handles_tiny_input():
    model = FakeBagOfWordsModel()
    points = project_sentence_level(model, "hello", ["world"])
    assert len(points) == 2


def test_project_token_level_requires_token_support():
    model = FakeBagOfWordsModel()
    try:
        project_token_level(model, "hello", ["world"])
        assert False, "expected ValueError"
    except ValueError:
        pass
