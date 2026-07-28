from whymatched.attribution.occlusion import occlusion_attribution

from .fakes import FakeBagOfWordsModel


def test_occlusion_weights_reflect_shared_words():
    model = FakeBagOfWordsModel()
    query = "remote work policy"
    chunk = "remote work is banned for contractors"
    result = occlusion_attribution(model, query, chunk)

    assert result.method == "occlusion"
    assert 0.0 < result.base_score <= 1.0

    # Removing a word the two texts share should hurt the score (positive weight);
    # removing a word absent from the query barely changes anything (weight ~0).
    by_token = {t.token: t.weight for t in result.chunk_tokens}
    assert by_token["remote"] > by_token["banned"]
    assert by_token["work"] > by_token["banned"]


def test_occlusion_returns_scores_for_every_word():
    model = FakeBagOfWordsModel()
    query = "a b c"
    chunk = "a b c d e"
    result = occlusion_attribution(model, query, chunk)
    assert len(result.query_tokens) == 3
    assert len(result.chunk_tokens) == 5
