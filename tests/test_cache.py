import numpy as np
import pytest

from whymatched.cache import EmbeddingCache

from .fakes import CountingModel, FakeBagOfWordsModel


def test_prime_dedupes_across_calls():
    model = CountingModel(FakeBagOfWordsModel())
    cache = EmbeddingCache(model)
    cache.prime(["hello world", "goodbye world"])
    assert model.calls == 1
    cache.prime(["hello world"])
    assert model.calls == 1


def test_prime_dedupes_within_call():
    model = CountingModel(FakeBagOfWordsModel())
    cache = EmbeddingCache(model)
    cache.prime(["hello world", "hello world", "hello world"])
    assert model.calls == 1
    assert cache.get("hello world") is not None


def test_get_raises_keyerror_when_not_primed():
    cache = EmbeddingCache(FakeBagOfWordsModel())
    with pytest.raises(KeyError):
        cache.get("never primed")


def test_score_matches_cosine_similarity():
    model = FakeBagOfWordsModel()
    cache = EmbeddingCache(model)
    cache.prime(["the cat sat", "the dog sat"])
    from whymatched.utils import cosine_similarity

    a, b = model.embed(["the cat sat", "the dog sat"])
    expected = float(cosine_similarity(a, b)[0, 0])
    assert cache.score("the cat sat", "the dog sat") == pytest.approx(expected)


def test_batching_respects_max_batch():
    model = CountingModel(FakeBagOfWordsModel())
    cache = EmbeddingCache(model, max_batch=2)
    cache.prime(["a", "b", "c", "d", "e"])
    assert model.calls == 3


def test_one_embed_call_invariant_for_small_batches():
    model = CountingModel(FakeBagOfWordsModel())
    cache = EmbeddingCache(model)
    texts = [f"text number {i}" for i in range(20)]
    cache.prime(texts)
    assert model.calls == 1
    for t in texts:
        assert isinstance(cache.get(t), np.ndarray)


def test_model_name_property():
    model = FakeBagOfWordsModel()
    cache = EmbeddingCache(model)
    assert cache.model_name == model.name


def test_model_name_property_falls_back_to_unknown():
    class NoNameModel:
        def embed(self, texts):
            return FakeBagOfWordsModel().embed(texts)

    cache = EmbeddingCache(NoNameModel())
    assert cache.model_name == "unknown"
