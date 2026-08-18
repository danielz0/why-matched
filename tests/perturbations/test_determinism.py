import dataclasses
import random

from whymatched.cache import EmbeddingCache
from whymatched.perturbations import evaluate, get_perturbations, propose_all

from ..fakes import FakeBagOfWordsModel


def test_same_seed_identical_output():
    query = "revenue grew after 2020 by 15% and contractors may never use the portal"
    chunk = "revenue is not allowed to grow before 2019 without approval from contractors"

    def run(seed: int):
        model = FakeBagOfWordsModel()
        cache = EmbeddingCache(model)
        perturbations = get_perturbations()
        rng = random.Random(seed)
        proposals = propose_all(query, chunk, perturbations, rng=rng)
        results = evaluate(cache, query, chunk, proposals)
        return [dataclasses.asdict(r) for r in results]

    first = run(0)
    second = run(0)
    assert first == second
    assert first
