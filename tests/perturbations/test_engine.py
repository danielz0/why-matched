import random

from whymatched.cache import EmbeddingCache
from whymatched.perturbations import evaluate, get_perturbations, propose_all
from whymatched.perturbations.engine import _arbitrate
from whymatched.perturbations.negation import NegationPerturbation

from ..fakes import CountingModel, FakeBagOfWordsModel


def test_evaluate_issues_exactly_one_embed_call():
    inner = FakeBagOfWordsModel()
    model = CountingModel(inner)
    cache = EmbeddingCache(model)

    query = "is remote work allowed for contractors"
    chunk = "remote work is not allowed for contractors"
    perturbations = [NegationPerturbation(legacy_rules=False)]
    rng = random.Random(0)
    proposals = propose_all(query, chunk, perturbations, rng=rng)
    assert proposals

    results = evaluate(cache, query, chunk, proposals)
    assert results
    assert model.calls == 1


def test_propose_all_drops_empty_after_apply():
    perturbations = [NegationPerturbation(legacy_rules=False)]
    rng = random.Random(0)
    proposals = propose_all("not", "irrelevant chunk text", perturbations, rng=rng)
    query_side = [c for side, c in proposals if side == "query"]
    assert query_side == []


def test_max_per_side_per_kind_caps_after_arbitration():
    text_query = " ".join(["not"] * 10)
    perturbations = [NegationPerturbation(legacy_rules=False, max_per_text=10)]
    rng = random.Random(0)
    proposals = propose_all(text_query, "chunk", perturbations, max_per_side_per_kind=2, rng=rng)
    query_side = [c for side, c in proposals if side == "query"]
    assert len(query_side) == 2


def test_arbitration_year_beats_bare_integer():
    text = "revenue grew after 2020 by 15%"
    perturbations = get_perturbations(kinds=("temporal", "numeric"))
    rng = random.Random(0)
    raw = []
    for p in perturbations:
        raw.extend(p.propose(text, rng=rng))
    resolved = _arbitrate(raw)

    year_span_candidates = [c for c in resolved if c.span.text == "2020"]
    assert year_span_candidates, "expected at least one surviving 2020 candidate"
    assert all(c.kind == "temporal" for c in year_span_candidates)

    percent_span_candidates = [c for c in resolved if "15" in c.span.text]
    assert percent_span_candidates
    assert all(c.kind == "numeric" for c in percent_span_candidates)
