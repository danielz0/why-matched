"""Integration tests against a real local sentence-transformers model.
Downloads ~90MB on first run; skipped automatically if unavailable."""
import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from whymatched import Debugger, LocalModel
from whymatched.attribution.gradient import gradient_attribution
from whymatched.attribution.maxsim import maxsim_attribution
from whymatched.utils import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def model():
    try:
        return LocalModel.from_sentence_transformers(MODEL_NAME)
    except Exception as e:
        pytest.skip(f"could not load {MODEL_NAME}: {e}")


def test_gradient_attribution_base_score_matches_embed(model):
    query = "Is remote work allowed for contractors?"
    chunk = "Remote work is not allowed for contractors under this policy."
    dbg = Debugger(model, method="integrated_gradients")
    result = dbg.analyze(query, [chunk], project=False, detect_collapse_flags=False)
    attribution = result.chunks[0].attribution

    vecs = model.embed([query, chunk])
    true_score = float(cosine_similarity(vecs[0], vecs[1])[0, 0])
    assert abs(attribution.base_score - true_score) < 1e-4
    assert len(attribution.query_tokens) > 0
    assert len(attribution.chunk_tokens) > 0


def test_negation_collapse_flagged_on_real_model(model):
    dbg = Debugger(model, method="occlusion")
    result = dbg.analyze(
        "Is remote work allowed for contractors?",
        ["Remote work is not allowed for contractors under this policy."],
        project=False,
    )
    kinds = {f.kind for f in result.chunks[0].collapse_flags}
    assert "negation_collapse" in kinds


def test_maxsim_matrix_shape(model):
    dbg = Debugger(model, method="maxsim")
    result = dbg.analyze("remote work", ["remote work policy"], project=False, detect_collapse_flags=False)
    attribution = result.chunks[0].attribution
    sim_matrix = np.array(attribution.extra["sim_matrix"])
    assert sim_matrix.shape == (len(attribution.query_tokens), len(attribution.chunk_tokens))
    assert sim_matrix.min() >= -1.0001 and sim_matrix.max() <= 1.0001


def test_token_level_projection(model):
    dbg = Debugger(model)
    result = dbg.analyze(
        "remote work", ["remote work policy"], projection_level="token", detect_collapse_flags=False
    )
    kinds = {p.kind for p in result.projection}
    assert "query_token" in kinds
    assert "chunk_token" in kinds


# -- dedicated gradient_attribution tests -----------------------------------

def test_gradient_attribution_token_counts_match_tokenizer(model):
    pytest.importorskip("captum")
    query, chunk = "remote work policy", "the policy covers remote work"
    result = gradient_attribution(model, query, chunk)
    special = set(model.tokenizer.all_special_tokens)
    expected_query_len = len([t for t in model.tokenize(query) if t not in special])
    expected_chunk_len = len([t for t in model.tokenize(chunk) if t not in special])
    assert len(result.query_tokens) == expected_query_len
    assert len(result.chunk_tokens) == expected_chunk_len
    assert result.method == "integrated_gradients"


def test_gradient_attribution_pad_vs_zero_baseline_differ(model):
    pytest.importorskip("captum")
    query, chunk = "is remote work allowed for contractors", "remote work is not allowed for contractors"
    pad_result = gradient_attribution(model, query, chunk, baseline="pad")
    zero_result = gradient_attribution(model, query, chunk, baseline="zero")

    # same underlying score either way -- baseline only affects attribution, not the score itself
    assert abs(pad_result.base_score - zero_result.base_score) < 1e-5

    pad_weights = [t.weight for t in pad_result.chunk_tokens]
    zero_weights = [t.weight for t in zero_result.chunk_tokens]
    assert len(pad_weights) == len(zero_weights)
    # different reference points should not produce numerically identical attributions
    assert any(abs(a - b) > 1e-4 for a, b in zip(pad_weights, zero_weights))


def test_gradient_attribution_unknown_baseline_raises(model):
    pytest.importorskip("captum")
    with pytest.raises(ValueError):
        gradient_attribution(model, "a", "b", baseline="bogus")


# -- dedicated maxsim_attribution tests --------------------------------------

def test_maxsim_attribution_direct_shapes_and_best_matches(model):
    query, chunk = "remote work", "remote work policy"
    result = maxsim_attribution(model, query, chunk)
    assert result.method == "maxsim"
    sim_matrix = np.array(result.extra["sim_matrix"])
    assert sim_matrix.shape == (len(result.query_tokens), len(result.chunk_tokens))
    assert len(result.extra["query_best_match"]) == len(result.query_tokens)
    assert len(result.extra["chunk_best_match"]) == len(result.chunk_tokens)
    # every query token's maxsim weight must equal its row max in the similarity matrix
    for i, tok in enumerate(result.query_tokens):
        assert abs(tok.weight - sim_matrix[i].max()) < 1e-5


def test_maxsim_attribution_rejects_model_without_token_embeddings():
    from whymatched.attribution.maxsim import maxsim_attribution as direct

    class _NoTokens:
        supports_token_embeddings = False
        name = "no-tokens"

    with pytest.raises(ValueError):
        direct(_NoTokens(), "a", "b")
