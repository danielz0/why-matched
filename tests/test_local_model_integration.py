"""Integration tests against a real local sentence-transformers model.
Downloads ~90MB on first run; skipped automatically if unavailable."""
import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from whymatched import Debugger, LocalModel
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
