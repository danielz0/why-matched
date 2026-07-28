from __future__ import annotations

from ..utils import cosine_similarity
from .base import AttributionResult, TokenScore


def _require_captum():
    try:
        from captum.attr import IntegratedGradients

        return IntegratedGradients
    except ImportError as e:  # pragma: no cover - exercised only when captum missing
        raise ImportError(
            "gradient attribution requires captum: pip install whymatched[local]"
        ) from e


def gradient_attribution(model, query: str, chunk: str, n_steps: int = 32) -> AttributionResult:
    """Integrated Gradients attribution of the cosine similarity score to
    each input token embedding. Requires a :class:`~whymatched.models.local.LocalModel`
    (full access to the differentiable forward pass).

    Unlike occlusion, this attributes the *exact* function that produced the
    score (embed -> pool -> normalize -> cosine), rather than approximating it
    via word deletion, at the cost of only being available for local models.
    """
    if not getattr(model, "supports_gradients", False):
        raise ValueError(
            f"{model.name} does not support gradient-based attribution; use method='occlusion' instead"
        )

    import torch

    IntegratedGradients = _require_captum()

    base_vecs = model.embed([query, chunk])
    base_score = float(cosine_similarity(base_vecs[0], base_vecs[1])[0, 0])
    query_vec_fixed = torch.tensor(base_vecs[0], dtype=torch.float32, device=model.device).unsqueeze(0)
    chunk_vec_fixed = torch.tensor(base_vecs[1], dtype=torch.float32, device=model.device).unsqueeze(0)

    def pooled_from_embeds(inputs_embeds, attention_mask):
        out = model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        pooled = model._pool(out.last_hidden_state, attention_mask)
        if model.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled

    def make_forward(fixed_vec):
        def forward(inputs_embeds, attention_mask):
            pooled = pooled_from_embeds(inputs_embeds, attention_mask)
            return torch.nn.functional.cosine_similarity(pooled, fixed_vec)

        return forward

    special = set(getattr(model.tokenizer, "all_special_tokens", []))

    def attribute_side(text, fixed_vec):
        enc = model._tokenize_batch([text])
        embedding_layer = model.model.get_input_embeddings()
        inputs_embeds = embedding_layer(enc["input_ids"]).detach()
        attention_mask = enc["attention_mask"]
        baseline = torch.zeros_like(inputs_embeds)

        ig = IntegratedGradients(make_forward(fixed_vec))
        attributions = ig.attribute(
            inputs_embeds,
            baselines=baseline,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
        )
        token_weights = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens = model.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        return [
            TokenScore(token=t, weight=float(w))
            for t, w in zip(tokens, token_weights)
            if t not in special
        ]

    query_tokens = attribute_side(query, chunk_vec_fixed)
    chunk_tokens = attribute_side(chunk, query_vec_fixed)

    return AttributionResult(
        method="integrated_gradients",
        query_tokens=query_tokens,
        chunk_tokens=chunk_tokens,
        base_score=base_score,
    )
