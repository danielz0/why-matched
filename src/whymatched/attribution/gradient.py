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


def _baseline_token_id(tokenizer, baseline: str) -> int:
    if baseline == "pad":
        token_id = tokenizer.pad_token_id
        if token_id is None:
            token_id = tokenizer.eos_token_id
        if token_id is None:
            raise ValueError(
                "tokenizer has no pad_token_id or eos_token_id to use as the IG baseline; "
                "pass baseline='zero' or baseline='mask' instead"
            )
        return token_id
    if baseline == "mask":
        token_id = tokenizer.mask_token_id
        if token_id is None:
            raise ValueError(
                "tokenizer has no mask_token_id to use as the IG baseline; "
                "pass baseline='pad' or baseline='zero' instead"
            )
        return token_id
    raise ValueError(f"unknown baseline: {baseline!r}; use 'pad', 'mask', or 'zero'")


def gradient_attribution(
    model, query: str, chunk: str, n_steps: int = 32, baseline: str = "pad"
) -> AttributionResult:
    """Integrated Gradients attribution of the cosine similarity score to
    each input token embedding. Requires a :class:`~whymatched.models.local.LocalModel`
    (full access to the differentiable forward pass).

    Unlike occlusion, this attributes the *exact* function that produced the
    score (embed -> pool -> normalize -> cosine), rather than approximating it
    via word deletion, at the cost of only being available for local models.

    ``baseline`` controls the IG reference point each token embedding is
    integrated from: ``"pad"`` (default) uses the model's pad-token embedding
    at every position, ``"mask"`` uses its mask-token embedding, and ``"zero"``
    uses the origin. The zero vector is not a point the model's forward pass
    was ever trained to see as "absence of a token," so it can produce
    misleading attributions; pad/mask are the standard choices for IG on
    transformers and are preferred unless you have a specific reason to use
    the origin.
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
        if baseline == "zero":
            baseline_embeds = torch.zeros_like(inputs_embeds)
        else:
            baseline_token_id = _baseline_token_id(model.tokenizer, baseline)
            baseline_ids = torch.full_like(enc["input_ids"], baseline_token_id)
            baseline_embeds = embedding_layer(baseline_ids).detach()

        ig = IntegratedGradients(make_forward(fixed_vec))
        attributions = ig.attribute(
            inputs_embeds,
            baselines=baseline_embeds,
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
