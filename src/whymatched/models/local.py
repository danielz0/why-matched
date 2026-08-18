from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .base import EmbeddingModel


class LocalModel(EmbeddingModel):
    """Wraps a local HuggingFace transformer with an explicit pooling strategy.

    Because we own the full forward pass (tokenizer -> AutoModel -> pooling ->
    optional L2 normalize), ``embed()`` and the gradient/token-embedding paths
    are guaranteed to compute the exact same score, which is what makes
    Integrated Gradients and MaxSim attribution faithful rather than
    approximate.
    """

    supports_gradients = True
    supports_token_embeddings = True

    def __init__(
        self,
        model_name_or_path: str,
        pooling: str = "mean",
        normalize: bool = True,
        device: str = "cpu",
        max_length: int = 256,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.name = model_name_or_path
        self.pooling = pooling
        self.normalize = normalize
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        self.model.eval()

    @classmethod
    def from_sentence_transformers(
        cls, model_name_or_path: str, device: str = "cpu", max_length: int = 256
    ) -> "LocalModel":
        """Load via sentence-transformers to auto-detect pooling mode, then
        keep only the underlying tokenizer + AutoModel so every subsequent
        computation goes through our own (differentiable) forward pass."""
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(model_name_or_path, device=device)

        transformer_module = None
        pooling_module = None
        for module in st._modules.values():
            cls_name = module.__class__.__name__
            if cls_name == "Transformer" and transformer_module is None:
                transformer_module = module
            elif cls_name == "Pooling" and pooling_module is None:
                pooling_module = module

        if transformer_module is None:
            raise ValueError(
                f"could not find a Transformer submodule in sentence-transformers model "
                f"'{model_name_or_path}'; construct LocalModel(...) directly instead"
            )

        pooling = "mean"
        if pooling_module is not None:
            if getattr(pooling_module, "pooling_mode_cls_token", False):
                pooling = "cls"
            elif getattr(pooling_module, "pooling_mode_max_tokens", False):
                pooling = "max"

        inst = cls.__new__(cls)
        inst.name = str(model_name_or_path)
        inst.pooling = pooling
        inst.normalize = True
        inst.device = device
        inst.max_length = getattr(transformer_module, "max_seq_length", None) or max_length
        inst.tokenizer = transformer_module.tokenizer
        inst.model = transformer_module.auto_model.to(device)
        inst.model.eval()
        return inst


    def _tokenize_batch(self, texts: Sequence[str]):
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

    def _pool(self, hidden_states, attention_mask):
        import torch

        if self.pooling == "cls":
            return hidden_states[:, 0]
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        if self.pooling == "max":
            masked = hidden_states.masked_fill(mask == 0, -1e9)
            return masked.max(dim=1).values
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        if len(texts) == 0:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        with torch.no_grad():
            enc = self._tokenize_batch(texts)
            out = self.model(**enc)
            pooled = self._pool(out.last_hidden_state, enc["attention_mask"])
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled.cpu().numpy().astype(np.float32)

    def tokenize(self, text: str) -> List[str]:
        ids = self.tokenizer(text, truncation=True, max_length=self.max_length)["input_ids"]
        return self.tokenizer.convert_ids_to_tokens(ids)

    def token_embeddings(self, text: str) -> Tuple[List[str], np.ndarray]:
        import torch

        with torch.no_grad():
            enc = self._tokenize_batch([text])
            out = self.model(**enc)
            hidden = out.last_hidden_state[0]
            if self.normalize:
                hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
            tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            return tokens, hidden.cpu().numpy().astype(np.float32)

    def embed_with_grad(self, text: str):
        """Return (pooled[1,d] with grad graph attached, inputs_embeds leaf
        tensor, attention_mask, tokens) for gradient-based attribution."""
        import torch

        enc = self._tokenize_batch([text])
        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(enc["input_ids"]).detach().clone()
        inputs_embeds.requires_grad_(True)
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=enc["attention_mask"])
        pooled = self._pool(out.last_hidden_state, enc["attention_mask"])
        if self.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        return pooled, inputs_embeds, enc["attention_mask"], tokens
