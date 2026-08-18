"""Entity perturbation: swap a named entity for a surrogate of the same
label. Requires spacy + a downloaded model; there is deliberately no
heuristic fallback (a capitalized-token guesser misfires on sentence-initial
and title-case text and would poison the benchmark with noisy candidates),
so `available()` returns False -- never raises -- when spacy isn't
importable or the model isn't downloaded, and the category is cleanly
skipped by the registry/engine.
"""
from __future__ import annotations

import importlib.resources
import json
import random
from typing import Dict, List, Tuple

from ..text import Span
from .base import Candidate

_NLP = None
_LABELS = {"PERSON", "ORG", "PRODUCT", "GPE", "LOC"}


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy
    except Exception:
        _NLP = False
        return _NLP
    try:
        _NLP = spacy.load("en_core_web_sm")
    except Exception:
        _NLP = False
    return _NLP


def _load_surrogates() -> Dict[str, List[str]]:
    raw = importlib.resources.files("whymatched.data").joinpath("entity_surrogates.json").read_text()
    return json.loads(raw)


class EntityPerturbation:
    name = "entity"
    kind = "entity"
    priority = 95
    requires: Tuple[str, ...] = ("spacy",)

    def __init__(self, model_name: str = "en_core_web_sm", max_per_text: int = 8):
        self.model_name = model_name
        self.max_per_text = max_per_text

    def available(self) -> bool:
        try:
            nlp = _get_nlp()
        except Exception:
            return False
        return nlp is not False

    def propose(self, text: str, *, rng: random.Random) -> List[Candidate]:
        if not self.available():
            return []
        nlp = _get_nlp()
        surrogates = _load_surrogates()
        doc = nlp(text)
        lowered_text = text.lower()
        out: List[Candidate] = []
        for ent in doc.ents:
            if len(out) >= self.max_per_text:
                break
            if ent.label_ not in _LABELS:
                continue
            pool = surrogates.get(ent.label_, [])
            candidates_pool = [
                s for s in pool if s.lower() != ent.text.lower() and s.lower() not in lowered_text
            ]
            if not candidates_pool:
                continue
            replacement = candidates_pool[rng.randrange(len(candidates_pool))]
            span = Span(ent.text, ent.start_char, ent.end_char)
            out.append(
                Candidate(
                    kind="entity",
                    span=span,
                    replacement=replacement,
                    trigger=f"{ent.text} -> {replacement}",
                    rationale=f"label={ent.label_}",
                )
            )
        return out
