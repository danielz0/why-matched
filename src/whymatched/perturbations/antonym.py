"""Antonym perturbation: propose swapping a word for its antonym from a
curated dictionary (optionally falling back to WordNet)."""
from __future__ import annotations

import importlib.resources
import json
import random
from typing import Dict, List, Optional

from ..text import split_words
from .base import Candidate

_ANTONYM_LOOKUP: Optional[Dict[str, str]] = None


def _load_antonyms() -> Dict[str, str]:
    global _ANTONYM_LOOKUP
    if _ANTONYM_LOOKUP is None:
        raw = importlib.resources.files("whymatched.data").joinpath("antonym_pairs.json").read_text()
        pairs = json.loads(raw)
        lookup: Dict[str, str] = {}
        for a, b in pairs:
            lookup[a.lower()] = b
            lookup[b.lower()] = a
        _ANTONYM_LOOKUP = lookup
    return _ANTONYM_LOOKUP


def _wordnet_antonym(word: str) -> Optional[str]:
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return None
    try:
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    return lemma.antonyms()[0].name().replace("_", " ")
    except Exception:
        return None
    return None


class AntonymPerturbation:
    name = "antonym"
    kind = "antonym"
    priority = 90
    requires: tuple = ()

    def __init__(self, *, use_wordnet: bool = False, max_per_text: int = 8):
        self.use_wordnet = use_wordnet
        self.max_per_text = max_per_text

    def available(self) -> bool:
        return True

    def propose(self, text: str, *, rng: random.Random) -> List[Candidate]:
        del rng
        lookup = _load_antonyms()
        out: List[Candidate] = []
        for word in split_words(text):
            if len(out) >= self.max_per_text:
                break
            lw = word.text.lower()
            antonym = lookup.get(lw)
            if antonym is None and self.use_wordnet:
                antonym = _wordnet_antonym(lw)
            if antonym:
                out.append(
                    Candidate(kind="antonym", span=word, replacement=antonym, trigger=f"{word.text} -> {antonym}")
                )
        return out
