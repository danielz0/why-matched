"""Comparative perturbation: swap a magnitude-comparison word/phrase for its
opposite (migrated out of antonym_pairs.json, spec bug B3 -- these were
already double-firing from the antonym file before this migration)."""
from __future__ import annotations

import random
from typing import List, Tuple

from ._dictionary import build_lookup, compile_phrase_regex, propose_swaps
from .base import Candidate

COMPARATIVE_PAIRS = [
    ("higher", "lower"),
    ("more", "less"),
    ("greater", "fewer"),
    ("above", "below"),
    ("exceeds", "falls below"),
    ("at least", "at most"),
    ("maximum", "minimum"),
    ("faster", "slower"),
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("over", "under"),
    ("increase", "reduce"),
    ("increase", "diminish"),
]

_LOOKUP = build_lookup(COMPARATIVE_PAIRS)
_REGEX = compile_phrase_regex(_LOOKUP.keys())


class ComparativePerturbation:
    name = "comparative"
    kind = "comparative"
    priority = 80
    requires: Tuple[str, ...] = ()

    def __init__(self, max_per_text: int = 8):
        self.max_per_text = max_per_text

    def available(self) -> bool:
        return True

    def propose(self, text: str, *, rng: random.Random) -> List[Candidate]:
        return propose_swaps(text, _LOOKUP, _REGEX, "comparative", max_per_text=self.max_per_text, rng=rng)

    @classmethod
    def _lookup_keys(cls):
        return set(_LOOKUP.keys())
