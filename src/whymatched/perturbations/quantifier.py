"""Quantifier perturbation: one matched quantifier can fan out to *multiple*
replacement candidates (all -> some / all -> none), unlike the 1:1 swaps
used elsewhere. Only swaps that remain grammatical without re-inflection are
included (all->some, not all->any which would need article/verb agreement
in some contexts); this is a known recall gap, not a bug.
"""
from __future__ import annotations

import random
from typing import List, Tuple

from ..text import Span
from ._dictionary import compile_phrase_regex
from .base import Candidate

QUANTIFIER_FANOUT = {
    "all": ["some", "none"],
    "none": ["all"],
    "every": ["some"],
    "always": ["sometimes"],
    "never": ["always"],
    "most": ["few"],
    "both": ["neither"],
}

_REGEX = compile_phrase_regex(QUANTIFIER_FANOUT.keys())


class QuantifierPerturbation:
    name = "quantifier"
    kind = "quantifier"
    priority = 75
    requires: Tuple[str, ...] = ()

    def __init__(self, max_per_text: int = 8):
        self.max_per_text = max_per_text

    def available(self) -> bool:
        return True

    def propose(self, text: str, *, rng: random.Random) -> List[Candidate]:
        del rng
        out: List[Candidate] = []
        for m in _REGEX.finditer(text):
            surface = m.group(0)
            replacements = QUANTIFIER_FANOUT.get(surface.lower())
            if not replacements:
                continue
            span = Span(surface, m.start(), m.end())
            for repl in replacements:
                if len(out) >= self.max_per_text:
                    return out
                out.append(Candidate(kind="quantifier", span=span, replacement=repl, trigger=f"{surface} -> {repl}"))
        return out

    @classmethod
    def _lookup_keys(cls):
        return set(QUANTIFIER_FANOUT.keys())
