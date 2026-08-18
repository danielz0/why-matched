"""Negation perturbation: propose removing (or, for contractions, expanding)
a negation cue so the text's polarity flips.

Bug B1 (fixed here): the original `_NEG_RE` wrapped every cue in `\\b...\\b`,
so `\\bn't\\b` could never match inside "don't" -- there is no word boundary
between "o" and "n". All contracted negations (don't, won't, isn't, can't,
didn't, ...) were silently missed. The fix is a dedicated contraction rule
that matches `\\w+n't` (no leading boundary before "n't") and replaces the
whole contraction with its expanded, de-negated form.
"""
from __future__ import annotations

import random
import re
from typing import List

from ..text import Span
from .base import Candidate

NEGATION_CUES = [
    "not", "no", "never", "none", "nobody", "nothing", "nowhere",
    "neither", "nor", "without", "except", "unless", "lack", "lacks", "lacking",
    "absence of", "fail to", "failed to", "unable to", "cannot", "can not",
]

_NEG_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in sorted(NEGATION_CUES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_CONTRACTION_RE = re.compile(r"\b(\w+)n't\b", re.IGNORECASE)
_CONTRACTION_IRREGULAR = {"wo": "will", "ca": "can", "sha": "shall"}


def _expand_contraction(stem: str) -> str:
    return _CONTRACTION_IRREGULAR.get(stem.lower(), stem)


def _find_negation_spans(text: str, *, legacy_rules: bool) -> List[Span]:
    spans = [Span(m.group(0), m.start(), m.end()) for m in _NEG_RE.finditer(text)]
    if not legacy_rules:
        for m in _CONTRACTION_RE.finditer(text):
            spans.append(Span(m.group(0), m.start(), m.end()))
        spans.sort(key=lambda s: s.start)
    return spans


class NegationPerturbation:
    name = "negation"
    kind = "negation"
    priority = 85
    requires: tuple = ()

    def __init__(self, *, legacy_rules: bool = False, max_per_text: int = 8):
        self.legacy_rules = legacy_rules
        self.max_per_text = max_per_text

    def available(self) -> bool:
        return True

    def propose(self, text: str, *, rng: random.Random) -> List[Candidate]:
        del rng
        out: List[Candidate] = []
        for span in _find_negation_spans(text, legacy_rules=self.legacy_rules)[: self.max_per_text]:
            match = _CONTRACTION_RE.fullmatch(span.text)
            if match:
                replacement = _expand_contraction(match.group(1))
                trigger = f"{span.text} -> {replacement}"
            else:
                replacement = ""
                trigger = span.text
            out.append(Candidate(kind="negation", span=span, replacement=replacement, trigger=trigger))
        return out
