"""Shared machinery for perturbation kinds that are just a bidirectional
word/phrase swap table: negation cues aside, antonym/comparative/quantifier/
modal/temporal-direction all reduce to "find a phrase from a lookup table,
propose swapping it for its pair." Written once, reused by each."""
from __future__ import annotations

import random
import re
from typing import Dict, Iterable, List, Sequence, Tuple, Union

from ..text import Span
from .base import Candidate, PerturbationKind

PairLike = Union[Tuple[str, str], List[str]]


def build_lookup(pairs: Sequence[PairLike]) -> Dict[str, str]:
    """Bidirectional lowercase-keyed lookup; value is the *other* side of the
    pair, in its as-authored form."""
    lookup: Dict[str, str] = {}
    for a, b in pairs:
        lookup[a.lower()] = b
        lookup[b.lower()] = a
    return lookup


def compile_phrase_regex(keys: Iterable[str]) -> "re.Pattern":
    """Longest-match-first, word-boundary guarded, case-insensitive. Handles
    multiword keys (e.g. 'at least') as literal phrases."""
    ordered = sorted(set(keys), key=len, reverse=True)
    return re.compile(
        r"\b(" + "|".join(re.escape(k) for k in ordered) + r")\b",
        re.IGNORECASE,
    )


def propose_swaps(
    text: str,
    lookup: Dict[str, str],
    regex: "re.Pattern",
    kind: PerturbationKind,
    *,
    max_per_text: int,
    rng: random.Random,
) -> List[Candidate]:
    """One candidate per non-overlapping regex match, left-to-right document
    order, capped at max_per_text. `rng` is accepted for signature symmetry
    with kinds that need randomness; a plain swap doesn't."""
    del rng
    out: List[Candidate] = []
    for m in regex.finditer(text):
        if len(out) >= max_per_text:
            break
        surface = m.group(0)
        repl = lookup.get(surface.lower())
        if repl is None:
            continue
        span = Span(surface, m.start(), m.end())
        out.append(
            Candidate(kind=kind, span=span, replacement=repl, trigger=f"{surface} -> {repl}")
        )
    return out
