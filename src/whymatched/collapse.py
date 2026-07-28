"""Negation / near-synonym ("antonym") collapse detection.

The core idea: build a counterfactual variant of the query or chunk that
*flips its meaning* (remove a negation cue, or swap a word for its antonym),
re-embed, and check whether the similarity score barely moves. If flipping
the meaning doesn't change the score, the embedding model can't distinguish
the two — a "collapse" that directly explains why a retriever surfaces a
chunk that says the opposite of what the query asked.
"""
from __future__ import annotations

import importlib.resources
import json
import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from .text import Span, remove_span, replace_span, split_words
from .utils import cosine_similarity

NEGATION_CUES = [
    "not", "n't", "no", "never", "none", "nobody", "nothing", "nowhere",
    "neither", "nor", "without", "except", "unless", "lack", "lacks", "lacking",
    "absence of", "fail to", "failed to", "unable to", "cannot", "can not",
]

_NEG_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in sorted(NEGATION_CUES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_ANTONYM_LOOKUP: Optional[dict] = None


def _load_antonyms() -> dict:
    global _ANTONYM_LOOKUP
    if _ANTONYM_LOOKUP is None:
        raw = importlib.resources.files("whymatched.data").joinpath("antonym_pairs.json").read_text()
        pairs = json.loads(raw)
        lookup = {}
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


@dataclass
class CollapseFlag:
    kind: Literal["negation_collapse", "antonym_collapse"]
    side: Literal["query", "chunk"]
    trigger: str
    counterfactual_snippet: str
    base_score: float
    counterfactual_score: float
    relative_delta: float


def _find_negation_spans(text: str) -> List[Span]:
    return [Span(m.group(0), m.start(), m.end()) for m in _NEG_RE.finditer(text)]


def _find_antonym_words(text: str, use_wordnet: bool) -> List[Tuple[Span, str]]:
    lookup = _load_antonyms()
    found = []
    for word in split_words(text):
        lw = word.text.lower()
        antonym = lookup.get(lw)
        if antonym is None and use_wordnet:
            antonym = _wordnet_antonym(lw)
        if antonym:
            found.append((word, antonym))
    return found


def detect_collapse(
    model,
    query: str,
    chunk: str,
    threshold: float = 0.10,
    use_wordnet: bool = False,
    max_checks_per_side: int = 6,
) -> List[CollapseFlag]:
    """Flag negation/antonym counterfactuals that barely move the similarity
    score (relative change below ``threshold``)."""
    base_vecs = model.embed([query, chunk])
    base_score = float(cosine_similarity(base_vecs[0], base_vecs[1])[0, 0])

    # candidates: (side, kind, trigger, counterfactual_query, counterfactual_chunk)
    candidates: List[Tuple[str, str, str, str, str]] = []

    for span in _find_negation_spans(query)[:max_checks_per_side]:
        cf_query = remove_span(query, span) or query
        candidates.append(("query", "negation_collapse", span.text, cf_query, chunk))
    for span in _find_negation_spans(chunk)[:max_checks_per_side]:
        cf_chunk = remove_span(chunk, span) or chunk
        candidates.append(("chunk", "negation_collapse", span.text, query, cf_chunk))

    for word, antonym in _find_antonym_words(query, use_wordnet)[:max_checks_per_side]:
        cf_query = replace_span(query, word, antonym)
        candidates.append(("query", "antonym_collapse", f"{word.text} -> {antonym}", cf_query, chunk))
    for word, antonym in _find_antonym_words(chunk, use_wordnet)[:max_checks_per_side]:
        cf_chunk = replace_span(chunk, word, antonym)
        candidates.append(("chunk", "antonym_collapse", f"{word.text} -> {antonym}", query, cf_chunk))

    if not candidates:
        return []

    cf_queries = [c[3] for c in candidates]
    cf_chunks = [c[4] for c in candidates]
    all_vecs = model.embed(cf_queries + cf_chunks)
    n = len(candidates)
    cf_query_vecs, cf_chunk_vecs = all_vecs[:n], all_vecs[n:]

    flags: List[CollapseFlag] = []
    for i, (side, kind, trigger, cf_query, cf_chunk) in enumerate(candidates):
        cf_score = float(cosine_similarity(cf_query_vecs[i], cf_chunk_vecs[i])[0, 0])
        relative_delta = abs(base_score - cf_score) / max(abs(base_score), 1e-6)
        if relative_delta < threshold:
            snippet = cf_query if side == "query" else cf_chunk
            flags.append(
                CollapseFlag(
                    kind=kind,
                    side=side,
                    trigger=trigger,
                    counterfactual_snippet=snippet[:160],
                    base_score=base_score,
                    counterfactual_score=cf_score,
                    relative_delta=relative_delta,
                )
            )
    return flags
