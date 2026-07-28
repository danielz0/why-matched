"""Model-agnostic word splitting used for occlusion attribution and collapse
detection. Deliberately independent of any tokenizer's subword scheme so the
resulting explanations read as whole words regardless of embedding backend."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)


@dataclass
class Span:
    text: str
    start: int
    end: int


def split_words(text: str) -> List[Span]:
    return [Span(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def remove_span(text: str, span: Span) -> str:
    result = text[: span.start] + text[span.end :]
    return re.sub(r"\s{2,}", " ", result).strip()


def replace_span(text: str, span: Span, replacement: str) -> str:
    return text[: span.start] + replacement + text[span.end :]
