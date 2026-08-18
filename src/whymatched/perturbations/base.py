"""Shared abstraction every perturbation category implements. A Perturbation
proposes Candidates against a single text (query OR chunk) without ever
touching an embedding model -- proposal is offline, which is what lets the
engine batch every candidate's embed call into one."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Protocol, Tuple

from ..text import Span, remove_span, replace_span

PerturbationKind = Literal[
    "negation", "antonym", "numeric", "entity",
    "temporal", "comparative", "quantifier", "modal",
]


@dataclass(frozen=True)
class Candidate:
    kind: PerturbationKind
    span: Span
    replacement: str
    trigger: str
    meaning_preserving: bool = False
    rationale: str = ""

    def apply(self, text: str) -> str:
        if self.replacement == "":
            return remove_span(text, self.span)
        return replace_span(text, self.span, self.replacement)


class Perturbation(Protocol):
    name: str
    kind: PerturbationKind
    priority: int
    requires: Tuple[str, ...]

    def available(self) -> bool: ...

    def propose(self, text: str, *, rng: random.Random) -> list: ...
