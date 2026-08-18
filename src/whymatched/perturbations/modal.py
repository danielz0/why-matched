"""Modal perturbation: swap a deontic/permission-strength word for its
opposite (E6 -- must/may is modal, not a plain antonym pair).

"required" deliberately overlaps antonym.py's required<->optional pair: the
two are distinct, legitimate lexical contrasts sharing a surface form
(required<->optional is a general synonym-of-antonym pair; required<->
permitted is the deontic contrast this module owns). Arbitration priority
(modal=70 < antonym=90) always resolves the runtime conflict in modal's
favor -- see tests/perturbations/test_modal.py.
"""
from __future__ import annotations

import random
from typing import List, Tuple

from ._dictionary import build_lookup, compile_phrase_regex, propose_swaps
from .base import Candidate

MODAL_PAIRS = [
    ("must", "may"),
    ("shall", "should"),
    ("required", "permitted"),
    ("mandatory", "optional"),
]

_LOOKUP = build_lookup(MODAL_PAIRS)
_REGEX = compile_phrase_regex(_LOOKUP.keys())


class ModalPerturbation:
    name = "modal"
    kind = "modal"
    priority = 70
    requires: Tuple[str, ...] = ()

    def __init__(self, max_per_text: int = 8):
        self.max_per_text = max_per_text

    def available(self) -> bool:
        return True

    def propose(self, text: str, *, rng: random.Random) -> List[Candidate]:
        return propose_swaps(text, _LOOKUP, _REGEX, "modal", max_per_text=self.max_per_text, rng=rng)

    @classmethod
    def _lookup_keys(cls):
        return set(_LOOKUP.keys())
