from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .antonym import AntonymPerturbation
from .base import Candidate, Perturbation, PerturbationKind
from .comparative import ComparativePerturbation
from .engine import PerturbationResult, evaluate, propose_all
from .entity import EntityPerturbation
from .modal import ModalPerturbation
from .negation import NegationPerturbation
from .numeric import NumericPerturbation
from .quantifier import QuantifierPerturbation
from .temporal import TemporalPerturbation

_REGISTRY: Dict[str, Perturbation] = {}


def register(p: Perturbation) -> None:
    _REGISTRY[p.name] = p


def get_perturbations(
    kinds: Optional[Sequence[str]] = None,
    *,
    include_unavailable: bool = False,
) -> List[Perturbation]:
    kinds_set = set(kinds) if kinds is not None else None
    out = []
    for p in _REGISTRY.values():
        if kinds_set is not None and p.kind not in kinds_set:
            continue
        if not include_unavailable and not p.available():
            continue
        out.append(p)
    return out


DEFAULT_KINDS = (
    "negation", "antonym", "numeric", "temporal",
    "comparative", "quantifier", "modal",
)


def _register_builtins() -> None:
    for cls in (
        NegationPerturbation,
        AntonymPerturbation,
        NumericPerturbation,
        TemporalPerturbation,
        ComparativePerturbation,
        QuantifierPerturbation,
        ModalPerturbation,
        EntityPerturbation,
    ):
        register(cls())


_register_builtins()

__all__ = [
    "Candidate",
    "Perturbation",
    "PerturbationKind",
    "PerturbationResult",
    "evaluate",
    "propose_all",
    "register",
    "get_perturbations",
    "DEFAULT_KINDS",
]
