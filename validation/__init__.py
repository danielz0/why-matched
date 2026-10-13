"""Root-level validation harness -- NOT shipped in the wheel
(``[tool.setuptools.packages.find] where=["src"]`` in pyproject.toml
excludes anything outside ``src/``). Ground-truth adapters plus a
precision/recall measurement for whymatched's collapse detector against
externally-labelled contrastive pairs.

Today this covers NevIR (negation-only) via ``load_nevir()``. See
``validation/README.md`` for why ``load_capretrieval()`` is a documented
stub rather than a working loader.
"""

from .datasets import LabelledPair, load_capretrieval, load_nevir
from .metrics import ConfusionCounts, DetectorMetrics, evaluate_detector

__all__ = [
    "LabelledPair",
    "load_nevir",
    "load_capretrieval",
    "ConfusionCounts",
    "DetectorMetrics",
    "evaluate_detector",
]
