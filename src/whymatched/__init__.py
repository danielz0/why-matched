from .attribution.base import AttributionResult, TokenScore
from .batch import BatchReport, CaseReport, EvalCase, load_cases, scan
from .collapse import CollapseFlag
from .core import AnalysisResult, ChunkAnalysis, Debugger
from .models.api import APIEmbeddingModel
from .models.base import EmbeddingModel
from .models.local import LocalModel
from .perturbations import PerturbationResult
from .projection import ProjectedPoint
from .report import render_html, write_html

__version__ = "0.3.0"

__all__ = [
    "Debugger",
    "AnalysisResult",
    "ChunkAnalysis",
    "EmbeddingModel",
    "LocalModel",
    "APIEmbeddingModel",
    "TokenScore",
    "AttributionResult",
    "CollapseFlag",
    "ProjectedPoint",
    "render_html",
    "write_html",
    "EvalCase",
    "CaseReport",
    "BatchReport",
    "scan",
    "load_cases",
    "PerturbationResult",
]
