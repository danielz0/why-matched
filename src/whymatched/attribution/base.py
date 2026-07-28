from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TokenScore:
    """A single token's contribution to the query-chunk similarity score.

    Sign and scale depend on ``method``:
    - occlusion / integrated_gradients: signed delta, positive = removing
      this token would *lower* the similarity (it was driving the match up).
    - maxsim: unsigned affinity (0..1ish) = this token's best cosine match
      against any token on the other side.
    """

    token: str
    weight: float


@dataclass
class AttributionResult:
    method: str
    query_tokens: List[TokenScore]
    chunk_tokens: List[TokenScore]
    base_score: float
    extra: Dict[str, Any] = field(default_factory=dict)
