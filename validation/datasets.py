"""Ground-truth adapters (WS-A / A1). Each loader returns a list of
LabelledPair -- one contrastive pair with human-labelled ground truth,
mapped from an external dataset's own schema onto whymatched's own
PerturbationKind taxonomy.

Licensing note (checked before writing this file, per the guide's own
warning that a redistribution-license miss "has sunk projects before"):
NevIR (orionweller/NevIR on the Hugging Face Hub) is MIT-licensed, which
permits redistribution -- but this module still never vendors any of its
rows. load_nevir() downloads at runtime via the `datasets` library
(the `validation` extra); the repo ships only the loader, never the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from whymatched.perturbations import PerturbationKind


@dataclass(frozen=True)
class LabelledPair:
    """One contrastive pair with human ground truth.

    query_a is relevant to doc_a, not doc_b; query_b is relevant to doc_b,
    not doc_a. doc_a/doc_b differ by a meaning-changing edit of the kind
    named in `category`.
    """

    pair_id: str
    query_a: str
    query_b: str
    doc_a: str
    doc_b: str
    source: Literal["nevir", "capretrieval", "custom"]
    category: PerturbationKind


def load_nevir(split: str = "test", limit: Optional[int] = None) -> List[LabelledPair]:
    """Load NevIR (Weller et al., EACL 2024) from the Hugging Face Hub.

    Dataset: `orionweller/NevIR` (MIT license, confirmed redistribution-safe;
    nothing from it is vendored in this repo -- this function downloads it
    at import/call time). Splits: train (948) / validation (225) /
    test (1383, empirically confirmed via `load_dataset` -- some secondary
    sources cite 1386).

    Schema mapping: NevIR's `q1`/`q2`/`doc1`/`doc2` -> this module's
    `query_a`/`query_b`/`doc_a`/`doc_b`. Every row is negation-only by
    construction (the dataset is built from CondaQA negation edits), so
    `category` is always "negation" here -- per-category reporting built
    from load_nevir() alone will only ever have a populated "negation" row
    until a second, differently-categorized dataset is added.

    Requires the `validation` extra (`pip install -e ".[validation]"`);
    the `datasets` import is deliberately lazy (mirrors
    `LocalModel.from_sentence_transformers`'s lazy `sentence_transformers`
    import in `src/whymatched/models/local.py`) so importing this module
    doesn't require the extra -- only calling this function does.
    """
    from datasets import load_dataset

    ds = load_dataset("orionweller/NevIR", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    pairs = []
    for row in ds:
        pairs.append(
            LabelledPair(
                pair_id=f"nevir-{split}-{row['id']}",
                query_a=row["q1"],
                query_b=row["q2"],
                doc_a=row["doc1"],
                doc_b=row["doc2"],
                source="nevir",
                category="negation",
            )
        )
    return pairs


def load_capretrieval(*args, **kwargs) -> List[LabelledPair]:
    """Documented stub -- do not treat this as a working loader.

    CapRetrieval (lxucs/CapRetrieval, Apache-2.0) is a single-query,
    graded-relevance retrieval dataset for image-caption granularity: one
    query maps to many candidate captions with relevance labels in
    {0, 1, 2}, not a meaning-reversed query_a/query_b pair. It does not
    have the contrastive shape LabelledPair requires, so there is no
    faithful field mapping to write here.

    Raises NotImplementedError rather than shipping a loader that
    silently mis-maps the schema (e.g. picking an arbitrary low-relevance
    caption as a stand-in "doc_b" would not actually test meaning-reversal,
    and would corrupt any precision/recall number computed from it).

    See validation/README.md's "Known gaps" section for follow-up options:
    either a genuinely contrastive dataset, or a bespoke synthesis method
    designed and reviewed on its own merits.
    """
    raise NotImplementedError(
        "load_capretrieval() is a documented stub: CapRetrieval's schema "
        "(one query -> many graded-relevance documents) doesn't map onto "
        "LabelledPair's contrastive query_a/query_b + doc_a/doc_b shape. "
        "See validation/README.md's 'Known gaps' section."
    )
