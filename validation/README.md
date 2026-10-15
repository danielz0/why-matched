# Validation harness (WS-A)

**Not shipped in the wheel.** `pyproject.toml`'s
`[tool.setuptools.packages.find] where=["src"]` only packages `src/`, so
this directory is excluded from the built package automatically.

This package answers the question the rest of the roadmap is gated on: how
often does whymatched's collapse detector agree with reality? It compares
`whymatched.collapse.detect_collapse()`'s flags against externally-labelled
contrastive pairs, where "reality" is the model's own actual ranking
behavior on each pair.

## Install

```bash
pip install -e ".[local,validation]"
```

`validation` pulls in the [`datasets`](https://pypi.org/project/datasets/)
library (Hugging Face Hub loading). `local` is needed to run real
sentence-transformers models via `whymatched.LocalModel`.

## Licensing

Checked before any loader was written, per the guide's own warning that a
licensing miss "has sunk projects before":

- **NevIR** (`orionweller/NevIR` on the Hugging Face Hub) is
  **MIT-licensed** — redistribution is permitted. Nothing from it is
  vendored in this repo regardless: `load_nevir()` downloads at call time,
  and only the loader is shipped.

## Usage

```bash
python -m validation.runner --help

python -m validation.runner \
    --dataset nevir --split test --sample-size 200 --seed 0 \
    --model sentence-transformers/all-MiniLM-L6-v2 --calibrate --markdown \
    --out validation/results/all-MiniLM-L6-v2.json
```

Python API:

```python
from validation import load_nevir, evaluate_detector
from whymatched import LocalModel

pairs = load_nevir(split="test", limit=50)
model = LocalModel.from_sentence_transformers("sentence-transformers/all-MiniLM-L6-v2")
metrics = evaluate_detector(model, pairs, calibrate=True)
print(metrics.summary())
```

## Known gaps

- **`load_capretrieval()` is a documented stub, not a working loader.**
  CapRetrieval (`lxucs/CapRetrieval`, Apache-2.0) turned out, on inspection,
  to be a single-query graded-relevance retrieval dataset for image-caption
  granularity — one query maps to many candidate captions with relevance
  labels in `{0, 1, 2}`, not a meaning-reversed `query_a`/`query_b` pair.
  It doesn't have the contrastive shape `LabelledPair` requires, and there
  is no faithful field mapping to write. Calling `load_capretrieval()`
  raises `NotImplementedError` explaining this rather than shipping a
  loader that silently mis-maps the schema. Follow-up options: find a
  genuinely contrastive second dataset (ideally covering a
  `PerturbationKind` other than negation), or design and separately review
  a bespoke synthesis method for deriving meaning-reversed pairs from
  graded-relevance data.
- **Per-category reporting is honest-but-thin.** NevIR is negation-only by
  construction (built from CondaQA negation edits), so `DetectorMetrics.
  by_category` only ever has a populated `"negation"` entry until a second,
  differently-categorized dataset is added. Don't read the absence of
  other categories as "no collapses of that kind" — it's an absence of
  ground truth for those kinds, not a null result.
- **`--sample-size` is a plain random sample, not stratified.** NevIR has
  no `PerturbationKind` axis to stratify on (see above), so
  `validation/runner.py`'s sampling is a fixed-seed
  `random.Random(seed).shuffle()` over the split's indices — documented
  here rather than silently implied to be stratified.

See [`REPORT.md`](REPORT.md) for the first real gate run's results.
