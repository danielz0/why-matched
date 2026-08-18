# WhyMatched

[![License: MIT](https://img.shields.io/github/license/danielz0/why-matched?color=FF5CC8)](https://github.com/danielz0/why-matched/blob/main/LICENSE)
[![Coverage](coverage.svg)](https://github.com/danielz0/why-matched/actions/workflows/ci.yml)
[![GitHub Stars](https://img.shields.io/github/stars/danielz0/why-matched?style=social)](https://github.com/danielz0/why-matched)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/danielz0?color=FF5CC8)](https://github.com/sponsors/danielz0)

**Why did this chunk match?**

A diagnostic and quality-control tool for embedding-based search and RAG
(retrieval-augmented generation) systems.

---

## For decision makers

### What it is

Search and RAG systems retrieve information by comparing the meaning of a
query against a database of text chunks using embeddings (numeric
representations of meaning). WhyMatched inspects those comparisons and
explains, in plain terms, *why* a particular chunk was judged relevant —
and flags the cases where the embedding model got it dangerously wrong.

### The problem it addresses

Embedding models are good at topic similarity, but they are not reliable
at telling apart:

- **Opposites** — "the machine runs without supervision" vs. "the machine
  requires supervision" can score as nearly identical.
- **Numbers and dates** — "$50" vs. "$500", or "Q3 2024" vs. "Q3 2025",
  can barely move the similarity score.
- **Comparisons and qualifiers** — "revenue increased" vs. "revenue
  decreased", "always" vs. "never", "required" vs. "optional".
- **Names and entities** — swapping the person, company, or place a
  sentence is about.

When that happens, a search or RAG system can confidently retrieve a
chunk that states the *opposite* of what's true, and nothing in a normal
similarity score reveals it. This is a silent, hard-to-catch failure mode
that erodes trust in AI-powered search and can surface incorrect
information to end users.

### What WhyMatched gives you

- **Per-query diagnostics**: for any (query, retrieved chunks) pair, a
  readable report showing which words drove the match and which
  retrieved chunks are flagged as unreliable — including a statistically
  calibrated confidence measure, not just a guess.
- **Fleet-wide quality checks**: scan a whole test set of queries at once
  and get a single number — the collapse rate — that tells you what
  fraction of your retrieval results have this failure mode.
- **A CI gate**: wire that collapse rate into your build pipeline so a
  regression in retrieval quality fails the build automatically, the same
  way a failing test suite would.
- **Works with your existing stack**: plugs into local, open-source
  embedding models or hosted APIs (OpenAI, Cohere, Voyage) without
  requiring any change to how your search/RAG system already works —
  it's a read-only diagnostic layer.

---

## For developers

### File tree

```
whymatched/
├── src/whymatched/
│   ├── core.py                  # Debugger, analyze(), AnalysisResult
│   ├── collapse.py              # detect_collapse(): threshold or calibrated
│   ├── calibration.py           # null generators, quantile rule, CalibrationProfile
│   ├── batch.py                 # EvalCase, BatchReport, scan(), load_cases()
│   ├── testing.py               # assert_collapse_rate / assert_no_collapse / assert_not_worse_than
│   ├── cache.py                 # EmbeddingCache: content-addressed, batched
│   ├── perturbations/           # negation, antonym, numeric, temporal,
│   │                            # comparative, quantifier, modal, entity + arbitration
│   ├── projection.py            # PCA / t-SNE / UMAP
│   ├── report.py                # self-contained HTML report (single + batch)
│   ├── cli.py                   # `whymatched run|scan|calibrate ...`
│   ├── text.py, utils.py
│   ├── attribution/
│   │   ├── occlusion.py         # leave-one-word-out (any backend)
│   │   ├── gradient.py          # Captum Integrated Gradients (local models)
│   │   └── maxsim.py            # ColBERT-style token-pair attribution
│   ├── models/
│   │   ├── local.py             # LocalModel (sentence-transformers/HF)
│   │   └── api.py               # APIEmbeddingModel (OpenAI/Cohere/Voyage/custom)
│   └── data/                    # antonym_pairs.json, synonym_pairs.json, entity_surrogates.json
├── examples/
│   ├── quickstart.py
│   ├── demo_negation_and_synonym_pairs.py
│   └── sample_report.html
├── benchmarks/                  # negation/antonym benchmark + control set
└── tests/
```

### Data flow

```
query, chunks, model, options
  -> embed query + chunks, rank by cosine similarity
  -> for each chunk (ranked):
       - attribute(method): occlusion | integrated_gradients | maxsim
       - detect_collapse(): propose counterfactuals across the enabled
         perturbation kinds, re-embed (via a shared EmbeddingCache), and
         flag either by fixed threshold or by a calibrated quantile rule
         (relative_delta compared against meaning-preserving "null" edits)
  -> project(): PCA / t-SNE / UMAP over query + chunk embeddings
  -> AnalysisResult { query, model_name, method, chunks[], projection }

scan(model, cases) -> BatchReport, for many (query, chunks) pairs at once
  -> whymatched.testing.assert_collapse_rate(...) as a CI gate
```

### Requirements

- Python 3.9+
- Core install has no ML framework dependency (`numpy`, `scikit-learn` only). Add an extra for the backend you need:
  - `local` — `torch`, `transformers`, `sentence-transformers`, `captum` (enables Integrated Gradients + MaxSim attribution on local models)
  - `openai` / `cohere` / `voyage` — the respective hosted API client
  - `entity` — `spacy` (entity-swap collapse detection; also needs `python -m spacy download en_core_web_sm`)
  - `viz` — `umap-learn` (adds UMAP as a projection method)
  - `all` — everything above

### Install

```bash
pip install whymatched                # core only 
pip install whymatched[local]         # + Integrated Gradients + MaxSim
pip install whymatched[openai]        # + hosted API client
pip install whymatched[entity]        # + entity-swap collapse detection 
```

`entity` also needs a model download: `python -m spacy download en_core_web_sm`.

### Using a hosted API provider (OpenAI, Cohere, Voyage)

Install the extra for the provider you want, then get an API key from
that provider's dashboard:

| Provider | Extra | Get a key at |
|---|---|---|
| OpenAI | `whymatched[openai]` | https://platform.openai.com/api-keys |
| Cohere | `whymatched[cohere]` | https://dashboard.cohere.com/api-keys |
| Voyage AI | `whymatched[voyage]` | https://dashboard.voyageai.com/organization/api-keys |

**CLI**: set the key as an environment variable and pass `--provider`. If
you don't pass `--api-key-env`, each provider's SDK reads its own default
variable — `OPENAI_API_KEY`, `CO_API_KEY`, or `VOYAGE_API_KEY`:

```bash
export OPENAI_API_KEY=sk-...
whymatched run \
  --provider openai --model text-embedding-3-small \
  --query "Does the machine need supervision to run?" \
  --chunk "The machine runs without supervision." \
  --out report.html
```

To read the key from a differently-named variable, pass `--api-key-env`:

```bash
export MY_OPENAI_KEY=sk-...
whymatched run --provider openai --api-key-env MY_OPENAI_KEY --query ... --chunk ...
```

The same applies to `whymatched scan`/`whymatched calibrate`, which take
the same `--provider`/`--model`/`--api-key-env` flags.

**Python API**: either rely on the environment variable, or pass the key
directly — nothing forces you to use an env var at all:

```python
import os
from whymatched import APIEmbeddingModel, Debugger

# option 1: read from whatever environment variable you like
model = APIEmbeddingModel.openai(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
)

# option 2: rely on the OpenAI SDK's own default (OPENAI_API_KEY) by
# omitting api_key entirely
model = APIEmbeddingModel.openai(model="text-embedding-3-small")

debugger = Debugger(model)
result = debugger.analyze(query, chunks)
```

`APIEmbeddingModel.cohere(...)` and `APIEmbeddingModel.voyage(...)` work
the same way. For any other hosted or self-hosted embedding endpoint
(Azure OpenAI, a self-hosted TEI server, etc.), wrap a plain
`texts -> vectors` callable with `APIEmbeddingModel.custom(embed_fn, name=...)`.

API keys are only ever read from the environment or passed in by you —
whymatched never stores, logs, or transmits them anywhere beyond the
provider SDK call itself.

### Set up for local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[local]"
pip install pytest
```

### Run the tests

```bash
pytest
```

Some tests are marked `optional_deps` (require `torch`/`spacy`/etc.) and
are skipped automatically if those aren't installed.

### Run the example

```bash
python examples/quickstart.py
```

or via CLI:

```bash
whymatched run \
  --query "Does the machine need supervision to run?" \
  --chunk "The machine runs without supervision." \
  --chunk "Only certified technicians are permitted to operate the machine." \
  --out report.html
```

Flags: `--provider openai --model text-embedding-3-small` for a hosted API,
`--json` for raw JSON output, `--input file.json` for
`{"query": ..., "chunks": [...]}`, `--method`, `--projection-level`,
`--projection-method`.

### Collapse detection: kinds and calibration

By default, `Debugger`/`detect_collapse()` check negation and antonym
collapse against a fixed relative-score threshold. Enable the full
multi-category engine and the calibrated decision rule explicitly:

```python
from whymatched import Debugger

debugger = Debugger(
    model,
    legacy_rules=False,               # negation, antonym, numeric, temporal,
                                       # comparative, quantifier, modal (entity
                                       # needs the [entity] extra + a spaCy model)
    calibrate_collapse=True,          # quantile rule instead of a fixed threshold
    quantile=0.10, n_null=50, alpha=0.05,
)
result = debugger.analyze(query, chunks)
```

Under calibration, a candidate's `relative_delta` is compared against the
distribution of deltas produced by curated, meaning-preserving synonym
substitutions of the same text (with an orthographic noise floor and a
content-deletion reference reported alongside, never averaged together).
Texts too short to produce enough null samples fall back to the plain
threshold rule rather than reporting statistics from a handful of samples.
Power users who want the full statistical detail (the quantile, its
bootstrap CI, ratio, z-score, per-generator null distributions) instead of
just which candidates got flagged should call
`whymatched.calibration.evaluate_calibrated()` directly.

### Batch scanning and CI gates

Run collapse detection across many `(query, chunks)` pairs at once and
gate a build on a collapse-rate threshold:

```bash
whymatched scan --input cases.jsonl \
  --max-collapse-rate 0.20 \
  --out batch_report.html --json report.json
```

`cases.jsonl` is one `{"query": ..., "chunks": [...]}` object per line (or
a JSON array). Exit codes: `0` thresholds met, `1` threshold exceeded, `2`
usage/IO error, `3` completed but some cases errored -- suitable for a CI
step. `--baseline previous_report.json --fail-on-regression` gates on "no
worse than last time" instead of (or in addition to) a fixed threshold.

Pre-compute null-delta quantiles once against a corpus and reuse them
across scans instead of resampling per call:

```bash
whymatched calibrate --corpus texts.txt --out profile.json
whymatched scan --input cases.jsonl --profile profile.json
```

For programmatic use, `whymatched.testing.assert_collapse_rate`/
`assert_no_collapse`/`assert_not_worse_than` raise `AssertionError` with a
message naming the threshold, actual rate, denominator, and worst
offenders -- drop them straight into a pytest-based CI check against a
`whymatched.scan(...)` result.

---

## Author & License

Created by Daniel Zimnicki.

Released under the [MIT License](LICENSE) — a permissive open-source license: free to use, modify, and
distribute (including commercially), with no warranty. See the [LICENSE](LICENSE) file for the full text.
