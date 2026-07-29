# whymatched

[![CI](https://github.com/danielz0/why-matched/actions/workflows/ci.yml/badge.svg)](https://github.com/danielz0/why-matched/actions/workflows/ci.yml)
[![coverage](coverage.svg)](https://github.com/danielz0/why-matched/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/whymatched.svg)](https://pypi.org/project/whymatched/)
[![Python versions](https://img.shields.io/pypi/pyversions/whymatched.svg)](https://pypi.org/project/whymatched/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Why did this chunk match?** A retrieval debugger for embedding-based
search/RAG. Its core feature — the reason it exists — is catching
**negation/antonym collapse**: cases where a retrieved chunk says the
*opposite* of what the query asked, but the embedding model barely notices
the difference (a huge, under-diagnosed source of bad retrieval, and
something no other pip-installable tool currently checks for). On top of
that it gives you token-level attribution (which words actually drove the
similarity score) and a 2D projection of the query and chunks, so you can
see the geometry behind a ranking.

Author: Daniel Zimnicki. Licensed under the [MIT License](LICENSE) — free and open source, use it however you like.

It's a plain Python library first — `pip install` it and call `Debugger(...).analyze(...)`
from inside your own RAG/search app to get structured data back. A CLI and
self-contained HTML report are included as a thin convenience layer on top.

## Install

```bash
# core (occlusion attribution + collapse detection + PCA projection, any embedding backend)
pip install whymatched

# + local HuggingFace/sentence-transformers models (adds Integrated Gradients + MaxSim attribution)
pip install whymatched[local]

# + a hosted embedding API client
pip install whymatched[openai]   # or [cohere] / [voyage]

# + UMAP as a projection option
pip install whymatched[viz]
```

## Quickstart

```python
from whymatched import Debugger, LocalModel

model = LocalModel.from_sentence_transformers("sentence-transformers/all-MiniLM-L6-v2")
debugger = Debugger(model)  # method="auto": Integrated Gradients for local models

result = debugger.analyze(
    query="Does the machine need supervision to run?",
    chunks=[
        "The machine runs without supervision.",
        "Only certified technicians are permitted to operate the machine.",
    ],
)

for c in result.chunks:
    print(c.rank, c.score, c.chunk)
    print("driving tokens:", [(t.token, t.weight) for t in c.top_chunk_tokens[:5]])
    for flag in c.collapse_flags:
        print("COLLAPSE:", flag.kind, flag.trigger, flag.base_score, "->", flag.counterfactual_score)
```

Run it and the top hit — which says the *opposite* of what's true — scores
0.826 similarity, and `collapse_flags` tells you exactly why: removing
"without" barely moves the score at all, from 0.826 to 0.828 — a **0.1%
relative change**. The embedding model is essentially blind to the one word
that determines whether this chunk answers the question or contradicts it.
See `examples/quickstart.py` and `examples/demo_negation_and_synonym_pairs.py`
— see [`examples/README.md`](examples/README.md) for a field-by-field guide
to interpreting the output.

This isn't a cherry-picked anecdote — see
[`benchmarks/`](benchmarks/README.md) for a 30-case negation/antonym
benchmark (plus a neutral control set) that measures how often the real
detector catches this failure mode, and how often it stays quiet when it
should.

## Using a hosted embedding API instead of a local model

```python
from whymatched import Debugger, APIEmbeddingModel

model = APIEmbeddingModel.openai(model="text-embedding-3-small")  # reads OPENAI_API_KEY
debugger = Debugger(model)  # auto-selects occlusion; no gradients/token vectors available via API
result = debugger.analyze(query, chunks)
```

Any callable `texts -> vectors` works too: `APIEmbeddingModel.custom(embed_fn, name=...)`,
for Azure OpenAI, a self-hosted TEI server, etc.

## What you get back

`Debugger.analyze(query, chunks)` returns an `AnalysisResult`:

- `chunks: List[ChunkAnalysis]` — one per retrieved chunk, ranked by score, each with:
  - `score` — cosine similarity
  - `attribution` — `AttributionResult` with per-token `TokenScore`s for both the query and the chunk
  - `collapse_flags` — `List[CollapseFlag]`, one per negation/antonym counterfactual that barely moved the score
- `projection: List[ProjectedPoint]` — 2D coordinates for the query and every chunk (or every token, see below)

Everything is a plain dataclass (`dataclasses.asdict` friendly via `.to_dict()`) — no
framework objects leak into your application.

## Attribution methods

| method | requires | what it measures |
|---|---|---|
| `occlusion` (default for API models) | any backend, just `embed()` | leave-one-word-out: how much removing each word changes the cosine score. Universal, word-granularity. |
| `integrated_gradients` (default for local models) | `LocalModel` | Captum Integrated Gradients w.r.t. input token embeddings — attributes the *exact* function that produced the score, at subword granularity. |
| `maxsim` | `LocalModel` | ColBERT-style late interaction: full query-token × chunk-token cosine matrix, each token scored by its best match on the other side. Answers "which specific token pair caused this," not "how much did this token move the pooled score." |

Pass `method=` to `Debugger(...)` to force one; `"auto"` (default) picks
Integrated Gradients for local models and occlusion for API models.

**Caveat:** occlusion "importance" is partly confounded by length — removing
*any* word shifts a mean-pooled sentence vector somewhat, regardless of that
word's actual meaning — so occlusion weights are a directional hint about
what the model is keying on, not a clean decomposition of "semantic
importance." Integrated Gradients is more faithful (it attributes the exact
function that produced the score) but is only available for local models.

## How this relates to prior work

Token-level attribution by itself is not new — `occlusion` and
`integrated_gradients` are standard XAI techniques (the kind Captum and SHAP
already implement) applied here to bi-encoder similarity instead of a
classifier's logits. Two lines of prior work do something similar for
embedding/retrieval models specifically: **BiLRP** (Vu et al., layer-wise
relevance propagation for explaining pairwise similarity between two
transformer encodings) and **MaxSimE / ColBERT-style late interaction**
(token-level attribution via per-token max-similarity, which `whymatched`'s
`maxsim` method is directly modeled on). If you need a deep, faithful
explanation of *why two vectors are close*, those are worth reading.

What isn't covered by that prior art — and what `whymatched` actually adds —
is turning attribution into an automated **diagnostic**: systematically
building negation/antonym counterfactuals and flagging when the model fails
to react to a meaning-reversing change. That's the part with no existing
pip-installable equivalent, and the part worth trusting the benchmark in
[`benchmarks/`](benchmarks/README.md) over rather than taking on faith.

## Negation / antonym collapse detection

For each recognized negation cue (`not`, `never`, `without`, `cannot`, `unless`, ...)
or antonym-eligible word (~60 built-in pairs — `allowed`/`prohibited`,
`increase`/`decrease`, `legal`/`illegal`, ... — optionally extended with
WordNet via `use_wordnet=True` if `nltk` + its wordnet corpus are installed),
`whymatched` builds a counterfactual (negation removed, or word swapped for
its antonym), re-embeds it, and checks whether the similarity score moved by
less than `collapse_threshold` (default 10% relative change). If it didn't,
that's flagged: the embedding can't tell the two apart, which is exactly the
mechanism behind a retriever surfacing a chunk that contradicts the query.

This is a per-word diagnostic, not an overall relevance judgment — a chunk
can be flagged even when its ranking is otherwise reasonable. Tune
`collapse_threshold` (looser = fewer, more confident flags) and
`max_checks_per_side` (in `detect_collapse`) to taste.

## Projection (query vs. chunks in reduced space)

```python
result = debugger.analyze(query, chunks, projection_level="sentence")  # default: one point per chunk
result = debugger.analyze(query, chunks, projection_level="token")     # local models only: one point per token
```

`projection_method` is `"pca"` (default, no extra deps), `"tsne"` (scikit-learn), or
`"umap"` (`pip install whymatched[viz]`).

## CLI + HTML report

```bash
whymatched run \
  --query "Does the machine need supervision to run?" \
  --chunk "The machine runs without supervision." \
  --chunk "Only certified technicians are permitted to operate the machine." \
  --out report.html
```

Add `--provider openai --model text-embedding-3-small` to use a hosted API
instead of the default local MiniLM model, or `--json` to print the full
result as JSON instead of writing a report. See `whymatched run --help` for
all options (`--input file.json` for `{"query": ..., "chunks": [...]}`,
`--method`, `--projection-level`, `--projection-method`, ...).

## Known limitations

- Occlusion and collapse detection operate at whole-word granularity
  (regex word splitting), independent of the model's own subword tokenizer —
  this keeps explanations human-readable but means very short words or
  multi-word idioms aren't handled specially.
- The antonym dictionary is a curated ~60-pair list covering common
  policy/instruction-style contrasts; it is not exhaustive. Set
  `use_wordnet=True` (after `nltk.download("wordnet")`) to widen coverage.
- `collapse_threshold` is a heuristic, not a calibrated statistical test —
  treat flags as "worth a human look," not ground truth.
- Gradient and MaxSim attribution require `LocalModel` (full access to the
  differentiable forward pass); hosted APIs only support occlusion.
- Occlusion importance is confounded by sentence length: removing *any* word
  shifts a mean-pooled vector somewhat regardless of that word's actual
  meaning, so occlusion weights are a directional hint about what the model
  is keying on, not a clean decomposition of semantic importance. Integrated
  Gradients doesn't have this issue, but is local-model-only (see above).
- The collapse-detection recall rate on curated hard cases is measured, not
  assumed — see [`benchmarks/`](benchmarks/README.md) for the current numbers
  and how to reproduce them against your own embedding model.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[local]"
pip install pytest
pytest
```

## Author & License

Created by Daniel Zimnicki.

Released under the [MIT License](LICENSE) — a permissive open-source license: free to use, modify, and
distribute (including commercially), with no warranty. See the [LICENSE](LICENSE) file for the full text.
