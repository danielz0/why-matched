# whymatched

**Why did this chunk match?** A retrieval debugger for embedding-based
search/RAG: given a query, a set of retrieved chunks, and an embedding
model, it tells you which words actually drove the similarity score, flags
cases where negation or antonym flips barely move the score (a huge and
under-diagnosed source of bad retrieval), and projects the query and chunks
into 2D so you can see the geometry behind a ranking.

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
    query="Is remote work allowed for contractors?",
    chunks=[
        "Remote work is not allowed for contractors under this policy.",
        "Employees may take unlimited vacation days.",
    ],
)

for c in result.chunks:
    print(c.rank, c.score, c.chunk)
    print("driving tokens:", [(t.token, t.weight) for t in c.top_chunk_tokens[:5]])
    for flag in c.collapse_flags:
        print("COLLAPSE:", flag.kind, flag.trigger, flag.base_score, "->", flag.counterfactual_score)
```

Run it and the top hit — which says the *opposite* of what's true — scores
0.86 similarity, and `collapse_flags` tells you exactly why: removing "not"
only moves the score from 0.860 to 0.925 (7.5% relative change). See
`examples/quickstart.py` and `examples/demo_negation_and_synonym_pairs.py`
— see [`examples/README.md`](examples/README.md) for a field-by-field guide
to interpreting the output.

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
  --query "Is remote work allowed for contractors?" \
  --chunk "Remote work is not allowed for contractors under this policy." \
  --chunk "Employees may take unlimited vacation days." \
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
