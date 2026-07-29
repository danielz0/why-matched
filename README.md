# WhyMatched

[![License: MIT](https://img.shields.io/github/license/danielz0/why-matched?color=FF5CC8)](https://github.com/danielz0/why-matched/blob/main/LICENSE)
[![Coverage](coverage.svg)](https://github.com/danielz0/why-matched/actions/workflows/ci.yml)
[![GitHub Stars](https://img.shields.io/github/stars/danielz0/why-matched?style=social)](https://github.com/danielz0/why-matched)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/danielz0?color=FF5CC8)](https://github.com/sponsors/danielz0)
[![PyPI](https://img.shields.io/pypi/v/whymatched?color=7B61FF)](https://pypi.org/project/whymatched/)


**Why did this chunk match?** 

A retrieval debugger for embedding-based search/RAG. Given a query and a set
of retrieved chunks, it tells you which words drove the similarity score,
flags negation/antonym collapse (a chunk saying the opposite of what the
query asked, with the embedding barely noticing), and projects query +
chunks into 2D.

## Data flow

```
query, chunks, model, options
  -> embed query + chunks, rank by cosine similarity
  -> for each chunk (ranked):
       - attribute(method): occlusion | integrated_gradients | maxsim
       - detect_collapse(): find negations/antonyms, build counterfactuals,
         re-embed, flag if score barely moves
  -> project(): PCA / t-SNE / UMAP over query + chunk embeddings
  -> AnalysisResult { query, model_name, method, chunks[], projection }
```

## File tree

```
whymatched/
├── src/whymatched/
│   ├── core.py                  # Debugger, analyze(), AnalysisResult
│   ├── collapse.py              # negation/antonym counterfactual detection
│   ├── projection.py            # PCA / t-SNE / UMAP
│   ├── report.py                # self-contained HTML report
│   ├── cli.py                   # `whymatched run ...`
│   ├── text.py, utils.py
│   ├── attribution/
│   │   ├── occlusion.py         # leave-one-word-out (any backend)
│   │   ├── gradient.py          # Captum Integrated Gradients (local models)
│   │   └── maxsim.py            # ColBERT-style token-pair attribution
│   ├── models/
│   │   ├── local.py             # LocalModel (sentence-transformers/HF)
│   │   └── api.py               # APIEmbeddingModel (OpenAI/Cohere/Voyage/custom)
│   └── data/antonym_pairs.json
├── examples/
│   ├── quickstart.py
│   ├── demo_negation_and_synonym_pairs.py
│   └── sample_report.html
├── benchmarks/                  # negation/antonym benchmark + control set
└── tests/
```

## Build

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[local]"
pip install pytest
pytest
```

## Install

```bash
pip install whymatched                # core: occlusion + collapse + PCA
pip install whymatched[local]         # + Integrated Gradients + MaxSim
pip install whymatched[openai]        # + hosted API client
```

## Run the example

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

## Author & License

Created by Daniel Zimnicki.

Released under the [MIT License](LICENSE) — a permissive open-source license: free to use, modify, and
distribute (including commercially), with no warranty. See the [LICENSE](LICENSE) file for the full text.
