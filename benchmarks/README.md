# Validation benchmark

`whymatched`'s central claim is that embedding models often can't tell "X"
from "not X," or "X" from its opposite — and that the collapse detector
catches this. That claim shouldn't rest on a single anecdote in the README.
This directory runs the real detector (`whymatched.collapse.detect_collapse`,
the same code path `Debugger.analyze()` uses) over two curated sets of
`(query, chunk)` pairs against a real embedding model and reports how it did.

```bash
pip install "whymatched[local]"
python benchmarks/negation_benchmark.py
```

## Method

- **30 "hard" cases** — the chunk is a plausible, high-lexical-overlap
  "wrong answer" to the query: it negates or antonym-flips the one word that
  determines whether it actually answers the question (e.g. query "Does the
  machine need supervision to run?" / chunk "The machine runs without
  supervision."). These cover 15 distinct negation cues (`not`, `never`,
  `no`, `none`, `nobody`, `nothing`, `nowhere`, `neither`, `without`,
  `unless`, `lacks`, `failed to`, `unable to`, `cannot`, `except`) and 15
  antonym pairs from the built-in dictionary. A retriever ranking on lexical
  or topical overlap alone would happily surface these; the question is
  whether the collapse detector flags them.
- **8 neutral control cases** — unrelated `(query, chunk)` pairs sharing no
  negation or antonym vocabulary. A detector that flags everything would
  score 100% on the hard set for free, so the control set's job is to show
  the detector isn't just noise — it should stay quiet here.

Both sets are hand-built rather than pulled from an external dataset, to
keep this script dependency-free (no `datasets` package, no network access
beyond the model download) and reproducible offline. They're written in the
same spirit as the contrastive negation pairs in **NevIR** (Weller et al.,
["NevIR: Negation in Neural Information Retrieval"](https://aclanthology.org/2024.findings-eacl.111/),
EACL Findings 2024), which found that even strong retrieval and reranking
models perform close to random chance on negation-flipped query/document
pairs — this benchmark doesn't reuse NevIR's dataset, but is a smaller,
self-contained check in the same vein, and validates that `whymatched`'s
own diagnostic actually catches the failure mode it was built to catch.

## Results (`sentence-transformers/all-MiniLM-L6-v2`, default `threshold=0.10`)

| | flagged | rate |
|---|---|---|
| Hard cases (should be flagged) | 17 / 30 | 57% |
| Control cases (should stay clean) | 0 / 8 | 0% |

Reproduce with `python benchmarks/negation_benchmark.py --markdown`.

**Reading these numbers honestly:** 57% recall on curated hard cases is not
"solved" — it means the detector catches roughly half of the negation/antonym
flips it's explicitly designed to catch on the *default* threshold, with the
default MiniLM model, and zero false positives on the control set. That's the
real, current behavior of the shipped code, not a cherry-picked example. A
few things worth knowing:

- **Which cases it misses tend to share a pattern**: cues like `never`, `no`,
  `none`, `nobody`, `nothing`, `lacks`, `failed to`, and `cannot` didn't fire
  in this run — for many of these, the counterfactual (removing the cue)
  changes little else about the sentence, so the *raw* score moves by more
  than 10% even though the meaning-critical word is what changed; this
  benchmark's `threshold=0.10` is the same default `Debugger` ships with, so
  it's a fair test of what you get out of the box.
- **Recall improves with a looser `collapse_threshold`** (e.g. `0.15`–`0.2`)
  at the cost of more false positives elsewhere — tune it for your corpus
  rather than trusting the default blindly.
- **A different base model will produce different numbers.** Run
  `python benchmarks/negation_benchmark.py --model <your-model>` against
  whatever embedding model your application actually uses; that number, not
  this one, is the one that matters for your retrieval pipeline.

## Full per-case results

<details>
<summary>Expand for the case-by-case table (regenerate with <code>--markdown</code>)</summary>

### Hard cases

| case | flagged | detail |
|---|---|---|
| negation: not | ✅ flagged | negation_collapse (chunk) 'not': 0.873 -> 0.877 (0.4%) |
| negation: never | clean | |
| negation: no | clean | |
| negation: none | clean | |
| negation: nobody | clean | |
| negation: nothing | clean | |
| negation: nowhere | ✅ flagged | antonym_collapse (query) 'Can -> cannot': 0.593 -> 0.649 (9.6%) |
| negation: neither | ✅ flagged | negation_collapse (chunk) 'Neither': 0.763 -> 0.834 (9.3%) |
| negation: without | ✅ flagged | negation_collapse (chunk) 'without': 0.826 -> 0.828 (0.1%) |
| negation: unless | ✅ flagged | negation_collapse (chunk) 'unless': 0.744 -> 0.716 (3.7%) |
| negation: lacks | clean | |
| negation: failed to | clean | |
| negation: unable to | ✅ flagged | antonym_collapse (chunk) 'support -> oppose': 0.783 -> 0.707 (9.7%) |
| negation: cannot | clean | |
| negation: except | ✅ flagged | negation_collapse (chunk) 'except': 0.835 -> 0.761 (8.9%) |
| antonym: allowed/prohibited | ✅ flagged | antonym_collapse (query) 'allowed -> prohibited': 0.936 -> 0.956 (2.1%) |
| antonym: legal/illegal | ✅ flagged | antonym_collapse (query) 'legal -> illegal': 0.948 -> 0.963 (1.7%) |
| antonym: required/optional | ✅ flagged | antonym_collapse (query) 'required -> optional': 0.877 -> 0.941 (7.2%) |
| antonym: increase/decrease | ✅ flagged | antonym_collapse (query) 'increase -> diminish': 0.875 -> 0.865 (1.2%) |
| antonym: more/less | ✅ flagged | antonym_collapse (query) 'more -> less': 0.920 -> 0.935 (1.7%) |
| antonym: higher/lower | ✅ flagged | antonym_collapse (query) 'higher -> lower': 0.822 -> 0.898 (9.2%) |
| antonym: before/after | ✅ flagged | antonym_collapse (query) 'before -> after': 0.879 -> 0.894 (1.7%) |
| antonym: true/false | ✅ flagged | antonym_collapse (query) 'true -> false': 0.869 -> 0.948 (9.1%) |
| antonym: open/closed | clean | |
| antonym: included/excluded | clean | |
| antonym: possible/impossible | clean | |
| antonym: win/lose | ✅ flagged | antonym_collapse (query) 'win -> lose': 0.916 -> 0.932 (1.7%) |
| antonym: approved/rejected | clean | |
| antonym: above/below | ✅ flagged | antonym_collapse (query) 'above -> below': 0.850 -> 0.876 (3.1%) |
| antonym: present/absent | clean | |

### Control cases

| case | flagged | detail |
|---|---|---|
| control: unrelated topic 1 | clean | |
| control: unrelated topic 2 | clean | |
| control: unrelated topic 3 | clean | |
| control: unrelated topic 4 | clean | |
| control: unrelated topic 5 | clean | |
| control: unrelated topic 6 | clean | |
| control: unrelated topic 7 | clean | |
| control: unrelated topic 8 | clean | |

</details>
