# Examples

## `quickstart.py`

Minimal end-to-end usage of the library: load a local model, run
`Debugger.analyze()`, and print the result.

```bash
pip install "whymatched[local]"
python examples/quickstart.py
```

## `demo_negation_and_synonym_pairs.py`

A small hand-crafted set of `(query, chunk)` pairs designed to exercise
negation collapse and antonym collapse against a real embedding model, plus
one neutral control pair that should not trigger any flag. Useful as a quick
regression check and as a demo of the tool's core value proposition.

## `sample_report.html`

A pre-generated HTML report (see the CLI's `--out` option) for the same
example query, showing the token-highlighted attribution view and the
query/chunk projection.

---

## Interpreting `quickstart.py`'s output

Running the script produces output like this:

```
query: Does the machine need supervision to run?
model: sentence-transformers/all-MiniLM-L6-v2  method: integrated_gradients

#1  score=0.826  'The machine runs without supervision.'
   driving tokens: [('supervision', 0.257), ('machine', 0.189), ('runs', 0.142), ('the', 0.062), ('without', 0.046)]
   ! negation_collapse (chunk): 'without' barely moves the score (0.826 -> 0.828, 0.1% relative change)

#2  score=0.430  'Only certified technicians are permitted to operate the machine.'
   driving tokens: [('machine', 0.128), ('operate', 0.076), ('permitted', 0.044), ('technicians', 0.025), ('the', 0.016)]

#3  score=0.159  'Routine maintenance should be scheduled every six months.'
   driving tokens: [('scheduled', 0.073), ('maintenance', 0.032), ('every', 0.016), ('should', 0.009), ('.', 0.004)]
```

Below is a field-by-field guide to reading it.

### Header

```
query: Does the machine need supervision to run?
model: sentence-transformers/all-MiniLM-L6-v2  method: integrated_gradients
```

The query text, the embedding model used, and the attribution method
`Debugger` selected automatically. `integrated_gradients` is chosen for
local models (full access to gradients); an API-backed model would fall
back to `occlusion`.

### Per-chunk block

```
#1  score=0.860  'Remote work is not allowed for contractors under this policy.'
```

- **`#1`** — rank by similarity, highest first.
- **`score`** — cosine similarity between the query and this chunk's
  embedding, typically in the 0–1 range for related text. This is the same
  number your retriever ranks on. It reflects the model's judgment of
  similarity, not factual correctness or relevance to the user's intent.

```
driving tokens: [('supervision', 0.257), ('machine', 0.189), ('runs', 0.142), ('the', 0.062), ('without', 0.046)]
```

The chunk's words ranked by their contribution to the score above. With
`integrated_gradients`, these weights approximately decompose the score
itself: a positive weight means the token pushed the similarity **up**; a
negative weight (not shown here, since only the top 5 are printed) would
mean it pulled the score **down**. Magnitude indicates relative importance.
In this example, "supervision" contributed the most to the match — and
"without" the *least* among the top five. That's not a coincidence: it's
the same blind spot the collapse flag below calls out from a different
angle. The one word that reverses this chunk's meaning barely registers in
either the attribution or the score.

```
! negation_collapse (chunk): 'without' barely moves the score (0.826 -> 0.828, 0.1% relative change)
```

This is the key diagnostic line. It means: the word "without" was deleted
from the chunk, the result was re-embedded, and the score barely moved at
all — from 0.826 to 0.828, a 0.1% relative change. Since "without" is
precisely the word that makes this chunk assert the **opposite** of what
the query asks ("does it need supervision" vs. "it runs without
supervision"), a near-zero relative change means the model is essentially
blind to its presence. That is a genuine blind spot: the embedding cannot
reliably distinguish "needs supervision" from "runs without supervision,"
yet this chunk is still ranked first.

### Why chunks #2 and #3 have no flags

Chunks #2 and #3 print no `!` lines at all. That's not the same as "verified
correct" — collapse detection only fires a check when it finds a
recognized negation cue or antonym-eligible word to test in the first
place. Neither chunk contains one (`permitted`, `certified`, `scheduled`,
etc. aren't in the negation-cue list or the antonym dictionary), so no
counterfactual was built and nothing was tested. Read a clean chunk as "no
applicable test was triggered," not as a positive guarantee — see
`collapse_threshold` and the antonym dictionary's coverage limits in the
main [README](../README.md#known-limitations).

### Practical guidance

- **High score + collapse flag** — worth investigating. The retriever may
  be surfacing a chunk that contradicts the query's intent.
- **Low score + collapse flag** — usually low priority. The chunk was not
  competitive for retrieval regardless of the flagged word.
- **Driving tokens** show what the model is actually keying on, which
  helps distinguish a score driven by the substantive answer word from one
  driven by generic overlapping vocabulary (e.g. "machine," "runs").
- **This isn't a cherry-picked example.** See [`../benchmarks/`](../benchmarks/README.md)
  for a 30-case negation/antonym benchmark, plus a neutral control set, that
  measures how often the real detector catches this failure mode and how
  often it stays quiet when it should.

For a visual equivalent of this same information — color-coded token
highlighting and a query/chunk projection plot — generate an HTML report
instead of using the library directly:

```bash
whymatched run \
  --query "Does the machine need supervision to run?" \
  --chunk "The machine runs without supervision." \
  --chunk "Only certified technicians are permitted to operate the machine." \
  --out report.html
```

See `sample_report.html` in this directory for a pre-generated example.
