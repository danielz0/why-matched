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
query: Is remote work allowed for contractors?
model: sentence-transformers/all-MiniLM-L6-v2  method: integrated_gradients

#1  score=0.860  'Remote work is not allowed for contractors under this policy.'
   driving tokens: [('contractors', 0.162), ('work', 0.138), ('remote', 0.126), ('allowed', 0.092), ('policy', 0.066)]
   ! negation_collapse (chunk): 'not' barely moves the score (0.860 -> 0.925, 7.5% relative change)
   ! antonym_collapse (query): 'allowed -> prohibited' barely moves the score (0.860 -> 0.891, 3.6% relative change)
   ! antonym_collapse (chunk): 'allowed -> prohibited' barely moves the score (0.860 -> 0.836, 2.8% relative change)

#2  score=0.368  'Contractors must complete onboarding within their first week.'
   driving tokens: [('contractors', 0.135), ('onboard', 0.083), ('must', 0.057), ('complete', 0.02), ('week', 0.016)]
   ! antonym_collapse (query): 'allowed -> prohibited' barely moves the score (0.368 -> 0.344, 6.5% relative change)
   ! antonym_collapse (chunk): 'must -> may' barely moves the score (0.368 -> 0.365, 0.9% relative change)

#3  score=0.237  'Employees may take unlimited vacation days.'
   driving tokens: [('employees', 0.092), ('vacation', 0.051), ('unlimited', 0.038), ('days', 0.03), ('take', 0.025)]
   ! antonym_collapse (query): 'allowed -> prohibited' barely moves the score (0.237 -> 0.253, 6.5% relative change)
   ! antonym_collapse (chunk): 'may -> must' barely moves the score (0.237 -> 0.245, 3.2% relative change)
```

Below is a field-by-field guide to reading it.

### Header

```
query: Is remote work allowed for contractors?
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
driving tokens: [('contractors', 0.162), ('work', 0.138), ('remote', 0.126), ('allowed', 0.092), ('policy', 0.066)]
```

The chunk's words ranked by their contribution to the score above. With
`integrated_gradients`, these weights approximately decompose the score
itself: a positive weight means the token pushed the similarity **up**; a
negative weight (not shown here, since only the top 5 are printed) would
mean it pulled the score **down**. Magnitude indicates relative importance.
In this example, "contractors" contributed the most to the match, "policy"
the least among the top five.

```
! negation_collapse (chunk): 'not' barely moves the score (0.860 -> 0.925, 7.5% relative change)
```

This is the key diagnostic line. It means: the word "not" was deleted from
the chunk, the result was re-embedded, and the score only moved from 0.860
to 0.925 — a 7.5% relative change. Since "not" is precisely the word that
makes this chunk assert the **opposite** of what the query asks, a small
relative change indicates the model is largely insensitive to its presence.
That is a genuine blind spot: the embedding cannot reliably distinguish
"allowed" from "not allowed," yet this chunk is still ranked first.

```
! antonym_collapse (query): 'allowed -> prohibited' barely moves the score
! antonym_collapse (chunk): 'allowed -> prohibited' barely moves the score
```

The same test applied differently: instead of deleting a negation, a word
is swapped for its antonym (`allowed -> prohibited`), once on the query
side and once on the chunk side. Both again show only a small relative
change — two independent confirmations of the same underlying weakness.

### Why the same flag appears on chunks #2 and #3

```
#2  score=0.368  ... ! antonym_collapse (query): 'allowed -> prohibited' ...
#3  score=0.237  ... ! antonym_collapse (query): 'allowed -> prohibited' ...
```

The `(query)`-side check swaps "allowed" for "prohibited" in the query
only, holding each chunk fixed. It measures whether the query's own
embedding is sensitive to that meaning flip — a property of the query, not
of any particular chunk — so it fires against every chunk in the batch.
This should not be read as "chunk #3 is a dangerous mismatch": chunk #3
scored low (0.237) and was never competitive for the top rank. **A collapse
flag is far more actionable when it appears on a high-scoring chunk than on
a low-scoring one.**

### Practical guidance

- **High score + collapse flag** — worth investigating. The retriever may
  be surfacing a chunk that contradicts the query's intent.
- **Low score + collapse flag** — usually low priority. The chunk was not
  competitive for retrieval regardless of the flagged word.
- **Driving tokens** show what the model is actually keying on, which
  helps distinguish a score driven by the substantive answer word from one
  driven by generic overlapping vocabulary (e.g. "work," "contractors").

For a visual equivalent of this same information — color-coded token
highlighting and a query/chunk projection plot — generate an HTML report
instead of using the library directly:

```bash
whymatched run \
  --query "Is remote work allowed for contractors?" \
  --chunk "Remote work is not allowed for contractors under this policy." \
  --chunk "Employees may take unlimited vacation days." \
  --out report.html
```

See `sample_report.html` in this directory for a pre-generated example.
