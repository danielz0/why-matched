"""A small hand-crafted eval set of (query, chunk) pairs designed to trigger
negation collapse and near-synonym/antonym collapse, for demoing and
regression-checking the flagging logic against a real embedding model.

Run: python examples/demo_negation_and_synonym_pairs.py
"""
from whymatched import Debugger, LocalModel

CASES = [
    {
        "name": "negation flip (policy)",
        "query": "Is remote work allowed for contractors?",
        "chunk": "Remote work is not allowed for contractors under this policy.",
    },
    {
        "name": "negation flip (safety)",
        "query": "Is this chemical safe to store near open flame?",
        "chunk": "This chemical is not safe to store near open flame.",
    },
    {
        "name": "antonym flip (legal status)",
        "query": "Is this activity legal in the state of California?",
        "chunk": "This activity is illegal in the state of California.",
    },
    {
        "name": "antonym flip (direction)",
        "query": "Did revenue increase in Q3?",
        "chunk": "Revenue decreased sharply in Q3.",
    },
    {
        "name": "unrelated control (no negation/antonym vocabulary, should NOT flag)",
        "query": "Where is the office located?",
        "chunk": "The office is located in downtown Seattle.",
    },
]

if __name__ == "__main__":
    model = LocalModel.from_sentence_transformers("sentence-transformers/all-MiniLM-L6-v2")
    debugger = Debugger(model)

    for case in CASES:
        result = debugger.analyze(case["query"], [case["chunk"]], project=False)
        c = result.chunks[0]
        flags = ", ".join(f.kind for f in c.collapse_flags) or "none"
        print(f"[{case['name']}]  score={c.score:.3f}  flags={flags}")
