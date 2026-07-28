"""Run: python examples/quickstart.py
(needs `pip install whymatched[local]`, downloads a small model on first run)"""
from whymatched import Debugger, LocalModel

model = LocalModel.from_sentence_transformers("sentence-transformers/all-MiniLM-L6-v2")
debugger = Debugger(model)  # method="auto" -> Integrated Gradients for local models

# Classic RAG failure: the top-scoring chunk says the OPPOSITE of what's true.
result = debugger.analyze(
    query="Is remote work allowed for contractors?",
    chunks=[
        "Remote work is not allowed for contractors under this policy.",
        "Employees may take unlimited vacation days.",
        "Contractors must complete onboarding within their first week.",
    ],
)

print(f"query: {result.query}")
print(f"model: {result.model_name}  method: {result.method}\n")

for c in result.chunks:
    print(f"#{c.rank + 1}  score={c.score:.3f}  {c.chunk!r}")
    top = c.top_chunk_tokens[:5]
    print("   driving tokens:", [(t.token, round(t.weight, 3)) for t in top])
    for flag in c.collapse_flags:
        print(
            f"   ! {flag.kind} ({flag.side}): '{flag.trigger}' barely moves the score "
            f"({flag.base_score:.3f} -> {flag.counterfactual_score:.3f}, "
            f"{flag.relative_delta*100:.1f}% relative change)"
        )
    print()
