"""Validation benchmark for whymatched's negation/antonym collapse detector.

The premise of this whole library is a claim: "embedding models often can't
tell 'X' from 'not X' or 'X' from its opposite, and this tool catches that."
That claim needs evidence, not just a single anecdote in the README.

This script runs the *real* collapse detector (`whymatched.collapse.detect_collapse`,
the same code path `Debugger.analyze()` uses) over two curated sets of
hand-built (query, chunk) pairs against a real embedding model:

  1. "hard" cases  - the chunk is a near-paraphrase of a plausible answer to
     the query, but negated or flipped to its antonym, so a bag-of-overlapping-
     words retriever would rank it highly despite it being the wrong answer.
     We report how many of these the detector flags (recall on known-hard cases).

  2. "control" cases - unrelated (query, chunk) pairs sharing no negation or
     antonym vocabulary. We report how many are (incorrectly) flagged, to show
     the detector isn't just flagging everything (false-positive rate).

These pairs are hand-built rather than pulled from a benchmark dataset, to
keep this script dependency-free (no extra download, no `datasets` package)
and reproducible offline. They're deliberately in the same spirit as the
contrastive negation pairs in NevIR (Weller et al., "NevIR: Negation in
Neural Information Retrieval", EACL 2024, https://aclanthology.org/2024.findings-eacl.111/)
which showed that even strong retrieval/rerank models are close to
random-chance on negation-flipped query/document pairs. NevIR itself isn't
bundled here; treat this script as our own smaller-scale check.

Usage:
    python benchmarks/negation_benchmark.py
    python benchmarks/negation_benchmark.py --model sentence-transformers/all-mpnet-base-v2
    python benchmarks/negation_benchmark.py --markdown   # emit a Markdown table too
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional

from whymatched.collapse import detect_collapse

# Each hard case: (label naming the cue/antonym being tested, query, chunk).
# The chunk is a plausible-looking, high-lexical-overlap "wrong answer" that
# a retriever could easily surface: it negates or flips the very word that
# determines whether it actually answers the query.
HARD_CASES = [
    ("negation: not", "Is the pool open in winter?", "The pool is not open in winter."),
    ("negation: never", "Does the office close early on Fridays?", "The office never closes early on Fridays."),
    ("negation: no", "Are there any seats left for the show?", "There are no seats left for the show."),
    ("negation: none", "Were any of the applicants qualified?", "None of the applicants were qualified."),
    ("negation: nobody", "Is someone monitoring the alarms overnight?", "Nobody is monitoring the alarms overnight."),
    ("negation: nothing", "Was anything found during the inspection?", "Nothing was found during the inspection."),
    ("negation: nowhere", "Can this part be bought somewhere nearby?", "This part can be found nowhere nearby."),
    ("negation: neither", "Did both candidates pass the exam?", "Neither candidate passed the exam."),
    ("negation: without", "Does the machine need supervision to run?", "The machine runs without supervision."),
    ("negation: unless", "Is the warranty valid for all repairs?", "The warranty is valid unless the repair was done by a third party."),
    ("negation: lacks", "Does the report include supporting data?", "The report lacks supporting data."),
    ("negation: failed to", "Did the vendor deliver the parts on schedule?", "The vendor failed to deliver the parts on schedule."),
    ("negation: unable to", "Can the support team resolve tickets same-day?", "The support team is unable to resolve tickets same-day."),
    ("negation: cannot", "Can visitors park in the staff lot?", "Visitors cannot park in the staff lot."),
    ("negation: except", "Are all rooms available for booking?", "All rooms are available for booking except the conference suite."),
    ("antonym: allowed/prohibited", "Is smoking allowed on the patio?", "Smoking is prohibited on the patio."),
    ("antonym: legal/illegal", "Is it legal to park here overnight?", "It is illegal to park here overnight."),
    ("antonym: required/optional", "Is a helmet required for the ride?", "A helmet is optional for the ride."),
    ("antonym: increase/decrease", "Will the new policy increase response times?", "The new policy will decrease response times."),
    ("antonym: more/less", "Does the premium plan give more storage?", "The premium plan gives less storage."),
    ("antonym: higher/lower", "Is the interest rate higher this year?", "The interest rate is lower this year."),
    ("antonym: before/after", "Should the form be submitted before the deadline?", "The form should be submitted after the deadline."),
    ("antonym: true/false", "Is the statement about the merger true?", "The statement about the merger is false."),
    ("antonym: open/closed", "Is the border open to travelers?", "The border is closed to travelers."),
    ("antonym: included/excluded", "Are taxes included in the price?", "Taxes are excluded from the price."),
    ("antonym: possible/impossible", "Is it possible to upgrade mid-contract?", "It is impossible to upgrade mid-contract."),
    ("antonym: win/lose", "Will the underdog win the final?", "The underdog will lose the final."),
    ("antonym: approved/rejected", "Was the proposal approved by the committee?", "The proposal was rejected by the committee."),
    ("antonym: above/below", "Is the temperature above freezing tonight?", "The temperature is below freezing tonight."),
    ("antonym: present/absent", "Was the CEO present at the meeting?", "The CEO was absent at the meeting."),
]

# Unrelated pairs with no negation cues or antonym-eligible vocabulary in
# common between query and chunk. A well-behaved detector should stay quiet.
CONTROL_CASES = [
    ("control: unrelated topic 1", "Where is the office located?", "The office is located in downtown Seattle."),
    ("control: unrelated topic 2", "How do I reset my password?", "Click the link in the email to choose a new password."),
    ("control: unrelated topic 3", "What time does the shuttle leave?", "The shuttle departs from the main lobby at 8am."),
    ("control: unrelated topic 4", "Who is the keynote speaker?", "Dr. Rivera will present the opening keynote."),
    ("control: unrelated topic 5", "What is the capital of France?", "Paris is the capital and largest city of France."),
    ("control: unrelated topic 6", "How many people attended the conference?", "Around three hundred people registered this year."),
    ("control: unrelated topic 7", "What color is the new logo?", "The redesigned logo uses a deep teal color."),
    ("control: unrelated topic 8", "When was the library built?", "Construction of the library finished in 1987."),
]


@dataclass
class CaseResult:
    label: str
    query: str
    chunk: str
    flagged: bool
    detail: Optional[str]


def _run_cases(model, cases, threshold: float) -> List[CaseResult]:
    results = []
    for label, query, chunk in cases:
        flags = detect_collapse(model, query, chunk, threshold=threshold)
        detail = None
        if flags:
            f = flags[0]
            detail = f"{f.kind} ({f.side}) '{f.trigger}': {f.base_score:.3f} -> {f.counterfactual_score:.3f} ({f.relative_delta*100:.1f}%)"
        results.append(CaseResult(label=label, query=query, chunk=chunk, flagged=bool(flags), detail=detail))
    return results


def _print_table(title: str, results: List[CaseResult]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for r in results:
        mark = "FLAGGED" if r.flagged else "clean"
        print(f"  [{mark:7}] {r.label:32} {r.detail or ''}")


def _markdown_table(title: str, results: List[CaseResult]) -> str:
    lines = [f"### {title}", "", "| case | flagged | detail |", "|---|---|---|"]
    for r in results:
        mark = "✅ flagged" if r.flagged else "clean"
        detail = (r.detail or "").replace("|", "\\|")
        lines.append(f"| {r.label} | {mark} | {detail} |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="sentence-transformers model to evaluate")
    parser.add_argument("--threshold", type=float, default=0.10, help="collapse_threshold (default: 0.10, matches Debugger's default)")
    parser.add_argument("--markdown", action="store_true", help="also print Markdown tables (for pasting into docs)")
    args = parser.parse_args(argv)

    from whymatched import LocalModel

    model = LocalModel.from_sentence_transformers(args.model)

    hard_results = _run_cases(model, HARD_CASES, args.threshold)
    control_results = _run_cases(model, CONTROL_CASES, args.threshold)

    n_hard_flagged = sum(r.flagged for r in hard_results)
    n_control_flagged = sum(r.flagged for r in control_results)

    print(f"model: {args.model}  threshold: {args.threshold}")
    _print_table("Hard cases (negation/antonym flips that should be flagged)", hard_results)
    _print_table("Control cases (unrelated pairs that should stay clean)", control_results)

    print("\nSummary")
    print("-------")
    print(f"  hard cases flagged:    {n_hard_flagged}/{len(hard_results)} ({100*n_hard_flagged/len(hard_results):.0f}%)")
    print(f"  control false positives: {n_control_flagged}/{len(control_results)} ({100*n_control_flagged/len(control_results):.0f}%)")

    if args.markdown:
        print("\n" + _markdown_table("Hard cases", hard_results))
        print("\n" + _markdown_table("Control cases", control_results))


if __name__ == "__main__":
    main()
