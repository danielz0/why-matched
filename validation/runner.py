"""CLI for running evaluate_detector() against a real embedding model over
a ground-truth dataset (WS-A / A3, "the gate").

CLI shape mirrors benchmarks/negation_benchmark.py deliberately
(--model/--threshold/--markdown reused verbatim) plus flags for the
dataset/sampling/calibration knobs a validation run needs to be
reproducible.

Usage:
    python -m validation.runner \\
        --dataset nevir --split test --sample-size 200 --seed 0 \\
        --model sentence-transformers/all-MiniLM-L6-v2 --calibrate --markdown \\
        --out validation/results/all-MiniLM-L6-v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import random
from typing import List, Optional

from .datasets import LabelledPair, load_nevir
from .metrics import evaluate_detector


def _sample(pairs: List[LabelledPair], sample_size: Optional[int], seed: int) -> List[LabelledPair]:
    """Plain fixed-seed random sample of `pairs`, via a local RNG (never
    the global `random` module, per CONTRIBUTING.md's determinism
    requirement). NevIR has no PerturbationKind-category axis to stratify
    on (every row is "negation" by construction), so "stratified across
    categories" honestly reduces to a reproducible random sample of the
    split -- documented here rather than silently glossed over."""
    if sample_size is None or sample_size >= len(pairs):
        return list(pairs)
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    chosen = sorted(indices[:sample_size])
    return [pairs[i] for i in chosen]


def _sample_sha256(pairs: List[LabelledPair]) -> str:
    """Hash of the sorted sampled pair_ids -- a pragmatic identity hash
    that's always computable without extra HF-Hub revision plumbing,
    deterministic given (dataset, split, seed, sample_size)."""
    joined = "".join(sorted(p.pair_id for p in pairs))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        choices=["nevir"],
        default="nevir",
        help="ground-truth dataset to load (capretrieval is a documented stub -- "
        "see validation/README.md -- and deliberately excluded here)",
    )
    parser.add_argument("--split", default="test", help="dataset split (default: test)")
    parser.add_argument("--sample-size", type=int, default=None, help="sample this many pairs (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="sampling + detector RNG seed (default: 0)")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model to evaluate",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="collapse threshold (default: 0.10, matches Debugger's default)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="use calibrated (null-distribution) detection instead of the fixed threshold rule",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=None,
        help="restrict detect_collapse() to these perturbation kinds (default: all of "
        "DEFAULT_KINDS -- note this means a flag from an unrelated kind, e.g. antonym, "
        "still counts as 'flagged' against a purely negation-labelled dataset like NevIR; "
        "pass --kinds negation to isolate the pair's own ground-truth category)",
    )
    parser.add_argument("--n-null", type=int, default=50, help="nulls per side when --calibrate (default: 50)")
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.10,
        help="null-distribution quantile when --calibrate (default: 0.10)",
    )
    parser.add_argument(
        "--correction",
        choices=["none", "bh"],
        default="none",
        help="multiple-comparison correction when --calibrate (default: none)",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="alpha for --correction bh (default: 0.05)")
    parser.add_argument(
        "--fit-profile-split",
        choices=["train", "validation"],
        default=None,
        help="fit a whymatched.calibration.fit_pair_profile() corpus-level null profile from "
        "this NevIR split's (query, chunk) pairs and pass it into every detect_collapse() call "
        "-- lets short pairs that can't produce enough per-call SynonymNull candidates on their "
        "own (see validation/REPORT.md) still calibrate. 'test' is deliberately excluded from "
        "the choices: fitting and gating on the same split would leak the evaluated pairs into "
        "their own null reference. Incompatible with --correction bh (raised by "
        "evaluate_calibrated itself).",
    )
    parser.add_argument(
        "--noise-eps",
        type=float,
        default=0.01,
        help="ground-truth noise band: scores within this are treated as a genuine tie/failure (default: 0.01)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="also print a Markdown table (for pasting into validation/REPORT.md)",
    )
    parser.add_argument("--out", default=None, help="write the full DetectorMetrics JSON to this path")
    args = parser.parse_args(argv)

    from whymatched import LocalModel
    from whymatched.calibration import fit_pair_profile

    if args.dataset == "nevir":
        pairs = load_nevir(split=args.split)
    else:  # pragma: no cover - unreachable given `choices=["nevir"]` above
        raise ValueError(f"unknown dataset: {args.dataset!r}")

    pairs = _sample(pairs, args.sample_size, args.seed)
    sample_hash = _sample_sha256(pairs)

    model = LocalModel.from_sentence_transformers(args.model)

    profile = None
    if args.fit_profile_split is not None:
        fit_pairs_source = load_nevir(split=args.fit_profile_split)
        fit_pairs = []
        for p in fit_pairs_source:
            fit_pairs.extend([(p.query_a, p.doc_a), (p.query_b, p.doc_b)])
        profile = fit_pair_profile(model, fit_pairs, n_null=args.n_null, seed=args.seed)
        print(
            f"fit_pair_profile: split={args.fit_profile_split} n_pairs={len(fit_pairs)} "
            f"n_samples={profile.n_samples} p10={profile.null_delta_quantiles['p10']:.4f} "
            f"p50={profile.null_delta_quantiles['p50']:.4f}\n"
        )

    metrics = evaluate_detector(
        model,
        pairs,
        threshold=args.threshold,
        calibrate=args.calibrate,
        n_null=args.n_null,
        quantile=args.quantile,
        correction=args.correction,
        alpha=args.alpha,
        seed=args.seed,
        noise_eps=args.noise_eps,
        kinds=args.kinds,
        profile=profile,
    )
    metrics.config.update(
        dataset=args.dataset,
        split=args.split,
        sample_size=len(pairs),
        sample_seed=args.seed,
        sample_sha256=sample_hash,
        fit_profile_split=args.fit_profile_split,
    )

    print(metrics.summary())
    if args.markdown:
        print()
        print("| category | tp | fp | fn | tn | precision | recall | f1 |")
        print("|---|---|---|---|---|---|---|---|")
        print(
            f"| overall | {metrics.overall.tp} | {metrics.overall.fp} | {metrics.overall.fn} | "
            f"{metrics.overall.tn} | {metrics.overall.precision:.3f} | {metrics.overall.recall:.3f} | "
            f"{metrics.overall.f1:.3f} |"
        )
        for category in sorted(metrics.by_category):
            c = metrics.by_category[category]
            print(
                f"| {category} | {c.tp} | {c.fp} | {c.fn} | {c.tn} | {c.precision:.3f} | {c.recall:.3f} | {c.f1:.3f} |"
            )

    if args.out:
        metrics.save(args.out)
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
