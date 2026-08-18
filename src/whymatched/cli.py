from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

from .batch import CaseLoadError, load_cases, scan
from .calibration import CalibrationProfile, fit_profile
from .core import AnalysisResult, Debugger
from .report import render_batch_html, write_html
from .testing import assert_collapse_rate, assert_not_worse_than


def _load_chunks(args: argparse.Namespace) -> tuple:
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        query = data["query"]
        chunks = data["chunks"]
        return query, chunks

    if not args.query:
        raise SystemExit("--query is required unless --input is given")

    chunks: List[str] = list(args.chunk or [])
    if args.chunks_file:
        with open(args.chunks_file, "r", encoding="utf-8") as f:
            text = f.read()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                chunks.extend(parsed)
            else:
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            chunks.extend([line for line in text.splitlines() if line.strip()])

    if not chunks:
        raise SystemExit("no chunks provided: use --chunk (repeatable), --chunks-file, or --input")

    return args.query, chunks


class _HashFakeModel:
    """Deterministic, offline, torch-free embedding model used ONLY when
    WHYMATCHED_TEST_FAKE_MODEL is set. Internal test scaffolding, not a
    public provider -- intentionally absent from --provider's choices so
    it can never be selected by a real user. Exists so subprocess-based
    CLI tests can spawn `python -m whymatched.cli ...` as a real process
    and run fast, fully offline, without a HF download, torch, or an API
    key."""

    name = "hash-fake"
    supports_gradients = False
    supports_token_embeddings = False

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def embed(self, texts):
        import hashlib

        import numpy as np

        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for w in t.lower().split():
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % self.dim
                v[h] += 1.0
            vecs.append(v)
        return np.stack(vecs)

    def tokenize(self, text):
        return text.split()


def _build_model(args: argparse.Namespace):
    if os.environ.get("WHYMATCHED_TEST_FAKE_MODEL"):
        return _HashFakeModel()

    if args.provider == "local":
        from .models.local import LocalModel

        return LocalModel.from_sentence_transformers(args.model)

    from .models.api import APIEmbeddingModel

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.provider == "openai":
        return APIEmbeddingModel.openai(model=args.model, api_key=api_key)
    if args.provider == "cohere":
        return APIEmbeddingModel.cohere(model=args.model, api_key=api_key)
    if args.provider == "voyage":
        return APIEmbeddingModel.voyage(model=args.model, api_key=api_key)
    raise SystemExit(f"unknown provider: {args.provider}")


def _result_to_json(result: AnalysisResult) -> dict:
    d = result.to_dict()
    return d


def _cmd_run(args: argparse.Namespace) -> None:
    query, chunks = _load_chunks(args)
    model = _build_model(args)
    debugger = Debugger(model, method=args.method, gradient_baseline=args.gradient_baseline)
    result = debugger.analyze(
        query,
        chunks,
        project=not args.no_projection,
        projection_level=args.projection_level,
        projection_method=args.projection_method,
        detect_collapse_flags=not args.no_collapse,
    )
    if args.json:
        json.dump(_result_to_json(result), sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        write_html(result, args.out)
        print(f"wrote {args.out}")


def _parse_kind_rate_pairs(pairs: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--max-collapse-rate-kind expects KIND=RATE, got {p!r}")
        kind, _, raw = p.partition("=")
        if not kind:
            raise ValueError(f"--max-collapse-rate-kind expects KIND=RATE, got {p!r} (empty kind)")
        try:
            out[kind] = float(raw)
        except ValueError:
            raise ValueError(f"--max-collapse-rate-kind: {raw!r} is not a valid float (kind={kind!r})") from None
    return out


def _load_batch_report_json(path: str):
    from .batch import BatchReport, CaseReport

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = [
        CaseReport(
            case_id=c["case_id"], query=c["query"], n_chunks=c["n_chunks"], top_score=c["top_score"],
            results=[], error=c.get("error"),
        )
        for c in data["cases"]
    ]
    from .perturbations import PerturbationResult

    for c, raw in zip(cases, data["cases"]):
        c.results = [
            PerturbationResult(
                kind=r["kind"], side=r["side"], trigger=r["trigger"],
                counterfactual_snippet=r["counterfactual_snippet"], base_score=r["base_score"],
                counterfactual_score=r["counterfactual_score"], relative_delta=r["relative_delta"],
                null_quantile=r.get("null_quantile"), null_quantile_ci=tuple(r["null_quantile_ci"]) if r.get("null_quantile_ci") else None,
                ratio=r.get("ratio"), z_score=r.get("z_score"), p_value=r.get("p_value"), q_value=r.get("q_value"),
                decision_rule=r.get("decision_rule", "threshold"), calibration_status=r.get("calibration_status", "ok"),
                is_collapse=r.get("is_collapse", False),
            )
            for r in raw["results"]
        ]
    return BatchReport(
        model_name=data["model_name"], whymatched_version=data["whymatched_version"],
        created_at=data["created_at"], config=data["config"], cases=cases,
        n_skipped=data["n_skipped"], n_errored=data["n_errored"],
    )


def _render_summary_table(report) -> str:
    lines = [
        f"model: {report.model_name}  whymatched: {report.whymatched_version}",
        f"cases: {len(report.cases)}  skipped: {report.n_skipped}  errored: {report.n_errored}",
        f"collapse_rate: {report.collapse_rate():.4f}  candidate_collapse_rate: {report.candidate_collapse_rate():.4f}",
    ]
    rates = report.rate_by_kind()
    if rates:
        lines.append("rate_by_kind: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(rates.items())))
    return "\n".join(lines)


def _cmd_scan(args: argparse.Namespace) -> int:
    try:
        cases = load_cases(args.input)
    except (CaseLoadError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        kind_rates = _parse_kind_rate_pairs(args.max_collapse_rate_kind)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    profile = None
    if args.profile:
        try:
            profile = CalibrationProfile.load(args.profile)
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    baseline = None
    if args.baseline:
        try:
            baseline = _load_batch_report_json(args.baseline)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    if args.fail_on_regression and baseline is None:
        print("error: --fail-on-regression requires --baseline", file=sys.stderr)
        return 2

    model = _build_model(args)
    kinds = [k.strip() for k in args.kinds.split(",")] if args.kinds else None
    top_k = None if args.top_k == 0 else args.top_k

    report = scan(
        model, cases, kinds=kinds, calibrate=args.calibrate, n_null=args.n_null,
        quantile=args.quantile, correction=args.correction, alpha=args.alpha,
        profile=profile, top_k_chunks=top_k, max_workers=args.max_workers, seed=args.seed,
    )

    if args.out:
        render_batch_html(report, args.out)
    if args.json:
        payload = json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str)
        if args.json == "-":
            sys.stdout.write(payload + "\n")
        else:
            with open(args.json, "w", encoding="utf-8") as f:
                f.write(payload)

    print(_render_summary_table(report), file=sys.stderr)

    try:
        if args.max_collapse_rate is not None:
            assert_collapse_rate(report, max_rate=args.max_collapse_rate)
        for kind, rate in kind_rates.items():
            assert_collapse_rate(report, max_rate=rate, kind=kind)
        if args.fail_on_regression:
            assert_not_worse_than(report, baseline, tolerance=args.regression_tolerance)
    except AssertionError as e:
        print(str(e), file=sys.stderr)
        return 1

    return 3 if report.n_errored > 0 else 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    try:
        with open(args.corpus, "r", encoding="utf-8") as f:
            corpus_texts = [line.strip() for line in f if line.strip()]
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not corpus_texts:
        print(f"error: {args.corpus}: no non-empty lines", file=sys.stderr)
        return 2

    model = _build_model(args)
    profile = fit_profile(model, corpus_texts, n_null=args.n_null, seed=args.seed)
    try:
        profile.save(args.out)
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _add_provider_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--provider", default="local", choices=["local", "openai", "cohere", "voyage"],
        help="embedding backend (default: local sentence-transformers model)",
    )
    p.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2",
        help="model name (HF hub id for --provider local, API model name otherwise)",
    )
    p.add_argument("--api-key-env", default=None, help="env var holding the API key (defaults per-provider)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="whymatched", description="Embedding retrieval debugger")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Analyze a (query, chunks) pair and emit a report or JSON")
    run.add_argument("--query", help="the search query")
    run.add_argument("--chunk", action="append", help="a retrieved chunk (repeatable)")
    run.add_argument("--chunks-file", help="file with one chunk per line, or a JSON list of strings")
    run.add_argument("--input", help="JSON file with {\"query\": ..., \"chunks\": [...]}, overrides --query/--chunk")
    _add_provider_args(run)
    run.add_argument(
        "--method", default="auto", choices=["auto", "occlusion", "integrated_gradients", "maxsim"],
        help="attribution method (default: auto = gradients for local models, occlusion for API models)",
    )
    run.add_argument(
        "--gradient-baseline", default="pad", choices=["pad", "mask", "zero"],
        help="Integrated Gradients reference point (default: pad-token embedding)",
    )
    run.add_argument("--no-collapse", action="store_true", help="skip negation/antonym collapse detection")
    run.add_argument("--no-projection", action="store_true", help="skip the 2D projection")
    run.add_argument("--projection-level", default="sentence", choices=["sentence", "token"])
    run.add_argument("--projection-method", default="pca", choices=["pca", "tsne", "umap"])
    run.add_argument("--out", default="whymatched_report.html", help="output HTML report path")
    run.add_argument("--json", action="store_true", help="print JSON to stdout instead of writing an HTML report")

    scan_p = sub.add_parser("scan", help="Batch-scan (query, chunks) cases and gate on a collapse-rate threshold")
    scan_p.add_argument("--input", required=True, help="JSONL (or JSON array) file of eval cases")
    _add_provider_args(scan_p)
    scan_p.add_argument("--kinds", default=None, help="comma-separated perturbation kinds (default: all non-entity kinds)")
    scan_p.add_argument("--top-k", type=int, default=3, help="top-K chunks per case to evaluate; 0 = all chunks")
    scan_p.add_argument("--calibrate", dest="calibrate", action="store_true", default=True, help="use the calibrated quantile rule (default)")
    scan_p.add_argument("--no-calibrate", dest="calibrate", action="store_false", help="use the fixed threshold rule instead")
    scan_p.add_argument("--n-null", type=int, default=50)
    scan_p.add_argument("--quantile", type=float, default=0.10)
    scan_p.add_argument("--correction", default="none", choices=["none", "bh"])
    scan_p.add_argument("--alpha", type=float, default=0.05)
    scan_p.add_argument("--profile", default=None, help="CalibrationProfile JSON path (from `whymatched calibrate`)")
    scan_p.add_argument("--seed", type=int, default=0)
    scan_p.add_argument("--max-workers", type=int, default=1)
    scan_p.add_argument("--out", default=None, help="batch HTML report path")
    scan_p.add_argument("--json", default=None, metavar="PATH", help="write JSON here, or '-' for stdout")
    scan_p.add_argument("--max-collapse-rate", type=float, default=None)
    scan_p.add_argument("--max-collapse-rate-kind", action="append", default=[], metavar="KIND=RATE")
    scan_p.add_argument("--baseline", default=None, help="previous scan's --json output, for --fail-on-regression")
    scan_p.add_argument("--fail-on-regression", action="store_true")
    scan_p.add_argument("--regression-tolerance", type=float, default=0.0)

    calib = sub.add_parser("calibrate", help="Fit a CalibrationProfile against a corpus")
    calib.add_argument("--corpus", required=True, help="text file, one text per line")
    calib.add_argument("--out", required=True, help="output CalibrationProfile JSON path")
    _add_provider_args(calib)
    calib.add_argument("--n-null", type=int, default=200)
    calib.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    if args.command == "run":
        _cmd_run(args)
        return 0
    if args.command == "scan":
        return _cmd_scan(args)
    if args.command == "calibrate":
        return _cmd_calibrate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
