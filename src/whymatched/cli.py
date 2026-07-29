from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .core import AnalysisResult, Debugger
from .report import write_html


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


def _build_model(args: argparse.Namespace):
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


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="whymatched", description="Embedding retrieval debugger")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Analyze a (query, chunks) pair and emit a report or JSON")
    run.add_argument("--query", help="the search query")
    run.add_argument("--chunk", action="append", help="a retrieved chunk (repeatable)")
    run.add_argument("--chunks-file", help="file with one chunk per line, or a JSON list of strings")
    run.add_argument("--input", help="JSON file with {\"query\": ..., \"chunks\": [...]}, overrides --query/--chunk")
    run.add_argument(
        "--provider", default="local", choices=["local", "openai", "cohere", "voyage"],
        help="embedding backend (default: local sentence-transformers model)",
    )
    run.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2",
        help="model name (HF hub id for --provider local, API model name otherwise)",
    )
    run.add_argument("--api-key-env", default=None, help="env var holding the API key (defaults per-provider)")
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

    args = parser.parse_args(argv)

    if args.command == "run":
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


if __name__ == "__main__":
    main()
