"""Command line interface for the HEI-SETI pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .logging_conf import setup_logging
from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hei-seti", description="High-Energy Astrobiology toolkit")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to pipeline config YAML")

    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch HEASARC tables")
    fetch_parser.add_argument("--tables", nargs="*", help="Override tables to fetch")
    fetch_parser.add_argument("--output", default="data/raw.parquet")

    featurize_parser = subparsers.add_parser("featurize", help="Engineer features and KB metrics")
    featurize_parser.add_argument("--input", default="data/raw.parquet")
    featurize_parser.add_argument("--output", default="data/features.parquet")

    train_parser = subparsers.add_parser("train", help="Train the anomaly detector")
    train_parser.add_argument("--input", default="data/features.parquet")
    train_parser.add_argument("--model", default="models/iforest.joblib")

    score_parser = subparsers.add_parser("score", help="Score and rank candidates")
    score_parser.add_argument("--model", required=True)
    score_parser.add_argument("--input", default="data/features.parquet")
    score_parser.add_argument("--output", default="results/candidates.csv")
    score_parser.add_argument("--top", type=int, default=50)

    plot_parser = subparsers.add_parser("plot", help="Visualise KB space")
    plot_parser.add_argument("--input", default="data/features.parquet")
    plot_parser.add_argument("--candidates", default="results/candidates.csv")
    plot_parser.add_argument("--output", default="results/kb_space.png")

    return parser


def _load_pipeline(config_path: str | Path) -> Pipeline:
    setup_logging(None)
    return Pipeline.from_yaml(config_path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pipeline = _load_pipeline(args.config)

    if args.command == "fetch":
        df = pipeline.fetch(tables=args.tables, output=args.output)
        print(f"Fetched {len(df)} rows -> {args.output}")
        return 0

    if args.command == "featurize":
        df = pipeline.featurize(input_path=args.input, output=args.output)
        print(f"Featurized {len(df)} rows -> {args.output}")
        return 0

    if args.command == "train":
        model_path = pipeline.train(input_path=args.input, model_path=args.model)
        print(f"Model saved to {model_path}")
        return 0

    if args.command == "score":
        scores = pipeline.score(
            model_path=args.model,
            input_path=args.input,
            top=args.top,
            output=args.output,
        )
        print(f"Wrote top {len(scores)} candidates -> {args.output}")
        return 0

    if args.command == "plot":
        features = pd.read_parquet(args.input)
        candidates = pd.read_csv(args.candidates) if Path(args.candidates).exists() else None
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(features["K"], features["B"], alpha=0.3, label="catalogue")
        if candidates is not None:
            ax.scatter(candidates["K"], candidates["B"], marker="x", s=80, label="candidates")
        ax.set_xlabel("Kardashev K")
        ax.set_ylabel("Barrow level")
        ax.set_title("K×B candidate landscape")
        ax.legend()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.output)
        print(f"Plot saved -> {args.output}")
        return 0

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
