from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project2.task2.smoothing_core import (  # noqa: E402
    build_task2_artifacts,
    parse_float_grid,
    write_task2_outputs,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task2 smoothing pipeline: Laplace, Interpolation, Backoff, Kneser-Ney."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="project2/poems_translated.parquet",
        help="Path to input parquet file.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="modern_text",
        help="Text column for language modeling.",
    )
    parser.add_argument(
        "--author-col",
        type=str,
        default="author",
        help="Author column used for stratified split.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Outer test split size.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Inner validation split size from outer train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for all splits/shuffles.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum train token frequency for vocabulary retention before mapping to <unk>.",
    )
    parser.add_argument(
        "--interp-bigram-grid",
        type=str,
        default="0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated lambda2 values for bigram interpolation.",
    )
    parser.add_argument(
        "--interp-trigram-step",
        type=float,
        default=0.1,
        help="Step size for trigram interpolation lambda simplex.",
    )
    parser.add_argument(
        "--discount-grid",
        type=str,
        default="0.5,0.75,1.0",
        help="Comma-separated discount values for backoff and Kneser-Ney.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="project2/task2/results",
        help="Output directory for Task2 artifacts.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    interp_bigram_grid = parse_float_grid(args.interp_bigram_grid, "--interp-bigram-grid")
    discount_grid = parse_float_grid(args.discount_grid, "--discount-grid")

    artifacts = build_task2_artifacts(
        input_path=args.input,
        text_col=args.text_col,
        author_col=args.author_col,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        min_freq=args.min_freq,
        interp_bigram_grid=interp_bigram_grid,
        interp_trigram_step=args.interp_trigram_step,
        discount_grid=discount_grid,
    )
    output_paths = write_task2_outputs(artifacts, args.out_dir)

    print(f"Wrote: {output_paths['metrics']}")
    print(f"Wrote: {output_paths['tuning']}")
    print(f"Wrote: {output_paths['comparison']}")
    print(f"Wrote: {output_paths['ranking']}")

    defaults = artifacts["metrics"]["best_method_defaults"]
    print(
        "Best method (default rule "
        f"{defaults['default_rule']}): {defaults['best_method']}"
    )


if __name__ == "__main__":
    main()
