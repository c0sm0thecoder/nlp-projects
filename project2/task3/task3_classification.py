from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project2.task3.classification_core import (  # noqa: E402
    build_task3_artifacts,
    write_task3_outputs,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task3 classification pipeline: Naive Bayes, Binary NB, "
            "Logistic Regression with BoW + sentiment-lexicon features."
        )
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
        help="Text column.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="author",
        help="Label column (author as class).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Max BoW vocabulary size (None = unlimited).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="project2/task3/results",
        help="Output directory.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    artifacts = build_task3_artifacts(
        input_path=args.input,
        text_col=args.text_col,
        label_col=args.label_col,
        test_size=args.test_size,
        seed=args.seed,
        max_features=args.max_features,
    )

    paths = write_task3_outputs(artifacts, args.out_dir)

    print(f"Wrote: {paths['metrics']}")
    print(f"Wrote: {paths['summary']}")
    print(f"Wrote: {paths['significance']}")
    print(
        f"Best classifier: {artifacts['best']['classifier']} "
        f"({artifacts['best']['feature_set']}) "
        f"macro_f1={artifacts['best']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
