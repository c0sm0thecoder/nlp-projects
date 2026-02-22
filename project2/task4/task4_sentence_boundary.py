from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project2.task4.sentence_boundary_core import (  # noqa: E402
    C_GRID,
    build_task4_artifacts,
    write_task4_outputs,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task4: Sentence boundary detection with logistic regression. "
            "Compare L1 vs L2 regularization with C tuning on validation set."
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
        help="Author column (for stratified split by poem).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split fraction (by poem).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction (by poem).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--c-grid",
        type=str,
        default=None,
        help="Comma-separated C values for grid search. Default: 0.001,0.01,0.1,0.5,1.0,5.0,10.0",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="project2/task4/results",
        help="Output directory.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    c_grid = None
    if args.c_grid:
        c_grid = [float(x.strip()) for x in args.c_grid.split(",")]

    artifacts = build_task4_artifacts(
        input_path=args.input,
        text_col=args.text_col,
        label_col=args.label_col,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        c_grid=c_grid,
    )

    paths = write_task4_outputs(artifacts, args.out_dir)

    for p in paths.values():
        print(f"Wrote: {p}")

    ds = artifacts["dataset_stats"]
    print(f"\nSplit: train={ds['train_dots']}  val={ds['val_dots']}  test={ds['test_dots']}")

    for penalty in ("l1", "l2"):
        r = artifacts["results"][penalty]
        tr = r["train_metrics"]
        va = r["val_metrics"]
        te = r["test_metrics"]
        print(
            f"\n  {penalty.upper()} (best C={r['best_C']}):\n"
            f"    Train — acc={tr['accuracy']:.4f}  F1={tr['f1']:.4f}\n"
            f"    Val   — acc={va['accuracy']:.4f}  F1={va['f1']:.4f}\n"
            f"    Test  — acc={te['accuracy']:.4f}  F1={te['f1']:.4f}  "
            f"P={te['precision']:.4f}  R={te['recall']:.4f}\n"
            f"    Non-zero coefs: {r['n_nonzero_coefs']}"
        )

    sig = artifacts["significance"]
    best = artifacts["best_penalty"]
    print(
        f"\nMcNemar L1 vs L2: chi2={sig['statistic']:.4f} p={sig['p_value']:.6f} "
        f"significant={sig['significant_0.05']}"
    )
    print(f"Best penalty: {best.upper()}")


if __name__ == "__main__":
    main()
