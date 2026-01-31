from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import pandas as pd


def load_tokenizer():
    # Reuse the exact tokenizer from task1 to keep results consistent.
    root_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root_dir))
    from task1.task1_tokenize import tokenize  # type: ignore

    return tokenize


def linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float, float | None]:
    # Simple least squares for y = a + b*x. Returns (a, b, r2).
    n = len(xs)
    if n == 0:
        raise ValueError("No data points for regression")

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        raise ValueError("Zero variance in x; cannot fit regression")

    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = cov_xy / var_x
    a = mean_y - b * mean_x

    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return a, b, r2


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Heaps' law V = k * N^beta")
    parser.add_argument("--input", type=str, default="poems_cleaned.parquet")
    parser.add_argument("--step", type=int, default=1000, help="Sample every N tokens")
    parser.add_argument("--out-dir", type=str, default="task2")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    input_path = root_dir / args.input
    out_dir = root_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenize = load_tokenizer()

    df = pd.read_parquet(input_path)
    texts = df["text"].fillna("")

    total_tokens = 0
    vocab: set[str] = set()
    points: list[tuple[int, int]] = []
    next_sample = args.step

    for text in texts:
        tokens = tokenize(text)
        for token in tokens:
            total_tokens += 1
            if token not in vocab:
                vocab.add(token)
            if total_tokens >= next_sample:
                points.append((total_tokens, len(vocab)))
                while total_tokens >= next_sample:
                    next_sample += args.step

    if not points or points[-1][0] != total_tokens:
        points.append((total_tokens, len(vocab)))

    xs = [math.log(n) for n, v in points if n > 0 and v > 0]
    ys = [math.log(v) for n, v in points if n > 0 and v > 0]

    a, b, r2 = linear_regression(xs, ys)
    k = math.exp(a)
    beta = b

    stats = {
        "dataset": str(input_path.name),
        "rows": int(len(df)),
        "tokenizer": "simple-space (task1_tokenize.py)",
        "step": int(args.step),
        "total_tokens": int(total_tokens),
        "unique_types": int(len(vocab)),
        "k": k,
        "beta": beta,
        "r2": r2,
        "num_points": int(len(points)),
    }

    stats_path = out_dir / "heaps_fit.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    points_path = out_dir / "heaps_points.csv"
    with points_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tokens", "types", "log_tokens", "log_types"])
        for n, v in points:
            writer.writerow([n, v, math.log(n), math.log(v)])


if __name__ == "__main__":
    main()
