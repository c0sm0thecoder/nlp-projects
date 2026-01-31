from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    points_path = out_dir / "heaps_points.csv"
    fit_path = out_dir / "heaps_fit.json"

    points = pd.read_csv(points_path)
    fit = json.loads(fit_path.read_text(encoding="utf-8"))

    k = float(fit["k"])
    beta = float(fit["beta"])

    tokens = points["tokens"].astype(float)
    types = points["types"].astype(float)

    pred = k * (tokens ** beta)

    plt.figure(figsize=(8, 6))
    plt.loglog(tokens, types, "o", markersize=3, alpha=0.7, label="Observed")
    plt.loglog(tokens, pred, "-", linewidth=2, label=f"Fit: V = {k:.3f} * N^{beta:.3f}")
    plt.xlabel("Tokens (N)")
    plt.ylabel("Types (V)")
    plt.title("Heaps' Law (log-log)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    out_path = out_dir / "heaps_plot.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


if __name__ == "__main__":
    main()
