from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json

import pandas as pd


def tokenize(text: str) -> list[str]:
    # Simple space-based tokenizer: normalize newlines/tabs to spaces, then split on spaces.
    return text.replace("\n", " ").replace("\t", " ").split()


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    data_path = root_dir / "poems_cleaned.parquet"
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    texts = df["text"].fillna("")

    token_counts: Counter[str] = Counter()
    total_tokens = 0

    for text in texts:
        tokens = tokenize(text)
        token_counts.update(tokens)
        total_tokens += len(tokens)

    stats = {
        "dataset": str(data_path.name),
        "rows": int(len(df)),
        "text_column": "text",
        "tokenizer": "simple-space",
        "total_tokens": int(total_tokens),
        "unique_types": int(len(token_counts)),
    }

    stats_path = out_dir / "token_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")

    freq_path = out_dir / "token_frequencies.csv"
    with freq_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "frequency"])
        for token, freq in token_counts.most_common():
            writer.writerow([token, freq])


if __name__ == "__main__":
    main()
