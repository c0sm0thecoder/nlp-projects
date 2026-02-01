from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json

import pandas as pd


def tokenize(text: str) -> list[str]:
    # Simple space-based tokenizer: normalize newlines/tabs to spaces, then split on spaces.
    return text.replace("\n", " ").replace("\t", " ").split()


def process_dataset(data_path: Path, text_column: str, output_suffix: str = "") -> dict:
    """Process a dataset and return statistics."""
    df = pd.read_parquet(data_path)
    texts = df[text_column].fillna("")

    token_counts: Counter[str] = Counter()
    total_tokens = 0

    for text in texts:
        tokens = tokenize(text)
        token_counts.update(tokens)
        total_tokens += len(tokens)

    stats = {
        "dataset": str(data_path.name),
        "rows": int(len(df)),
        "text_column": text_column,
        "tokenizer": "simple-space",
        "total_tokens": int(total_tokens),
        "unique_types": int(len(token_counts)),
    }

    return stats, token_counts


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process original cleaned dataset
    cleaned_data_path = root_dir / "poems_cleaned.parquet"
    cleaned_stats, cleaned_token_counts = process_dataset(cleaned_data_path, "text")

    # Save cleaned dataset results
    stats_path = out_dir / "token_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned_stats, f, ensure_ascii=False, indent=2)
        f.write("\n")

    freq_path = out_dir / "token_frequencies.csv"
    with freq_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "frequency"])
        for token, freq in cleaned_token_counts.most_common():
            writer.writerow([token, freq])

    # Process translated dataset
    translated_data_path = root_dir / "data_translation" / "poems_translated.parquet"
    if translated_data_path.exists():
        # Process both original and modern text columns
        original_stats, original_token_counts = process_dataset(translated_data_path, "text", "_original")
        modern_stats, modern_token_counts = process_dataset(translated_data_path, "modern_text", "_modern")

        # Save translated dataset results (original text)
        translated_stats_path = out_dir / "token_stats_translated_original.json"
        with translated_stats_path.open("w", encoding="utf-8") as f:
            json.dump(original_stats, f, ensure_ascii=False, indent=2)
            f.write("\n")

        translated_freq_path = out_dir / "token_frequencies_translated_original.csv"
        with translated_freq_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["token", "frequency"])
            for token, freq in original_token_counts.most_common():
                writer.writerow([token, freq])

        # Save translated dataset results (modern text)
        modern_stats_path = out_dir / "token_stats_translated_modern.json"
        with modern_stats_path.open("w", encoding="utf-8") as f:
            json.dump(modern_stats, f, ensure_ascii=False, indent=2)
            f.write("\n")

        modern_freq_path = out_dir / "token_frequencies_translated_modern.csv"
        with modern_freq_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["token", "frequency"])
            for token, freq in modern_token_counts.most_common():
                writer.writerow([token, freq])

        # Print summary
        print(f"Processed datasets:")
        print(f"1. Cleaned dataset: {cleaned_stats['total_tokens']:,} total tokens, {cleaned_stats['unique_types']:,} unique tokens")
        print(f"2. Translated dataset (original): {original_stats['total_tokens']:,} total tokens, {original_stats['unique_types']:,} unique tokens")  
        print(f"3. Translated dataset (modern): {modern_stats['total_tokens']:,} total tokens, {modern_stats['unique_types']:,} unique tokens")
    else:
        print(f"Processed cleaned dataset: {cleaned_stats['total_tokens']:,} total tokens, {cleaned_stats['unique_types']:,} unique tokens")
        print("Translated dataset not found.")


if __name__ == "__main__":
    main()
