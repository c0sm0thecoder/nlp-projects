from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
UNK_TOKEN = "<unk>"
START_TOKEN = "<s>"
END_TOKEN = "</s>"


def tokenize(text: str) -> list[str]:
    """Lowercase text and tokenize with Unicode-safe word boundaries."""
    return TOKEN_RE.findall(text.lower())


def validate_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_and_tokenize(df: pd.DataFrame, text_col: str) -> tuple[pd.DataFrame, dict[str, int]]:
    rows_input = int(len(df))
    text_series = df[text_col].fillna("").astype(str)
    non_empty_mask = text_series.str.strip().ne("")

    non_empty_df = df.loc[non_empty_mask].copy()
    non_empty_df["tokens"] = non_empty_df[text_col].astype(str).map(tokenize)
    token_non_empty_mask = non_empty_df["tokens"].map(len).gt(0)
    clean_df = non_empty_df.loc[token_non_empty_mask].copy()
    clean_df = clean_df.reset_index(drop=True)

    stats = {
        "rows_input": rows_input,
        "rows_after_non_empty_text": int(len(non_empty_df)),
        "rows_after_tokenization": int(len(clean_df)),
        "dropped_empty_text_rows": int(rows_input - len(non_empty_df)),
        "dropped_no_word_token_rows": int(len(non_empty_df) - len(clean_df)),
    }
    return clean_df, stats


def stratified_split(
    df: pd.DataFrame,
    author_col: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in df.groupby(author_col, sort=False):
        shuffled = group.sample(frac=1.0, random_state=seed)
        if len(shuffled) <= 1:
            n_test = 0
        else:
            n_test = max(1, int(round(len(shuffled) * test_size)))
            n_test = min(n_test, len(shuffled) - 1)

        test_parts.append(shuffled.iloc[:n_test].copy())
        train_parts.append(shuffled.iloc[n_test:].copy())

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Train split is empty. Check dataset size or split settings.")
    if test_df.empty:
        raise ValueError("Test split is empty. Check dataset size or split settings.")

    return train_df, test_df


def flatten_sequences(sequences: Iterable[Sequence[str]]) -> list[str]:
    flat: list[str] = []
    for seq in sequences:
        flat.extend(seq)
    return flat


def apply_unk_policy(
    train_sequences: list[list[str]],
    test_sequences: list[list[str]],
    min_freq: int,
) -> tuple[list[list[str]], list[list[str]], dict[str, int]]:
    train_token_counts = Counter(flatten_sequences(train_sequences))
    retained_vocab = {token for token, count in train_token_counts.items() if count >= min_freq}

    train_mapped = [
        [token if token in retained_vocab else UNK_TOKEN for token in seq]
        for seq in train_sequences
    ]
    test_mapped = [
        [token if token in retained_vocab else UNK_TOKEN for token in seq]
        for seq in test_sequences
    ]

    train_unk_count = sum(1 for seq in train_mapped for token in seq if token == UNK_TOKEN)
    test_unk_count = sum(1 for seq in test_mapped for token in seq if token == UNK_TOKEN)
    test_oov_before_mapping = sum(
        1 for seq in test_sequences for token in seq if token not in retained_vocab
    )

    vocab_stats = {
        "raw_train_vocab_size": int(len(train_token_counts)),
        "retained_vocab_size": int(len(retained_vocab)),
        "effective_vocab_size_with_unk": int(len(retained_vocab) + 1),
        "rare_train_token_types_mapped_to_unk": int(
            sum(1 for _, count in train_token_counts.items() if count < min_freq)
        ),
        "train_unk_token_count": int(train_unk_count),
        "test_unk_token_count": int(test_unk_count),
        "test_oov_before_unk_count": int(test_oov_before_mapping),
        "min_freq": int(min_freq),
    }
    return train_mapped, test_mapped, vocab_stats


def generate_ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n == 1:
        return [(token,) for token in tokens]

    padded = [START_TOKEN] * (n - 1) + list(tokens) + [END_TOKEN]
    return [tuple(padded[i : i + n]) for i in range(len(padded) - n + 1)]


def build_ngram_model(
    sequences: list[list[str]],
    n: int,
) -> tuple[Counter[tuple[str, ...]], Counter[tuple[str, ...]], int]:
    ngram_counts: Counter[tuple[str, ...]] = Counter()
    context_counts: Counter[tuple[str, ...]] = Counter()
    total_events = 0

    for seq in sequences:
        ngrams = generate_ngrams(seq, n)
        total_events += len(ngrams)
        ngram_counts.update(ngrams)
        if n > 1:
            for ng in ngrams:
                context_counts[ng[:-1]] += 1

    return ngram_counts, context_counts, total_events


def perplexity_diagnostics(
    sequences: list[list[str]],
    n: int,
    ngram_counts: Counter[tuple[str, ...]],
    context_counts: Counter[tuple[str, ...]],
    unigram_total: int,
) -> dict[str, float | int]:
    total_events = 0
    zero_prob_events = 0
    log_prob_sum = 0.0

    for seq in sequences:
        ngrams = generate_ngrams(seq, n)
        total_events += len(ngrams)

        for ng in ngrams:
            if n == 1:
                numerator = ngram_counts.get(ng, 0)
                denominator = unigram_total
            else:
                numerator = ngram_counts.get(ng, 0)
                denominator = context_counts.get(ng[:-1], 0)

            if numerator == 0 or denominator == 0:
                zero_prob_events += 1
            else:
                log_prob_sum += math.log(numerator / denominator)

    if total_events == 0:
        perplexity = math.inf
    elif zero_prob_events > 0:
        perplexity = math.inf
    else:
        perplexity = math.exp(-log_prob_sum / total_events)

    unseen_rate = zero_prob_events / total_events if total_events > 0 else 0.0
    return {
        "perplexity": perplexity,
        "events": int(total_events),
        "zero_prob_events": int(zero_prob_events),
        "unseen_rate": unseen_rate,
    }


def serialize_perplexity(value: float) -> float | str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return float(value)


def ngram_to_text(ngram: tuple[str, ...]) -> str:
    return " ".join(ngram)


def write_top_ngrams_csv(
    output_path: Path,
    model_counts: dict[str, Counter[tuple[str, ...]]],
    top_k: int,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "n", "rank", "ngram", "count"])
        for model_name, counter in model_counts.items():
            n = {"unigram": 1, "bigram": 2, "trigram": 3}[model_name]
            for rank, (ngram, count) in enumerate(counter.most_common(top_k), start=1):
                writer.writerow([model_name, n, rank, ngram_to_text(ngram), int(count)])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train unsmoothed unigram/bigram/trigram models and compute perplexity."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="project2/poems_translated.parquet",
        help="Path to input parquet dataset.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="modern_text",
        help="Text column to model.",
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
        help="Fraction for held-out test split per author group.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split/shuffle.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum train token frequency to keep; lower counts map to <unk>.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="project2/task1/results",
        help="Directory for JSON/CSV outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top n-grams per model for CSV export.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not (0 < args.test_size < 1):
        raise ValueError("--test-size must be between 0 and 1.")
    if args.min_freq < 1:
        raise ValueError("--min-freq must be >= 1.")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    validate_columns(df, [args.author_col, args.text_col])
    df = df.reset_index(drop=False).rename(columns={"index": "row_id"})

    clean_df, clean_stats = clean_and_tokenize(df, args.text_col)
    if clean_df.empty:
        raise ValueError("No usable rows after text cleaning and tokenization.")

    train_df, test_df = stratified_split(clean_df, args.author_col, args.test_size, args.seed)

    train_sequences_raw = train_df["tokens"].tolist()
    test_sequences_raw = test_df["tokens"].tolist()

    train_sequences, test_sequences, vocab_stats = apply_unk_policy(
        train_sequences=train_sequences_raw,
        test_sequences=test_sequences_raw,
        min_freq=args.min_freq,
    )

    train_token_count = int(sum(len(seq) for seq in train_sequences))
    test_token_count = int(sum(len(seq) for seq in test_sequences))

    if train_token_count == 0:
        raise ValueError("Train split has zero tokens after preprocessing.")
    if test_token_count == 0:
        raise ValueError("Test split has zero tokens after preprocessing.")

    ngram_sizes = {"unigram": 1, "bigram": 2, "trigram": 3}
    model_counts: dict[str, Counter[tuple[str, ...]]] = {}
    model_metrics: dict[str, dict[str, float | str | int]] = {}

    unigram_total = 0
    context_cache: dict[str, Counter[tuple[str, ...]]] = {}
    for model_name, n in ngram_sizes.items():
        counts, contexts, total_events = build_ngram_model(train_sequences, n)
        model_counts[model_name] = counts
        context_cache[model_name] = contexts
        if n == 1:
            unigram_total = total_events

    for model_name, n in ngram_sizes.items():
        counts = model_counts[model_name]
        contexts = context_cache[model_name]

        train_diag = perplexity_diagnostics(
            sequences=train_sequences,
            n=n,
            ngram_counts=counts,
            context_counts=contexts,
            unigram_total=unigram_total,
        )
        test_diag = perplexity_diagnostics(
            sequences=test_sequences,
            n=n,
            ngram_counts=counts,
            context_counts=contexts,
            unigram_total=unigram_total,
        )

        model_metrics[model_name] = {
            "train_perplexity": serialize_perplexity(float(train_diag["perplexity"])),
            "test_perplexity": serialize_perplexity(float(test_diag["perplexity"])),
            "zero_prob_events_train": int(train_diag["zero_prob_events"]),
            "zero_prob_events_test": int(test_diag["zero_prob_events"]),
            "unseen_rate_train": float(train_diag["unseen_rate"]),
            "unseen_rate_test": float(test_diag["unseen_rate"]),
            "train_event_count": int(train_diag["events"]),
            "test_event_count": int(test_diag["events"]),
            "ngram_type_count": int(len(counts)),
        }

    metrics = {
        "config": {
            "input": str(input_path),
            "text_col": args.text_col,
            "author_col": args.author_col,
            "test_size": args.test_size,
            "seed": args.seed,
            "min_freq": args.min_freq,
            "tokenizer": "unicode-word-boundary-lowercase",
            "smoothing": "none (unsmoothed MLE)",
            "boundary_tokens": [START_TOKEN, END_TOKEN],
        },
        "corpus": {
            **clean_stats,
            "authors": int(clean_df[args.author_col].nunique()),
            "total_tokens_before_unk": int(sum(len(seq) for seq in train_sequences_raw + test_sequences_raw)),
            "train_tokens_after_unk": train_token_count,
            "test_tokens_after_unk": test_token_count,
        },
        "split": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_author_counts": {
                str(k): int(v) for k, v in train_df[args.author_col].value_counts().sort_index().items()
            },
            "test_author_counts": {
                str(k): int(v) for k, v in test_df[args.author_col].value_counts().sort_index().items()
            },
        },
        "vocab": vocab_stats,
        "models": model_metrics,
    }

    metrics_path = out_dir / "task1_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    split_stats = {
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "train_row_ids": [int(x) for x in train_df["row_id"].tolist()],
        "test_row_ids": [int(x) for x in test_df["row_id"].tolist()],
        "train_author_counts": metrics["split"]["train_author_counts"],
        "test_author_counts": metrics["split"]["test_author_counts"],
    }
    split_stats_path = out_dir / "task1_split_stats.json"
    split_stats_path.write_text(json.dumps(split_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    top_ngrams_path = out_dir / "task1_top_ngrams.csv"
    write_top_ngrams_csv(top_ngrams_path, model_counts=model_counts, top_k=args.top_k)

    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {split_stats_path}")
    print(f"Wrote: {top_ngrams_path}")
    for model_name in ("unigram", "bigram", "trigram"):
        m = model_metrics[model_name]
        print(
            f"{model_name}: "
            f"train_ppl={m['train_perplexity']} "
            f"test_ppl={m['test_perplexity']} "
            f"zero_prob_test={m['zero_prob_events_test']}"
        )


if __name__ == "__main__":
    main()
