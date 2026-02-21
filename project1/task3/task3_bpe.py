from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def load_tokenizer():
    # Reuse the exact tokenizer from task1 to keep results consistent.
    root_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root_dir))
    from task1.task1_tokenize import tokenize  # type: ignore

    return tokenize


def build_word_vocab(texts: pd.Series, tokenize) -> Counter[str]:
    word_counts: Counter[str] = Counter()
    for text in texts:
        tokens = tokenize(text)
        word_counts.update(tokens)
    return word_counts


def get_pair_counts(vocab: dict[tuple[str, ...], int]) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    for word, freq in vocab.items():
        if len(word) < 2:
            continue
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_word(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    merged = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def learn_bpe(
    word_counts: Counter[str],
    vocab_size: int,
    min_pair_freq: int,
    progress_every: int,
) -> tuple[list[tuple[str, str]], dict[tuple[str, ...], int]]:
    # Initialize vocabulary with characters + end-of-word marker.
    vocab: dict[tuple[str, ...], int] = {}
    symbols: set[str] = set()
    for word, freq in word_counts.items():
        chars = list(word)
        chars.append("</w>")
        word_tuple = tuple(chars)
        vocab[word_tuple] = freq
        symbols.update(chars)

    initial_symbols = len(symbols)
    target_merges = max(0, vocab_size - initial_symbols)

    merges: list[tuple[str, str]] = []
    for i in range(target_merges):
        pair_counts = get_pair_counts(vocab)
        if not pair_counts:
            break

        best_pair, best_freq = pair_counts.most_common(1)[0]
        if best_freq < min_pair_freq:
            break

        merges.append(best_pair)

        new_vocab: dict[tuple[str, ...], int] = {}
        for word, freq in vocab.items():
            merged_word = merge_word(word, best_pair)
            new_vocab[merged_word] = new_vocab.get(merged_word, 0) + freq
        vocab = new_vocab

        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"Learned merges: {i + 1}/{target_merges}")

    return merges, vocab


def apply_bpe_to_word(word: str, bpe_ranks: dict[tuple[str, str], int]) -> list[str]:
    if not word:
        return []
    symbols = tuple(list(word) + ["</w>"])

    while True:
        pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
        ranked = [(bpe_ranks[p], p) for p in pairs if p in bpe_ranks]
        if not ranked:
            break
        _, best_pair = min(ranked)
        symbols = merge_word(symbols, best_pair)

    out: list[str] = []
    for sym in symbols:
        if sym == "</w>":
            continue
        if sym.endswith("</w>"):
            out.append(sym[: -len("</w>")])
        else:
            out.append(sym)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train word-level BPE and tokenize corpus")
    parser.add_argument("--input", type=str, default="poems_translated.parquet")
    parser.add_argument("--text-col", type=str, default="modern_text")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--min-pair-freq", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default="task3")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    input_path = root_dir / args.input
    out_dir = root_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenize = load_tokenizer()

    df = pd.read_parquet(input_path)
    if args.text_col not in df.columns:
        raise SystemExit(f"Missing column '{args.text_col}' in {input_path.name}")
    texts = df[args.text_col].fillna("")

    word_counts = build_word_vocab(texts, tokenize)
    merges, _ = learn_bpe(
        word_counts=word_counts,
        vocab_size=args.vocab_size,
        min_pair_freq=args.min_pair_freq,
        progress_every=args.progress_every,
    )

    bpe_ranks = {pair: i for i, pair in enumerate(merges)}

    # Tokenize corpus with learned BPE.
    bpe_cache: dict[str, list[str]] = {}
    bpe_token_counts: Counter[str] = Counter()
    bpe_texts: list[str] = []
    total_bpe_tokens = 0

    for text in texts:
        words = tokenize(text)
        out_tokens: list[str] = []
        for word in words:
            if word not in bpe_cache:
                bpe_cache[word] = apply_bpe_to_word(word, bpe_ranks)
            tokens = bpe_cache[word]
            out_tokens.extend(tokens)
            bpe_token_counts.update(tokens)
            total_bpe_tokens += len(tokens)
        bpe_texts.append(" ".join(out_tokens))

    df_out = df.copy()
    df_out["bpe_text"] = bpe_texts

    # Save outputs
    merges_path = out_dir / "bpe_merges.txt"
    with merges_path.open("w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for a, b in merges:
            f.write(f"{a} {b}\n")

    vocab_items = sorted(bpe_token_counts.items(), key=lambda x: (-x[1], x[0]))
    vocab = {token: i for i, (token, _) in enumerate(vocab_items)}

    vocab_path = out_dir / "bpe_vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    freq_path = out_dir / "bpe_token_frequencies.csv"
    with freq_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "frequency"])
        for token, freq in vocab_items:
            writer.writerow([token, freq])

    stats = {
        "dataset": str(input_path.name),
        "rows": int(len(df)),
        "text_column": args.text_col,
        "tokenizer": "word-level BPE (task1_tokenize.py base)",
        "vocab_size_target": int(args.vocab_size),
        "min_pair_freq": int(args.min_pair_freq),
        "num_merges": int(len(merges)),
        "total_bpe_tokens": int(total_bpe_tokens),
        "unique_bpe_types": int(len(bpe_token_counts)),
    }

    stats_path = out_dir / "bpe_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    out_path = out_dir / "poems_bpe.parquet"
    df_out.to_parquet(out_path, index=False)


if __name__ == "__main__":
    main()
