from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
DEFAULT_RARE_THRESHOLD = 1
DEFAULT_FREQUENT_THRESHOLD = 10
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path(r"C:\Windows\Fonts\verdana.ttf"),
]


@dataclass(frozen=True)
class SimpleCSRMatrix:
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, int]

    def toarray(self) -> np.ndarray:
        dense = np.zeros(self.shape, dtype=self.data.dtype if self.data.size else np.int32)
        for row_id in range(self.shape[0]):
            start = int(self.indptr[row_id])
            end = int(self.indptr[row_id + 1])
            if start == end:
                continue
            dense[row_id, self.indices[start:end]] = self.data[start:end]
        return dense

    def diagonal(self) -> np.ndarray:
        diagonal = np.zeros(min(self.shape), dtype=self.data.dtype if self.data.size else np.int32)
        for row_id in range(len(diagonal)):
            start = int(self.indptr[row_id])
            end = int(self.indptr[row_id + 1])
            row_indices = self.indices[start:end]
            row_values = self.data[start:end]
            matches = row_values[row_indices == row_id]
            if matches.size:
                diagonal[row_id] = matches[0]
        return diagonal

    def _resolve_selector(self, selector: slice | int | Sequence[int], axis_size: int) -> list[int]:
        if isinstance(selector, slice):
            return list(range(*selector.indices(axis_size)))
        if isinstance(selector, int):
            if selector < 0:
                selector += axis_size
            return [selector]
        return [int(value) for value in selector]

    def __getitem__(self, key: tuple[slice | int | Sequence[int], slice | int | Sequence[int]]) -> "SimpleCSRMatrix":
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Matrix slicing expects row and column selectors.")

        selected_rows = self._resolve_selector(key[0], self.shape[0])
        selected_cols = self._resolve_selector(key[1], self.shape[1])
        col_map = {old_col: new_col for new_col, old_col in enumerate(selected_cols)}

        rows: list[int] = []
        cols: list[int] = []
        data: list[int] = []
        for new_row, old_row in enumerate(selected_rows):
            start = int(self.indptr[old_row])
            end = int(self.indptr[old_row + 1])
            for old_col, value in zip(self.indices[start:end], self.data[start:end]):
                mapped_col = col_map.get(int(old_col))
                if mapped_col is None:
                    continue
                rows.append(new_row)
                cols.append(mapped_col)
                data.append(int(value))

        return make_csr_matrix(rows, cols, data, (len(selected_rows), len(selected_cols)))


def make_csr_matrix(
    rows: Sequence[int],
    cols: Sequence[int],
    data: Sequence[int],
    shape: tuple[int, int],
) -> SimpleCSRMatrix:
    entries: dict[tuple[int, int], int] = {}
    for row_id, col_id, value in zip(rows, cols, data):
        key = (int(row_id), int(col_id))
        entries[key] = entries.get(key, 0) + int(value)

    row_buckets: list[list[tuple[int, int]]] = [[] for _ in range(shape[0])]
    for (row_id, col_id), value in entries.items():
        row_buckets[row_id].append((col_id, value))

    csr_data: list[int] = []
    csr_indices: list[int] = []
    csr_indptr = [0]
    for bucket in row_buckets:
        bucket.sort(key=lambda item: item[0])
        for col_id, value in bucket:
            csr_indices.append(int(col_id))
            csr_data.append(int(value))
        csr_indptr.append(len(csr_data))

    return SimpleCSRMatrix(
        data=np.asarray(csr_data, dtype=np.int32),
        indices=np.asarray(csr_indices, dtype=np.int32),
        indptr=np.asarray(csr_indptr, dtype=np.int64),
        shape=(int(shape[0]), int(shape[1])),
    )


def save_npz(output_path: Path, matrix: SimpleCSRMatrix) -> None:
    np.savez_compressed(
        output_path,
        format=np.asarray("csr"),
        shape=np.asarray(matrix.shape, dtype=np.int64),
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
    )


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = FONT_CANDIDATES
    if bold:
        bold_candidates = [
            Path(r"C:\Windows\Fonts\segoeuib.ttf"),
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
            Path(r"C:\Windows\Fonts\tahomabd.ttf"),
            Path(r"C:\Windows\Fonts\verdanab.ttf"),
        ]
        candidates = bold_candidates + candidates

    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)

    return ImageFont.load_default()


def make_rotated_text(label: str, font: ImageFont.ImageFont, *, fill: str = "black") -> Image.Image:
    scratch = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), label, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    image = Image.new("RGBA", (width + 6, height + 6), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.text((3 - bbox[0], 3 - bbox[1]), label, fill=fill, font=font)
    return image.rotate(90, expand=True)


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text with Unicode-aware word boundaries."""
    return TOKEN_RE.findall(str(text).lower())


def validate_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_documents(
    df: pd.DataFrame,
    *,
    text_col: str,
    author_col: str,
    title_col: str,
) -> pd.DataFrame:
    validate_columns(df, [text_col, author_col, title_col])

    prepared = df.reset_index(drop=False).rename(columns={"index": "row_id"}).copy()
    prepared[text_col] = prepared[text_col].fillna("").astype(str)
    prepared["tokens"] = prepared[text_col].map(tokenize)
    return prepared


def _normalize_number(value: float | int) -> float | int:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def compute_frequency_tables(tokenized_documents: Iterable[Sequence[str]]) -> tuple[Counter[str], Counter[str]]:
    term_frequency: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()

    for tokens in tokenized_documents:
        term_frequency.update(tokens)
        document_frequency.update(set(tokens))

    return term_frequency, document_frequency


def compute_corpus_statistics(
    documents_df: pd.DataFrame,
    *,
    author_col: str,
    rare_threshold: int = DEFAULT_RARE_THRESHOLD,
    frequent_threshold: int = DEFAULT_FREQUENT_THRESHOLD,
    top_k: int = 10,
) -> tuple[dict[str, object], Counter[str], Counter[str]]:
    tokenized_documents = documents_df["tokens"].tolist()
    term_frequency, document_frequency = compute_frequency_tables(tokenized_documents)
    doc_lengths = [len(tokens) for tokens in tokenized_documents]

    average_length = float(np.mean(doc_lengths)) if doc_lengths else 0.0
    median_length = float(np.median(doc_lengths)) if doc_lengths else 0.0

    stats: dict[str, object] = {
        "documents": int(len(documents_df)),
        "authors": int(documents_df[author_col].nunique()),
        "total_tokens": int(sum(doc_lengths)),
        "unique_tokens": int(len(term_frequency)),
        "average_document_length": round(average_length, 6),
        "median_document_length": _normalize_number(median_length),
        "min_document_length": int(min(doc_lengths)) if doc_lengths else 0,
        "max_document_length": int(max(doc_lengths)) if doc_lengths else 0,
        "rare_words_freq_eq_1": int(sum(1 for count in term_frequency.values() if count == rare_threshold)),
        f"frequent_words_freq_ge_{frequent_threshold}": int(
            sum(1 for count in term_frequency.values() if count >= frequent_threshold)
        ),
        "top_terms": [
            {
                "token": token,
                "term_frequency": int(freq),
                "document_frequency": int(document_frequency[token]),
            }
            for token, freq in term_frequency.most_common(top_k)
        ],
    }
    return stats, term_frequency, document_frequency


def build_vocabulary(
    term_frequency: Counter[str],
    document_frequency: Counter[str],
    *,
    min_frequency: int = 1,
) -> list[dict[str, int | str]]:
    ordered_tokens = sorted(
        (
            token,
            int(freq),
            int(document_frequency[token]),
        )
        for token, freq in term_frequency.items()
        if freq >= min_frequency
    )
    ordered_tokens.sort(key=lambda item: (-item[1], item[0]))

    return [
        {
            "term_id": index,
            "token": token,
            "term_frequency": term_freq,
            "document_frequency": doc_freq,
        }
        for index, (token, term_freq, doc_freq) in enumerate(ordered_tokens)
    ]


def build_term_document_matrix(
    tokenized_documents: Sequence[Sequence[str]],
    vocabulary: Sequence[dict[str, int | str]],
) -> SimpleCSRMatrix:
    vocab_index = {str(entry["token"]): int(entry["term_id"]) for entry in vocabulary}

    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []

    for doc_id, tokens in enumerate(tokenized_documents):
        counts = Counter(tokens)
        for token, count in counts.items():
            rows.append(doc_id)
            cols.append(vocab_index[token])
            data.append(int(count))

    return make_csr_matrix(rows, cols, data, (len(tokenized_documents), len(vocabulary)))


def build_word_word_matrix(
    tokenized_documents: Sequence[Sequence[str]],
    vocabulary: Sequence[dict[str, int | str]],
    *,
    window_size: int,
) -> SimpleCSRMatrix:
    vocab_index = {str(entry["token"]): int(entry["term_id"]) for entry in vocabulary}
    pair_counts: Counter[tuple[int, int]] = Counter()

    for tokens in tokenized_documents:
        token_ids = [vocab_index.get(token, -1) for token in tokens]
        for left_pos, left_id in enumerate(token_ids):
            if left_id < 0:
                continue
            right_limit = min(len(token_ids), left_pos + window_size + 1)
            for right_pos in range(left_pos + 1, right_limit):
                right_id = token_ids[right_pos]
                if right_id < 0 or left_id == right_id:
                    continue
                pair = (left_id, right_id) if left_id < right_id else (right_id, left_id)
                pair_counts[pair] += 1

    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    for (left_id, right_id), count in pair_counts.items():
        rows.extend([left_id, right_id])
        cols.extend([right_id, left_id])
        data.extend([int(count), int(count)])

    size = len(vocabulary)
    return make_csr_matrix(rows, cols, data, (size, size))


def dense_term_document_view(
    matrix: SimpleCSRMatrix,
    documents_df: pd.DataFrame,
    vocabulary: Sequence[dict[str, int | str]],
    *,
    top_n: int,
) -> pd.DataFrame:
    selected = list(vocabulary[:top_n])
    token_labels = [str(entry["token"]) for entry in selected]
    dense = matrix[:, : len(selected)].toarray()
    dense_df = pd.DataFrame(dense, columns=token_labels, index=documents_df["row_id"].tolist())
    dense_df.index.name = "row_id"
    return dense_df


def select_term_document_heatmap_rows(
    dense_df: pd.DataFrame,
    *,
    doc_limit: int,
) -> pd.Index:
    row_scores = dense_df.sum(axis=1)
    order = sorted(dense_df.index.tolist(), key=lambda row_id: (-int(row_scores.loc[row_id]), int(row_id)))
    return pd.Index(order[:doc_limit], name="row_id")


def dense_word_word_view(
    matrix: SimpleCSRMatrix,
    vocabulary: Sequence[dict[str, int | str]],
    *,
    top_n: int,
) -> pd.DataFrame:
    selected = list(vocabulary[:top_n])
    token_labels = [str(entry["token"]) for entry in selected]
    dense = matrix[: len(selected), : len(selected)].toarray()
    return pd.DataFrame(dense, index=token_labels, columns=token_labels)


def write_summary_markdown(
    output_path: Path,
    summary: dict[str, object],
    *,
    word_min_freq: int,
    window_size: int,
) -> None:
    top_terms = summary.get("top_terms", [])
    lines = [
        "# Task 1 Summary",
        "",
        "## Corpus Statistics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Documents | {summary['documents']} |",
        f"| Authors | {summary['authors']} |",
        f"| Total tokens | {summary['total_tokens']} |",
        f"| Unique tokens | {summary['unique_tokens']} |",
        f"| Average document length | {summary['average_document_length']} |",
        f"| Median document length | {summary['median_document_length']} |",
        f"| Rare words (freq = 1) | {summary['rare_words_freq_eq_1']} |",
        f"| Frequent words (freq >= 10) | {summary['frequent_words_freq_ge_10']} |",
        "",
        "## Matrix Configuration",
        "",
        f"- Term-document matrix vocabulary: full corpus vocabulary ({summary['unique_tokens']} terms)",
        f"- Word-word matrix vocabulary: terms with corpus frequency >= {word_min_freq}",
        f"- Word-word sliding window size: {window_size}",
        "",
        "## Top Terms",
        "",
        "| Rank | Token | Term Frequency | Document Frequency |",
        "| --- | --- | ---: | ---: |",
    ]

    for rank, entry in enumerate(top_terms, start=1):
        lines.append(
            f"| {rank} | {entry['token']} | {entry['term_frequency']} | {entry['document_frequency']} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if df.empty:
        raise ValueError("Cannot render a heatmap for an empty matrix view.")

    values = df.to_numpy(dtype=float)
    min_value = float(values.min())
    max_value = float(values.max())
    font_size = 12 if max(df.shape) > 40 else 16
    cell_size = max(20, font_size + 8)
    font = load_font(font_size)
    title_font = load_font(font_size + 4, bold=True)
    axis_font = load_font(font_size + 1, bold=True)

    scratch = Image.new("RGB", (1, 1), "white")
    scratch_draw = ImageDraw.Draw(scratch)
    row_labels = [str(label) for label in df.index]
    col_labels = [str(label) for label in df.columns]

    rotated_y_label = make_rotated_text(ylabel, axis_font)
    rotated_col_labels = [make_rotated_text(label, font) for label in col_labels]

    row_widths = [
        scratch_draw.textbbox((0, 0), label, font=font)[2] - scratch_draw.textbbox((0, 0), label, font=font)[0]
        for label in row_labels
    ]
    title_box = scratch_draw.textbbox((0, 0), title, font=title_font)
    x_label_box = scratch_draw.textbbox((0, 0), xlabel, font=axis_font)

    max_row_width = max(row_widths, default=0)
    max_col_height = max((image.height for image in rotated_col_labels), default=0)
    title_height = title_box[3] - title_box[1]
    x_label_height = x_label_box[3] - x_label_box[1]

    left_margin = rotated_y_label.width + max_row_width + 28
    top_margin = title_height + max_col_height + 28
    right_margin = 20
    bottom_margin = x_label_height + 24
    grid_width = df.shape[1] * cell_size
    grid_height = df.shape[0] * cell_size

    image = Image.new(
        "RGB",
        (left_margin + grid_width + right_margin, top_margin + grid_height + bottom_margin),
        "white",
    )
    draw = ImageDraw.Draw(image)

    title_width = title_box[2] - title_box[0]
    title_x = max(8, (image.width - title_width) // 2)
    draw.text((title_x, 8), title, fill="black", font=title_font)

    x_label_width = x_label_box[2] - x_label_box[0]
    x_label_x = left_margin + max((grid_width - x_label_width) // 2, 0)
    draw.text((x_label_x, top_margin + grid_height + 8), xlabel, fill="black", font=axis_font)

    y_label_x = 8
    y_label_y = top_margin + max((grid_height - rotated_y_label.height) // 2, 0)
    image.paste(rotated_y_label, (y_label_x, y_label_y), rotated_y_label)

    def color_for(value: float) -> tuple[int, int, int]:
        if max_value == min_value:
            ratio = 0.0
        else:
            ratio = (value - min_value) / (max_value - min_value)
        low = np.array((245, 250, 240), dtype=float)
        high = np.array((17, 94, 89), dtype=float)
        rgb = low + ratio * (high - low)
        return tuple(int(channel) for channel in rgb)

    for row_id in range(df.shape[0]):
        y0 = top_margin + row_id * cell_size
        y1 = y0 + cell_size
        label = row_labels[row_id]
        label_box = draw.textbbox((0, 0), label, font=font)
        label_y = y0 + max((cell_size - (label_box[3] - label_box[1])) // 2, 0)
        draw.text((left_margin - 8 - (label_box[2] - label_box[0]), label_y), label, fill="black", font=font)

        for col_id in range(df.shape[1]):
            x0 = left_margin + col_id * cell_size
            x1 = x0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color_for(values[row_id, col_id]), outline=(225, 225, 225))

    for col_id, rotated in enumerate(rotated_col_labels):
        x = left_margin + col_id * cell_size + max((cell_size - rotated.width) // 2, 0)
        y = top_margin - rotated.height - 6
        image.paste(rotated, (x, y), rotated)

    image.save(output_path, format="PNG")


def write_outputs(
    *,
    out_dir: Path,
    documents_df: pd.DataFrame,
    author_col: str,
    title_col: str,
    summary: dict[str, object],
    term_frequency: Counter[str],
    document_frequency: Counter[str],
    term_vocabulary: Sequence[dict[str, int | str]],
    word_vocabulary: Sequence[dict[str, int | str]],
    term_document_matrix: SimpleCSRMatrix,
    word_word_matrix: SimpleCSRMatrix,
    dense_term_document_df: pd.DataFrame,
    dense_word_word_df: pd.DataFrame,
    heatmap_term_document_df: pd.DataFrame,
    window_size: int,
    word_min_freq: int,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = out_dir / "task1_summary.json"
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_md_path = out_dir / "task1_summary.md"
    write_summary_markdown(summary_md_path, summary, word_min_freq=word_min_freq, window_size=window_size)

    frequencies_df = pd.DataFrame(
        [
            {
                "token": token,
                "term_frequency": int(term_frequency[token]),
                "document_frequency": int(document_frequency[token]),
            }
            for token in [str(entry["token"]) for entry in term_vocabulary]
        ]
    )
    token_frequencies_path = out_dir / "task1_token_frequencies.csv"
    frequencies_df.to_csv(token_frequencies_path, index=False, encoding="utf-8")

    documents_path = out_dir / "task1_documents.csv"
    documents_export_df = documents_df.loc[:, ["row_id", author_col, title_col]].rename(
        columns={author_col: "author", title_col: "title"}
    )
    documents_export_df.to_csv(documents_path, index=False, encoding="utf-8")

    term_vocab_path = out_dir / "task1_term_document_terms.csv"
    pd.DataFrame(term_vocabulary).to_csv(term_vocab_path, index=False, encoding="utf-8")

    term_matrix_path = out_dir / "task1_term_document_matrix.npz"
    save_npz(term_matrix_path, term_document_matrix)

    dense_term_document_path = out_dir / "task1_term_document_dense.csv"
    dense_term_document_df.to_csv(dense_term_document_path, encoding="utf-8")

    term_heatmap_path = out_dir / "task1_term_document_heatmap.png"
    save_heatmap(
        heatmap_term_document_df,
        term_heatmap_path,
        title="Term-Document Matrix Heatmap",
        xlabel="Terms",
        ylabel="Document row_id",
    )

    word_vocab_path = out_dir / "task1_word_word_terms.csv"
    pd.DataFrame(word_vocabulary).to_csv(word_vocab_path, index=False, encoding="utf-8")

    word_matrix_path = out_dir / "task1_word_word_matrix.npz"
    save_npz(word_matrix_path, word_word_matrix)

    dense_word_word_path = out_dir / "task1_word_word_dense.csv"
    dense_word_word_df.to_csv(dense_word_word_path, encoding="utf-8", index_label="token")

    word_heatmap_path = out_dir / "task1_word_word_heatmap.png"
    save_heatmap(
        dense_word_word_df,
        word_heatmap_path,
        title="Word-Word Matrix Heatmap",
        xlabel="Context word",
        ylabel="Center word",
    )

    return {
        "summary_json": summary_json_path,
        "summary_md": summary_md_path,
        "token_frequencies": token_frequencies_path,
        "documents": documents_path,
        "term_vocab": term_vocab_path,
        "term_matrix": term_matrix_path,
        "term_dense": dense_term_document_path,
        "term_heatmap": term_heatmap_path,
        "word_vocab": word_vocab_path,
        "word_matrix": word_matrix_path,
        "word_dense": dense_word_word_path,
        "word_heatmap": word_heatmap_path,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    if args.window_size < 1:
        raise ValueError("--window-size must be >= 1.")
    if args.word_min_freq < 1:
        raise ValueError("--word-min-freq must be >= 1.")
    if args.viz_top_n < 1:
        raise ValueError("--viz-top-n must be >= 1.")
    if args.viz_doc_limit < 1:
        raise ValueError("--viz-doc-limit must be >= 1.")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    documents_df = prepare_documents(
        pd.read_parquet(input_path),
        text_col=args.text_col,
        author_col=args.author_col,
        title_col=args.title_col,
    )

    summary, term_frequency, document_frequency = compute_corpus_statistics(
        documents_df,
        author_col=args.author_col,
    )

    term_vocabulary = build_vocabulary(term_frequency, document_frequency)
    word_vocabulary = build_vocabulary(
        term_frequency,
        document_frequency,
        min_frequency=args.word_min_freq,
    )
    if not term_vocabulary:
        raise ValueError("No vocabulary could be built from the input corpus.")
    if not word_vocabulary:
        raise ValueError(
            "Word-word vocabulary is empty. Lower --word-min-freq or verify the corpus contains tokens."
        )

    tokenized_documents = documents_df["tokens"].tolist()
    term_document_matrix = build_term_document_matrix(tokenized_documents, term_vocabulary)
    word_word_matrix = build_word_word_matrix(
        tokenized_documents,
        word_vocabulary,
        window_size=args.window_size,
    )

    dense_term_document_df = dense_term_document_view(
        term_document_matrix,
        documents_df,
        term_vocabulary,
        top_n=min(args.viz_top_n, len(term_vocabulary)),
    )
    heatmap_rows = select_term_document_heatmap_rows(
        dense_term_document_df,
        doc_limit=min(args.viz_doc_limit, len(dense_term_document_df)),
    )
    heatmap_term_document_df = dense_term_document_df.loc[heatmap_rows]

    dense_word_word_df = dense_word_word_view(
        word_word_matrix,
        word_vocabulary,
        top_n=min(args.viz_top_n, len(word_vocabulary)),
    )

    out_dir = Path(args.out_dir)
    artifact_paths = write_outputs(
        out_dir=out_dir,
        documents_df=documents_df,
        author_col=args.author_col,
        title_col=args.title_col,
        summary=summary,
        term_frequency=term_frequency,
        document_frequency=document_frequency,
        term_vocabulary=term_vocabulary,
        word_vocabulary=word_vocabulary,
        term_document_matrix=term_document_matrix,
        word_word_matrix=word_word_matrix,
        dense_term_document_df=dense_term_document_df,
        dense_word_word_df=dense_word_word_df,
        heatmap_term_document_df=heatmap_term_document_df,
        window_size=args.window_size,
        word_min_freq=args.word_min_freq,
    )

    return {
        "summary": summary,
        "artifact_paths": artifact_paths,
        "term_document_shape": tuple(term_document_matrix.shape),
        "word_word_shape": tuple(word_word_matrix.shape),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Task 1 corpus statistics and matrix artifacts.")
    parser.add_argument("--input", default="project3/poems_cleaned.parquet", help="Input parquet path.")
    parser.add_argument("--text-col", default="text", help="Text column name.")
    parser.add_argument("--author-col", default="author", help="Author column name.")
    parser.add_argument("--title-col", default="title", help="Title column name.")
    parser.add_argument("--window-size", type=int, default=2, help="Sliding window size for word co-occurrence.")
    parser.add_argument(
        "--word-min-freq",
        type=int,
        default=10,
        help="Minimum corpus frequency for words included in the word-word matrix.",
    )
    parser.add_argument("--viz-top-n", type=int, default=30, help="Top terms used in dense matrix exports.")
    parser.add_argument(
        "--viz-doc-limit",
        type=int,
        default=60,
        help="Number of documents shown in the term-document heatmap.",
    )
    parser.add_argument("--out-dir", default="project3/task1/results", help="Output directory for result artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_pipeline(args)

    print(f"Wrote artifacts to: {Path(args.out_dir)}")
    print(f"Term-document matrix shape: {result['term_document_shape']}")
    print(f"Word-word matrix shape: {result['word_word_shape']}")
    print(
        "Summary:",
        json.dumps(result["summary"], ensure_ascii=False),
    )


if __name__ == "__main__":
    main()
