from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from project3.task1.task1_corpus_matrices import tokenize
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from project3.task1.task1_corpus_matrices import tokenize

DEFAULT_QUERY_WORDS = [
    "aşıq",
    "can",
    "dil",
    "yar",
    "gözəl",
    "könül",
    "gül",
    "göz",
    "dərd",
    "sultan",
]


@dataclass(frozen=True)
class VocabularyItem:
    token_id: int
    token: str
    corpus_frequency: int


@dataclass(frozen=True)
class CorpusBundle:
    tokenized_documents: list[list[str]]
    encoded_documents: list[list[int]]
    term_frequency: Counter[str]
    vocabulary: list[VocabularyItem]
    token_to_id: dict[str, int]
    id_to_token: list[str]
    kept_token_count: int
    total_token_count: int


@dataclass(frozen=True)
class CooccurrenceBundle:
    row_ids: np.ndarray
    col_ids: np.ndarray
    counts: np.ndarray
    directed_nonzero_pairs: int
    total_pair_events: int
    total_weighted_count: float


@dataclass
class GloveParameters:
    word_vectors: np.ndarray
    context_vectors: np.ndarray
    word_biases: np.ndarray
    context_biases: np.ndarray
    word_vec_grad_sq: np.ndarray
    context_vec_grad_sq: np.ndarray
    word_bias_grad_sq: np.ndarray
    context_bias_grad_sq: np.ndarray


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def load_corpus(input_path: Path, *, text_col: str) -> list[list[str]]:
    df = pd.read_parquet(input_path)
    if text_col not in df.columns:
        raise ValueError(f"Missing required column: {text_col}")
    return [tokenize(text) for text in df[text_col].fillna("").astype(str)]


def build_vocabulary(term_frequency: Counter[str], *, min_count: int) -> list[VocabularyItem]:
    ordered = sorted((token, int(freq)) for token, freq in term_frequency.items() if freq >= min_count)
    ordered.sort(key=lambda item: (-item[1], item[0]))
    return [
        VocabularyItem(token_id=index, token=token, corpus_frequency=freq)
        for index, (token, freq) in enumerate(ordered)
    ]


def encode_documents(
    tokenized_documents: Iterable[Sequence[str]],
    token_to_id: dict[str, int],
) -> list[list[int]]:
    return [[token_to_id[token] for token in document if token in token_to_id] for document in tokenized_documents]


def prepare_corpus(
    input_path: Path,
    *,
    text_col: str,
    min_count: int,
) -> CorpusBundle:
    tokenized_documents = load_corpus(input_path, text_col=text_col)
    term_frequency: Counter[str] = Counter()
    for document in tokenized_documents:
        term_frequency.update(document)

    vocabulary = build_vocabulary(term_frequency, min_count=min_count)
    token_to_id = {item.token: item.token_id for item in vocabulary}
    id_to_token = [item.token for item in vocabulary]
    encoded_documents = encode_documents(tokenized_documents, token_to_id)

    return CorpusBundle(
        tokenized_documents=tokenized_documents,
        encoded_documents=encoded_documents,
        term_frequency=term_frequency,
        vocabulary=vocabulary,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        kept_token_count=sum(len(document) for document in encoded_documents),
        total_token_count=sum(len(document) for document in tokenized_documents),
    )


def build_cooccurrence_pairs(
    encoded_documents: Sequence[Sequence[int]],
    *,
    window_size: int,
) -> CooccurrenceBundle:
    pair_counts: defaultdict[tuple[int, int], float] = defaultdict(float)
    total_pair_events = 0

    for document in encoded_documents:
        doc_len = len(document)
        for center_index, center_id in enumerate(document):
            left = max(0, center_index - window_size)
            right = min(doc_len, center_index + window_size + 1)
            for context_index in range(left, right):
                if context_index == center_index:
                    continue
                distance = abs(center_index - context_index)
                context_id = int(document[context_index])
                pair_counts[(int(center_id), context_id)] += 1.0 / distance
                total_pair_events += 1

    if not pair_counts:
        return CooccurrenceBundle(
            row_ids=np.asarray([], dtype=np.int32),
            col_ids=np.asarray([], dtype=np.int32),
            counts=np.asarray([], dtype=np.float32),
            directed_nonzero_pairs=0,
            total_pair_events=0,
            total_weighted_count=0.0,
        )

    ordered_pairs = sorted(pair_counts.items(), key=lambda item: (item[0][0], item[0][1]))
    row_ids = np.asarray([pair[0] for pair, _ in ordered_pairs], dtype=np.int32)
    col_ids = np.asarray([pair[1] for pair, _ in ordered_pairs], dtype=np.int32)
    counts = np.asarray([weight for _, weight in ordered_pairs], dtype=np.float32)

    return CooccurrenceBundle(
        row_ids=row_ids,
        col_ids=col_ids,
        counts=counts,
        directed_nonzero_pairs=len(ordered_pairs),
        total_pair_events=total_pair_events,
        total_weighted_count=float(np.sum(counts, dtype=np.float64)),
    )


def glove_weighting(counts: np.ndarray, *, x_max: float, alpha: float) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)
    weights = np.ones_like(counts, dtype=np.float32)
    mask = counts < x_max
    weights[mask] = np.power(counts[mask] / x_max, alpha, dtype=np.float32)
    return weights


def initialize_glove_parameters(
    *,
    vocab_size: int,
    embedding_dim: int,
    seed: int,
) -> GloveParameters:
    rng = np.random.default_rng(seed)
    bound = 0.5 / max(embedding_dim, 1)
    word_vectors = rng.uniform(-bound, bound, size=(vocab_size, embedding_dim)).astype(np.float32)
    context_vectors = rng.uniform(-bound, bound, size=(vocab_size, embedding_dim)).astype(np.float32)
    word_biases = np.zeros(vocab_size, dtype=np.float32)
    context_biases = np.zeros(vocab_size, dtype=np.float32)

    return GloveParameters(
        word_vectors=word_vectors,
        context_vectors=context_vectors,
        word_biases=word_biases,
        context_biases=context_biases,
        word_vec_grad_sq=np.ones((vocab_size, embedding_dim), dtype=np.float32),
        context_vec_grad_sq=np.ones((vocab_size, embedding_dim), dtype=np.float32),
        word_bias_grad_sq=np.ones(vocab_size, dtype=np.float32),
        context_bias_grad_sq=np.ones(vocab_size, dtype=np.float32),
    )


def train_glove_epoch(
    *,
    params: GloveParameters,
    row_ids: np.ndarray,
    col_ids: np.ndarray,
    counts: np.ndarray,
    x_max: float,
    alpha: float,
    batch_size: int,
    learning_rate: float,
    rng: np.random.Generator,
    epsilon: float = 1e-8,
) -> tuple[float, int]:
    if len(row_ids) == 0:
        raise ValueError("GloVe training received zero co-occurrence pairs.")

    order = rng.permutation(len(row_ids))
    log_counts = np.log(counts.astype(np.float32))
    weight_values = glove_weighting(counts, x_max=x_max, alpha=alpha)

    loss_sum = 0.0
    pairs_seen = 0

    for start in range(0, len(order), batch_size):
        batch_indices = order[start : start + batch_size]
        batch_rows = row_ids[batch_indices]
        batch_cols = col_ids[batch_indices]
        batch_log_counts = log_counts[batch_indices]
        batch_weights = weight_values[batch_indices]

        word_vectors = params.word_vectors[batch_rows]
        context_vectors = params.context_vectors[batch_cols]
        predictions = (
            np.sum(word_vectors * context_vectors, axis=1)
            + params.word_biases[batch_rows]
            + params.context_biases[batch_cols]
        )
        residual = predictions - batch_log_counts
        weighted_loss = batch_weights * residual * residual
        grad_scale = batch_weights * residual

        grad_word = grad_scale[:, None] * context_vectors
        grad_context = grad_scale[:, None] * word_vectors

        unique_rows, row_inverse = np.unique(batch_rows, return_inverse=True)
        row_grad = np.zeros((len(unique_rows), params.word_vectors.shape[1]), dtype=np.float32)
        row_bias_grad = np.zeros(len(unique_rows), dtype=np.float32)
        np.add.at(row_grad, row_inverse, grad_word)
        np.add.at(row_bias_grad, row_inverse, grad_scale)

        unique_cols, col_inverse = np.unique(batch_cols, return_inverse=True)
        col_grad = np.zeros((len(unique_cols), params.context_vectors.shape[1]), dtype=np.float32)
        col_bias_grad = np.zeros(len(unique_cols), dtype=np.float32)
        np.add.at(col_grad, col_inverse, grad_context)
        np.add.at(col_bias_grad, col_inverse, grad_scale)

        params.word_vec_grad_sq[unique_rows] += row_grad * row_grad
        params.context_vec_grad_sq[unique_cols] += col_grad * col_grad
        params.word_bias_grad_sq[unique_rows] += row_bias_grad * row_bias_grad
        params.context_bias_grad_sq[unique_cols] += col_bias_grad * col_bias_grad

        params.word_vectors[unique_rows] -= (
            learning_rate * row_grad / np.sqrt(params.word_vec_grad_sq[unique_rows] + epsilon)
        )
        params.context_vectors[unique_cols] -= (
            learning_rate * col_grad / np.sqrt(params.context_vec_grad_sq[unique_cols] + epsilon)
        )
        params.word_biases[unique_rows] -= (
            learning_rate * row_bias_grad / np.sqrt(params.word_bias_grad_sq[unique_rows] + epsilon)
        )
        params.context_biases[unique_cols] -= (
            learning_rate * col_bias_grad / np.sqrt(params.context_bias_grad_sq[unique_cols] + epsilon)
        )

        batch_size_actual = len(batch_indices)
        loss_sum += float(np.sum(weighted_loss, dtype=np.float64))
        pairs_seen += batch_size_actual

    return loss_sum / max(pairs_seen, 1), pairs_seen


def train_glove(
    *,
    params: GloveParameters,
    cooccurrence: CooccurrenceBundle,
    x_max: float,
    alpha: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    metrics: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        avg_loss, pairs_seen = train_glove_epoch(
            params=params,
            row_ids=cooccurrence.row_ids,
            col_ids=cooccurrence.col_ids,
            counts=cooccurrence.counts,
            x_max=x_max,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            rng=rng,
        )
        metrics.append(
            {
                "model": "glove",
                "epoch": epoch,
                "avg_weighted_loss": avg_loss,
                "pairs_seen": pairs_seen,
            }
        )

    return metrics


def export_normalized_embeddings(params: GloveParameters) -> np.ndarray:
    combined = params.word_vectors + params.context_vectors
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized = combined / norms
    return normalized.astype(np.float32, copy=False)


def most_similar(
    embeddings: np.ndarray,
    token_to_id: dict[str, int],
    id_to_token: Sequence[str],
    query_word: str,
    *,
    top_k: int,
    excluded_words: set[str] | None = None,
) -> list[dict[str, object]]:
    if query_word not in token_to_id:
        return []

    query_id = token_to_id[query_word]
    scores = embeddings @ embeddings[query_id]
    scores = scores.astype(np.float64, copy=True)
    scores[query_id] = -np.inf

    if excluded_words:
        for word in excluded_words:
            word_id = token_to_id.get(word)
            if word_id is not None:
                scores[word_id] = -np.inf

    ranked_ids = np.argsort(-scores)
    rows: list[dict[str, object]] = []
    for token_id in ranked_ids:
        if len(rows) >= top_k:
            break
        if not np.isfinite(scores[token_id]):
            continue
        rows.append(
            {
                "similar_word": id_to_token[int(token_id)],
                "cosine_similarity": float(scores[token_id]),
            }
        )
    return rows


def evaluate_query_neighbors(
    *,
    embeddings: np.ndarray,
    query_words: Sequence[str],
    token_to_id: dict[str, int],
    id_to_token: Sequence[str],
    top_k: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for query_word in query_words:
        for rank, result in enumerate(
            most_similar(
                embeddings,
                token_to_id,
                id_to_token,
                query_word,
                top_k=top_k,
            ),
            start=1,
        ):
            rows.append(
                {
                    "model": "glove",
                    "query_word": query_word,
                    "rank": rank,
                    "similar_word": result["similar_word"],
                    "cosine_similarity": result["cosine_similarity"],
                }
            )
    return rows


def build_exclusion_set(
    term_frequency: Counter[str],
    *,
    task1_token_frequencies_path: Path | None,
    top_n: int = 20,
) -> set[str]:
    if task1_token_frequencies_path and task1_token_frequencies_path.exists():
        df = pd.read_csv(task1_token_frequencies_path)
        if "token" in df.columns:
            return set(df["token"].head(top_n).astype(str).tolist())
    return {token for token, _ in term_frequency.most_common(top_n)}


def evaluate_vector_equations(
    *,
    embeddings: np.ndarray,
    query_words: Sequence[str],
    token_to_id: dict[str, int],
    id_to_token: Sequence[str],
    exclusion_set: set[str],
    neighbor_top_k: int,
    equation_count: int,
    equation_result_top_k: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    accepted_equations = 0

    for query_word in query_words:
        neighbors = most_similar(
            embeddings,
            token_to_id,
            id_to_token,
            query_word,
            top_k=neighbor_top_k,
        )
        eligible_neighbors = [
            result["similar_word"]
            for result in neighbors
            if len(str(result["similar_word"])) >= 3 and str(result["similar_word"]) not in exclusion_set
        ]
        if len(eligible_neighbors) < 2:
            continue

        base_word = eligible_neighbors[0]
        subtract_word = eligible_neighbors[1]
        add_word = query_word

        vector = (
            embeddings[token_to_id[base_word]]
            - embeddings[token_to_id[subtract_word]]
            + embeddings[token_to_id[add_word]]
        )
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            continue
        vector = vector / norm
        scores = embeddings @ vector

        for excluded in {base_word, subtract_word, add_word}:
            token_id = token_to_id.get(excluded)
            if token_id is not None:
                scores[token_id] = -np.inf

        ranked_ids = np.argsort(-scores)
        equation_id = f"glove_{accepted_equations + 1}"
        rank = 0
        for token_id in ranked_ids:
            if rank >= equation_result_top_k:
                break
            if not np.isfinite(scores[token_id]):
                continue
            rank += 1
            rows.append(
                {
                    "model": "glove",
                    "equation_id": equation_id,
                    "base_word": base_word,
                    "subtract_word": subtract_word,
                    "add_word": add_word,
                    "rank": rank,
                    "result_word": id_to_token[int(token_id)],
                    "cosine_similarity": float(scores[token_id]),
                }
            )

        accepted_equations += 1
        if accepted_equations >= equation_count:
            break

    return rows


def write_summary_markdown(
    output_path: Path,
    *,
    config: dict[str, object],
    training_metrics_df: pd.DataFrame,
    neighbors_df: pd.DataFrame,
    equations_df: pd.DataFrame,
) -> None:
    lines = [
        "# Task 3 GloVe Summary",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
    ]

    for key in [
        "input",
        "text_col",
        "min_count",
        "embedding_dim",
        "window_size",
        "x_max",
        "alpha",
        "epochs",
        "batch_size",
        "learning_rate",
        "seed",
        "vocab_size",
        "tokens_after_min_count",
        "directed_nonzero_pairs",
        "total_pair_events",
    ]:
        lines.append(f"| {key} | {config[key]} |")

    lines.extend(
        [
            "",
            "## Training Loss",
            "",
            "| Model | Final Average Weighted Loss | Pairs Seen |",
            "| --- | ---: | ---: |",
        ]
    )

    if not training_metrics_df.empty:
        final_row = training_metrics_df.sort_values("epoch").iloc[-1]
        lines.append(
            f"| glove | {final_row['avg_weighted_loss']:.6f} | {int(final_row['pairs_seen'])} |"
        )

    lines.extend(
        [
            "",
            "## Similarity Notes",
            "",
            "The nearest-neighbor results mostly capture topical and poetic co-occurrence structure rather than strict dictionary synonymy.",
            "",
            "| Query Word | Top 3 Similar Words |",
            "| --- | --- |",
        ]
    )

    for query_word in config["query_words"]:
        query_neighbors = neighbors_df[neighbors_df["query_word"] == query_word].sort_values("rank")
        if query_neighbors.empty:
            continue
        top_words = ", ".join(query_neighbors.head(3)["similar_word"].astype(str).tolist())
        lines.append(f"| {query_word} | {top_words} |")

    lines.extend(
        [
            "",
            "## Vector Arithmetic Notes",
            "",
            "Each probe uses `neighbor_1 - neighbor_2 + query_word` with neighbors chosen deterministically from the learned top-10 results after excluding very common tokens.",
            "",
            "| Equation | Top Result |",
            "| --- | --- |",
        ]
    )

    for equation_id in equations_df["equation_id"].drop_duplicates().tolist():
        equation_rows = equations_df[equations_df["equation_id"] == equation_id].sort_values("rank")
        if equation_rows.empty:
            continue
        first_row = equation_rows.iloc[0]
        equation_text = (
            f"{first_row['base_word']} - {first_row['subtract_word']} + {first_row['add_word']}"
        )
        lines.append(f"| {equation_text} | {first_row['result_word']} |")

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    if args.min_count < 1:
        raise ValueError("--min-count must be >= 1.")
    if args.embedding_dim < 1:
        raise ValueError("--embedding-dim must be >= 1.")
    if args.window_size < 1:
        raise ValueError("--window-size must be >= 1.")
    if args.x_max <= 0:
        raise ValueError("--x-max must be > 0.")
    if args.alpha <= 0:
        raise ValueError("--alpha must be > 0.")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0.")
    if args.neighbor_top_k < 1:
        raise ValueError("--neighbor-top-k must be >= 1.")
    if args.equation_count < 1:
        raise ValueError("--equation-count must be >= 1.")
    if args.equation_result_top_k < 1:
        raise ValueError("--equation-result-top-k must be >= 1.")

    set_global_seed(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = prepare_corpus(input_path, text_col=args.text_col, min_count=args.min_count)
    if not corpus.vocabulary:
        raise ValueError("Vocabulary is empty. Lower --min-count or verify the corpus.")

    cooccurrence = build_cooccurrence_pairs(corpus.encoded_documents, window_size=args.window_size)
    if cooccurrence.directed_nonzero_pairs == 0:
        raise ValueError("No co-occurrence pairs were produced. Check the corpus or window size.")

    params = initialize_glove_parameters(
        vocab_size=len(corpus.vocabulary),
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )
    training_metrics = train_glove(
        params=params,
        cooccurrence=cooccurrence,
        x_max=args.x_max,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    embeddings = export_normalized_embeddings(params)

    embedding_path = out_dir / "task3_glove_embeddings.npy"
    np.save(embedding_path, embeddings)

    cooccurrence_path = out_dir / "task3_cooccurrence_pairs.npz"
    np.savez_compressed(
        cooccurrence_path,
        row_ids=cooccurrence.row_ids,
        col_ids=cooccurrence.col_ids,
        counts=cooccurrence.counts,
    )

    cooccurrence_stats = {
        "vocab_size": len(corpus.vocabulary),
        "tokens_after_min_count": corpus.kept_token_count,
        "directed_nonzero_pairs": cooccurrence.directed_nonzero_pairs,
        "total_pair_events": cooccurrence.total_pair_events,
        "total_weighted_count": cooccurrence.total_weighted_count,
    }
    cooccurrence_stats_path = out_dir / "task3_cooccurrence_stats.json"
    cooccurrence_stats_path.write_text(
        json.dumps(cooccurrence_stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    query_words = list(args.query_words or DEFAULT_QUERY_WORDS)
    neighbors_df = pd.DataFrame(
        evaluate_query_neighbors(
            embeddings=embeddings,
            query_words=query_words,
            token_to_id=corpus.token_to_id,
            id_to_token=corpus.id_to_token,
            top_k=args.neighbor_top_k,
        ),
        columns=["model", "query_word", "rank", "similar_word", "cosine_similarity"],
    )

    exclusion_set = build_exclusion_set(
        corpus.term_frequency,
        task1_token_frequencies_path=Path(args.task1_token_frequencies) if args.task1_token_frequencies else None,
        top_n=20,
    )
    equations_df = pd.DataFrame(
        evaluate_vector_equations(
            embeddings=embeddings,
            query_words=query_words,
            token_to_id=corpus.token_to_id,
            id_to_token=corpus.id_to_token,
            exclusion_set=exclusion_set,
            neighbor_top_k=args.neighbor_top_k,
            equation_count=args.equation_count,
            equation_result_top_k=args.equation_result_top_k,
        ),
        columns=[
            "model",
            "equation_id",
            "base_word",
            "subtract_word",
            "add_word",
            "rank",
            "result_word",
            "cosine_similarity",
        ],
    )

    config = {
        "input": str(input_path),
        "text_col": args.text_col,
        "min_count": args.min_count,
        "embedding_dim": args.embedding_dim,
        "window_size": args.window_size,
        "x_max": args.x_max,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "vocab_size": len(corpus.vocabulary),
        "tokens_after_min_count": corpus.kept_token_count,
        "directed_nonzero_pairs": cooccurrence.directed_nonzero_pairs,
        "total_pair_events": cooccurrence.total_pair_events,
        "total_weighted_count": cooccurrence.total_weighted_count,
        "query_words": query_words,
        "task1_exclusion_set": sorted(exclusion_set),
        "neighbor_top_k": args.neighbor_top_k,
        "equation_count": args.equation_count,
        "equation_result_top_k": args.equation_result_top_k,
    }

    config_path = out_dir / "task3_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    vocab_df = pd.DataFrame(
        [
            {
                "token_id": item.token_id,
                "token": item.token,
                "corpus_frequency": item.corpus_frequency,
            }
            for item in corpus.vocabulary
        ]
    )
    vocab_path = out_dir / "task3_vocab.csv"
    vocab_df.to_csv(vocab_path, index=False, encoding="utf-8")

    training_metrics_df = pd.DataFrame(training_metrics)
    metrics_path = out_dir / "task3_training_metrics.csv"
    training_metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    neighbors_path = out_dir / "task3_neighbors.csv"
    neighbors_df.to_csv(neighbors_path, index=False, encoding="utf-8")

    equations_path = out_dir / "task3_equations.csv"
    equations_df.to_csv(equations_path, index=False, encoding="utf-8")

    summary_path = out_dir / "task3_summary.md"
    write_summary_markdown(
        summary_path,
        config=config,
        training_metrics_df=training_metrics_df,
        neighbors_df=neighbors_df,
        equations_df=equations_df,
    )

    return {
        "config": config,
        "artifact_paths": {
            "config": config_path,
            "summary": summary_path,
            "vocab": vocab_path,
            "metrics": metrics_path,
            "neighbors": neighbors_path,
            "equations": equations_path,
            "embeddings": embedding_path,
            "cooccurrence_pairs": cooccurrence_path,
            "cooccurrence_stats": cooccurrence_stats_path,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate a local GloVe model for Task 3.")
    parser.add_argument("--input", default="project3/poems_cleaned.parquet", help="Input parquet path.")
    parser.add_argument("--text-col", default="text", help="Text column name.")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum token frequency to keep.")
    parser.add_argument("--embedding-dim", type=int, default=100, help="Embedding dimension.")
    parser.add_argument("--window-size", type=int, default=5, help="Co-occurrence window size.")
    parser.add_argument("--x-max", type=float, default=100.0, help="GloVe weighting cutoff.")
    parser.add_argument("--alpha", type=float, default=0.75, help="GloVe weighting exponent.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Mini-batch size for sparse pair updates.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="AdaGrad learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--task1-token-frequencies",
        default="project3/task1/results/task1_token_frequencies.csv",
        help="Optional Task 1 token-frequency CSV used to build the exclusion set.",
    )
    parser.add_argument(
        "--query-words",
        nargs="*",
        default=DEFAULT_QUERY_WORDS,
        help="Query words used for similarity evaluation.",
    )
    parser.add_argument("--neighbor-top-k", type=int, default=10, help="Neighbors per query word.")
    parser.add_argument("--equation-count", type=int, default=5, help="Number of vector-arithmetic probes.")
    parser.add_argument(
        "--equation-result-top-k",
        type=int,
        default=5,
        help="Returned results per vector-arithmetic probe.",
    )
    parser.add_argument("--out-dir", default="project3/task3/results", help="Output directory for result artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_pipeline(args)
    config = result["config"]

    print(f"Wrote artifacts to: {Path(args.out_dir)}")
    print(f"Vocabulary size: {config['vocab_size']}")
    print(f"Tokens after min_count: {config['tokens_after_min_count']}")
    print(f"Directed nonzero pairs: {config['directed_nonzero_pairs']}")
    print(f"Total pair events: {config['total_pair_events']}")


if __name__ == "__main__":
    main()
