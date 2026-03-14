from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SkipGramNegSamplingModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.reset_parameters(embedding_dim)

    def reset_parameters(self, embedding_dim: int) -> None:
        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.input_embeddings.weight, -bound, bound)
        nn.init.zeros_(self.output_embeddings.weight)

    def forward(
        self,
        center_ids: torch.Tensor,
        target_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        center_vectors = self.input_embeddings(center_ids)
        target_vectors = self.output_embeddings(target_ids)
        negative_vectors = self.output_embeddings(negative_ids)

        positive_scores = torch.sum(center_vectors * target_vectors, dim=1)
        negative_scores = torch.sum(center_vectors.unsqueeze(1) * negative_vectors, dim=2)

        positive_loss = -F.logsigmoid(positive_scores)
        negative_loss = -F.logsigmoid(-negative_scores).sum(dim=1)
        return (positive_loss + negative_loss).mean()

    def export_normalized_embeddings(self) -> np.ndarray:
        weights = self.input_embeddings.weight.detach().cpu()
        normalized = F.normalize(weights, p=2, dim=1)
        return np.asarray(normalized.tolist(), dtype=np.float32)


class CbowNegSamplingModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.input_embeddings = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean", sparse=True)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.reset_parameters(embedding_dim)

    def reset_parameters(self, embedding_dim: int) -> None:
        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.input_embeddings.weight, -bound, bound)
        nn.init.zeros_(self.output_embeddings.weight)

    def forward(
        self,
        context_ids: torch.Tensor,
        offsets: torch.Tensor,
        target_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        context_vectors = self.input_embeddings(context_ids, offsets)
        target_vectors = self.output_embeddings(target_ids)
        negative_vectors = self.output_embeddings(negative_ids)

        positive_scores = torch.sum(context_vectors * target_vectors, dim=1)
        negative_scores = torch.sum(context_vectors.unsqueeze(1) * negative_vectors, dim=2)

        positive_loss = -F.logsigmoid(positive_scores)
        negative_loss = -F.logsigmoid(-negative_scores).sum(dim=1)
        return (positive_loss + negative_loss).mean()

    def export_normalized_embeddings(self) -> np.ndarray:
        weights = self.input_embeddings.weight.detach().cpu()
        normalized = F.normalize(weights, p=2, dim=1)
        return np.asarray(normalized.tolist(), dtype=np.float32)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    encoded_documents: list[list[int]] = []
    for document in tokenized_documents:
        encoded_documents.append([token_to_id[token] for token in document if token in token_to_id])
    return encoded_documents


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


def compute_subsampling_keep_probs(
    vocabulary: Sequence[VocabularyItem],
    *,
    total_token_count: int,
    subsample_t: float,
) -> np.ndarray:
    if subsample_t <= 0:
        return np.ones(len(vocabulary), dtype=np.float32)

    keep_probs = np.ones(len(vocabulary), dtype=np.float32)
    total = max(total_token_count, 1)
    for item in vocabulary:
        freq = item.corpus_frequency / total
        keep_prob = min((np.sqrt(freq / subsample_t) + 1.0) * (subsample_t / freq), 1.0)
        keep_probs[item.token_id] = np.float32(keep_prob)
    return keep_probs


def subsample_encoded_documents(
    encoded_documents: Sequence[Sequence[int]],
    keep_probs: np.ndarray,
    *,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    subsampled: list[list[int]] = []
    for document in encoded_documents:
        if not document:
            subsampled.append([])
            continue
        document_array = np.asarray(document, dtype=np.int64)
        draws = rng.random(len(document_array))
        kept = document_array[draws < keep_probs[document_array]]
        subsampled.append(kept.tolist())
    return subsampled


def generate_skipgram_pairs(
    encoded_documents: Sequence[Sequence[int]],
    *,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    centers: list[int] = []
    targets: list[int] = []

    for document in encoded_documents:
        doc_len = len(document)
        for center_index, center_id in enumerate(document):
            left = max(0, center_index - window_size)
            right = min(doc_len, center_index + window_size + 1)
            for context_index in range(left, right):
                if context_index == center_index:
                    continue
                centers.append(int(center_id))
                targets.append(int(document[context_index]))

    return np.asarray(centers, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def generate_cbow_examples(
    encoded_documents: Sequence[Sequence[int]],
    *,
    window_size: int,
) -> tuple[list[list[int]], np.ndarray]:
    contexts: list[list[int]] = []
    targets: list[int] = []

    for document in encoded_documents:
        doc_len = len(document)
        for target_index, target_id in enumerate(document):
            left = max(0, target_index - window_size)
            right = min(doc_len, target_index + window_size + 1)
            context = [
                int(document[context_index])
                for context_index in range(left, right)
                if context_index != target_index
            ]
            if not context:
                continue
            contexts.append(context)
            targets.append(int(target_id))

    return contexts, np.asarray(targets, dtype=np.int64)


def build_negative_sampling_distribution(vocabulary: Sequence[VocabularyItem]) -> torch.Tensor:
    counts = torch.tensor([item.corpus_frequency for item in vocabulary], dtype=torch.float32)
    weights = counts.pow(0.75)
    return weights / weights.sum()


def sample_negative_ids(
    distribution: torch.Tensor,
    *,
    batch_size: int,
    negative_samples: int,
    forbidden_ids: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    sampled = torch.multinomial(
        distribution,
        batch_size * negative_samples,
        replacement=True,
        generator=generator,
    ).view(batch_size, negative_samples)

    forbidden = forbidden_ids.view(batch_size, 1)
    invalid_mask = sampled.eq(forbidden)
    while invalid_mask.any():
        resampled = torch.multinomial(
            distribution,
            int(invalid_mask.sum().item()),
            replacement=True,
            generator=generator,
        )
        sampled[invalid_mask] = resampled
        invalid_mask = sampled.eq(forbidden)
    return sampled


def _batched_flatten_contexts(contexts: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    flat_contexts: list[int] = []
    offsets: list[int] = []
    offset = 0
    for context in contexts:
        offsets.append(offset)
        flat_contexts.extend(int(token_id) for token_id in context)
        offset += len(context)
    return torch.tensor(flat_contexts, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


def train_skipgram_model(
    centers: np.ndarray,
    targets: np.ndarray,
    *,
    vocab_size: int,
    embedding_dim: int,
    negative_samples: int,
    distribution: torch.Tensor,
    epochs: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
) -> tuple[SkipGramNegSamplingModel, list[dict[str, object]]]:
    model = SkipGramNegSamplingModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(seed + 1000)
    metrics: list[dict[str, object]] = []

    if len(centers) == 0:
        raise ValueError("Skip-gram training received zero positive pairs.")

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(centers))
        loss_sum = 0.0
        examples_seen = 0

        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            center_batch = torch.tensor(centers[batch_indices], dtype=torch.long)
            target_batch = torch.tensor(targets[batch_indices], dtype=torch.long)
            negative_batch = sample_negative_ids(
                distribution,
                batch_size=len(batch_indices),
                negative_samples=negative_samples,
                forbidden_ids=target_batch,
                generator=sampling_generator,
            )

            optimizer.zero_grad()
            loss = model(center_batch, target_batch, negative_batch)
            loss.backward()
            optimizer.step()

            batch_size_actual = len(batch_indices)
            loss_sum += float(loss.item()) * batch_size_actual
            examples_seen += batch_size_actual

        metrics.append(
            {
                "model": "skipgram",
                "epoch": epoch,
                "avg_loss": loss_sum / max(examples_seen, 1),
                "examples_seen": examples_seen,
            }
        )

    return model, metrics


def train_cbow_model(
    contexts: Sequence[Sequence[int]],
    targets: np.ndarray,
    *,
    vocab_size: int,
    embedding_dim: int,
    negative_samples: int,
    distribution: torch.Tensor,
    epochs: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
) -> tuple[CbowNegSamplingModel, list[dict[str, object]]]:
    model = CbowNegSamplingModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(seed + 2000)
    metrics: list[dict[str, object]] = []

    if len(contexts) == 0:
        raise ValueError("CBOW training received zero training examples.")

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(contexts))
        loss_sum = 0.0
        examples_seen = 0

        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch_contexts = [contexts[int(index)] for index in batch_indices]
            batch_targets = torch.tensor(targets[batch_indices], dtype=torch.long)
            flat_contexts, offsets = _batched_flatten_contexts(batch_contexts)
            negative_batch = sample_negative_ids(
                distribution,
                batch_size=len(batch_indices),
                negative_samples=negative_samples,
                forbidden_ids=batch_targets,
                generator=sampling_generator,
            )

            optimizer.zero_grad()
            loss = model(flat_contexts, offsets, batch_targets, negative_batch)
            loss.backward()
            optimizer.step()

            batch_size_actual = len(batch_indices)
            loss_sum += float(loss.item()) * batch_size_actual
            examples_seen += batch_size_actual

        metrics.append(
            {
                "model": "cbow",
                "epoch": epoch,
                "avg_loss": loss_sum / max(examples_seen, 1),
                "examples_seen": examples_seen,
            }
        )

    return model, metrics


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
    results: list[dict[str, object]] = []
    for token_id in ranked_ids:
        if len(results) >= top_k:
            break
        if not np.isfinite(scores[token_id]):
            continue
        results.append(
            {
                "similar_word": id_to_token[int(token_id)],
                "cosine_similarity": float(scores[token_id]),
            }
        )
    return results


def evaluate_query_neighbors(
    *,
    model_name: str,
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
                    "model": model_name,
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
    model_name: str,
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
        equation_id = f"{model_name}_{accepted_equations + 1}"
        rank = 0
        for token_id in ranked_ids:
            if rank >= equation_result_top_k:
                break
            if not np.isfinite(scores[token_id]):
                continue
            rank += 1
            rows.append(
                {
                    "model": model_name,
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
        "# Task 2 Word2Vec Summary",
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
        "negative_samples",
        "subsample_t",
        "epochs",
        "batch_size",
        "seed",
        "vocab_size",
        "tokens_after_min_count",
        "tokens_after_subsampling",
    ]:
        lines.append(f"| {key} | {config[key]} |")

    lines.extend(
        [
            "",
            "## Training Loss",
            "",
            "| Model | Final Average Loss | Examples Seen |",
            "| --- | ---: | ---: |",
        ]
    )

    for model_name in ["skipgram", "cbow"]:
        model_metrics = training_metrics_df[training_metrics_df["model"] == model_name]
        if model_metrics.empty:
            continue
        final_row = model_metrics.sort_values("epoch").iloc[-1]
        lines.append(
            f"| {model_name} | {final_row['avg_loss']:.6f} | {int(final_row['examples_seen'])} |"
        )

    lines.extend(
        [
            "",
            "## Similarity Notes",
            "",
            "The nearest-neighbor results mostly capture poetic association, style, and lexical relatedness rather than strict dictionary synonymy.",
            "",
        ]
    )

    for model_name in ["skipgram", "cbow"]:
        lines.extend(
            [
                f"### {model_name.capitalize()}",
                "",
                "| Query Word | Top 3 Similar Words |",
                "| --- | --- |",
            ]
        )
        model_neighbors = neighbors_df[neighbors_df["model"] == model_name]
        for query_word in config["query_words"]:
            query_neighbors = model_neighbors[model_neighbors["query_word"] == query_word].sort_values("rank")
            if query_neighbors.empty:
                continue
            top_words = ", ".join(query_neighbors.head(3)["similar_word"].astype(str).tolist())
            lines.append(f"| {query_word} | {top_words} |")
        lines.append("")

    lines.extend(
        [
            "## Vector Arithmetic Notes",
            "",
            "Each probe uses `neighbor_1 - neighbor_2 + query_word` with neighbors chosen deterministically from the learned top-10 results after excluding very common tokens.",
            "",
            "| Model | Equation | Top Result |",
            "| --- | --- | --- |",
        ]
    )

    for model_name in ["skipgram", "cbow"]:
        model_equations = equations_df[equations_df["model"] == model_name]
        for equation_id in model_equations["equation_id"].drop_duplicates().tolist():
            equation_rows = model_equations[model_equations["equation_id"] == equation_id].sort_values("rank")
            if equation_rows.empty:
                continue
            first_row = equation_rows.iloc[0]
            equation_text = (
                f"{first_row['base_word']} - {first_row['subtract_word']} + {first_row['add_word']}"
            )
            lines.append(f"| {model_name} | {equation_text} | {first_row['result_word']} |")

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    if args.min_count < 1:
        raise ValueError("--min-count must be >= 1.")
    if args.embedding_dim < 1:
        raise ValueError("--embedding-dim must be >= 1.")
    if args.window_size < 1:
        raise ValueError("--window-size must be >= 1.")
    if args.negative_samples < 1:
        raise ValueError("--negative-samples must be >= 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
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

    corpus = prepare_corpus(
        input_path,
        text_col=args.text_col,
        min_count=args.min_count,
    )
    if not corpus.vocabulary:
        raise ValueError("Vocabulary is empty. Lower --min-count or verify the corpus.")

    keep_probs = compute_subsampling_keep_probs(
        corpus.vocabulary,
        total_token_count=corpus.kept_token_count,
        subsample_t=args.subsample_t,
    )
    subsampled_documents = subsample_encoded_documents(
        corpus.encoded_documents,
        keep_probs,
        seed=args.seed,
    )
    subsampled_token_count = sum(len(document) for document in subsampled_documents)
    training_documents = subsampled_documents

    skipgram_centers, skipgram_targets = generate_skipgram_pairs(training_documents, window_size=args.window_size)
    cbow_contexts, cbow_targets = generate_cbow_examples(training_documents, window_size=args.window_size)
    subsampling_applied = True
    if len(skipgram_centers) == 0 or len(cbow_contexts) == 0:
        training_documents = [list(document) for document in corpus.encoded_documents]
        subsampled_token_count = corpus.kept_token_count
        skipgram_centers, skipgram_targets = generate_skipgram_pairs(
            training_documents,
            window_size=args.window_size,
        )
        cbow_contexts, cbow_targets = generate_cbow_examples(
            training_documents,
            window_size=args.window_size,
        )
        subsampling_applied = False

    distribution = build_negative_sampling_distribution(corpus.vocabulary)
    learning_rate = 0.01

    skipgram_model, skipgram_metrics = train_skipgram_model(
        skipgram_centers,
        skipgram_targets,
        vocab_size=len(corpus.vocabulary),
        embedding_dim=args.embedding_dim,
        negative_samples=args.negative_samples,
        distribution=distribution,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=learning_rate,
    )
    cbow_model, cbow_metrics = train_cbow_model(
        cbow_contexts,
        cbow_targets,
        vocab_size=len(corpus.vocabulary),
        embedding_dim=args.embedding_dim,
        negative_samples=args.negative_samples,
        distribution=distribution,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed + 1,
        learning_rate=learning_rate,
    )

    skipgram_embeddings = skipgram_model.export_normalized_embeddings()
    cbow_embeddings = cbow_model.export_normalized_embeddings()

    skipgram_embedding_path = out_dir / "task2_skipgram_embeddings.npy"
    cbow_embedding_path = out_dir / "task2_cbow_embeddings.npy"
    np.save(skipgram_embedding_path, skipgram_embeddings)
    np.save(cbow_embedding_path, cbow_embeddings)

    query_words = list(args.query_words or DEFAULT_QUERY_WORDS)
    neighbors_rows = evaluate_query_neighbors(
        model_name="skipgram",
        embeddings=skipgram_embeddings,
        query_words=query_words,
        token_to_id=corpus.token_to_id,
        id_to_token=corpus.id_to_token,
        top_k=args.neighbor_top_k,
    )
    neighbors_rows.extend(
        evaluate_query_neighbors(
            model_name="cbow",
            embeddings=cbow_embeddings,
            query_words=query_words,
            token_to_id=corpus.token_to_id,
            id_to_token=corpus.id_to_token,
            top_k=args.neighbor_top_k,
        )
    )
    neighbors_df = pd.DataFrame(
        neighbors_rows,
        columns=["model", "query_word", "rank", "similar_word", "cosine_similarity"],
    )

    exclusion_set = build_exclusion_set(
        corpus.term_frequency,
        task1_token_frequencies_path=Path(args.task1_token_frequencies) if args.task1_token_frequencies else None,
        top_n=20,
    )
    equation_rows = evaluate_vector_equations(
        model_name="skipgram",
        embeddings=skipgram_embeddings,
        query_words=query_words,
        token_to_id=corpus.token_to_id,
        id_to_token=corpus.id_to_token,
        exclusion_set=exclusion_set,
        neighbor_top_k=args.neighbor_top_k,
        equation_count=args.equation_count,
        equation_result_top_k=args.equation_result_top_k,
    )
    equation_rows.extend(
        evaluate_vector_equations(
            model_name="cbow",
            embeddings=cbow_embeddings,
            query_words=query_words,
            token_to_id=corpus.token_to_id,
            id_to_token=corpus.id_to_token,
            exclusion_set=exclusion_set,
            neighbor_top_k=args.neighbor_top_k,
            equation_count=args.equation_count,
            equation_result_top_k=args.equation_result_top_k,
        )
    )
    equations_df = pd.DataFrame(
        equation_rows,
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
        "negative_samples": args.negative_samples,
        "subsample_t": args.subsample_t,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": learning_rate,
        "vocab_size": len(corpus.vocabulary),
        "total_tokens": corpus.total_token_count,
        "tokens_after_min_count": corpus.kept_token_count,
        "tokens_after_subsampling": subsampled_token_count,
        "subsampling_applied": subsampling_applied,
        "skipgram_positive_pairs": int(len(skipgram_centers)),
        "cbow_examples": int(len(cbow_contexts)),
        "query_words": query_words,
        "task1_exclusion_set": sorted(exclusion_set),
        "neighbor_top_k": args.neighbor_top_k,
        "equation_count": args.equation_count,
        "equation_result_top_k": args.equation_result_top_k,
    }

    config_path = out_dir / "task2_config.json"
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
    vocab_path = out_dir / "task2_vocab.csv"
    vocab_df.to_csv(vocab_path, index=False, encoding="utf-8")

    training_metrics_df = pd.DataFrame(skipgram_metrics + cbow_metrics)
    training_metrics_path = out_dir / "task2_training_metrics.csv"
    training_metrics_df.to_csv(training_metrics_path, index=False, encoding="utf-8")

    neighbors_path = out_dir / "task2_neighbors.csv"
    neighbors_df.to_csv(neighbors_path, index=False, encoding="utf-8")

    equations_path = out_dir / "task2_equations.csv"
    equations_df.to_csv(equations_path, index=False, encoding="utf-8")

    summary_path = out_dir / "task2_summary.md"
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
            "metrics": training_metrics_path,
            "neighbors": neighbors_path,
            "equations": equations_path,
            "skipgram_embeddings": skipgram_embedding_path,
            "cbow_embeddings": cbow_embedding_path,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate Word2Vec models for Task 2.")
    parser.add_argument("--input", default="project3/poems_cleaned.parquet", help="Input parquet path.")
    parser.add_argument("--text-col", default="text", help="Text column name.")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum token frequency to keep in the vocabulary.")
    parser.add_argument("--embedding-dim", type=int, default=100, help="Embedding dimension.")
    parser.add_argument("--window-size", type=int, default=5, help="Context window size.")
    parser.add_argument("--negative-samples", type=int, default=5, help="Number of negative samples per example.")
    parser.add_argument("--subsample-t", type=float, default=1e-4, help="Subsampling threshold.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs for each model.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Mini-batch size.")
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
    parser.add_argument("--equation-count", type=int, default=5, help="Number of vector-arithmetic probes per model.")
    parser.add_argument(
        "--equation-result-top-k",
        type=int,
        default=5,
        help="Returned results per vector-arithmetic probe.",
    )
    parser.add_argument("--out-dir", default="project3/task2/results", help="Output directory for result artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_pipeline(args)
    config = result["config"]

    print(f"Wrote artifacts to: {Path(args.out_dir)}")
    print(f"Vocabulary size: {config['vocab_size']}")
    print(f"Tokens after min_count: {config['tokens_after_min_count']}")
    print(f"Tokens after subsampling: {config['tokens_after_subsampling']}")
    print(f"Skip-gram positive pairs: {config['skipgram_positive_pairs']}")
    print(f"CBOW examples: {config['cbow_examples']}")


if __name__ == "__main__":
    main()
