from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from task1.task1_corpus_matrices import tokenize
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from task1.task1_corpus_matrices import tokenize


@dataclass(frozen=True)
class FeatureBundle:
    name: str
    train: np.ndarray
    test: np.ndarray


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        architecture: str,
    ) -> None:
        super().__init__()
        self.architecture = architecture

        if architecture == "rnn":
            self.recurrent = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            output_dim = hidden_dim
        elif architecture == "birnn":
            self.recurrent = nn.RNN(
                input_size=input_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            output_dim = hidden_dim * 2
        elif architecture == "lstm":
            self.recurrent = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            output_dim = hidden_dim
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Treat fixed-length feature vectors as one-step sequences.
        sequence_inputs = inputs.unsqueeze(1)

        if self.architecture == "lstm":
            _, (hidden, _) = self.recurrent(sequence_inputs)
            final_state = hidden[-1]
        else:
            _, hidden = self.recurrent(sequence_inputs)
            if self.architecture == "birnn":
                final_state = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                final_state = hidden[-1]

        return self.classifier(final_state)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(input_path: Path, *, text_col: str, label_col: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    df = pd.read_parquet(input_path)
    missing = [column for column in [text_col, label_col] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    dataset = df[[text_col, label_col]].copy()
    dataset[text_col] = dataset[text_col].fillna("").astype(str)
    dataset[label_col] = dataset[label_col].fillna("unknown").astype(str)
    dataset["tokens"] = dataset[text_col].map(tokenize)
    dataset["tokenized_text"] = dataset["tokens"].map(lambda items: " ".join(items))
    return dataset


def filter_classes(dataset: pd.DataFrame, *, label_col: str, min_docs_per_class: int) -> pd.DataFrame:
    counts = dataset[label_col].value_counts()
    kept_labels = counts[counts >= min_docs_per_class].index.tolist()
    filtered = dataset[dataset[label_col].isin(kept_labels)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No classes left after filtering. Lower --min-docs-per-class.")
    return filtered


def build_count_features(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    *,
    max_features: int,
) -> FeatureBundle:
    vectorizer = CountVectorizer(max_features=max_features)
    train = vectorizer.fit_transform(train_texts).toarray().astype(np.float32)
    test = vectorizer.transform(test_texts).toarray().astype(np.float32)
    return FeatureBundle(name="count", train=train, test=test)


def build_tfidf_features(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    *,
    max_features: int,
) -> FeatureBundle:
    vectorizer = TfidfVectorizer(max_features=max_features)
    train = vectorizer.fit_transform(train_texts).toarray().astype(np.float32)
    test = vectorizer.transform(test_texts).toarray().astype(np.float32)
    return FeatureBundle(name="tfidf", train=train, test=test)


def _binary_presence_matrix(texts: Sequence[str], *, max_features: int) -> tuple[np.ndarray, CountVectorizer]:
    vectorizer = CountVectorizer(binary=True, max_features=max_features)
    matrix = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    return matrix, vectorizer


def _compute_class_word_ppmi(binary_matrix: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    # binary_matrix: [N_docs, V], labels: [N_docs]
    n_docs, vocab_size = binary_matrix.shape
    epsilon = 1e-9

    class_doc_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    word_doc_counts = binary_matrix.sum(axis=0, dtype=np.float64)  # [V]

    p_word = (word_doc_counts + epsilon) / (n_docs + epsilon)
    p_class = (class_doc_counts + epsilon) / (n_docs + epsilon)

    ppmi = np.zeros((vocab_size, num_classes), dtype=np.float32)
    for class_id in range(num_classes):
        class_mask = labels == class_id
        joint_counts = binary_matrix[class_mask].sum(axis=0, dtype=np.float64)  # [V]
        p_joint = (joint_counts + epsilon) / (n_docs + epsilon)
        pmi = np.log(p_joint / (p_word * p_class[class_id] + epsilon))
        ppmi[:, class_id] = np.maximum(pmi, 0.0).astype(np.float32)

    return ppmi


def build_pmi_features(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    train_labels: np.ndarray,
    *,
    num_classes: int,
    max_features: int,
) -> FeatureBundle:
    train_binary, vectorizer = _binary_presence_matrix(train_texts, max_features=max_features)
    test_binary = vectorizer.transform(test_texts).toarray().astype(np.float32)

    ppmi = _compute_class_word_ppmi(train_binary, train_labels, num_classes)

    train_scores = train_binary @ ppmi
    test_scores = test_binary @ ppmi

    train_norm = np.maximum(train_binary.sum(axis=1, keepdims=True), 1.0)
    test_norm = np.maximum(test_binary.sum(axis=1, keepdims=True), 1.0)

    train = (train_scores / train_norm).astype(np.float32)
    test = (test_scores / test_norm).astype(np.float32)
    return FeatureBundle(name="pmi", train=train, test=test)


def load_embedding_matrix(embedding_path: Path, vocab_path: Path) -> tuple[np.ndarray, dict[str, int]]:
    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {embedding_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocabulary file: {vocab_path}")

    embeddings = np.load(embedding_path).astype(np.float32)
    vocab_df = pd.read_csv(vocab_path)
    if "token" not in vocab_df.columns:
        raise ValueError(f"Expected a 'token' column in vocabulary file: {vocab_path}")

    token_to_id = {token: idx for idx, token in enumerate(vocab_df["token"].astype(str).tolist())}
    if embeddings.shape[0] != len(token_to_id):
        raise ValueError(
            f"Embedding-vocabulary mismatch for {embedding_path.name}: "
            f"{embeddings.shape[0]} rows vs {len(token_to_id)} vocab tokens"
        )

    return embeddings, token_to_id


def average_embedding_features(token_lists: Sequence[Sequence[str]], embeddings: np.ndarray, token_to_id: dict[str, int]) -> np.ndarray:
    features = np.zeros((len(token_lists), embeddings.shape[1]), dtype=np.float32)
    for index, tokens in enumerate(token_lists):
        vectors = [embeddings[token_to_id[token]] for token in tokens if token in token_to_id]
        if vectors:
            features[index] = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
    return features


def build_embedding_features(
    *,
    name: str,
    train_tokens: Sequence[Sequence[str]],
    test_tokens: Sequence[Sequence[str]],
    embeddings: np.ndarray,
    token_to_id: dict[str, int],
) -> FeatureBundle:
    train = average_embedding_features(train_tokens, embeddings, token_to_id)
    test = average_embedding_features(test_tokens, embeddings, token_to_id)
    return FeatureBundle(name=name, train=train, test=test)


def train_model(
    *,
    architecture: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[RecurrentClassifier, list[dict[str, object]]]:
    set_global_seed(seed)

    x_train = torch.tensor(train_features, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)

    model = RecurrentClassifier(
        input_dim=x_train.shape[1],
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        architecture=architecture,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    rng = np.random.default_rng(seed)
    history: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(x_train))
        epoch_loss = 0.0
        examples_seen = 0

        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch_x = x_train[batch_indices]
            batch_y = y_train[batch_indices]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_size_actual = len(batch_indices)
            epoch_loss += float(loss.item()) * batch_size_actual
            examples_seen += batch_size_actual

        history.append(
            {
                "epoch": epoch,
                "avg_loss": epoch_loss / max(examples_seen, 1),
                "examples_seen": examples_seen,
            }
        )

    return model, history


def evaluate_model(model: RecurrentClassifier, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32))
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def write_summary_markdown(
    path: Path,
    *,
    config: dict[str, object],
    results_df: pd.DataFrame,
) -> None:
    lines = [
        "# Task 5 Classification Summary",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
    ]

    for key in [
        "input",
        "text_col",
        "label_col",
        "min_docs_per_class",
        "test_size",
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "hidden_dim",
        "max_vectorizer_features",
        "train_docs",
        "test_docs",
        "num_classes",
    ]:
        lines.append(f"| {key} | {config[key]} |")

    lines.extend(
        [
            "",
            "## Performance Table",
            "",
            "| Feature | Model | Accuracy | Macro F1 |",
            "| --- | --- | ---: | ---: |",
        ]
    )

    for _, row in results_df.sort_values(["feature", "model"]).iterrows():
        lines.append(
            f"| {row['feature']} | {row['model']} | {float(row['test_accuracy']):.4f} | {float(row['test_macro_f1']):.4f} |"
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.hidden_dim < 1:
        raise ValueError("--hidden-dim must be >= 1.")
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test-size must be in (0, 1).")

    set_global_seed(args.seed)

    dataset = load_dataset(Path(args.input), text_col=args.text_col, label_col=args.label_col)
    dataset = filter_classes(dataset, label_col=args.label_col, min_docs_per_class=args.min_docs_per_class)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(dataset[args.label_col].astype(str))

    train_df, test_df, y_train, y_test = train_test_split(
        dataset,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    train_texts = train_df["tokenized_text"].tolist()
    test_texts = test_df["tokenized_text"].tolist()
    train_tokens = train_df["tokens"].tolist()
    test_tokens = test_df["tokens"].tolist()
    num_classes = len(label_encoder.classes_)

    count_features = build_count_features(train_texts, test_texts, max_features=args.max_vectorizer_features)
    tfidf_features = build_tfidf_features(train_texts, test_texts, max_features=args.max_vectorizer_features)
    pmi_features = build_pmi_features(
        train_texts,
        test_texts,
        y_train,
        num_classes=num_classes,
        max_features=args.max_vectorizer_features,
    )

    w2v_embeddings, w2v_token_to_id = load_embedding_matrix(
        Path(args.task2_embeddings),
        Path(args.task2_vocab),
    )
    glove_embeddings, glove_token_to_id = load_embedding_matrix(
        Path(args.task3_embeddings),
        Path(args.task3_vocab),
    )

    w2v_features = build_embedding_features(
        name="word2vec",
        train_tokens=train_tokens,
        test_tokens=test_tokens,
        embeddings=w2v_embeddings,
        token_to_id=w2v_token_to_id,
    )
    glove_features = build_embedding_features(
        name="glove",
        train_tokens=train_tokens,
        test_tokens=test_tokens,
        embeddings=glove_embeddings,
        token_to_id=glove_token_to_id,
    )

    feature_bundles = [count_features, tfidf_features, pmi_features, w2v_features, glove_features]
    architectures = ["rnn", "birnn", "lstm"]

    result_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for feature_bundle in feature_bundles:
        for architecture in architectures:
            model, history = train_model(
                architecture=architecture,
                train_features=feature_bundle.train,
                train_labels=y_train,
                num_classes=num_classes,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
            )

            train_metrics = evaluate_model(model, feature_bundle.train, y_train)
            test_metrics = evaluate_model(model, feature_bundle.test, y_test)

            result_rows.append(
                {
                    "feature": feature_bundle.name,
                    "model": architecture,
                    "input_dim": int(feature_bundle.train.shape[1]),
                    "train_accuracy": train_metrics["accuracy"],
                    "train_macro_f1": train_metrics["macro_f1"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "train_docs": int(len(train_df)),
                    "test_docs": int(len(test_df)),
                    "num_classes": int(num_classes),
                }
            )

            for item in history:
                metric_rows.append(
                    {
                        "feature": feature_bundle.name,
                        "model": architecture,
                        "epoch": int(item["epoch"]),
                        "avg_loss": float(item["avg_loss"]),
                        "examples_seen": int(item["examples_seen"]),
                    }
                )

    results_df = pd.DataFrame(result_rows)
    metrics_df = pd.DataFrame(metric_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "input": args.input,
        "text_col": args.text_col,
        "label_col": args.label_col,
        "min_docs_per_class": args.min_docs_per_class,
        "test_size": args.test_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "max_vectorizer_features": args.max_vectorizer_features,
        "train_docs": int(len(train_df)),
        "test_docs": int(len(test_df)),
        "num_classes": int(num_classes),
        "class_names": label_encoder.classes_.tolist(),
        "features": [bundle.name for bundle in feature_bundles],
        "models": architectures,
        "task2_embeddings": args.task2_embeddings,
        "task2_vocab": args.task2_vocab,
        "task3_embeddings": args.task3_embeddings,
        "task3_vocab": args.task3_vocab,
    }

    config_path = out_dir / "task5_config.json"
    results_path = out_dir / "task5_results.csv"
    metrics_path = out_dir / "task5_training_metrics.csv"
    summary_path = out_dir / "task5_summary.md"

    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    results_df.to_csv(results_path, index=False, encoding="utf-8")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")
    write_summary_markdown(summary_path, config=config, results_df=results_df)

    return {
        "config": config,
        "artifact_paths": {
            "config": config_path,
            "results": results_path,
            "training_metrics": metrics_path,
            "summary": summary_path,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train RNN, BiRNN, and LSTM text classifiers using Count, TF-IDF, PMI, Word2Vec, and GloVe features."
        )
    )
    parser.add_argument("--input", default="project3/poems_cleaned.parquet", help="Input parquet path.")
    parser.add_argument("--text-col", default="text", help="Text column name.")
    parser.add_argument("--label-col", default="author", help="Target label column name.")
    parser.add_argument("--min-docs-per-class", type=int, default=5, help="Minimum documents required per label.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion.")
    parser.add_argument("--max-vectorizer-features", type=int, default=3000, help="Max features for Count/TF-IDF/PMI.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension for recurrent encoders.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs for each model-feature pair.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--task2-embeddings",
        default="project3/task2/results/task2_skipgram_embeddings.npy",
        help="Task 2 Word2Vec embedding file.",
    )
    parser.add_argument(
        "--task2-vocab",
        default="project3/task2/results/task2_vocab.csv",
        help="Task 2 vocabulary CSV.",
    )
    parser.add_argument(
        "--task3-embeddings",
        default="project3/task3/results/task3_glove_embeddings.npy",
        help="Task 3 GloVe embedding file.",
    )
    parser.add_argument(
        "--task3-vocab",
        default="project3/task3/results/task3_vocab.csv",
        help="Task 3 vocabulary CSV.",
    )
    parser.add_argument("--out-dir", default="project3/task5/results", help="Output directory for Task 5 artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_pipeline(args)
    config = result["config"]

    print(f"Wrote artifacts to: {Path(args.out_dir)}")
    print(f"Train docs: {config['train_docs']}")
    print(f"Test docs: {config['test_docs']}")
    print(f"Classes: {config['num_classes']}")


if __name__ == "__main__":
    main()
