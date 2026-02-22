from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

# Sentiment lexicon setup

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.corpus import opinion_lexicon

    _VADER = SentimentIntensityAnalyzer()
    _POS_WORDS: set[str] = set(opinion_lexicon.positive())
    _NEG_WORDS: set[str] = set(opinion_lexicon.negative())
except Exception:  # pragma: no cover – graceful degradation
    _VADER = None  # type: ignore[assignment]
    _POS_WORDS = set()
    _NEG_WORDS = set()

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

CLASSIFIERS: tuple[str, ...] = ("naive_bayes", "binary_naive_bayes", "logistic_regression")
FEATURE_SETS: tuple[str, ...] = ("bow", "lexicon", "bow_lexicon")


# Text tokenization and validation

def tokenize(text: str) -> str:
    """Lowercase, return space-joined tokens (for CountVectorizer analyzer='word')."""
    return " ".join(TOKEN_RE.findall(text.lower()))


def validate_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# Cleaning and splitting data

def clean_dataframe(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop empty texts, return cleaned df and stats."""
    rows_input = len(df)
    text_series = df[text_col].fillna("").astype(str)
    mask = text_series.str.strip().ne("")
    clean_df = df.loc[mask].copy()
    clean_df["clean_text"] = clean_df[text_col].astype(str).map(tokenize)
    non_empty = clean_df["clean_text"].str.strip().ne("")
    clean_df = clean_df.loc[non_empty].reset_index(drop=True)
    stats = {
        "rows_input": int(rows_input),
        "rows_after_cleaning": int(len(clean_df)),
        "dropped_rows": int(rows_input - len(clean_df)),
    }
    return clean_df, stats


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-label stratified split identical to task1 logic."""
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in df.groupby(label_col, sort=False):
        shuffled = group.sample(frac=1.0, random_state=seed)
        if len(shuffled) <= 1:
            n_test = 0
        else:
            n_test = max(1, int(round(len(shuffled) * test_size)))
            n_test = min(n_test, len(shuffled) - 1)
        test_parts.append(shuffled.iloc[:n_test].copy())
        train_parts.append(shuffled.iloc[n_test:].copy())

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Train split is empty.")
    if test_df.empty:
        raise ValueError("Test split is empty.")
    return train_df, test_df


# Building feature vectors

def build_bow_features(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    binary: bool = False,
    max_features: int | None = None,
) -> tuple[np.ndarray, np.ndarray, CountVectorizer]:
    """Standard or binary bag-of-words via CountVectorizer."""
    vec = CountVectorizer(
        analyzer="word",
        binary=binary,
        max_features=max_features,
        token_pattern=r"\b\w+\b",
    )
    X_train = vec.fit_transform(train_texts).toarray()
    X_test = vec.transform(test_texts).toarray()
    return X_train, X_test, vec


def _lexicon_features_single(text: str) -> list[float]:
    """Compute sentiment-lexicon features for a single document."""
    tokens = TOKEN_RE.findall(text.lower())
    n_tokens = max(len(tokens), 1)
    pos_count = sum(1 for t in tokens if t in _POS_WORDS)
    neg_count = sum(1 for t in tokens if t in _NEG_WORDS)

    feats = [
        float(pos_count),
        float(neg_count),
        float(pos_count - neg_count),
        float(pos_count / n_tokens),
        float(neg_count / n_tokens),
    ]
    if _VADER is not None:
        scores = _VADER.polarity_scores(text)
        feats.extend([scores["pos"], scores["neg"], scores["neu"], scores["compound"]])
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])
    return feats


LEXICON_FEATURE_NAMES: list[str] = [
    "lex_pos_count",
    "lex_neg_count",
    "lex_polarity_diff",
    "lex_pos_ratio",
    "lex_neg_ratio",
    "vader_pos",
    "vader_neg",
    "vader_neu",
    "vader_compound",
]


def build_lexicon_features(texts: Sequence[str]) -> np.ndarray:
    """Build a (n_docs, 9) sentiment-lexicon feature matrix."""
    return np.array([_lexicon_features_single(t) for t in texts], dtype=np.float64)


def build_combined_features(
    bow_matrix: np.ndarray,
    lexicon_matrix: np.ndarray,
) -> np.ndarray:
    """Concatenate BoW and lexicon features."""
    return np.hstack([bow_matrix, lexicon_matrix])


def extract_features(
    train_texts: list[str],
    test_texts: list[str],
    feature_set: str,
    binary_bow: bool = False,
    max_features: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X_train, X_test, feature_names) for the chosen feature_set."""
    if feature_set == "bow":
        X_tr, X_te, vec = build_bow_features(train_texts, test_texts, binary=binary_bow, max_features=max_features)
        return X_tr, X_te, vec.get_feature_names_out().tolist()

    if feature_set == "lexicon":
        X_tr = build_lexicon_features(train_texts)
        X_te = build_lexicon_features(test_texts)
        return X_tr, X_te, list(LEXICON_FEATURE_NAMES)

    if feature_set == "bow_lexicon":
        X_tr_bow, X_te_bow, vec = build_bow_features(train_texts, test_texts, binary=binary_bow, max_features=max_features)
        X_tr_lex = build_lexicon_features(train_texts)
        X_te_lex = build_lexicon_features(test_texts)
        X_tr = build_combined_features(X_tr_bow, X_tr_lex)
        X_te = build_combined_features(X_te_bow, X_te_lex)
        names = vec.get_feature_names_out().tolist() + list(LEXICON_FEATURE_NAMES)
        return X_tr, X_te, names

    raise ValueError(f"Unknown feature_set: {feature_set}")


# Training classifiers

def train_classifier(
    classifier_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    max_iter: int = 1000,
    C: float = 1.0,
) -> Any:
    """Train and return a fitted classifier."""
    if classifier_name == "naive_bayes":
        model = MultinomialNB(alpha=1.0)
        # MultinomialNB needs non-negative features
        X_train_safe = np.clip(X_train, 0, None)
        model.fit(X_train_safe, y_train)
        return model

    if classifier_name == "binary_naive_bayes":
        model = MultinomialNB(alpha=1.0)
        X_train_bin = (X_train > 0).astype(np.float64)
        model.fit(X_train_bin, y_train)
        return model

    if classifier_name == "logistic_regression":
        model = LogisticRegression(
            max_iter=max_iter,
            random_state=seed,
            C=C,
            solver="lbfgs",
        )
        model.fit(X_train, y_train)
        return model

    raise ValueError(f"Unknown classifier: {classifier_name}")


def predict(
    classifier_name: str,
    model: Any,
    X_test: np.ndarray,
) -> np.ndarray:
    """Predict labels."""
    if classifier_name == "binary_naive_bayes":
        X_test_bin = (X_test > 0).astype(np.float64)
        return model.predict(X_test_bin)
    if classifier_name == "naive_bayes":
        X_test_safe = np.clip(X_test, 0, None)
        return model.predict(X_test_safe)
    return model.predict(X_test)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    """Compute accuracy, macro/weighted precision, recall, F1, confusion matrix."""
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": acc,
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm,
        "classification_report": report,
        "label_names": label_names,
    }


# Comparing classifiers with significance tests

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, Any]:
    """McNemar's test to compare two classifiers on same test set.

    Builds the 2×2 contingency table:
        - b: A correct, B wrong
        - c: A wrong, B correct
    Under H0 (no difference), b and c should be equal.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B right

    n = b + c
    if n == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "b": b,
            "c": c,
            "significant_0.05": False,
            "note": "Both classifiers agree on all samples.",
        }

    # Use continuity-corrected McNemar
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1.0 - scipy_stats.chi2.cdf(chi2, df=1))

    return {
        "statistic": float(chi2),
        "p_value": p_value,
        "b": b,
        "c": c,
        "significant_0.05": p_value < 0.05,
    }


def pairwise_significance_tests(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Run McNemar's test for every pair of classifiers."""
    names = sorted(predictions.keys())
    results: list[dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a, name_b = names[i], names[j]
            test_result = mcnemar_test(y_true, predictions[name_a], predictions[name_b])
            results.append({
                "classifier_a": name_a,
                "classifier_b": name_b,
                **test_result,
            })
    return results


# Running the full pipeline

def run_single_experiment(
    classifier_name: str,
    feature_set: str,
    train_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    seed: int = 42,
    max_features: int | None = None,
) -> dict[str, Any]:
    """Train one classifier with one feature set, return metrics + predictions."""
    binary_bow = (classifier_name == "binary_naive_bayes")

    X_train, X_test, feat_names = extract_features(
        train_texts, test_texts, feature_set, binary_bow=binary_bow, max_features=max_features,
    )

    model = train_classifier(classifier_name, X_train, y_train, seed=seed)
    y_pred = predict(classifier_name, model, X_test)
    metrics = evaluate_predictions(y_test, y_pred, label_names)

    return {
        "classifier": classifier_name,
        "feature_set": feature_set,
        "n_features": int(X_train.shape[1]),
        "metrics": metrics,
        "y_pred": y_pred,
        "model": model,
    }


def build_task3_artifacts(
    input_path: str,
    text_col: str = "modern_text",
    label_col: str = "author",
    test_size: float = 0.2,
    seed: int = 42,
    max_features: int | None = None,
) -> dict[str, Any]:
    """Full Task 3 pipeline: train all classifiers × feature sets, test, significance."""
    parquet_path = Path(input_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    validate_columns(df, [text_col, label_col])

    clean_df, clean_stats = clean_dataframe(df, text_col, label_col)
    if clean_df.empty:
        raise ValueError("No usable rows after cleaning.")

    train_df, test_df = stratified_split(clean_df, label_col, test_size, seed)

    le = LabelEncoder()
    le.fit(clean_df[label_col].values)
    label_names = le.classes_.tolist()

    y_train = le.transform(train_df[label_col].values)
    y_test = le.transform(test_df[label_col].values)
    train_texts = train_df["clean_text"].tolist()
    test_texts = test_df["clean_text"].tolist()

    # Run all experiments
    experiment_results: list[dict[str, Any]] = []
    all_predictions: dict[str, np.ndarray] = {}

    for clf_name in CLASSIFIERS:
        for feat_set in FEATURE_SETS:
            result = run_single_experiment(
                classifier_name=clf_name,
                feature_set=feat_set,
                train_texts=train_texts,
                test_texts=test_texts,
                y_train=y_train,
                y_test=y_test,
                label_names=label_names,
                seed=seed,
                max_features=max_features,
            )
            experiment_results.append(result)
            key = f"{clf_name}__{feat_set}"
            all_predictions[key] = result["y_pred"]

    # Statistical significance (McNemar pairwise)
    significance_results = pairwise_significance_tests(y_test, all_predictions)

    # Build summary rows
    summary_rows: list[dict[str, Any]] = []
    for exp in experiment_results:
        m = exp["metrics"]
        summary_rows.append({
            "classifier": exp["classifier"],
            "feature_set": exp["feature_set"],
            "n_features": exp["n_features"],
            "accuracy": m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
        })

    # Rank by macro_f1
    ranked = sorted(summary_rows, key=lambda r: r["macro_f1"], reverse=True)
    for rank_idx, row in enumerate(ranked, start=1):
        row["rank"] = rank_idx

    best = ranked[0]

    split_info = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "n_labels": int(len(label_names)),
        "label_names": label_names,
        "train_label_counts": {
            str(k): int(v) for k, v in train_df[label_col].value_counts().sort_index().items()
        },
        "test_label_counts": {
            str(k): int(v) for k, v in test_df[label_col].value_counts().sort_index().items()
        },
    }

    return {
        "config": {
            "input": str(parquet_path),
            "text_col": text_col,
            "label_col": label_col,
            "test_size": test_size,
            "seed": seed,
            "max_features": max_features,
        },
        "clean_stats": clean_stats,
        "split": split_info,
        "summary_rows": ranked,
        "experiment_results": experiment_results,
        "significance_results": significance_results,
        "best": {
            "classifier": best["classifier"],
            "feature_set": best["feature_set"],
            "macro_f1": best["macro_f1"],
            "accuracy": best["accuracy"],
        },
        # Pass-through for Streamlit
        "train_texts": train_texts,
        "test_texts": test_texts,
        "y_train": y_train,
        "y_test": y_test,
        "label_names": label_names,
    }


# Saving results to disk

def _safe_json(obj: Any) -> Any:
    """Make numpy types JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_task3_outputs(artifacts: dict[str, Any], out_dir: str) -> dict[str, str]:
    """Write JSON + CSV results, return paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Metrics JSON
    metrics = {
        "config": artifacts["config"],
        "clean_stats": artifacts["clean_stats"],
        "split": artifacts["split"],
        "best": artifacts["best"],
        "experiments": [],
    }
    for exp in artifacts["experiment_results"]:
        entry = {
            "classifier": exp["classifier"],
            "feature_set": exp["feature_set"],
            "n_features": exp["n_features"],
            "accuracy": exp["metrics"]["accuracy"],
            "macro_precision": exp["metrics"]["macro_precision"],
            "macro_recall": exp["metrics"]["macro_recall"],
            "macro_f1": exp["metrics"]["macro_f1"],
            "weighted_f1": exp["metrics"]["weighted_f1"],
            "confusion_matrix": exp["metrics"]["confusion_matrix"],
        }
        metrics["experiments"].append(entry)

    metrics_path = out / "task3_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=_safe_json),
        encoding="utf-8",
    )

    # 2. Summary CSV
    summary_path = out / "task3_summary.csv"
    fieldnames = [
        "rank", "classifier", "feature_set", "n_features",
        "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in artifacts["summary_rows"]:
            writer.writerow({k: row[k] for k in fieldnames})

    # 3. Significance JSON
    sig_path = out / "task3_significance.json"
    sig_path.write_text(
        json.dumps(artifacts["significance_results"], ensure_ascii=False, indent=2, default=_safe_json),
        encoding="utf-8",
    )

    return {
        "metrics": str(metrics_path),
        "summary": str(summary_path),
        "significance": str(sig_path),
    }
