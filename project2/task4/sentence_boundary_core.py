from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

# Constants and abbreviations

PENALTIES: tuple[str, ...] = ("l1", "l2")
C_GRID: list[float] = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

# Known Azerbaijani abbreviations (lowercase, without trailing dot)
_KNOWN_ABBREVIATIONS: set[str] = {
    "mr", "mrs", "dr", "prof", "vs", "etc", "inc", "jr", "sr",
    "b", "e", "h", "m",  # common single-letter abbreviations
    "məs", "və", "s",    # Azerbaijani: məsələn (for example), və sairə
}

# Non-leaky features (previous ones like next_char_is_newline directly
# mirrored the labeling heuristic, so we removed them)
FEATURE_NAMES: list[str] = [
    "word_len",               # length of the token containing the dot
    "left_word_len",          # length of the word immediately before the dot-token
    "right_word_len",         # length of the word immediately after the dot-token
    "next_char_is_space",     # 1 if character after the dot is whitespace (space/tab)
    "prev_char_is_alpha",     # 1 if char before dot is alphabetic
    "prev_char_is_digit",     # 1 if char before dot is a digit
    "is_ellipsis",            # 1 if the dot is part of a multi-dot sequence (.., ...)
    "is_known_abbreviation",  # 1 if text before dot is a known abbreviation
    "dot_position_ratio",     # position of the dot / total text length
    "words_since_last_dot",   # token count since the previous dot (or start)
    "left_word_is_upper",     # 1 if the word before the dot starts with uppercase
    "right_word_is_upper",    # 1 if the word after the dot starts with uppercase
    "prev_is_comma_or_semi",  # 1 if there is a comma/semicolon shortly before the dot
    "next_is_quote",          # 1 if char after dot (skip space) is a quote mark
    "left_word_freq_rank",    # frequency-rank bucket of left word (0-4, from corpus)
    "right_word_freq_rank",   # frequency-rank bucket of right word (0-4, from corpus)
    "chars_since_last_dot",   # character distance since the previous dot
    "left_has_vowel",         # 1 if left word contains a vowel (real word heuristic)
]


# Extracting and labeling dot positions

_AZ_VOWELS = set("aeıioöuüəAEIİOÖUÜƏ")


def _has_vowel(word: str) -> bool:
    return any(ch in _AZ_VOWELS for ch in word)


def _find_dot_positions(text: str) -> list[int]:
    """Return character indices of all '.' in *text*."""
    return [i for i, ch in enumerate(text) if ch == "."]


def _word_around(text: str, pos: int, direction: int, max_len: int = 30) -> str:
    """Extract the word immediately before (direction=-1) or after (direction=+1) *pos*."""
    chars: list[str] = []
    step = direction
    i = pos + step
    while 0 <= i < len(text) and not text[i].isalnum():
        i += step
    while 0 <= i < len(text) and (text[i].isalnum() or text[i] == "'"):
        chars.append(text[i])
        i += step
        if len(chars) >= max_len:
            break
    if direction == -1:
        chars.reverse()
    return "".join(chars)


def _is_sentence_boundary(text: str, dot_pos: int) -> bool:
    """Heuristic ground truth: a dot is a sentence boundary if it is followed by
    a newline, end-of-text, or a capitalized word after whitespace, **and** is
    NOT part of an ellipsis or a decimal number like ``3.14``.
    """
    if dot_pos + 1 < len(text) and text[dot_pos + 1] == ".":
        return False
    if dot_pos > 0 and text[dot_pos - 1] == ".":
        return False
    if dot_pos > 0 and dot_pos + 1 < len(text):
        if text[dot_pos - 1].isdigit() and text[dot_pos + 1].isdigit():
            return False

    after = text[dot_pos + 1:] if dot_pos + 1 < len(text) else ""
    stripped = after.lstrip()

    if not stripped:
        return True
    if after and after[0] == "\n":
        return True
    if stripped and stripped[0].isupper():
        return True
    return False


def extract_dot_samples(
    text: str,
    text_id: int = 0,
) -> list[dict[str, Any]]:
    """For every '.' in *text*, extract features and a binary label."""
    dot_positions = _find_dot_positions(text)
    if not dot_positions:
        return []

    samples: list[dict[str, Any]] = []
    prev_dot_idx = -1

    for pos in dot_positions:
        label = int(_is_sentence_boundary(text, pos))

        # token containing the dot
        start = pos
        while start > 0 and text[start - 1] not in (" ", "\n", "\t"):
            start -= 1
        end = pos + 1
        while end < len(text) and text[end] not in (" ", "\n", "\t"):
            end += 1
        dot_token = text[start:end]

        left_word = _word_around(text, pos, direction=-1)
        right_word = _word_around(text, pos, direction=+1)

        prev_ch = text[pos - 1] if pos > 0 else ""
        next_ch = text[pos + 1] if pos + 1 < len(text) else ""

        between = text[prev_dot_idx + 1: pos] if prev_dot_idx >= 0 else text[:pos]
        words_since = len(re.findall(r"\b\w+\b", between))
        chars_since = pos - prev_dot_idx - 1 if prev_dot_idx >= 0 else pos

        window_before = text[max(0, pos - 30): pos]
        prev_comma_semi = int(bool(re.search(r"[,;]", window_before)))

        after_stripped = text[pos + 1:].lstrip() if pos + 1 < len(text) else ""
        next_is_quote = int(
            bool(after_stripped)
            and after_stripped[0] in "\"'«»\u201c\u201d\u2018\u2019\u201e"
        )

        features = {
            "word_len": len(dot_token),
            "left_word_len": len(left_word),
            "right_word_len": len(right_word),
            "next_char_is_space": int(next_ch in (" ", "\t")),
            "prev_char_is_alpha": int(prev_ch.isalpha()),
            "prev_char_is_digit": int(prev_ch.isdigit()),
            "is_ellipsis": int(
                (pos > 0 and text[pos - 1] == ".")
                or (pos + 1 < len(text) and text[pos + 1] == ".")
            ),
            "is_known_abbreviation": int(left_word.lower() in _KNOWN_ABBREVIATIONS),
            "dot_position_ratio": round(pos / max(len(text), 1), 6),
            "words_since_last_dot": words_since,
            "left_word_is_upper": int(bool(left_word) and left_word[0].isupper()),
            "right_word_is_upper": int(bool(right_word) and right_word[0].isupper()),
            "prev_is_comma_or_semi": prev_comma_semi,
            "next_is_quote": next_is_quote,
            # placeholders — filled after split via enrich_with_corpus_features
            "left_word_freq_rank": 0,
            "right_word_freq_rank": 0,
            "chars_since_last_dot": chars_since,
            "left_has_vowel": int(_has_vowel(left_word)),
        }

        samples.append({
            "text_id": text_id,
            "dot_pos": pos,
            "label": label,
            "left_word": left_word.lower(),
            "right_word": right_word.lower(),
            **features,
        })
        prev_dot_idx = pos

    return samples


# Enriching features with corpus statistics

def _word_freq_rank(word: str, freq_map: dict[str, int], total: int) -> int:
    f = freq_map.get(word, 0)
    ratio = f / max(total, 1)
    if ratio >= 0.01:
        return 4
    if ratio >= 0.005:
        return 3
    if ratio >= 0.001:
        return 2
    if f > 0:
        return 1
    return 0


def enrich_with_corpus_features(
    dot_df: pd.DataFrame,
    word_freq: dict[str, int],
    total_words: int,
) -> None:
    """Fill in left/right word frequency rank features in-place."""
    dot_df["left_word_freq_rank"] = dot_df["left_word"].map(
        lambda w: _word_freq_rank(w, word_freq, total_words)
    )
    dot_df["right_word_freq_rank"] = dot_df["right_word"].map(
        lambda w: _word_freq_rank(w, word_freq, total_words)
    )


def build_word_freq_from_texts(texts: Sequence[str]) -> tuple[dict[str, int], int]:
    from collections import Counter
    freq: Counter[str] = Counter()
    for t in texts:
        freq.update(re.findall(r"\b\w+\b", t.lower()))
    return dict(freq), sum(freq.values())


# Preparing the data

def validate_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_dot_dataset(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    all_samples: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        if not text.strip():
            continue
        samples = extract_dot_samples(text, text_id=int(idx))
        all_samples.extend(samples)
    if not all_samples:
        raise ValueError("No dot candidates found in the dataset.")
    return pd.DataFrame(all_samples)


def _split_ids_stratified(
    id_label_df: pd.DataFrame,
    split_frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split text IDs into (main, split_off), stratified by label."""
    main_ids: list[int] = []
    split_ids: list[int] = []
    for _, group in id_label_df.groupby("label", sort=False):
        shuffled = group.sample(frac=1.0, random_state=seed)
        n_split = max(1, int(round(len(shuffled) * split_frac)))
        n_split = min(n_split, len(shuffled) - 1)
        split_ids.extend(shuffled["text_id"].iloc[:n_split].tolist())
        main_ids.extend(shuffled["text_id"].iloc[n_split:].tolist())
    return main_ids, split_ids


def stratified_three_way_split(
    dot_df: pd.DataFrame,
    source_df: pd.DataFrame,
    label_col: str,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dot samples into train / val / test by poem (text_id),
    stratified by the poem's author. All dots from one poem stay together."""
    text_ids = dot_df["text_id"].unique()
    id_label = source_df.loc[text_ids, label_col].reset_index()
    id_label.columns = ["text_id", "label"]

    # First split: separate test
    remaining_ids, test_ids = _split_ids_stratified(id_label, test_size, seed)

    # Second split: from remaining, separate val
    remaining_df = id_label[id_label["text_id"].isin(set(remaining_ids))]
    adjusted_val = val_size / (1.0 - test_size)
    train_ids, val_ids = _split_ids_stratified(remaining_df, adjusted_val, seed + 1)

    train_df = dot_df[dot_df["text_id"].isin(set(train_ids))].reset_index(drop=True)
    val_df = dot_df[dot_df["text_id"].isin(set(val_ids))].reset_index(drop=True)
    test_df = dot_df[dot_df["text_id"].isin(set(test_ids))].reset_index(drop=True)

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if split.empty:
            raise ValueError(f"{name} split is empty — check dataset size or split fractions.")

    return train_df, val_df, test_df


# Training and evaluating models

def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    penalty: str = "l2",
    C: float = 1.0,
    seed: int = 42,
    max_iter: int = 5000,
) -> LogisticRegression:
    l1_ratio = 1.0 if penalty == "l1" else 0.0
    model = LogisticRegression(
        l1_ratio=l1_ratio,
        C=C,
        solver="saga",
        max_iter=max_iter,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    y_pred = model.predict(X)
    labels = ["not_boundary", "boundary"]
    report = classification_report(
        y, y_pred, target_names=labels, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y, y_pred).tolist()
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
    }


def tune_C(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    penalty: str,
    c_grid: list[float],
    seed: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Grid-search over C values on the validation set. Returns best C and rows."""
    best_c = c_grid[0]
    best_f1 = -1.0
    rows: list[dict[str, Any]] = []

    for c_val in c_grid:
        model = train_logistic(X_train, y_train, penalty=penalty, C=c_val, seed=seed)
        train_ev = evaluate_model(model, X_train, y_train)
        val_ev = evaluate_model(model, X_val, y_val)
        row = {
            "penalty": penalty,
            "C": c_val,
            "train_accuracy": train_ev["accuracy"],
            "train_f1": train_ev["f1"],
            "val_accuracy": val_ev["accuracy"],
            "val_f1": val_ev["f1"],
        }
        rows.append(row)
        if val_ev["f1"] > best_f1:
            best_f1 = val_ev["f1"]
            best_c = c_val

    return best_c, rows


def feature_importance_table(
    model: LogisticRegression,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    coefs = model.coef_[0]
    rows = []
    for name, coef in zip(feature_names, coefs):
        rows.append({
            "feature": name,
            "coefficient": float(coef),
            "abs_coefficient": float(abs(coef)),
        })
    rows.sort(key=lambda r: r["abs_coefficient"], reverse=True)
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
    return rows


# Reconstructing sentences from dot predictions

def reconstruct_sentences(
    text: str,
    dot_positions: list[int],
    predictions: list[int],
) -> list[str]:
    boundary_positions = [p for p, pred in zip(dot_positions, predictions) if pred == 1]
    sentences: list[str] = []
    start = 0
    for bp in boundary_positions:
        sent = text[start: bp + 1].strip()
        if sent:
            sentences.append(sent)
        start = bp + 1
    remaining = text[start:].strip()
    if remaining:
        sentences.append(remaining)
    return sentences


# McNemar's statistical test

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, Any]:
    from scipy import stats as scipy_stats

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    n = b + c
    if n == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "b": b,
            "c": c,
            "significant_0.05": False,
            "note": "Both models agree on all samples.",
        }

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1.0 - scipy_stats.chi2.cdf(chi2, df=1))

    return {
        "statistic": float(chi2),
        "p_value": p_value,
        "b": b,
        "c": c,
        "significant_0.05": p_value < 0.05,
    }


# Full pipeline execution

def build_task4_artifacts(
    input_path: str,
    text_col: str = "modern_text",
    label_col: str = "author",
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
    c_grid: list[float] | None = None,
) -> dict[str, Any]:
    """Run the complete Task 4 pipeline with train/val/test split and C tuning."""
    if c_grid is None:
        c_grid = list(C_GRID)

    parquet_path = Path(input_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    validate_columns(df, [text_col])

    clean_mask = df[text_col].fillna("").astype(str).str.strip().ne("")
    clean_df = df.loc[clean_mask].copy()
    if clean_df.empty:
        raise ValueError("No non-empty texts found.")

    # Build dot-level dataset
    dot_df = prepare_dot_dataset(clean_df, text_col)

    # Three-way split (by poem, stratified by author)
    train_dot_df, val_dot_df, test_dot_df = stratified_three_way_split(
        dot_df, clean_df, label_col, val_size, test_size, seed,
    )

    # Build word freq from training poems only (no data leakage)
    train_text_ids = set(train_dot_df["text_id"].unique())
    train_texts = [
        str(clean_df.loc[tid, text_col])
        for tid in train_text_ids
        if tid in clean_df.index
    ]
    word_freq, total_words = build_word_freq_from_texts(train_texts)

    # Enrich all splits with corpus features (using train-only freqs)
    for split_df in (train_dot_df, val_dot_df, test_dot_df):
        enrich_with_corpus_features(split_df, word_freq, total_words)

    # Prepare matrices
    X_train = train_dot_df[FEATURE_NAMES].values.astype(np.float64)
    y_train = train_dot_df["label"].values.astype(np.int64)
    X_val = val_dot_df[FEATURE_NAMES].values.astype(np.float64)
    y_val = val_dot_df["label"].values.astype(np.int64)
    X_test = test_dot_df[FEATURE_NAMES].values.astype(np.float64)
    y_test = test_dot_df["label"].values.astype(np.int64)

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Tune C on validation, then retrain on train+val, evaluate on test
    results: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    all_tuning_rows: list[dict[str, Any]] = []

    for penalty in PENALTIES:
        best_c, tuning_rows = tune_C(
            X_train, y_train, X_val, y_val, penalty, c_grid, seed,
        )
        all_tuning_rows.extend(tuning_rows)

        # Retrain final model on train+val with best C
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])
        final_model = train_logistic(
            X_train_val, y_train_val, penalty=penalty, C=best_c, seed=seed,
        )

        train_eval = evaluate_model(final_model, X_train, y_train)
        val_eval = evaluate_model(final_model, X_val, y_val)
        test_eval = evaluate_model(final_model, X_test, y_test)
        feat_imp = feature_importance_table(final_model, FEATURE_NAMES)

        results[penalty] = {
            "penalty": penalty,
            "best_C": best_c,
            "train_metrics": {k: v for k, v in train_eval.items() if k != "y_pred"},
            "val_metrics": {k: v for k, v in val_eval.items() if k != "y_pred"},
            "test_metrics": {k: v for k, v in test_eval.items() if k != "y_pred"},
            "feature_importance": feat_imp,
            "n_nonzero_coefs": int(np.sum(np.abs(final_model.coef_[0]) > 1e-10)),
            "model": final_model,
        }
        predictions[penalty] = test_eval["y_pred"]

    # McNemar L1 vs L2 on test set
    sig_test = mcnemar_test(y_test, predictions["l1"], predictions["l2"])

    l1_f1 = results["l1"]["test_metrics"]["f1"]
    l2_f1 = results["l2"]["test_metrics"]["f1"]
    best_penalty = "l1" if l1_f1 > l2_f1 else "l2"
    if abs(l1_f1 - l2_f1) < 1e-8:
        best_penalty = "l2"

    dataset_stats = {
        "total_texts": int(len(clean_df)),
        "total_dots": int(len(dot_df)),
        "positive_dots": int(dot_df["label"].sum()),
        "negative_dots": int((dot_df["label"] == 0).sum()),
        "positive_rate": float(dot_df["label"].mean()),
        "train_dots": int(len(train_dot_df)),
        "val_dots": int(len(val_dot_df)),
        "test_dots": int(len(test_dot_df)),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "test_positive_rate": float(y_test.mean()),
    }

    return {
        "config": {
            "input": str(parquet_path),
            "text_col": text_col,
            "label_col": label_col,
            "val_size": val_size,
            "test_size": test_size,
            "seed": seed,
            "c_grid": c_grid,
            "n_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
        },
        "dataset_stats": dataset_stats,
        "results": {
            p: {k: v for k, v in r.items() if k != "model"}
            for p, r in results.items()
        },
        "models": {p: r["model"] for p, r in results.items()},
        "tuning_rows": all_tuning_rows,
        "significance": sig_test,
        "best_penalty": best_penalty,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_dot_df": train_dot_df,
        "val_dot_df": val_dot_df,
        "test_dot_df": test_dot_df,
        "source_df": clean_df,
        "scaler": scaler,
    }


# Writing results to files

def _safe_json(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_task4_outputs(artifacts: dict[str, Any], out_dir: str) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Metrics JSON
    metrics = {
        "config": artifacts["config"],
        "dataset_stats": artifacts["dataset_stats"],
        "results": artifacts["results"],
        "significance_l1_vs_l2": artifacts["significance"],
        "best_penalty": artifacts["best_penalty"],
    }
    metrics_path = out / "task4_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=_safe_json),
        encoding="utf-8",
    )

    # 2. Feature importance CSV
    fi_path = out / "task4_feature_importance.csv"
    with fi_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["penalty", "rank", "feature", "coefficient", "abs_coefficient"],
        )
        writer.writeheader()
        for penalty in PENALTIES:
            for row in artifacts["results"][penalty]["feature_importance"]:
                writer.writerow({
                    "penalty": penalty,
                    "rank": row["rank"],
                    "feature": row["feature"],
                    "coefficient": round(row["coefficient"], 6),
                    "abs_coefficient": round(row["abs_coefficient"], 6),
                })

    # 3. Comparison summary CSV
    comp_path = out / "task4_comparison.csv"
    with comp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "penalty", "best_C", "train_f1", "val_f1", "test_f1",
            "test_accuracy", "test_precision", "test_recall", "n_nonzero_coefs",
        ])
        writer.writeheader()
        for penalty in PENALTIES:
            r = artifacts["results"][penalty]
            writer.writerow({
                "penalty": penalty,
                "best_C": r["best_C"],
                "train_f1": round(r["train_metrics"]["f1"], 6),
                "val_f1": round(r["val_metrics"]["f1"], 6),
                "test_f1": round(r["test_metrics"]["f1"], 6),
                "test_accuracy": round(r["test_metrics"]["accuracy"], 6),
                "test_precision": round(r["test_metrics"]["precision"], 6),
                "test_recall": round(r["test_metrics"]["recall"], 6),
                "n_nonzero_coefs": r["n_nonzero_coefs"],
            })

    # 4. Tuning grid results CSV
    tuning_path = out / "task4_tuning.csv"
    with tuning_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "penalty", "C", "train_accuracy", "train_f1", "val_accuracy", "val_f1",
        ])
        writer.writeheader()
        for row in artifacts["tuning_rows"]:
            writer.writerow({
                "penalty": row["penalty"],
                "C": row["C"],
                "train_accuracy": round(row["train_accuracy"], 6),
                "train_f1": round(row["train_f1"], 6),
                "val_accuracy": round(row["val_accuracy"], 6),
                "val_f1": round(row["val_f1"], 6),
            })

    return {
        "metrics": str(metrics_path),
        "feature_importance": str(fi_path),
        "comparison": str(comp_path),
        "tuning": str(tuning_path),
    }
