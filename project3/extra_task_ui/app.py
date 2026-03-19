from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from scipy.spatial.distance import cosine

ROOT = Path(__file__).resolve().parents[1]
TASK1_RESULTS = ROOT / "task1" / "results"
TASK2_RESULTS = ROOT / "task2" / "results"
TASK3_RESULTS = ROOT / "task3" / "results"
TASK4_RESULTS = ROOT / "task4" / "results"
TASK5_RESULTS = ROOT / "task5" / "results"

app = FastAPI(title="NLP Project Dashboard", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


def _safe_read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default or {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default or {}


def _safe_read_csv(path: Path, default_columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=default_columns or [])
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=default_columns or [])


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False


# Load embeddings and vocabularies for analogy lab
_EMBEDDINGS_CACHE: dict[str, dict[str, Any]] = {}


def _load_embeddings() -> dict[str, dict[str, Any]]:
    """Load all embeddings and vocabularies on demand."""
    if _EMBEDDINGS_CACHE:
        return _EMBEDDINGS_CACHE

    # Load Task 2 embeddings (Skip-gram and CBOW)
    for model_name, emb_file in [
        ("skipgram", TASK2_RESULTS / "task2_skipgram_embeddings.npy"),
        ("cbow", TASK2_RESULTS / "task2_cbow_embeddings.npy"),
    ]:
        vocab_df = _safe_read_csv(
            TASK2_RESULTS / "task2_vocab.csv",
            default_columns=["token_id", "token", "corpus_frequency"],
        )
        if not vocab_df.empty and emb_file.exists():
            try:
                emb = np.load(str(emb_file))
                vocab_dict = {str(row["token"]).lower(): idx for idx, row in vocab_df.iterrows()}
                _EMBEDDINGS_CACHE[f"word2vec_{model_name}"] = {
                    "embeddings": emb,
                    "vocab": vocab_dict,
                    "reverse_vocab": {idx: str(row["token"]) for idx, row in vocab_df.iterrows()},
                }
            except Exception:
                pass

    # Load Task 3 embeddings (GloVe)
    vocab_df = _safe_read_csv(
        TASK3_RESULTS / "task3_vocab.csv",
        default_columns=["token_id", "token", "corpus_frequency"],
    )
    glove_file = TASK3_RESULTS / "task3_glove_embeddings.npy"
    if not vocab_df.empty and glove_file.exists():
        try:
            emb = np.load(str(glove_file))
            vocab_dict = {str(row["token"]).lower(): idx for idx, row in vocab_df.iterrows()}
            _EMBEDDINGS_CACHE["glove"] = {
                "embeddings": emb,
                "vocab": vocab_dict,
                "reverse_vocab": {idx: str(row["token"]) for idx, row in vocab_df.iterrows()},
            }
        except Exception:
            pass

    return _EMBEDDINGS_CACHE


def _compute_analogy(model: str, word_a: str, word_b: str, word_c: str, top_k: int = 5) -> dict[str, Any]:
    """
    Solve analogy: a:b::c:?
    Returns top_k candidates with cosine similarities.
    """
    embeddings = _load_embeddings()
    if model not in embeddings:
        return {"error": f"Model '{model}' not found"}

    model_data = embeddings[model]
    emb_matrix = model_data["embeddings"]
    vocab = model_data["vocab"]
    reverse_vocab = model_data["reverse_vocab"]

    # Normalize word inputs
    word_a = word_a.lower().strip()
    word_b = word_b.lower().strip()
    word_c = word_c.lower().strip()

    # Check all words exist in vocab
    if word_a not in vocab:
        return {"error": f"Word '{word_a}' not in vocabulary"}
    if word_b not in vocab:
        return {"error": f"Word '{word_b}' not in vocabulary"}
    if word_c not in vocab:
        return {"error": f"Word '{word_c}' not in vocabulary"}

    try:
        # Get word vectors
        vec_a = emb_matrix[vocab[word_a]]
        vec_b = emb_matrix[vocab[word_b]]
        vec_c = emb_matrix[vocab[word_c]]

        # Compute target vector: c + (b - a)
        target_vec = vec_c + (vec_b - vec_a)
        # Normalize target vector
        target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-10)

        # Compute cosine similarity with all words
        similarities = []
        exclude_words = {word_a, word_b, word_c}

        for idx in range(len(emb_matrix)):
            word = reverse_vocab.get(idx, "")
            if word.lower() in exclude_words:
                continue
            word_vec = emb_matrix[idx]
            word_vec = word_vec / (np.linalg.norm(word_vec) + 1e-10)
            sim = float(1 - cosine(target_vec, word_vec))
            similarities.append((word, sim))

        # Sort by similarity and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = [
            {"word": word, "similarity": round(sim, 4)} for word, sim in similarities[:top_k]
        ]

        return {
            "analogy": f"{word_a}:{word_b}::{word_c}:?",
            "results": top_results,
        }

    except Exception as e:
        return {"error": str(e)}


def _build_task_cards() -> list[dict[str, Any]]:
    task1_summary = _safe_read_json(TASK1_RESULTS / "task1_summary.json")
    task2_config = _safe_read_json(TASK2_RESULTS / "task2_config.json")
    task3_config = _safe_read_json(TASK3_RESULTS / "task3_config.json")
    task4_config = _safe_read_json(TASK4_RESULTS / "task4_config.json")
    task5_config = _safe_read_json(TASK5_RESULTS / "task5_config.json")

    cards = [
        {
            "task": "Task 1",
            "title": "Corpus Matrices",
            "metric": _to_int(task1_summary.get("unique_tokens", 0)),
            "metric_label": "Unique Tokens",
            "sub": f"Docs: {_to_int(task1_summary.get('documents', 0))} · Authors: {_to_int(task1_summary.get('authors', 0))}",
            "gradient": "linear-gradient(135deg, #4f46e5, #06b6d4)",
        },
        {
            "task": "Task 2",
            "title": "Word2Vec",
            "metric": _to_int(task2_config.get("vocab_size", 0)),
            "metric_label": "Vocab Size",
            "sub": f"Epochs: {_to_int(task2_config.get('epochs', 0))}",
            "gradient": "linear-gradient(135deg, #ec4899, #8b5cf6)",
        },
        {
            "task": "Task 3",
            "title": "GloVe",
            "metric": _to_int(task3_config.get("directed_nonzero_pairs", 0)),
            "metric_label": "Directed Pairs",
            "sub": f"Epochs: {_to_int(task3_config.get('epochs', 0))}",
            "gradient": "linear-gradient(135deg, #f97316, #eab308)",
        },
        {
            "task": "Task 4",
            "title": "Embedding Comparison",
            "metric": _to_int(task4_config.get("query_word_count", 0)),
            "metric_label": "Query Words",
            "sub": (
                f"kNN k={_to_int(task4_config.get('neighbor_top_k', 0))} · "
                f"Eq k={_to_int(task4_config.get('equation_top_k', 0))}"
            ),
            "gradient": "linear-gradient(135deg, #22c55e, #14b8a6)",
        },
        {
            "task": "Task 5",
            "title": "Sequence Classification",
            "metric": _to_int(task5_config.get("num_classes", 0)),
            "metric_label": "Classes",
            "sub": (
                f"Train: {_to_int(task5_config.get('train_docs', 0))} · "
                f"Test: {_to_int(task5_config.get('test_docs', 0))}"
            ),
            "gradient": "linear-gradient(135deg, #0ea5e9, #a855f7)",
        },
    ]
    return cards


def _build_task5_payload() -> dict[str, Any]:
    results_df = _safe_read_csv(
        TASK5_RESULTS / "task5_results.csv",
        default_columns=[
            "feature",
            "model",
            "test_accuracy",
            "test_macro_f1",
            "train_accuracy",
            "train_macro_f1",
        ],
    )

    if results_df.empty:
        return {
            "rows": [],
            "best": {},
            "by_feature": [],
        }

    results_df = results_df.copy()
    results_df["test_accuracy"] = results_df["test_accuracy"].map(_to_float)
    results_df["test_macro_f1"] = results_df["test_macro_f1"].map(_to_float)

    best_row = results_df.sort_values(["test_accuracy", "test_macro_f1"], ascending=False).iloc[0]

    by_feature_df = (
        results_df.sort_values(["feature", "test_accuracy", "test_macro_f1"], ascending=[True, False, False])
        .groupby("feature", as_index=False)
        .first()
    )

    rows = []
    for _, row in results_df.iterrows():
        rows.append(
            {
                "feature": str(row["feature"]),
                "model": str(row["model"]),
                "test_accuracy": round(_to_float(row["test_accuracy"]), 4),
                "test_macro_f1": round(_to_float(row["test_macro_f1"]), 4),
            }
        )

    by_feature = []
    for _, row in by_feature_df.iterrows():
        by_feature.append(
            {
                "feature": str(row["feature"]),
                "best_model": str(row["model"]),
                "accuracy": round(_to_float(row["test_accuracy"]), 4),
                "macro_f1": round(_to_float(row["test_macro_f1"]), 4),
            }
        )

    return {
        "rows": rows,
        "best": {
            "feature": str(best_row["feature"]),
            "model": str(best_row["model"]),
            "accuracy": round(_to_float(best_row["test_accuracy"]), 4),
            "macro_f1": round(_to_float(best_row["test_macro_f1"]), 4),
        },
        "by_feature": by_feature,
    }


def _build_task5_training_payload() -> dict[str, Any]:
    metrics_df = _safe_read_csv(
        TASK5_RESULTS / "task5_training_metrics.csv",
        default_columns=["feature", "model", "epoch", "avg_loss", "examples_seen"],
    )
    if metrics_df.empty:
        return {"rows": [], "max_epoch": 0}

    metrics_df = metrics_df.copy()
    metrics_df["epoch"] = metrics_df["epoch"].map(_to_int)
    metrics_df["avg_loss"] = metrics_df["avg_loss"].map(_to_float)
    metrics_df["examples_seen"] = metrics_df["examples_seen"].map(_to_int)

    rows = []
    for _, row in metrics_df.iterrows():
        rows.append(
            {
                "feature": str(row["feature"]),
                "model": str(row["model"]),
                "epoch": _to_int(row["epoch"]),
                "avg_loss": round(_to_float(row["avg_loss"]), 6),
                "examples_seen": _to_int(row["examples_seen"]),
            }
        )

    return {
        "rows": rows,
        "max_epoch": max((item["epoch"] for item in rows), default=0),
    }


def _build_task4_payload() -> dict[str, Any]:
    neighbors_df = _safe_read_csv(
        TASK4_RESULTS / "task4_neighbors_overlap.csv",
        default_columns=[
            "word2vec_model",
            "query_word",
            "jaccard_similarity",
            "overlap_count",
            "overlap_words",
        ],
    )
    equations_df = _safe_read_csv(
        TASK4_RESULTS / "task4_equations_overlap.csv",
        default_columns=[
            "word2vec_model",
            "query_word",
            "jaccard_similarity",
            "overlap_count",
            "overlap_words",
        ],
    )

    def frame_to_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        out: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            if _is_missing(row.get("word2vec_model")) or _is_missing(row.get("query_word")):
                continue
            out.append(
                {
                    "word2vec_model": str(row.get("word2vec_model", "")),
                    "query_word": str(row.get("query_word", "")),
                    "jaccard_similarity": round(_to_float(row.get("jaccard_similarity", 0.0)), 4),
                    "overlap_count": _to_int(row.get("overlap_count", 0)),
                    "overlap_words": "" if _is_missing(row.get("overlap_words")) else str(row.get("overlap_words", "")),
                }
            )
        return out

    return {
        "neighbors": frame_to_rows(neighbors_df),
        "equations": frame_to_rows(equations_df),
    }


def _build_synonyms_payload() -> dict[str, Any]:
    task2_neighbors = _safe_read_csv(
        TASK2_RESULTS / "task2_neighbors.csv",
        default_columns=["model", "query_word", "rank", "similar_word", "cosine_similarity"],
    )
    task3_neighbors = _safe_read_csv(
        TASK3_RESULTS / "task3_neighbors.csv",
        default_columns=["model", "query_word", "rank", "similar_word", "cosine_similarity"],
    )

    merged = pd.concat([task2_neighbors, task3_neighbors], ignore_index=True)
    if merged.empty:
        return {"rows": [], "models": [], "query_words": []}

    out_rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        if _is_missing(row.get("model")) or _is_missing(row.get("query_word")) or _is_missing(row.get("similar_word")):
            continue
        rank = _to_int(row.get("rank", 0))
        cosine = _to_float(row.get("cosine_similarity", 0.0))
        if rank <= 0:
            continue
        out_rows.append(
            {
                "model": str(row.get("model", "")),
                "query_word": str(row.get("query_word", "")),
                "rank": rank,
                "similar_word": str(row.get("similar_word", "")),
                "cosine_similarity": round(cosine, 4),
            }
        )

    models = sorted({item["model"] for item in out_rows if item["model"]})
    query_words = sorted({item["query_word"] for item in out_rows if item["query_word"]})

    return {
        "rows": out_rows,
        "models": models,
        "query_words": query_words,
    }


def load_dashboard_data() -> dict[str, Any]:
    task_cards = _build_task_cards()
    task5 = _build_task5_payload()
    task5_training = _build_task5_training_payload()
    task4_detail = _build_task4_payload()
    synonyms = _build_synonyms_payload()

    task4_neighbors = _safe_read_csv(
        TASK4_RESULTS / "task4_neighbors_overlap.csv",
        default_columns=["word2vec_model", "jaccard_similarity"],
    )
    task4_equations = _safe_read_csv(
        TASK4_RESULTS / "task4_equations_overlap.csv",
        default_columns=["word2vec_model", "jaccard_similarity"],
    )

    neighbor_avg = (
        float(task4_neighbors["jaccard_similarity"].mean())
        if not task4_neighbors.empty and "jaccard_similarity" in task4_neighbors.columns
        else 0.0
    )
    equation_avg = (
        float(task4_equations["jaccard_similarity"].mean())
        if not task4_equations.empty and "jaccard_similarity" in task4_equations.columns
        else 0.0
    )

    # Analogy lab available models
    analogy_models = list(_load_embeddings().keys())

    return {
        "task_cards": task_cards,
        "task5": task5,
        "task5_training": task5_training,
        "task4_detail": task4_detail,
        "synonyms": synonyms,
        "task4_overlap": {
            "neighbor_avg_jaccard": round(neighbor_avg, 4),
            "equation_avg_jaccard": round(equation_avg, 4),
        },
        "analogy_lab": {
            "models": analogy_models,
        },
    }


@app.get("/")
def index(request: Request):
    data = load_dashboard_data()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "data": data,
        },
    )


@app.get("/api/dashboard")
def dashboard_api() -> JSONResponse:
    return JSONResponse(load_dashboard_data())


@app.get("/api/analogy")
def analogy_api(model: str = "word2vec_skipgram", a: str = "", b: str = "", c: str = "", top_k: int = 5) -> JSONResponse:
    """
    API endpoint for word analogy computation.
    Query params: model, a, b, c, top_k
    Example: /api/analogy?model=word2vec_skipgram&a=man&b=woman&c=king&top_k=5
    """
    if not all([a.strip(), b.strip(), c.strip()]):
        return JSONResponse({"error": "All four words (a, b, c, model) must be provided"}, status_code=400)

    result = _compute_analogy(model, a, b, c, top_k)
    return JSONResponse(result)
