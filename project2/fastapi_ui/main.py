from __future__ import annotations

import math
import pickle
import sys
import time
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from task1.task1_ngram import (  # noqa: E402
    END_TOKEN,
    START_TOKEN,
    UNK_TOKEN,
    apply_unk_policy,
    build_ngram_model,
    clean_and_tokenize,
    perplexity_diagnostics,
    stratified_split,
    validate_columns as validate_task1_columns,
)
from task2.smoothing_core import (  # noqa: E402
    build_task2_artifacts,
    map_text_to_tokens,
    model_probability,
)
from task3.classification_core import (  # noqa: E402
    build_lexicon_features,
    build_task3_artifacts,
    predict as task3_predict,
    train_classifier,
)
from task4.sentence_boundary_core import (  # noqa: E402
    FEATURE_NAMES,
    build_task4_artifacts,
    build_word_freq_from_texts,
    enrich_with_corpus_features,
    extract_dot_samples,
    reconstruct_sentences,
)

app = FastAPI(title="Project2 FastAPI UI", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
CACHE_DIR = Path(__file__).parent / ".model_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class Task1Request(BaseModel):
    input_path: str = "poems_translated.parquet"
    text_col: str = "modern_text"
    author_col: str = "author"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42
    min_freq: int = Field(default=2, ge=1)


class Task2Request(BaseModel):
    input_path: str = "poems_translated.parquet"
    text_col: str = "modern_text"
    author_col: str = "author"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    val_size: float = Field(default=0.1, gt=0.0, lt=1.0)
    seed: int = 42
    min_freq: int = Field(default=2, ge=1)
    interp_bigram_grid: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9]
    interp_trigram_step: float = Field(default=0.1, gt=0.0, le=1.0)
    discount_grid: list[float] = [0.5, 0.75, 1.0]


class Task3Request(BaseModel):
    dataset: str = "twitter_samples"
    input_path: str | None = "poems_translated.parquet"
    text_col: str = "text"
    label_col: str = "label"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42
    max_features: int | None = None


class Task4Request(BaseModel):
    input_path: str = "poems_translated.parquet"
    text_col: str = "modern_text"
    label_col: str = "author"
    val_size: float = Field(default=0.1, gt=0.0, lt=1.0)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42
    c_grid: list[float] = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]


class Task1PredictRequest(Task1Request):
    text: str
    model: str = "trigram"
    top_k: int = Field(default=5, ge=1, le=20)


class Task2PredictRequest(Task2Request):
    text: str
    method: str | None = None
    order: int = Field(default=3)
    top_k: int = Field(default=5, ge=1, le=20)


class Task3PredictRequest(Task3Request):
    text: str
    classifier: str | None = None
    feature_set: str | None = None


class Task4PredictRequest(Task4Request):
    text: str
    penalty: str | None = None


class WarmupRequest(BaseModel):
    task2: bool = True
    task3: bool = True
    task4: bool = True


_task1_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
_task2_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
_task3_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
_task4_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
_task3_predict_cache: dict[tuple[Any, ...], dict[str, Any]] = {}


def _mtime_ns_or_zero(path_str: str | None) -> int:
    if not path_str:
        return 0
    p = _resolve_input_path(path_str)
    if p is None or not p.exists():
        return 0
    return int(p.stat().st_mtime_ns)


def _cache_hash(parts: tuple[Any, ...]) -> str:
    data = repr(parts).encode("utf-8")
    return sha256(data).hexdigest()


def _cache_file(prefix: str, parts: tuple[Any, ...]) -> Path:
    return CACHE_DIR / f"{prefix}_{_cache_hash(parts)}.pkl"


def _load_persistent_cache(prefix: str, parts: tuple[Any, ...]) -> Any | None:
    path = _cache_file(prefix, parts)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_persistent_cache(prefix: str, parts: tuple[Any, ...], value: Any) -> None:
    path = _cache_file(prefix, parts)
    with path.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)


def _resolve_input_path(input_path: str | None) -> Path | None:
    if input_path is None:
        return None
    p = Path(input_path)
    if not p.is_absolute():
        p = ROOT_DIR / p
    return p


def _safe_metric(value: float) -> float | str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return float(value)


def _build_task1_summary(req: Task1Request) -> dict[str, Any]:
    input_path = _resolve_input_path(req.input_path)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)
    validate_task1_columns(df, [req.author_col, req.text_col])

    clean_df, clean_stats = clean_and_tokenize(df, req.text_col)
    train_df, test_df = stratified_split(clean_df, req.author_col, req.test_size, req.seed)

    train_sequences_raw = train_df["tokens"].tolist()
    test_sequences_raw = test_df["tokens"].tolist()

    train_sequences, test_sequences, vocab_stats = apply_unk_policy(
        train_sequences=train_sequences_raw,
        test_sequences=test_sequences_raw,
        min_freq=req.min_freq,
    )

    ngram_sizes = {"unigram": 1, "bigram": 2, "trigram": 3}
    model_counts: dict[str, Any] = {}
    context_cache: dict[str, Any] = {}
    unigram_total = 0

    for model_name, n in ngram_sizes.items():
        counts, contexts, total_events = build_ngram_model(train_sequences, n)
        model_counts[model_name] = counts
        context_cache[model_name] = contexts
        if n == 1:
            unigram_total = total_events

    models = {}
    for model_name, n in ngram_sizes.items():
        train_diag = perplexity_diagnostics(
            sequences=train_sequences,
            n=n,
            ngram_counts=model_counts[model_name],
            context_counts=context_cache[model_name],
            unigram_total=unigram_total,
        )
        test_diag = perplexity_diagnostics(
            sequences=test_sequences,
            n=n,
            ngram_counts=model_counts[model_name],
            context_counts=context_cache[model_name],
            unigram_total=unigram_total,
        )
        models[model_name] = {
            "train_perplexity": _safe_metric(float(train_diag["perplexity"])),
            "test_perplexity": _safe_metric(float(test_diag["perplexity"])),
            "zero_prob_events_test": int(test_diag["zero_prob_events"]),
            "unseen_rate_test": float(test_diag["unseen_rate"]),
        }

    return {
        "config": req.model_dump(),
        "clean_stats": clean_stats,
        "split": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "vocab": vocab_stats,
        "models": models,
    }


def _task1_cache_key(req: Task1Request) -> tuple[Any, ...]:
    return (
        req.input_path,
        req.text_col,
        req.author_col,
        float(req.test_size),
        int(req.seed),
        int(req.min_freq),
    )


def _task2_cache_key(req: Task2Request) -> tuple[Any, ...]:
    input_path = str(_resolve_input_path(req.input_path)) if req.input_path else req.input_path
    return (
        input_path,
        _mtime_ns_or_zero(req.input_path),
        req.text_col,
        req.author_col,
        float(req.test_size),
        float(req.val_size),
        int(req.seed),
        int(req.min_freq),
        tuple(float(x) for x in req.interp_bigram_grid),
        float(req.interp_trigram_step),
        tuple(float(x) for x in req.discount_grid),
    )


def _task3_cache_key(req: Task3Request) -> tuple[Any, ...]:
    input_path = str(_resolve_input_path(req.input_path)) if req.input_path else req.input_path
    return (
        req.dataset,
        input_path,
        _mtime_ns_or_zero(req.input_path),
        req.text_col,
        req.label_col,
        float(req.test_size),
        int(req.seed),
        req.max_features,
    )


def _task4_cache_key(req: Task4Request) -> tuple[Any, ...]:
    input_path = str(_resolve_input_path(req.input_path)) if req.input_path else req.input_path
    return (
        input_path,
        _mtime_ns_or_zero(req.input_path),
        req.text_col,
        req.label_col,
        float(req.val_size),
        float(req.test_size),
        int(req.seed),
        tuple(float(x) for x in req.c_grid),
    )


def _get_task1_state(req: Task1Request) -> dict[str, Any]:
    key = _task1_cache_key(req)
    cached = _task1_cache.get(key)
    if cached is not None:
        return cached

    input_path = _resolve_input_path(req.input_path)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)
    validate_task1_columns(df, [req.author_col, req.text_col])

    clean_df, _ = clean_and_tokenize(df, req.text_col)
    train_df, test_df = stratified_split(clean_df, req.author_col, req.test_size, req.seed)
    train_raw = train_df["tokens"].tolist()
    test_raw = test_df["tokens"].tolist()

    train_sequences, _, _ = apply_unk_policy(
        train_sequences=train_raw,
        test_sequences=test_raw,
        min_freq=req.min_freq,
    )

    raw_counts = Counter(token for seq in train_raw for token in seq)
    retained_vocab = {token for token, count in raw_counts.items() if count >= req.min_freq}

    unigram_counts, _, unigram_total = build_ngram_model(train_sequences, 1)
    bigram_counts, bigram_contexts, _ = build_ngram_model(train_sequences, 2)
    trigram_counts, trigram_contexts, _ = build_ngram_model(train_sequences, 3)

    candidate_words = [
        ng[0]
        for ng in unigram_counts.keys()
        if ng and ng[0] not in {START_TOKEN, END_TOKEN}
    ]

    state = {
        "retained_vocab": retained_vocab,
        "candidate_words": candidate_words,
        "unigram_counts": unigram_counts,
        "unigram_total": unigram_total,
        "bigram_counts": bigram_counts,
        "bigram_contexts": bigram_contexts,
        "trigram_counts": trigram_counts,
        "trigram_contexts": trigram_contexts,
    }
    _task1_cache[key] = state
    return state


def _get_task2_state(req: Task2Request) -> dict[str, Any]:
    key = _task2_cache_key(req)
    cached = _task2_cache.get(key)
    if cached is not None:
        return cached

    persisted = _load_persistent_cache("task2", key)
    if isinstance(persisted, dict):
        _task2_cache[key] = persisted
        return persisted

    input_path = _resolve_input_path(req.input_path)
    if input_path is None:
        raise ValueError("input_path is required")

    artifacts = build_task2_artifacts(
        str(input_path),
        req.text_col,
        req.author_col,
        req.test_size,
        req.val_size,
        req.seed,
        req.min_freq,
        req.interp_bigram_grid,
        req.interp_trigram_step,
        req.discount_grid,
    )
    _save_persistent_cache("task2", key, artifacts)
    _task2_cache[key] = artifacts
    return artifacts


def _get_task3_state(req: Task3Request) -> dict[str, Any]:
    key = _task3_cache_key(req)
    cached = _task3_cache.get(key)
    if cached is not None:
        return cached

    persisted = _load_persistent_cache("task3", key)
    if isinstance(persisted, dict):
        _task3_cache[key] = persisted
        return persisted

    input_path = _resolve_input_path(req.input_path) if req.input_path else None
    artifacts = build_task3_artifacts(
        str(input_path) if input_path else None,
        req.dataset,
        req.text_col,
        req.label_col,
        req.test_size,
        req.seed,
        req.max_features,
    )
    _save_persistent_cache("task3", key, artifacts)
    _task3_cache[key] = artifacts
    return artifacts


def _get_task4_state(req: Task4Request) -> dict[str, Any]:
    def _ensure_task4_derived(artifacts: dict[str, Any]) -> dict[str, Any]:
        if "_train_word_freq" in artifacts and "_train_total_words" in artifacts:
            return artifacts

        source_df = artifacts["source_df"]
        train_text_ids = set(artifacts["train_dot_df"]["text_id"].unique())
        train_texts = [
            str(source_df.loc[tid, req.text_col])
            for tid in train_text_ids
            if tid in source_df.index
        ]
        word_freq, total_words = build_word_freq_from_texts(train_texts)
        artifacts["_train_word_freq"] = word_freq
        artifacts["_train_total_words"] = total_words
        return artifacts

    key = _task4_cache_key(req)
    cached = _task4_cache.get(key)
    if cached is not None:
        return _ensure_task4_derived(cached)

    persisted = _load_persistent_cache("task4", key)
    if isinstance(persisted, dict):
        persisted = _ensure_task4_derived(persisted)
        _task4_cache[key] = persisted
        _save_persistent_cache("task4", key, persisted)
        return persisted

    input_path = _resolve_input_path(req.input_path)
    if input_path is None:
        raise ValueError("input_path is required")

    artifacts = build_task4_artifacts(
        str(input_path),
        req.text_col,
        req.label_col,
        req.val_size,
        req.test_size,
        req.seed,
        req.c_grid,
    )

    artifacts = _ensure_task4_derived(artifacts)

    _save_persistent_cache("task4", key, artifacts)
    _task4_cache[key] = artifacts
    return artifacts


def _build_task3_predict_bundle(
    req: Task3PredictRequest,
    artifacts: dict[str, Any],
    classifier: str,
    feature_set: str,
) -> dict[str, Any]:
    cache_key = (_task3_cache_key(req), classifier, feature_set)
    cached = _task3_predict_cache.get(cache_key)
    if cached is not None:
        return cached

    train_texts = artifacts["train_texts"]
    y_train = artifacts["y_train"]
    binary_bow = classifier == "binary_naive_bayes"

    if feature_set in {"bow", "bow_lexicon"}:
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(
            analyzer="word",
            binary=binary_bow,
            max_features=req.max_features,
            token_pattern=r"\b\w+\b",
        )
        X_train_bow = vectorizer.fit_transform(train_texts).toarray()
    else:
        vectorizer = None
        X_train_bow = None

    if feature_set == "bow":
        X_train = X_train_bow
    elif feature_set == "lexicon":
        X_train = build_lexicon_features(train_texts)
    elif feature_set == "bow_lexicon":
        X_train_lex = build_lexicon_features(train_texts)
        X_train = np.hstack([X_train_bow, X_train_lex])
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    if X_train is None:
        raise ValueError("Could not build Task 3 training features.")

    model = train_classifier(
        classifier_name=classifier,
        X_train=X_train,
        y_train=y_train,
        seed=req.seed,
    )

    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "feature_set": feature_set,
        "classifier": classifier,
    }
    _task3_predict_cache[cache_key] = bundle
    return bundle


def _task3_single_features(text: str, bundle: dict[str, Any]) -> np.ndarray:
    cleaned = " ".join(text.lower().split())
    feature_set = bundle["feature_set"]
    vectorizer = bundle["vectorizer"]

    if feature_set == "bow":
        return vectorizer.transform([cleaned]).toarray()
    if feature_set == "lexicon":
        return build_lexicon_features([cleaned])
    if feature_set == "bow_lexicon":
        X_bow = vectorizer.transform([cleaned]).toarray()
        X_lex = build_lexicon_features([cleaned])
        return np.hstack([X_bow, X_lex])

    raise ValueError(f"Unknown feature_set: {feature_set}")


def _top_next_words_task1(req: Task1PredictRequest) -> dict[str, Any]:
    model_name = req.model.lower().strip()
    if model_name not in {"unigram", "bigram", "trigram"}:
        raise ValueError("model must be one of: unigram, bigram, trigram")

    state = _get_task1_state(req)
    tokens = clean_and_tokenize(pd.DataFrame({"x": [req.text]}), "x")[0]
    raw_tokens = tokens.iloc[0]["tokens"] if len(tokens) else []
    if not raw_tokens:
        raise ValueError("Input text has no valid word tokens.")

    retained_vocab = state["retained_vocab"]
    mapped_tokens = [tok if tok in retained_vocab else UNK_TOKEN for tok in raw_tokens]
    candidates = state["candidate_words"]

    unigram_counts = state["unigram_counts"]
    unigram_total = state["unigram_total"]
    bigram_counts = state["bigram_counts"]
    bigram_contexts = state["bigram_contexts"]
    trigram_counts = state["trigram_counts"]
    trigram_contexts = state["trigram_contexts"]

    scored: list[dict[str, Any]] = []

    for word in candidates:
        if model_name == "unigram":
            prob = unigram_counts.get((word,), 0) / max(unigram_total, 1)
        elif model_name == "bigram":
            context = (mapped_tokens[-1],)
            denom = bigram_contexts.get(context, 0)
            if denom > 0:
                prob = bigram_counts.get((context[0], word), 0) / denom
            else:
                prob = unigram_counts.get((word,), 0) / max(unigram_total, 1)
        else:
            if len(mapped_tokens) >= 2:
                context = (mapped_tokens[-2], mapped_tokens[-1])
            elif len(mapped_tokens) == 1:
                context = (START_TOKEN, mapped_tokens[-1])
            else:
                context = (START_TOKEN, START_TOKEN)
            denom = trigram_contexts.get(context, 0)
            if denom > 0:
                prob = trigram_counts.get((context[0], context[1], word), 0) / denom
            else:
                prob = unigram_counts.get((word,), 0) / max(unigram_total, 1)

        scored.append({"word": word, "probability": float(prob)})

    scored.sort(key=lambda row: row["probability"], reverse=True)
    top = scored[: req.top_k]

    return {
        "model": model_name,
        "input_text": req.text,
        "tokens": raw_tokens,
        "mapped_tokens": mapped_tokens,
        "prediction": top[0]["word"] if top else None,
        "top_k": top,
    }


def _top_next_words_task2(req: Task2PredictRequest) -> dict[str, Any]:
    if req.order not in {2, 3}:
        raise ValueError("order must be 2 or 3")

    artifacts = _get_task2_state(req)
    default_method = artifacts["metrics"]["best_method_defaults"]["best_method"]
    method = (req.method or default_method).strip().lower()

    model_key = (method, req.order)
    if model_key not in artifacts["final_models"]:
        raise ValueError(f"No trained model found for method={method}, order={req.order}")

    mapped = map_text_to_tokens(req.text, artifacts["retained_vocab_final"])
    mapped_tokens = mapped["mapped_tokens"]
    model = artifacts["final_models"][model_key]

    if req.order == 2:
        context = (mapped_tokens[-1],)
    else:
        if len(mapped_tokens) >= 2:
            context = (mapped_tokens[-2], mapped_tokens[-1])
        elif len(mapped_tokens) == 1:
            context = (START_TOKEN, mapped_tokens[-1])
        else:
            context = (START_TOKEN, START_TOKEN)

    candidates = [w for w in model.counts.vocab if w not in {START_TOKEN, END_TOKEN}]
    scored = [
        {
            "word": word,
            "probability": float(model_probability(model, context, word)),
        }
        for word in candidates
    ]
    scored.sort(key=lambda row: row["probability"], reverse=True)
    top = scored[: req.top_k]

    return {
        "method": method,
        "order": req.order,
        "input_text": req.text,
        "tokens": mapped["raw_tokens"],
        "mapped_tokens": mapped_tokens,
        "prediction": top[0]["word"] if top else None,
        "top_k": top,
    }


def _predict_sentiment_task3(req: Task3PredictRequest) -> dict[str, Any]:
    artifacts = _get_task3_state(req)
    best = artifacts["best"]

    classifier = (req.classifier or best["classifier"]).strip().lower()
    feature_set = (req.feature_set or best["feature_set"]).strip().lower()

    if not req.text or not req.text.strip():
        raise ValueError("Input text is empty.")

    bundle = _build_task3_predict_bundle(req, artifacts, classifier, feature_set)
    X_single = _task3_single_features(req.text, bundle)
    y_pred = task3_predict(classifier, bundle["model"], X_single)
    pred_idx = int(y_pred[0])
    label_names = artifacts["label_names"]
    if pred_idx < 0 or pred_idx >= len(label_names):
        raise ValueError("Predicted label index is out of range.")

    confidence = None
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_single)
        if probs.shape[0] > 0:
            confidence = float(np.max(probs[0]))

    return {
        "input_text": req.text,
        "predicted_label": label_names[pred_idx],
        "predicted_index": pred_idx,
        "confidence": confidence,
        "classifier": classifier,
        "feature_set": feature_set,
    }


def _predict_boundaries_task4(req: Task4PredictRequest) -> dict[str, Any]:
    artifacts = _get_task4_state(req)
    penalty = (req.penalty or artifacts["best_penalty"]).strip().lower()
    if penalty not in {"l1", "l2"}:
        raise ValueError("penalty must be one of: l1, l2")

    model = artifacts["models"][penalty]
    scaler = artifacts["scaler"]

    samples = extract_dot_samples(req.text, text_id=0)
    if not samples:
        return {
            "penalty": penalty,
            "input_text": req.text,
            "sentences": [req.text.strip()] if req.text.strip() else [],
            "dot_predictions": [],
        }

    sample_df = pd.DataFrame(samples)
    enrich_with_corpus_features(
        sample_df,
        artifacts["_train_word_freq"],
        artifacts["_train_total_words"],
    )

    X = sample_df[FEATURE_NAMES].values.astype(np.float64)
    X = scaler.transform(X)
    y_pred = model.predict(X)

    dot_positions = sample_df["dot_pos"].tolist()
    sentences = reconstruct_sentences(req.text, dot_positions, y_pred.tolist())

    dot_predictions = []
    for idx, row in sample_df.iterrows():
        pos = int(row["dot_pos"])
        dot_predictions.append(
            {
                "dot_pos": pos,
                "predicted": "boundary" if int(y_pred[idx]) == 1 else "not_boundary",
                "context": req.text[max(0, pos - 15) : pos + 15],
            }
        )

    return {
        "penalty": penalty,
        "input_text": req.text,
        "sentences": sentences,
        "dot_predictions": dot_predictions,
    }


def _warmup_default_models(req: WarmupRequest) -> dict[str, Any]:
    timings: dict[str, float] = {}
    warmed: list[str] = []

    if req.task2:
        t0 = time.perf_counter()
        _get_task2_state(Task2Request())
        timings["task2_seconds"] = round(time.perf_counter() - t0, 4)
        warmed.append("task2")

    if req.task3:
        t0 = time.perf_counter()
        _get_task3_state(Task3Request())
        timings["task3_seconds"] = round(time.perf_counter() - t0, 4)
        warmed.append("task3")

    if req.task4:
        t0 = time.perf_counter()
        _get_task4_state(Task4Request())
        timings["task4_seconds"] = round(time.perf_counter() - t0, 4)
        warmed.append("task4")

    return {
        "status": "ok",
        "warmed": warmed,
        "timings": timings,
        "cache_dir": str(CACHE_DIR),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/task1/run")
async def run_task1(req: Task1Request) -> dict[str, Any]:
    try:
        return await run_in_threadpool(_build_task1_summary, req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/task2/run")
async def run_task2(req: Task2Request) -> dict[str, Any]:
    try:
        artifacts = await run_in_threadpool(_get_task2_state, req)

        return {
            "config": req.model_dump(),
            "best_method_defaults": artifacts["metrics"]["best_method_defaults"],
            "ranking_rows": artifacts["ranking_rows"],
            "method_rows": artifacts["method_rows"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/task3/run")
async def run_task3(req: Task3Request) -> dict[str, Any]:
    try:
        artifacts = await run_in_threadpool(_get_task3_state, req)

        return {
            "config": req.model_dump(),
            "clean_stats": artifacts["clean_stats"],
            "split": artifacts["split"],
            "best": artifacts["best"],
            "best_classifier": artifacts["best_classifier"],
            "classifier_analysis": artifacts["classifier_analysis"],
            "summary_rows": artifacts["summary_rows"],
            "significance_results": artifacts["significance_results"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/task4/run")
async def run_task4(req: Task4Request) -> dict[str, Any]:
    try:
        artifacts = await run_in_threadpool(_get_task4_state, req)

        results = {}
        for penalty in ("l1", "l2"):
            row = artifacts["results"][penalty]
            results[penalty] = {
                "best_C": row["best_C"],
                "train": row["train_metrics"],
                "val": row["val_metrics"],
                "test": row["test_metrics"],
                "n_nonzero_coefs": row["n_nonzero_coefs"],
            }

        return {
            "config": req.model_dump(),
            "dataset_stats": artifacts["dataset_stats"],
            "best_penalty": artifacts["best_penalty"],
            "significance": artifacts["significance"],
            "results": results,
            "tuning_rows": artifacts["tuning_rows"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/task1/predict-next")
async def predict_task1_next(req: Task1PredictRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(_top_next_words_task1, req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/task2/predict-next")
async def predict_task2_next(req: Task2PredictRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(_top_next_words_task2, req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/task3/predict-sentiment")
async def predict_task3_sentiment(req: Task3PredictRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(_predict_sentiment_task3, req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/task4/predict-sentences")
async def predict_task4_sentences(req: Task4PredictRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(_predict_boundaries_task4, req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/warmup")
async def warmup_models(req: WarmupRequest) -> dict[str, Any]:
    try:
        return await run_in_threadpool(_warmup_default_models, req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
