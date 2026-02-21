from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project2.task1.task1_ngram import (  # noqa: E402
    END_TOKEN,
    START_TOKEN,
    UNK_TOKEN,
    apply_unk_policy,
    clean_and_tokenize,
    serialize_perplexity,
    stratified_split,
    tokenize,
    validate_columns,
)

METHODS: tuple[str, ...] = ("laplace", "interpolation", "backoff", "kneser_ney")
ORDERS: tuple[int, ...] = (2, 3)
RANKING_RULES: tuple[str, ...] = (
    "trigram_test_ppl",
    "bigram_test_ppl",
    "avg_bigram_trigram_test_ppl",
)


@dataclass
class ModelCounts:
    unigram_counts: Counter[str]
    unigram_total: int
    vocab: set[str]
    vocab_size: int
    bigram_counts: Counter[tuple[str, str]]
    bigram_context_counts: Counter[tuple[str]]
    trigram_counts: Counter[tuple[str, str, str]]
    trigram_context_counts: Counter[tuple[str, str]]
    followers_bigram: dict[tuple[str], set[str]]
    followers_trigram: dict[tuple[str, str], set[str]]
    predecessors_by_word: dict[str, set[str]]
    continuation_counts: dict[str, int]
    continuation_total: int


@dataclass
class SmoothedModel:
    method: str
    order: int
    params: dict[str, float]
    counts: ModelCounts
    backoff_alpha_bigram: dict[tuple[str], float] = field(default_factory=dict)
    backoff_alpha_trigram: dict[tuple[str, str], float] = field(default_factory=dict)
    kn_lambda_bigram: dict[tuple[str], float] = field(default_factory=dict)
    kn_lambda_trigram: dict[tuple[str, str], float] = field(default_factory=dict)


def flatten_sequences(sequences: Iterable[Sequence[str]]) -> list[str]:
    flat: list[str] = []
    for seq in sequences:
        flat.extend(seq)
    return flat


def retained_vocab_from_raw(train_sequences_raw: list[list[str]], min_freq: int) -> set[str]:
    counts = Counter(flatten_sequences(train_sequences_raw))
    return {token for token, count in counts.items() if count >= min_freq}


def parse_float_grid(raw: str, name: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError(f"{name} produced an empty grid.")
    return values


def generate_trigram_lambda_grid(step: float) -> list[dict[str, float]]:
    if step <= 0 or step > 1:
        raise ValueError("--interp-trigram-step must be in (0, 1].")

    grid: list[dict[str, float]] = []
    steps = int(round(1.0 / step))
    seen: set[tuple[float, float, float]] = set()

    for i in range(steps + 1):
        l1 = round(i * step, 10)
        for j in range(steps + 1):
            l2 = round(j * step, 10)
            l3 = round(1.0 - l1 - l2, 10)
            if l3 < 0:
                continue
            key = (l1, l2, l3)
            if key in seen:
                continue
            seen.add(key)
            grid.append({"lambda1": l1, "lambda2": l2, "lambda3": l3})

    if not grid:
        raise ValueError("Interpolation trigram grid is empty. Check --interp-trigram-step.")
    return grid


def method_param_grid(
    method: str,
    order: int,
    interp_bigram_grid: list[float],
    interp_trigram_step: float,
    discount_grid: list[float],
) -> list[dict[str, float]]:
    if method == "laplace":
        return [{"alpha": 1.0}]

    if method == "interpolation":
        if order == 2:
            grid: list[dict[str, float]] = []
            for lambda2 in interp_bigram_grid:
                if not (0 <= lambda2 <= 1):
                    raise ValueError("Interpolation bigram lambda2 must be in [0,1].")
                lambda1 = 1.0 - lambda2
                grid.append({"lambda1": lambda1, "lambda2": lambda2})
            return grid
        if order == 3:
            return generate_trigram_lambda_grid(interp_trigram_step)

    if method in {"backoff", "kneser_ney"}:
        grid = []
        for discount in discount_grid:
            if discount <= 0:
                raise ValueError("Discount values must be > 0.")
            grid.append({"discount": discount})
        return grid

    raise ValueError(f"Unsupported method: {method}")


def build_model_counts(sequences: list[list[str]]) -> ModelCounts:
    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[tuple[str, str]] = Counter()
    trigram_counts: Counter[tuple[str, str, str]] = Counter()
    bigram_context_counts: Counter[tuple[str]] = Counter()
    trigram_context_counts: Counter[tuple[str, str]] = Counter()
    followers_bigram: defaultdict[tuple[str], set[str]] = defaultdict(set)
    followers_trigram: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    predecessors_by_word: defaultdict[str, set[str]] = defaultdict(set)

    for seq in sequences:
        unigram_counts.update(seq + [END_TOKEN])

        padded_bigram = [START_TOKEN] + list(seq) + [END_TOKEN]
        for i in range(len(padded_bigram) - 1):
            h, w = padded_bigram[i], padded_bigram[i + 1]
            bigram_counts[(h, w)] += 1
            bigram_context_counts[(h,)] += 1
            followers_bigram[(h,)].add(w)
            predecessors_by_word[w].add(h)

        padded_trigram = [START_TOKEN, START_TOKEN] + list(seq) + [END_TOKEN]
        for i in range(len(padded_trigram) - 2):
            h1, h2, w = padded_trigram[i], padded_trigram[i + 1], padded_trigram[i + 2]
            trigram_counts[(h1, h2, w)] += 1
            trigram_context_counts[(h1, h2)] += 1
            followers_trigram[(h1, h2)].add(w)

    continuation_counts = {word: len(preds) for word, preds in predecessors_by_word.items()}
    continuation_total = len(bigram_counts)
    vocab = set(unigram_counts.keys())

    return ModelCounts(
        unigram_counts=unigram_counts,
        unigram_total=int(sum(unigram_counts.values())),
        vocab=vocab,
        vocab_size=len(vocab),
        bigram_counts=bigram_counts,
        bigram_context_counts=bigram_context_counts,
        trigram_counts=trigram_counts,
        trigram_context_counts=trigram_context_counts,
        followers_bigram={k: set(v) for k, v in followers_bigram.items()},
        followers_trigram={k: set(v) for k, v in followers_trigram.items()},
        predecessors_by_word={k: set(v) for k, v in predecessors_by_word.items()},
        continuation_counts=continuation_counts,
        continuation_total=continuation_total,
    )


def iter_events(sequences: list[list[str]], order: int) -> Iterable[tuple[tuple[str, ...], str]]:
    if order == 2:
        for seq in sequences:
            padded = [START_TOKEN] + list(seq) + [END_TOKEN]
            for i in range(len(padded) - 1):
                yield (padded[i],), padded[i + 1]
        return

    if order == 3:
        for seq in sequences:
            padded = [START_TOKEN, START_TOKEN] + list(seq) + [END_TOKEN]
            for i in range(len(padded) - 2):
                yield (padded[i], padded[i + 1]), padded[i + 2]
        return

    raise ValueError(f"Unsupported order: {order}")


def unigram_mle(counts: ModelCounts, word: str) -> float:
    if counts.unigram_total <= 0:
        return 0.0
    return counts.unigram_counts.get(word, 0) / counts.unigram_total


def bigram_mle(counts: ModelCounts, context: tuple[str], word: str) -> float:
    ctx_count = counts.bigram_context_counts.get(context, 0)
    if ctx_count <= 0:
        return 0.0
    return counts.bigram_counts.get((context[0], word), 0) / ctx_count


def trigram_mle(counts: ModelCounts, context: tuple[str, str], word: str) -> float:
    ctx_count = counts.trigram_context_counts.get(context, 0)
    if ctx_count <= 0:
        return 0.0
    return counts.trigram_counts.get((context[0], context[1], word), 0) / ctx_count


def _backoff_bigram_prob(
    counts: ModelCounts,
    context: tuple[str],
    word: str,
    discount: float,
    alpha_bigram: dict[tuple[str], float],
) -> float:
    ctx_count = counts.bigram_context_counts.get(context, 0)
    if ctx_count <= 0:
        return unigram_mle(counts, word)

    observed = counts.bigram_counts.get((context[0], word), 0)
    if observed > 0:
        return max(observed - discount, 0.0) / ctx_count

    return alpha_bigram.get(context, 1.0) * unigram_mle(counts, word)


def precompute_backoff_alphas(
    counts: ModelCounts,
    order: int,
    discount: float,
) -> tuple[dict[tuple[str], float], dict[tuple[str, str], float]]:
    alpha_bigram: dict[tuple[str], float] = {}
    alpha_trigram: dict[tuple[str, str], float] = {}

    for context, ctx_count in counts.bigram_context_counts.items():
        seen_words = counts.followers_bigram.get(context, set())
        if ctx_count <= 0 or not seen_words:
            alpha_bigram[context] = 1.0
            continue

        seen_mass = 0.0
        lower_seen_mass = 0.0
        for word in seen_words:
            observed = counts.bigram_counts.get((context[0], word), 0)
            seen_mass += max(observed - discount, 0.0) / ctx_count
            lower_seen_mass += unigram_mle(counts, word)

        beta = max(0.0, 1.0 - seen_mass)
        denom = max(0.0, 1.0 - lower_seen_mass)
        alpha_bigram[context] = beta / denom if denom > 0 else 0.0

    if order == 2:
        return alpha_bigram, alpha_trigram

    for context, ctx_count in counts.trigram_context_counts.items():
        seen_words = counts.followers_trigram.get(context, set())
        if ctx_count <= 0 or not seen_words:
            alpha_trigram[context] = 1.0
            continue

        seen_mass = 0.0
        lower_seen_mass = 0.0
        suffix_context = (context[1],)
        for word in seen_words:
            observed = counts.trigram_counts.get((context[0], context[1], word), 0)
            seen_mass += max(observed - discount, 0.0) / ctx_count
            lower_seen_mass += _backoff_bigram_prob(
                counts,
                context=suffix_context,
                word=word,
                discount=discount,
                alpha_bigram=alpha_bigram,
            )

        beta = max(0.0, 1.0 - seen_mass)
        denom = max(0.0, 1.0 - lower_seen_mass)
        alpha_trigram[context] = beta / denom if denom > 0 else 0.0

    return alpha_bigram, alpha_trigram


def continuation_prob(counts: ModelCounts, word: str) -> float:
    if counts.continuation_total <= 0:
        return 0.0
    return counts.continuation_counts.get(word, 0) / counts.continuation_total


def precompute_kneser_ney_lambdas(
    counts: ModelCounts,
    order: int,
    discount: float,
) -> tuple[dict[tuple[str], float], dict[tuple[str, str], float]]:
    lambda_bigram: dict[tuple[str], float] = {}
    lambda_trigram: dict[tuple[str, str], float] = {}

    for context, ctx_count in counts.bigram_context_counts.items():
        followers = counts.followers_bigram.get(context, set())
        if ctx_count <= 0:
            lambda_bigram[context] = 0.0
        else:
            lambda_bigram[context] = discount * len(followers) / ctx_count

    if order == 2:
        return lambda_bigram, lambda_trigram

    for context, ctx_count in counts.trigram_context_counts.items():
        followers = counts.followers_trigram.get(context, set())
        if ctx_count <= 0:
            lambda_trigram[context] = 0.0
        else:
            lambda_trigram[context] = discount * len(followers) / ctx_count

    return lambda_bigram, lambda_trigram


def build_smoothed_model(
    sequences: list[list[str]],
    order: int,
    method: str,
    params: dict[str, float],
) -> SmoothedModel:
    if order not in ORDERS:
        raise ValueError(f"Unsupported order: {order}")
    if method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")

    counts = build_model_counts(sequences)
    model = SmoothedModel(method=method, order=order, params=dict(params), counts=counts)

    if method == "backoff":
        discount = float(params.get("discount", 0.75))
        alpha_bigram, alpha_trigram = precompute_backoff_alphas(counts, order, discount)
        model.backoff_alpha_bigram = alpha_bigram
        model.backoff_alpha_trigram = alpha_trigram

    if method == "kneser_ney":
        discount = float(params.get("discount", 0.75))
        lambda_bigram, lambda_trigram = precompute_kneser_ney_lambdas(counts, order, discount)
        model.kn_lambda_bigram = lambda_bigram
        model.kn_lambda_trigram = lambda_trigram

    return model


def _kneser_ney_bigram_prob(
    model: SmoothedModel,
    context: tuple[str],
    word: str,
) -> float:
    counts = model.counts
    discount = float(model.params.get("discount", 0.75))
    ctx_count = counts.bigram_context_counts.get(context, 0)
    if ctx_count <= 0:
        return continuation_prob(counts, word)

    observed = counts.bigram_counts.get((context[0], word), 0)
    lambda_term = model.kn_lambda_bigram.get(context, 0.0)
    return max(observed - discount, 0.0) / ctx_count + lambda_term * continuation_prob(counts, word)


def model_probability(model: SmoothedModel, context: tuple[str, ...], word: str) -> float:
    counts = model.counts
    order = model.order
    method = model.method

    if method == "laplace":
        alpha = float(model.params.get("alpha", 1.0))
        if order == 2:
            ctx = (context[0],)
            ctx_count = counts.bigram_context_counts.get(ctx, 0)
            numerator = counts.bigram_counts.get((ctx[0], word), 0) + alpha
            denominator = ctx_count + alpha * counts.vocab_size
            return numerator / denominator if denominator > 0 else 0.0
        ctx = (context[0], context[1])
        ctx_count = counts.trigram_context_counts.get(ctx, 0)
        numerator = counts.trigram_counts.get((ctx[0], ctx[1], word), 0) + alpha
        denominator = ctx_count + alpha * counts.vocab_size
        return numerator / denominator if denominator > 0 else 0.0

    if method == "interpolation":
        if order == 2:
            lambda2 = float(model.params.get("lambda2", 0.7))
            lambda1 = float(model.params.get("lambda1", 1.0 - lambda2))
            total = lambda1 + lambda2
            if total > 0:
                lambda1, lambda2 = lambda1 / total, lambda2 / total
            return lambda2 * bigram_mle(counts, (context[0],), word) + lambda1 * unigram_mle(counts, word)

        lambda1 = float(model.params.get("lambda1", 0.1))
        lambda2 = float(model.params.get("lambda2", 0.3))
        lambda3 = float(model.params.get("lambda3", 0.6))
        total = lambda1 + lambda2 + lambda3
        if total > 0:
            lambda1, lambda2, lambda3 = lambda1 / total, lambda2 / total, lambda3 / total
        return (
            lambda3 * trigram_mle(counts, (context[0], context[1]), word)
            + lambda2 * bigram_mle(counts, (context[1],), word)
            + lambda1 * unigram_mle(counts, word)
        )

    if method == "backoff":
        discount = float(model.params.get("discount", 0.75))
        if order == 2:
            return _backoff_bigram_prob(
                counts=counts,
                context=(context[0],),
                word=word,
                discount=discount,
                alpha_bigram=model.backoff_alpha_bigram,
            )

        trigram_context = (context[0], context[1])
        ctx_count = counts.trigram_context_counts.get(trigram_context, 0)
        suffix_context = (context[1],)
        if ctx_count <= 0:
            return _backoff_bigram_prob(
                counts=counts,
                context=suffix_context,
                word=word,
                discount=discount,
                alpha_bigram=model.backoff_alpha_bigram,
            )

        observed = counts.trigram_counts.get((trigram_context[0], trigram_context[1], word), 0)
        if observed > 0:
            return max(observed - discount, 0.0) / ctx_count

        alpha = model.backoff_alpha_trigram.get(trigram_context, 1.0)
        lower = _backoff_bigram_prob(
            counts=counts,
            context=suffix_context,
            word=word,
            discount=discount,
            alpha_bigram=model.backoff_alpha_bigram,
        )
        return alpha * lower

    if method == "kneser_ney":
        if order == 2:
            return _kneser_ney_bigram_prob(model, (context[0],), word)

        discount = float(model.params.get("discount", 0.75))
        trigram_context = (context[0], context[1])
        ctx_count = counts.trigram_context_counts.get(trigram_context, 0)
        if ctx_count <= 0:
            return _kneser_ney_bigram_prob(model, (context[1],), word)

        observed = counts.trigram_counts.get((trigram_context[0], trigram_context[1], word), 0)
        lambda_term = model.kn_lambda_trigram.get(trigram_context, 0.0)
        lower = _kneser_ney_bigram_prob(model, (context[1],), word)
        return max(observed - discount, 0.0) / ctx_count + lambda_term * lower

    raise ValueError(f"Unsupported method: {method}")


def evaluate_model(model: SmoothedModel, sequences: list[list[str]]) -> dict[str, float | int]:
    total_events = 0
    zero_prob_events = 0
    log_prob_sum = 0.0

    for context, word in iter_events(sequences, model.order):
        prob = model_probability(model, context, word)
        total_events += 1
        if prob <= 0:
            zero_prob_events += 1
        else:
            log_prob_sum += math.log(prob)

    if total_events <= 0:
        perplexity = math.inf
    elif zero_prob_events > 0:
        perplexity = math.inf
    else:
        perplexity = math.exp(-log_prob_sum / total_events)

    return {
        "perplexity": perplexity,
        "events": int(total_events),
        "zero_prob_events": int(zero_prob_events),
        "unseen_rate": zero_prob_events / total_events if total_events > 0 else 0.0,
    }


def _params_json(params: dict[str, float]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def _tuning_key(row: dict[str, Any]) -> tuple[float, str]:
    ppl = float(row["val_perplexity"])
    return ppl, row["params_json"]


def run_tuning(
    inner_train_sequences: list[list[str]],
    val_sequences: list[list[str]],
    interp_bigram_grid: list[float],
    interp_trigram_step: float,
    discount_grid: list[float],
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    tuning_rows: list[dict[str, Any]] = []
    best_by_method_order: dict[tuple[str, int], dict[str, Any]] = {}

    for order in ORDERS:
        for method in METHODS:
            params_grid = method_param_grid(
                method=method,
                order=order,
                interp_bigram_grid=interp_bigram_grid,
                interp_trigram_step=interp_trigram_step,
                discount_grid=discount_grid,
            )

            for params in params_grid:
                model = build_smoothed_model(inner_train_sequences, order, method, params)
                train_diag = evaluate_model(model, inner_train_sequences)
                val_diag = evaluate_model(model, val_sequences)

                row = {
                    "method": method,
                    "order": order,
                    "params": dict(params),
                    "params_json": _params_json(params),
                    "train_perplexity": float(train_diag["perplexity"]),
                    "val_perplexity": float(val_diag["perplexity"]),
                    "val_zero_prob_events": int(val_diag["zero_prob_events"]),
                    "val_unseen_rate": float(val_diag["unseen_rate"]),
                }
                tuning_rows.append(row)

            method_rows = [r for r in tuning_rows if r["method"] == method and r["order"] == order]
            method_rows = sorted(method_rows, key=_tuning_key)
            best_by_method_order[(method, order)] = method_rows[0]

    return best_by_method_order, tuning_rows


def train_final_models(
    final_train_sequences: list[list[str]],
    test_sequences: list[list[str]],
    best_by_method_order: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], SmoothedModel]]:
    method_rows: list[dict[str, Any]] = []
    models: dict[tuple[str, int], SmoothedModel] = {}

    for order in ORDERS:
        for method in METHODS:
            best = best_by_method_order[(method, order)]
            params = dict(best["params"])

            final_model = build_smoothed_model(final_train_sequences, order, method, params)
            models[(method, order)] = final_model

            train_diag = evaluate_model(final_model, final_train_sequences)
            test_diag = evaluate_model(final_model, test_sequences)

            method_rows.append(
                {
                    "method": method,
                    "order": order,
                    "best_params": params,
                    "train_perplexity": float(train_diag["perplexity"]),
                    "val_perplexity": float(best["val_perplexity"]),
                    "test_perplexity": float(test_diag["perplexity"]),
                    "zero_prob_events_test": int(test_diag["zero_prob_events"]),
                    "unseen_rate_test": float(test_diag["unseen_rate"]),
                }
            )

    return method_rows, models


def _method_score_map(method_rows: list[dict[str, Any]]) -> dict[str, dict[int, float]]:
    score_map: dict[str, dict[int, float]] = defaultdict(dict)
    for row in method_rows:
        score_map[row["method"]][int(row["order"])] = float(row["test_perplexity"])
    return score_map


def rank_methods(method_rows: list[dict[str, Any]], rule: str) -> list[dict[str, Any]]:
    if rule not in RANKING_RULES:
        raise ValueError(f"Unsupported ranking rule: {rule}")

    score_map = _method_score_map(method_rows)
    scored_rows: list[dict[str, Any]] = []

    for method in METHODS:
        bigram = score_map.get(method, {}).get(2, math.inf)
        trigram = score_map.get(method, {}).get(3, math.inf)
        if rule == "trigram_test_ppl":
            score = trigram
        elif rule == "bigram_test_ppl":
            score = bigram
        else:
            score = (bigram + trigram) / 2.0

        scored_rows.append(
            {
                "rule": rule,
                "method": method,
                "score": float(score),
                "bigram_test_perplexity": float(bigram),
                "trigram_test_perplexity": float(trigram),
            }
        )

    scored_rows.sort(key=lambda row: (row["score"], row["method"]))
    for idx, row in enumerate(scored_rows, start=1):
        row["rank"] = idx
    return scored_rows


def build_ranking_rows(method_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    best_by_rule: dict[str, str] = {}

    for rule in RANKING_RULES:
        rows = rank_methods(method_rows, rule)
        all_rows.extend(rows)
        best_by_rule[rule] = rows[0]["method"]

    defaults = {
        "default_rule": "trigram_test_ppl",
        "best_method": best_by_rule["trigram_test_ppl"],
        "best_method_by_rule": best_by_rule,
    }
    return all_rows, defaults


def _serialize_params(params: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in params.items()}


def _serialize_method_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "order": int(row["order"]),
        "method": str(row["method"]),
        "best_params": _serialize_params(row["best_params"]),
        "train_perplexity": serialize_perplexity(float(row["train_perplexity"])),
        "val_perplexity": serialize_perplexity(float(row["val_perplexity"])),
        "test_perplexity": serialize_perplexity(float(row["test_perplexity"])),
        "zero_prob_events_test": int(row["zero_prob_events_test"]),
        "unseen_rate_test": float(row["unseen_rate_test"]),
    }


def prepare_task2_data(
    input_path: str,
    text_col: str,
    author_col: str,
    test_size: float,
    val_size: float,
    seed: int,
    min_freq: int,
) -> dict[str, Any]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input parquet not found: {path}")
    if not (0 < test_size < 1):
        raise ValueError("--test-size must be between 0 and 1.")
    if not (0 < val_size < 1):
        raise ValueError("--val-size must be between 0 and 1.")
    if min_freq < 1:
        raise ValueError("--min-freq must be >= 1.")

    df = pd.read_parquet(path)
    validate_columns(df, [author_col, text_col])
    df = df.reset_index(drop=False).rename(columns={"index": "row_id"})

    clean_df, clean_stats = clean_and_tokenize(df, text_col)
    if clean_df.empty:
        raise ValueError("No usable rows after cleaning/tokenization.")

    outer_train_df, test_df = stratified_split(clean_df, author_col, test_size, seed)
    inner_train_df, val_df = stratified_split(outer_train_df, author_col, val_size, seed)

    inner_train_raw = inner_train_df["tokens"].tolist()
    val_raw = val_df["tokens"].tolist()
    final_train_raw = outer_train_df["tokens"].tolist()
    test_raw = test_df["tokens"].tolist()

    inner_train, val, inner_vocab_stats = apply_unk_policy(inner_train_raw, val_raw, min_freq)
    final_train, test, final_vocab_stats = apply_unk_policy(final_train_raw, test_raw, min_freq)
    retained_vocab_final = retained_vocab_from_raw(final_train_raw, min_freq)

    split = {
        "outer_train_rows": int(len(outer_train_df)),
        "inner_train_rows": int(len(inner_train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "outer_train_author_counts": {
            str(k): int(v) for k, v in outer_train_df[author_col].value_counts().sort_index().items()
        },
        "inner_train_author_counts": {
            str(k): int(v) for k, v in inner_train_df[author_col].value_counts().sort_index().items()
        },
        "val_author_counts": {
            str(k): int(v) for k, v in val_df[author_col].value_counts().sort_index().items()
        },
        "test_author_counts": {
            str(k): int(v) for k, v in test_df[author_col].value_counts().sort_index().items()
        },
    }

    corpus = {
        **clean_stats,
        "authors": int(clean_df[author_col].nunique()),
        "inner_train_tokens_after_unk": int(sum(len(seq) for seq in inner_train)),
        "val_tokens_after_unk": int(sum(len(seq) for seq in val)),
        "final_train_tokens_after_unk": int(sum(len(seq) for seq in final_train)),
        "test_tokens_after_unk": int(sum(len(seq) for seq in test)),
        "inner_vocab": inner_vocab_stats,
        "final_vocab": final_vocab_stats,
    }

    return {
        "input_path": str(path),
        "inner_train_sequences": inner_train,
        "val_sequences": val,
        "final_train_sequences": final_train,
        "test_sequences": test,
        "retained_vocab_final": retained_vocab_final,
        "corpus": corpus,
        "split": split,
    }


def build_task2_artifacts(
    input_path: str,
    text_col: str,
    author_col: str,
    test_size: float,
    val_size: float,
    seed: int,
    min_freq: int,
    interp_bigram_grid: list[float],
    interp_trigram_step: float,
    discount_grid: list[float],
) -> dict[str, Any]:
    data = prepare_task2_data(
        input_path=input_path,
        text_col=text_col,
        author_col=author_col,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        min_freq=min_freq,
    )

    best_by_method_order, tuning_rows = run_tuning(
        inner_train_sequences=data["inner_train_sequences"],
        val_sequences=data["val_sequences"],
        interp_bigram_grid=interp_bigram_grid,
        interp_trigram_step=interp_trigram_step,
        discount_grid=discount_grid,
    )

    method_rows, final_models = train_final_models(
        final_train_sequences=data["final_train_sequences"],
        test_sequences=data["test_sequences"],
        best_by_method_order=best_by_method_order,
    )

    ranking_rows, best_defaults = build_ranking_rows(method_rows)

    metrics = {
        "config": {
            "input": str(input_path),
            "text_col": text_col,
            "author_col": author_col,
            "test_size": float(test_size),
            "val_size": float(val_size),
            "seed": int(seed),
            "min_freq": int(min_freq),
            "orders": list(ORDERS),
            "methods": list(METHODS),
            "tokenizer": "unicode-word-boundary-lowercase",
            "boundary_tokens": [START_TOKEN, END_TOKEN],
            "unk_token": UNK_TOKEN,
            "interp_bigram_grid": [float(v) for v in interp_bigram_grid],
            "interp_trigram_step": float(interp_trigram_step),
            "discount_grid": [float(v) for v in discount_grid],
        },
        "corpus": data["corpus"],
        "split": data["split"],
        "methods": [_serialize_method_row(row) for row in method_rows],
        "best_method_defaults": best_defaults,
    }

    return {
        "metrics": metrics,
        "tuning_rows": tuning_rows,
        "method_rows": method_rows,
        "ranking_rows": ranking_rows,
        "best_by_method_order": best_by_method_order,
        "final_models": final_models,
        "retained_vocab_final": data["retained_vocab_final"],
        "inner_train_sequences": data["inner_train_sequences"],
        "val_sequences": data["val_sequences"],
        "final_train_sequences": data["final_train_sequences"],
        "test_sequences": data["test_sequences"],
    }


def evaluate_manual_params(
    method: str,
    order: int,
    params: dict[str, float],
    inner_train_sequences: list[list[str]],
    val_sequences: list[list[str]],
    final_train_sequences: list[list[str]],
    test_sequences: list[list[str]],
) -> dict[str, Any]:
    inner_model = build_smoothed_model(inner_train_sequences, order, method, params)
    val_diag = evaluate_model(inner_model, val_sequences)

    final_model = build_smoothed_model(final_train_sequences, order, method, params)
    train_diag = evaluate_model(final_model, final_train_sequences)
    test_diag = evaluate_model(final_model, test_sequences)

    return {
        "method": method,
        "order": order,
        "params": dict(params),
        "train_perplexity": float(train_diag["perplexity"]),
        "val_perplexity": float(val_diag["perplexity"]),
        "test_perplexity": float(test_diag["perplexity"]),
        "zero_prob_events_test": int(test_diag["zero_prob_events"]),
        "unseen_rate_test": float(test_diag["unseen_rate"]),
    }


def map_text_to_tokens(text: str, retained_vocab: set[str]) -> dict[str, Any]:
    raw_tokens = tokenize(text)
    if not raw_tokens:
        raise ValueError("Input has no valid word tokens.")
    mapped = [token if token in retained_vocab else UNK_TOKEN for token in raw_tokens]
    unk_count = sum(1 for token in mapped if token == UNK_TOKEN)
    return {
        "raw_tokens": raw_tokens,
        "mapped_tokens": mapped,
        "unk_count": unk_count,
        "unk_rate": unk_count / len(mapped) if mapped else 0.0,
    }


def evaluate_custom_text_with_models(
    mapped_tokens: list[str],
    models: dict[tuple[str, int], SmoothedModel],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for order in ORDERS:
            model = models[(method, order)]
            diag = evaluate_model(model, [mapped_tokens])
            rows.append(
                {
                    "method": method,
                    "order": order,
                    "perplexity": float(diag["perplexity"]),
                    "zero_prob_events": int(diag["zero_prob_events"]),
                    "unseen_rate": float(diag["unseen_rate"]),
                    "event_count": int(diag["events"]),
                }
            )
    rows.sort(key=lambda row: (row["order"], row["method"]))
    return rows


def write_task2_outputs(artifacts: dict[str, Any], out_dir: str) -> dict[str, Path]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "task2_metrics.json"
    tuning_path = output_dir / "task2_tuning_results.csv"
    comparison_path = output_dir / "task2_method_comparison.csv"
    ranking_path = output_dir / "task2_ranking.csv"

    metrics_path.write_text(
        json.dumps(artifacts["metrics"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with tuning_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "order",
                "params_json",
                "train_perplexity",
                "val_perplexity",
                "val_zero_prob_events",
                "val_unseen_rate",
            ]
        )
        for row in artifacts["tuning_rows"]:
            writer.writerow(
                [
                    row["method"],
                    row["order"],
                    row["params_json"],
                    row["train_perplexity"],
                    row["val_perplexity"],
                    row["val_zero_prob_events"],
                    row["val_unseen_rate"],
                ]
            )

    with comparison_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "order",
                "best_params_json",
                "train_perplexity",
                "val_perplexity",
                "test_perplexity",
                "zero_prob_events_test",
                "unseen_rate_test",
            ]
        )
        for row in artifacts["method_rows"]:
            writer.writerow(
                [
                    row["method"],
                    row["order"],
                    _params_json(row["best_params"]),
                    row["train_perplexity"],
                    row["val_perplexity"],
                    row["test_perplexity"],
                    row["zero_prob_events_test"],
                    row["unseen_rate_test"],
                ]
            )

    with ranking_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rule",
                "rank",
                "method",
                "score",
                "bigram_test_perplexity",
                "trigram_test_perplexity",
            ]
        )
        for row in artifacts["ranking_rows"]:
            writer.writerow(
                [
                    row["rule"],
                    row["rank"],
                    row["method"],
                    row["score"],
                    row["bigram_test_perplexity"],
                    row["trigram_test_perplexity"],
                ]
            )

    return {
        "metrics": metrics_path,
        "tuning": tuning_path,
        "comparison": comparison_path,
        "ranking": ranking_path,
    }
