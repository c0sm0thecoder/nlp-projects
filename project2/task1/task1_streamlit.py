from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from .task1_ngram import (
        END_TOKEN,
        START_TOKEN,
        UNK_TOKEN,
        apply_unk_policy,
        build_ngram_model,
        clean_and_tokenize,
        generate_ngrams,
        perplexity_diagnostics,
        stratified_split,
        tokenize,
        validate_columns,
    )
except ImportError:
    from task1_ngram import (  # type: ignore
        END_TOKEN,
        START_TOKEN,
        UNK_TOKEN,
        apply_unk_policy,
        build_ngram_model,
        clean_and_tokenize,
        generate_ngrams,
        perplexity_diagnostics,
        stratified_split,
        tokenize,
        validate_columns,
    )


def flatten_sequences(sequences: list[list[str]]) -> list[str]:
    flat: list[str] = []
    for seq in sequences:
        flat.extend(seq)
    return flat


def format_perplexity(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.6f}"


def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


@st.cache_resource(show_spinner=True)
def build_artifacts(
    input_path: str,
    text_col: str,
    author_col: str,
    test_size: float,
    seed: int,
    min_freq: int,
) -> dict[str, Any]:
    parquet_path = Path(input_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    validate_columns(df, [author_col, text_col])
    df = df.reset_index(drop=False).rename(columns={"index": "row_id"})

    clean_df, clean_stats = clean_and_tokenize(df, text_col)
    if clean_df.empty:
        raise ValueError("No usable rows after cleaning/tokenization.")

    train_df, test_df = stratified_split(clean_df, author_col, test_size, seed)
    train_sequences_raw = train_df["tokens"].tolist()
    test_sequences_raw = test_df["tokens"].tolist()

    raw_train_counts = Counter(flatten_sequences(train_sequences_raw))
    retained_vocab = {token for token, count in raw_train_counts.items() if count >= min_freq}

    train_sequences, test_sequences, vocab_stats = apply_unk_policy(
        train_sequences=train_sequences_raw,
        test_sequences=test_sequences_raw,
        min_freq=min_freq,
    )

    model_specs = {"unigram": 1, "bigram": 2, "trigram": 3}
    models: dict[str, dict[str, Any]] = {}
    unigram_total = 0

    for model_name, n in model_specs.items():
        counts, contexts, total_events = build_ngram_model(train_sequences, n)
        models[model_name] = {
            "n": n,
            "counts": counts,
            "contexts": contexts,
            "train_event_count": total_events,
            "type_count": len(counts),
        }
        if n == 1:
            unigram_total = total_events

    baseline: dict[str, dict[str, Any]] = {}
    for model_name, spec in models.items():
        train_diag = perplexity_diagnostics(
            sequences=train_sequences,
            n=spec["n"],
            ngram_counts=spec["counts"],
            context_counts=spec["contexts"],
            unigram_total=unigram_total,
        )
        test_diag = perplexity_diagnostics(
            sequences=test_sequences,
            n=spec["n"],
            ngram_counts=spec["counts"],
            context_counts=spec["contexts"],
            unigram_total=unigram_total,
        )
        baseline[model_name] = {
            "train_perplexity": float(train_diag["perplexity"]),
            "test_perplexity": float(test_diag["perplexity"]),
            "zero_prob_events_train": int(train_diag["zero_prob_events"]),
            "zero_prob_events_test": int(test_diag["zero_prob_events"]),
            "unseen_rate_test": float(test_diag["unseen_rate"]),
        }

    return {
        "clean_stats": clean_stats,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_author_counts": {
            str(k): int(v) for k, v in train_df[author_col].value_counts().sort_index().items()
        },
        "test_author_counts": {
            str(k): int(v) for k, v in test_df[author_col].value_counts().sort_index().items()
        },
        "vocab_stats": vocab_stats,
        "retained_vocab": retained_vocab,
        "models": models,
        "unigram_total": unigram_total,
        "baseline": baseline,
    }


def evaluate_text(
    text: str,
    artifacts: dict[str, Any],
    max_rows_per_model: int = 25,
) -> dict[str, Any]:
    raw_tokens = tokenize(text)
    if not raw_tokens:
        raise ValueError("Input has no valid word tokens.")

    retained_vocab: set[str] = artifacts["retained_vocab"]
    mapped_tokens = [tok if tok in retained_vocab else UNK_TOKEN for tok in raw_tokens]
    unk_count = sum(1 for tok in mapped_tokens if tok == UNK_TOKEN)

    results: dict[str, dict[str, Any]] = {}
    for model_name in ("unigram", "bigram", "trigram"):
        model = artifacts["models"][model_name]
        n = model["n"]
        counts = model["counts"]
        contexts = model["contexts"]
        unigram_total = artifacts["unigram_total"]

        diag = perplexity_diagnostics(
            sequences=[mapped_tokens],
            n=n,
            ngram_counts=counts,
            context_counts=contexts,
            unigram_total=unigram_total,
        )

        ngram_events = generate_ngrams(mapped_tokens, n)
        rows: list[dict[str, Any]] = []
        for ng in ngram_events[:max_rows_per_model]:
            if n == 1:
                numerator = counts.get(ng, 0)
                denominator = unigram_total
                context_text = "(all tokens)"
            else:
                ctx = ng[:-1]
                numerator = counts.get(ng, 0)
                denominator = contexts.get(ctx, 0)
                context_text = " ".join(ctx)

            prob = (numerator / denominator) if numerator > 0 and denominator > 0 else 0.0
            rows.append(
                {
                    "ngram": " ".join(ng),
                    "context": context_text,
                    "ngram_count": int(numerator),
                    "context_count": int(denominator),
                    "probability": prob,
                }
            )

        results[model_name] = {
            "perplexity": float(diag["perplexity"]),
            "zero_prob_events": int(diag["zero_prob_events"]),
            "event_count": int(diag["events"]),
            "unseen_rate": float(diag["unseen_rate"]),
            "events_preview": rows,
            "events_total": len(ngram_events),
        }

    return {
        "raw_tokens": raw_tokens,
        "mapped_tokens": mapped_tokens,
        "unk_count": unk_count,
        "unk_rate": unk_count / len(mapped_tokens),
        "models": results,
    }


def render_baseline_section(artifacts: dict[str, Any]) -> None:
    st.subheader("Dataset and baseline")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows (clean)", artifacts["clean_stats"]["rows_after_tokenization"])
    col2.metric("Train rows", artifacts["train_rows"])
    col3.metric("Test rows", artifacts["test_rows"])
    col4.metric("Vocab (+<unk>)", artifacts["vocab_stats"]["effective_vocab_size_with_unk"])

    model_cols = st.columns(3)
    for idx, model_name in enumerate(("unigram", "bigram", "trigram")):
        baseline = artifacts["baseline"][model_name]
        with model_cols[idx]:
            st.markdown(f"**{model_name.capitalize()} baseline**")
            st.write(f"Train ppl: `{format_perplexity(baseline['train_perplexity'])}`")
            st.write(f"Test ppl: `{format_perplexity(baseline['test_perplexity'])}`")
            st.write(f"Zero-prob test events: `{baseline['zero_prob_events_test']}`")
            st.write(f"Unseen rate test: `{baseline['unseen_rate_test']:.4f}`")

    with st.expander("Show split by author"):
        left, right = st.columns(2)
        left.markdown("**Train**")
        left.dataframe(
            pd.DataFrame(
                {
                    "author": list(artifacts["train_author_counts"].keys()),
                    "count": list(artifacts["train_author_counts"].values()),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        right.markdown("**Test**")
        right.dataframe(
            pd.DataFrame(
                {
                    "author": list(artifacts["test_author_counts"].keys()),
                    "count": list(artifacts["test_author_counts"].values()),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_result_section(result: dict[str, Any]) -> None:
    st.subheader("Input analysis")
    st.write(f"Token count: `{len(result['raw_tokens'])}`")
    st.write(f"Mapped to <unk>: `{result['unk_count']}` (`{result['unk_rate']:.2%}`)")

    with st.expander("Show tokens"):
        st.markdown("**Raw tokens**")
        st.code(" ".join(result["raw_tokens"]))
        st.markdown("**Mapped tokens**")
        st.code(" ".join(result["mapped_tokens"]))

    tabs = st.tabs(["Unigram", "Bigram", "Trigram"])
    for tab, model_name in zip(tabs, ("unigram", "bigram", "trigram")):
        model_result = result["models"][model_name]
        with tab:
            c1, c2, c3 = st.columns(3)
            c1.metric("Perplexity", format_perplexity(model_result["perplexity"]))
            c2.metric("Zero-prob events", model_result["zero_prob_events"])
            c3.metric("Unseen rate", f"{model_result['unseen_rate']:.2%}")
            st.caption(
                f"Showing first {len(model_result['events_preview'])} events "
                f"of {model_result['events_total']} total."
            )
            st.dataframe(
                pd.DataFrame(model_result["events_preview"]),
                use_container_width=True,
                hide_index=True,
            )


def render_top_ngrams_from_csv() -> None:
    csv_path = Path("project2/task1/results/task1_top_ngrams.csv")
    if not csv_path.exists():
        return

    st.subheader("Top n-grams from latest run")
    top_df = pd.read_csv(csv_path)
    model_name = st.selectbox(
        "Model",
        options=["unigram", "bigram", "trigram"],
        index=0,
        key="top_ngram_model",
    )
    filtered = top_df[top_df["model"] == model_name].head(20)
    st.dataframe(filtered, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Task1 N-gram Playground", layout="wide")
    st.title("Project2 Task1: N-gram model playground")
    st.caption("Test custom text against your unsmoothed unigram, bigram, and trigram models.")

    with st.sidebar:
        st.header("Settings")
        input_path = st.text_input("Parquet path", value="project2/poems_translated.parquet")
        text_col = st.text_input("Text column", value="modern_text")
        author_col = st.text_input("Author column", value="author")
        test_size = st.slider("Test size", min_value=0.05, max_value=0.40, value=0.20, step=0.05)
        seed = st.number_input("Seed", min_value=0, max_value=99999, value=42, step=1)
        min_freq = st.number_input("Min freq for <unk>", min_value=1, max_value=20, value=2, step=1)
        preview_rows = st.slider("Events preview per model", min_value=5, max_value=100, value=25, step=5)

        st.markdown("---")
        st.markdown("**Run command**")
        st.code("streamlit run project2/task1/task1_streamlit.py")

    try:
        artifacts = build_artifacts(
            input_path=input_path,
            text_col=text_col,
            author_col=author_col,
            test_size=float(test_size),
            seed=int(seed),
            min_freq=int(min_freq),
        )
    except Exception as exc:
        st.error(f"Failed to build model artifacts: {exc}")
        st.stop()

    render_baseline_section(artifacts)
    render_top_ngrams_from_csv()

    st.subheader("Try your own text")
    sample_text = st.text_area(
        "Input text",
        value="Yene bir teze gozele ashiq oldum",
        height=150,
        help="Paste any text. The app tokenizes it and computes model diagnostics.",
    )

    if st.button("Analyze text", type="primary"):
        try:
            result = evaluate_text(sample_text, artifacts, max_rows_per_model=int(preview_rows))
        except Exception as exc:
            st.error(f"Could not analyze text: {exc}")
        else:
            render_result_section(result)

    with st.expander("Model details"):
        st.write(
            {
                "tokenizer": r"\b\w+\b + lowercase",
                "smoothing": "none (MLE)",
                "boundary_tokens": [START_TOKEN, END_TOKEN],
                "unk_token": UNK_TOKEN,
            }
        )


if __name__ == "__main__":
    main()
