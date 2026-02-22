from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project2.task2.smoothing_core import (  # noqa: E402
    METHODS,
    ORDERS,
    RANKING_RULES,
    build_task2_artifacts,
    evaluate_custom_text_with_models,
    evaluate_manual_params,
    map_text_to_tokens,
    parse_float_grid,
    rank_methods,
)


def format_perplexity(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


@st.cache_resource(show_spinner=True)
def load_artifacts(
    input_path: str,
    text_col: str,
    author_col: str,
    test_size: float,
    val_size: float,
    seed: int,
    min_freq: int,
    interp_bigram_grid_raw: str,
    interp_trigram_step: float,
    discount_grid_raw: str,
) -> dict[str, Any]:
    interp_bigram_grid = parse_float_grid(interp_bigram_grid_raw, "--interp-bigram-grid")
    discount_grid = parse_float_grid(discount_grid_raw, "--discount-grid")

    return build_task2_artifacts(
        input_path=input_path,
        text_col=text_col,
        author_col=author_col,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        min_freq=min_freq,
        interp_bigram_grid=interp_bigram_grid,
        interp_trigram_step=interp_trigram_step,
        discount_grid=discount_grid,
    )


def render_summary(artifacts: dict[str, Any]) -> None:
    metrics = artifacts["metrics"]
    corpus = metrics["corpus"]
    split = metrics["split"]

    st.subheader("Dataset and split summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows (after tokenize)", corpus["rows_after_tokenization"])
    col2.metric("Outer train rows", split["outer_train_rows"])
    col3.metric("Inner val rows", split["val_rows"])
    col4.metric("Test rows", split["test_rows"])

    with st.expander("Show author distribution"):
        left, right = st.columns(2)
        left.markdown("**Inner Train**")
        left.dataframe(
            pd.DataFrame(
                {
                    "author": list(split["inner_train_author_counts"].keys()),
                    "count": list(split["inner_train_author_counts"].values()),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        right.markdown("**Test**")
        right.dataframe(
            pd.DataFrame(
                {
                    "author": list(split["test_author_counts"].keys()),
                    "count": list(split["test_author_counts"].values()),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_method_comparison(artifacts: dict[str, Any]) -> None:
    st.subheader("Method comparison (best tuned params)")
    rows = []
    for row in artifacts["method_rows"]:
        rows.append(
            {
                "method": row["method"],
                "order": row["order"],
                "best_params": json.dumps(row["best_params"], ensure_ascii=False, sort_keys=True),
                "train_perplexity": format_perplexity(float(row["train_perplexity"])),
                "val_perplexity": format_perplexity(float(row["val_perplexity"])),
                "test_perplexity": format_perplexity(float(row["test_perplexity"])),
                "zero_prob_events_test": row["zero_prob_events_test"],
                "unseen_rate_test": f"{float(row['unseen_rate_test']):.2%}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_ranking_panel(artifacts: dict[str, Any], ranking_rule: str) -> None:
    st.subheader("Ranking panel")
    ranked = rank_methods(artifacts["method_rows"], ranking_rule)

    best = ranked[0]
    st.info(
        f"Best method for `{ranking_rule}`: `{best['method']}` "
        f"(score={format_perplexity(float(best['score']))})"
    )
    ranking_df = pd.DataFrame(ranked)[
        ["rank", "method", "score", "bigram_test_perplexity", "trigram_test_perplexity"]
    ]
    ranking_df["score"] = ranking_df["score"].map(lambda x: format_perplexity(float(x)))
    ranking_df["bigram_test_perplexity"] = ranking_df["bigram_test_perplexity"].map(
        lambda x: format_perplexity(float(x))
    )
    ranking_df["trigram_test_perplexity"] = ranking_df["trigram_test_perplexity"].map(
        lambda x: format_perplexity(float(x))
    )
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)


def manual_params_ui(method: str, order: int, default_params: dict[str, float]) -> dict[str, float]:
    if method == "laplace":
        st.caption("Laplace uses fixed alpha=1.0.")
        return {"alpha": 1.0}

    if method == "interpolation" and order == 2:
        default_lambda2 = float(default_params.get("lambda2", 0.7))
        lambda2 = st.slider("lambda2 (bigram weight)", 0.0, 1.0, default_lambda2, 0.01)
        return {"lambda1": 1.0 - lambda2, "lambda2": lambda2}

    if method == "interpolation" and order == 3:
        default_l1 = float(default_params.get("lambda1", 0.2))
        default_l2 = float(default_params.get("lambda2", 0.3))

        lambda1 = st.slider("lambda1 (unigram)", 0.0, 1.0, default_l1, 0.01)
        lambda2_max = max(0.0, 1.0 - lambda1)
        lambda2 = st.slider(
            "lambda2 (bigram)",
            0.0,
            float(lambda2_max),
            float(min(default_l2, lambda2_max)),
            0.01,
        )
        lambda3 = max(0.0, 1.0 - lambda1 - lambda2)
        st.write(f"lambda3 (trigram) = `{lambda3:.2f}`")
        return {"lambda1": lambda1, "lambda2": lambda2, "lambda3": lambda3}

    default_discount = float(default_params.get("discount", 0.75))
    discount = st.slider("discount", 0.05, 1.5, default_discount, 0.01)
    return {"discount": discount}


def render_tuning_playground(artifacts: dict[str, Any]) -> None:
    st.subheader("Tuning playground")
    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox("Method", options=list(METHODS), index=0)
    with col2:
        order = st.selectbox("Order", options=list(ORDERS), index=1)

    default_params = artifacts["best_by_method_order"][(method, order)]["params"]
    st.caption(f"Default tuned params: `{json.dumps(default_params, ensure_ascii=False, sort_keys=True)}`")
    params = manual_params_ui(method, int(order), default_params)

    if st.button("Evaluate parameter set", key="eval_param_set"):
        result = evaluate_manual_params(
            method=method,
            order=int(order),
            params=params,
            inner_train_sequences=artifacts["inner_train_sequences"],
            val_sequences=artifacts["val_sequences"],
            final_train_sequences=artifacts["final_train_sequences"],
            test_sequences=artifacts["test_sequences"],
        )
        r1, r2, r3 = st.columns(3)
        r1.metric("Train perplexity", format_perplexity(float(result["train_perplexity"])))
        r2.metric("Validation perplexity", format_perplexity(float(result["val_perplexity"])))
        r3.metric("Test perplexity", format_perplexity(float(result["test_perplexity"])))
        st.write(
            {
                "zero_prob_events_test": result["zero_prob_events_test"],
                "unseen_rate_test": result["unseen_rate_test"],
                "params": result["params"],
            }
        )


def render_text_probe(artifacts: dict[str, Any]) -> None:
    st.subheader("Custom text probe")
    text = st.text_area(
        "Input text",
        value="Yenə bir təzə gözələ aşiq oldum",
        height=130,
        help="Text is tokenized and mapped to Task2 final training vocabulary (<unk> for OOV).",
    )
    if st.button("Analyze text", key="analyze_text"):
        mapped = map_text_to_tokens(text, artifacts["retained_vocab_final"])
        st.write(
            f"Token count: `{len(mapped['raw_tokens'])}`, "
            f"<unk> count: `{mapped['unk_count']}` ({mapped['unk_rate']:.2%})"
        )

        with st.expander("Show token mapping"):
            st.markdown("**Raw tokens**")
            st.code(" ".join(mapped["raw_tokens"]))
            st.markdown("**Mapped tokens**")
            st.code(" ".join(mapped["mapped_tokens"]))

        rows = evaluate_custom_text_with_models(
            mapped_tokens=mapped["mapped_tokens"],
            models=artifacts["final_models"],
        )
        result_df = pd.DataFrame(rows)
        result_df["perplexity"] = result_df["perplexity"].map(lambda x: format_perplexity(float(x)))
        result_df["unseen_rate"] = result_df["unseen_rate"].map(lambda x: f"{float(x):.2%}")

        tabs = st.tabs(["Bigram", "Trigram"])
        for tab, order in zip(tabs, [2, 3]):
            with tab:
                st.dataframe(
                    result_df[result_df["order"] == order][
                        ["method", "order", "perplexity", "zero_prob_events", "unseen_rate", "event_count"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


def main() -> None:
    st.set_page_config(page_title="Task2 Smoothing Playground", layout="wide")
    st.title("Project2 Task2: Smoothing methods playground")
    st.caption("Laplace, Interpolation, Backoff, and Kneser-Ney on bigram/trigram language models.")

    with st.sidebar:
        st.header("Data + split settings")
        input_path = st.text_input("Parquet path", value="poems_translated.parquet")
        text_col = st.text_input("Text column", value="modern_text")
        author_col = st.text_input("Author column", value="author")
        test_size = st.slider("Test size", min_value=0.05, max_value=0.40, value=0.20, step=0.05)
        val_size = st.slider("Validation size", min_value=0.05, max_value=0.40, value=0.10, step=0.05)
        seed = st.number_input("Seed", min_value=0, max_value=99999, value=42, step=1)
        min_freq = st.number_input("Min freq for <unk>", min_value=1, max_value=20, value=2, step=1)

        st.header("Tuning grids")
        interp_bigram_grid = st.text_input("Interp bigram grid", value="0.5,0.6,0.7,0.8,0.9")
        interp_trigram_step = st.slider(
            "Interp trigram step", min_value=0.05, max_value=0.50, value=0.10, step=0.05
        )
        discount_grid = st.text_input("Discount grid", value="0.5,0.75,1.0")

        st.header("Ranking")
        ranking_rule = st.selectbox("Ranking rule", options=list(RANKING_RULES), index=0)

        st.markdown("---")
        st.markdown("**Run command**")
        st.code("streamlit run project2/task2/task2_streamlit.py")

    try:
        artifacts = load_artifacts(
            input_path=input_path,
            text_col=text_col,
            author_col=author_col,
            test_size=float(test_size),
            val_size=float(val_size),
            seed=int(seed),
            min_freq=int(min_freq),
            interp_bigram_grid_raw=interp_bigram_grid,
            interp_trigram_step=float(interp_trigram_step),
            discount_grid_raw=discount_grid,
        )
    except Exception as exc:
        st.error(f"Could not build artifacts: {exc}")
        st.stop()

    render_summary(artifacts)
    render_method_comparison(artifacts)
    render_ranking_panel(artifacts, ranking_rule)
    render_tuning_playground(artifacts)
    render_text_probe(artifacts)


if __name__ == "__main__":
    main()
