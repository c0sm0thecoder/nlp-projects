from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project2.task4.sentence_boundary_core import (  # noqa: E402
    FEATURE_NAMES,
    PENALTIES,
    C_GRID,
    build_task4_artifacts,
    evaluate_model,
    extract_dot_samples,
    enrich_with_corpus_features,
    build_word_freq_from_texts,
    feature_importance_table,
    mcnemar_test,
    reconstruct_sentences,
    train_logistic,
)


# Load and cache model artifacts


@st.cache_resource(show_spinner=True)
def load_artifacts(
    input_path: str,
    text_col: str,
    label_col: str,
    val_size: float,
    test_size: float,
    seed: int,
    c_grid_str: str,
) -> dict[str, Any]:
    c_grid = [float(x.strip()) for x in c_grid_str.split(",") if x.strip()]
    return build_task4_artifacts(
        input_path=input_path,
        text_col=text_col,
        label_col=label_col,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        c_grid=c_grid,
    )


# UI sections for displaying results


def render_data_summary(artifacts: dict[str, Any]) -> None:
    st.subheader("Dataset summary")
    ds = artifacts["dataset_stats"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total texts", ds["total_texts"])
    c2.metric("Total dots", ds["total_dots"])
    c3.metric("Boundary dots", ds["positive_dots"])
    c4.metric("Non-boundary dots", ds["negative_dots"])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Positive rate", f"{ds['positive_rate']:.2%}")
    c6.metric("Train dots", ds["train_dots"])
    c7.metric("Val dots", ds["val_dots"])
    c8.metric("Test dots", ds["test_dots"])

    st.caption(
        f"Train pos rate: {ds['train_positive_rate']:.2%} | "
        f"Val pos rate: {ds['val_positive_rate']:.2%} | "
        f"Test pos rate: {ds['test_positive_rate']:.2%}"
    )


def render_comparison(artifacts: dict[str, Any]) -> None:
    st.subheader("L1 vs L2 comparison")

    rows = []
    for penalty in PENALTIES:
        r = artifacts["results"][penalty]
        tr = r["train_metrics"]
        va = r["val_metrics"]
        te = r["test_metrics"]
        rows.append({
            "Penalty": penalty.upper(),
            "Best C": r["best_C"],
            "Train F1": f"{tr['f1']:.4f}",
            "Val F1": f"{va['f1']:.4f}",
            "Test F1": f"{te['f1']:.4f}",
            "Test Acc": f"{te['accuracy']:.4f}",
            "Test P": f"{te['precision']:.4f}",
            "Test R": f"{te['recall']:.4f}",
            "Non-zero coefs": r["n_nonzero_coefs"],
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    best = artifacts["best_penalty"]
    st.success(f"Best penalty: **{best.upper()}** (by test F1)")

    # Overfitting check
    for penalty in PENALTIES:
        r = artifacts["results"][penalty]
        train_f1 = r["train_metrics"]["f1"]
        test_f1 = r["test_metrics"]["f1"]
        gap = train_f1 - test_f1
        if gap > 0.05:
            st.warning(
                f"{penalty.upper()}: train-test F1 gap = {gap:.4f} "
                f"(train={train_f1:.4f}, test={test_f1:.4f}) — possible overfitting."
            )


def render_confusion_matrices(artifacts: dict[str, Any]) -> None:
    st.subheader("Confusion matrices")
    labels = ["not_boundary", "boundary"]

    tabs = st.tabs(["L1", "L2"])
    for tab, penalty in zip(tabs, PENALTIES):
        with tab:
            r = artifacts["results"][penalty]
            for split_name in ("train_metrics", "val_metrics", "test_metrics"):
                nice = split_name.replace("_metrics", "").capitalize()
                cm = r[split_name]["confusion_matrix"]
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                m = r[split_name]
                st.markdown(
                    f"**{nice}:** acc={m['accuracy']:.4f} "
                    f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}"
                )
                st.dataframe(cm_df, use_container_width=True)


def render_feature_importance(artifacts: dict[str, Any]) -> None:
    st.subheader("Feature importance (by |coefficient|)")

    tabs = st.tabs(["L1", "L2"])
    for tab, penalty in zip(tabs, PENALTIES):
        with tab:
            fi = artifacts["results"][penalty]["feature_importance"]
            fi_df = pd.DataFrame(fi)[["rank", "feature", "coefficient", "abs_coefficient"]]
            fi_df["coefficient"] = fi_df["coefficient"].map(lambda x: f"{x:.6f}")
            fi_df["abs_coefficient"] = fi_df["abs_coefficient"].map(lambda x: f"{x:.6f}")
            st.dataframe(fi_df, use_container_width=True, hide_index=True)


def render_tuning(artifacts: dict[str, Any]) -> None:
    st.subheader("C tuning results (on validation set)")
    rows = artifacts["tuning_rows"]
    if not rows:
        st.info("No tuning results available.")
        return

    tuning_df = pd.DataFrame(rows)
    for col in ("train_accuracy", "train_f1", "val_accuracy", "val_f1"):
        tuning_df[col] = tuning_df[col].map(lambda x: f"{x:.4f}")
    st.dataframe(tuning_df, use_container_width=True, hide_index=True)


def render_significance(artifacts: dict[str, Any]) -> None:
    st.subheader("Statistical significance (McNemar)")
    sig = artifacts["significance"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Chi-squared", f"{sig['statistic']:.4f}")
    col2.metric("p-value", f"{sig['p_value']:.6f}")
    col3.metric("Significant (alpha=0.05)", "Yes" if sig["significant_0.05"] else "No")

    st.caption(
        f"b (L1 correct, L2 wrong) = {sig['b']} | "
        f"c (L1 wrong, L2 correct) = {sig['c']}"
    )

    if sig["significant_0.05"]:
        st.info("The difference between L1 and L2 is statistically significant at alpha = 0.05.")
    else:
        st.warning("No statistically significant difference between L1 and L2 at alpha = 0.05.")


def render_sentence_demo(artifacts: dict[str, Any]) -> None:
    st.subheader("Sentence detection demo")

    text = st.text_area(
        "Input text",
        value=(
            "Günəş doğdu. Göy üzü açıq idi. "
            "Bağda quşlar oxuyurdu.\n"
            "Mən 3.14 ədədini bildim. Nəticə yaxşıdır."
        ),
        height=120,
    )

    penalty = st.radio("Penalty", ["l1", "l2"], horizontal=True)

    if st.button("Detect sentences", key="detect_sents"):
        model = artifacts["models"][penalty]
        scaler = artifacts["scaler"]
        samples = extract_dot_samples(text, text_id=0)

        if not samples:
            st.warning("No dots found in the input text.")
            return

        sample_df = pd.DataFrame(samples)
        # Enrich with corpus features from training data
        source_df = artifacts["source_df"]
        train_text_ids = set(artifacts["train_dot_df"]["text_id"].unique())
        train_texts = [
            str(source_df.loc[tid, artifacts["config"]["text_col"]])
            for tid in train_text_ids
            if tid in source_df.index
        ]
        word_freq, total_words = build_word_freq_from_texts(train_texts)
        enrich_with_corpus_features(sample_df, word_freq, total_words)

        X = sample_df[FEATURE_NAMES].values.astype(np.float64)
        X = scaler.transform(X)
        y_pred = model.predict(X)

        dot_positions = sample_df["dot_pos"].tolist()
        sentences = reconstruct_sentences(text, dot_positions, y_pred.tolist())

        st.markdown("**Detected sentences:**")
        for i, sent in enumerate(sentences, 1):
            st.write(f"{i}. {sent}")

        with st.expander("Dot-level predictions"):
            pred_rows = []
            for ii, row in sample_df.iterrows():
                idx = row.name
                pred_rows.append({
                    "dot_pos": int(row["dot_pos"]),
                    "context": text[max(0, int(row["dot_pos"]) - 15): int(row["dot_pos"]) + 15],
                    "predicted": "boundary" if y_pred[idx] == 1 else "not_boundary",
                    "label (heuristic)": "boundary" if row["label"] == 1 else "not_boundary",
                })
            st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)


# Main app


def main() -> None:
    st.set_page_config(page_title="Task4 – Sentence Boundary Detection", layout="wide")
    st.title("Project2 Task4: Sentence Boundary Detection")
    st.caption(
        "Logistic Regression with L1 vs L2 regularization — "
        "train / val / test split with C tuning on validation."
    )

    with st.sidebar:
        st.header("Data settings")
        input_path = st.text_input("Parquet path", value="poems_translated.parquet")
        text_col = st.text_input("Text column", value="modern_text")
        label_col = st.text_input("Label column", value="author")
        val_size = st.slider("Val size", 0.05, 0.30, 0.10, 0.05)
        test_size = st.slider("Test size", 0.05, 0.40, 0.20, 0.05)
        seed = st.number_input("Seed", 0, 99999, 42, 1)
        c_grid_str = st.text_input(
            "C grid (comma-sep)", value="0.001,0.01,0.1,0.5,1.0,5.0,10.0"
        )

        st.markdown("---")
        st.markdown("**Run command**")
        st.code("streamlit run project2/task4/task4_streamlit.py")

    try:
        artifacts = load_artifacts(
            input_path=input_path,
            text_col=text_col,
            label_col=label_col,
            val_size=float(val_size),
            test_size=float(test_size),
            seed=int(seed),
            c_grid_str=c_grid_str,
        )
    except Exception as exc:
        st.error(f"Could not build artifacts: {exc}")
        st.stop()

    render_data_summary(artifacts)
    render_comparison(artifacts)
    render_tuning(artifacts)
    render_confusion_matrices(artifacts)
    render_feature_importance(artifacts)
    render_significance(artifacts)
    render_sentence_demo(artifacts)


if __name__ == "__main__":
    main()
