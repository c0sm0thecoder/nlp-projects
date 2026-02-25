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

from project2.task3.classification_core import (  # noqa: E402
    CLASSIFIERS,
    FEATURE_SETS,
    build_task3_artifacts,
    extract_features,
    evaluate_predictions,
    mcnemar_test,
    run_single_experiment,
    train_classifier,
    predict,
)


# Load model artifacts (cached)


@st.cache_resource(show_spinner=True)
def load_artifacts(
    dataset: str,
    input_path: str,
    text_col: str,
    label_col: str,
    test_size: float,
    seed: int,
    max_features: int | None,
) -> dict[str, Any]:
    return build_task3_artifacts(
        dataset=dataset,
        input_path=input_path,
        text_col=text_col,
        label_col=label_col,
        test_size=test_size,
        seed=seed,
        max_features=max_features,
    )


# Display functions


def render_data_summary(artifacts: dict[str, Any]) -> None:
    st.subheader("Dataset & split summary")
    cs = artifacts["clean_stats"]
    sp = artifacts["split"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (raw)", cs["rows_input"])
    c2.metric("Rows (clean)", cs["rows_after_cleaning"])
    c3.metric("Train rows", sp["train_rows"])
    c4.metric("Test rows", sp["test_rows"])

    with st.expander("Label distribution"):
        left, right = st.columns(2)
        left.markdown("**Train**")
        left.dataframe(
            pd.DataFrame({
                "label": list(sp["train_label_counts"].keys()),
                "count": list(sp["train_label_counts"].values()),
            }),
            use_container_width=True,
            hide_index=True,
        )
        right.markdown("**Test**")
        right.dataframe(
            pd.DataFrame({
                "label": list(sp["test_label_counts"].keys()),
                "count": list(sp["test_label_counts"].values()),
            }),
            use_container_width=True,
            hide_index=True,
        )


def render_results_table(artifacts: dict[str, Any]) -> None:
    st.subheader("Classification results (all experiments)")

    rows = []
    for row in artifacts["summary_rows"]:
        rows.append({
            "rank": row["rank"],
            "classifier": row["classifier"],
            "feature_set": row["feature_set"],
            "n_features": row["n_features"],
            "accuracy": f"{row['accuracy']:.4f}",
            "macro_P": f"{row['macro_precision']:.4f}",
            "macro_R": f"{row['macro_recall']:.4f}",
            "macro_F1": f"{row['macro_f1']:.4f}",
            "weighted_F1": f"{row['weighted_f1']:.4f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    best = artifacts["best"]
    st.success(
        f"Best: **{best['classifier']}** with **{best['feature_set']}** features "
        f"(macro F1 = {best['macro_f1']:.4f}, accuracy = {best['accuracy']:.4f})"
    )

    st.markdown("### Best algorithm by classifier")
    clf_rows = []
    for row in artifacts.get("classifier_analysis", []):
        clf_rows.append({
            "classifier": row["classifier"],
            "best_feature_set": row["best_feature_set"],
            "macro_F1": f"{row['macro_f1']:.4f}",
            "accuracy": f"{row['accuracy']:.4f}",
        })
    if clf_rows:
        st.dataframe(pd.DataFrame(clf_rows), use_container_width=True, hide_index=True)
        best_clf = artifacts["best_classifier"]
        st.info(
            f"Overall best algorithm: **{best_clf['classifier']}** "
            f"(with **{best_clf['best_feature_set']}** features, "
            f"macro F1 = {best_clf['macro_f1']:.4f})"
        )


def render_confusion_matrices(artifacts: dict[str, Any]) -> None:
    st.subheader("Confusion matrices")
    labels = artifacts["split"]["label_names"]

    for exp in artifacts["experiment_results"]:
        key = f"{exp['classifier']} + {exp['feature_set']}"
        with st.expander(key):
            cm = exp["metrics"]["confusion_matrix"]
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            st.dataframe(cm_df, use_container_width=True)


def render_significance(artifacts: dict[str, Any]) -> None:
    st.subheader("Statistical significance (McNemar's test)")
    st.caption(
        "McNemar's test compares paired classifier predictions. "
        "p < 0.05 ⟹ statistically significant difference."
    )

    sig = artifacts["significance_results"]
    if not sig:
        st.info("No significance results available.")
        return

    rows = []
    for r in sig:
        rows.append({
            "Classifier A": r["classifier_a"],
            "Classifier B": r["classifier_b"],
            "χ² statistic": f"{r['statistic']:.4f}",
            "p-value": f"{r['p_value']:.6f}",
            "b (A✓ B✗)": r["b"],
            "c (A✗ B✓)": r["c"],
            "significant (α=0.05)": "Yes" if r["significant_0.05"] else "No",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Highlight notable findings
    sig_pairs = [r for r in sig if r["significant_0.05"]]
    if sig_pairs:
        st.info(f"{len(sig_pairs)} / {len(sig)} classifier pairs show a statistically significant difference.")
    else:
        st.warning("No statistically significant differences found between any classifier pairs at α=0.05.")


def render_per_class_report(artifacts: dict[str, Any]) -> None:
    st.subheader("Per-class classification reports")

    for exp in artifacts["experiment_results"]:
        key = f"{exp['classifier']} + {exp['feature_set']}"
        with st.expander(key):
            report = exp["metrics"]["classification_report"]
            report_rows = []
            for label in exp["metrics"]["label_names"]:
                if label in report:
                    r = report[label]
                    report_rows.append({
                        "class": label,
                        "precision": f"{r['precision']:.4f}",
                        "recall": f"{r['recall']:.4f}",
                        "f1-score": f"{r['f1-score']:.4f}",
                        "support": int(r["support"]),
                    })
            if report_rows:
                st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)


def render_single_experiment(artifacts: dict[str, Any]) -> None:
    st.subheader("Single experiment playground")

    col1, col2 = st.columns(2)
    with col1:
        clf = st.selectbox("Classifier", list(CLASSIFIERS), index=0)
    with col2:
        feat = st.selectbox("Feature set", list(FEATURE_SETS), index=0)

    if st.button("Run experiment", key="run_exp"):
        result = run_single_experiment(
            classifier_name=clf,
            feature_set=feat,
            train_texts=artifacts["train_texts"],
            test_texts=artifacts["test_texts"],
            y_train=artifacts["y_train"],
            y_test=artifacts["y_test"],
            label_names=artifacts["label_names"],
            seed=artifacts["config"]["seed"],
            max_features=artifacts["config"]["max_features"],
        )
        m = result["metrics"]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Accuracy", f"{m['accuracy']:.4f}")
        r2.metric("Macro P", f"{m['macro_precision']:.4f}")
        r3.metric("Macro R", f"{m['macro_recall']:.4f}")
        r4.metric("Macro F1", f"{m['macro_f1']:.4f}")

        st.markdown("**Confusion matrix**")
        labels = m["label_names"]
        cm_df = pd.DataFrame(m["confusion_matrix"], index=labels, columns=labels)
        st.dataframe(cm_df, use_container_width=True)


# Main app


def main() -> None:
    st.set_page_config(page_title="Task3 – Text Classification", layout="wide")
    st.title("Project2 Task3: Text Classification")
    st.caption(
        "Naive Bayes · Binary Naive Bayes · Logistic Regression | "
        "BoW · Sentiment Lexicon · Combined features | McNemar's significance test"
    )

    with st.sidebar:
        st.header("Data settings")
        dataset = st.selectbox("Dataset", ["twitter_samples", "parquet"], index=0)
        input_path = st.text_input(
            "Parquet path",
            value="poems_translated.parquet",
            disabled=(dataset != "parquet"),
        )
        default_text_col = "text" if dataset == "twitter_samples" else "modern_text"
        default_label_col = "label" if dataset == "twitter_samples" else "author"
        text_col = st.text_input("Text column", value=default_text_col)
        label_col = st.text_input("Label column", value=default_label_col)
        test_size = st.slider("Test size", 0.05, 0.40, 0.20, 0.05)
        seed = st.number_input("Seed", 0, 99999, 42, 1)
        max_features_raw = st.number_input(
            "Max BoW features (0 = unlimited)", 0, 100000, 0, 100,
        )
        max_features = int(max_features_raw) if max_features_raw > 0 else None

        st.markdown("---")
        st.markdown("**Run command**")
        st.code("streamlit run project2/task3/task3_streamlit.py")

    try:
        artifacts = load_artifacts(
            dataset=dataset,
            input_path=input_path,
            text_col=text_col,
            label_col=label_col,
            test_size=float(test_size),
            seed=int(seed),
            max_features=max_features,
        )
    except Exception as exc:
        st.error(f"Could not build artifacts: {exc}")
        st.stop()

    render_data_summary(artifacts)
    render_results_table(artifacts)
    render_confusion_matrices(artifacts)
    render_per_class_report(artifacts)
    render_significance(artifacts)
    render_single_experiment(artifacts)


if __name__ == "__main__":
    main()
