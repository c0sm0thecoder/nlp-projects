from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path, *, required_columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV file: {path}")
    df = pd.read_csv(path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
    return df


def _final_training_rows(task2_metrics: pd.DataFrame, task3_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for model_name in ["skipgram", "cbow"]:
        model_df = task2_metrics[task2_metrics["model"] == model_name]
        if model_df.empty:
            continue
        final_row = model_df.sort_values("epoch").iloc[-1]
        rows.append(
            {
                "model": model_name,
                "objective": "negative_sampling",
                "final_loss": float(final_row["avg_loss"]),
                "rows_seen": int(final_row["examples_seen"]),
            }
        )

    glove_df = task3_metrics[task3_metrics["model"] == "glove"]
    if not glove_df.empty:
        final_row = glove_df.sort_values("epoch").iloc[-1]
        rows.append(
            {
                "model": "glove",
                "objective": "global_cooccurrence",
                "final_loss": float(final_row["avg_weighted_loss"]),
                "rows_seen": int(final_row["pairs_seen"]),
            }
        )

    return pd.DataFrame(rows, columns=["model", "objective", "final_loss", "rows_seen"])


def _compare_neighbors(task2_neighbors: pd.DataFrame, task3_neighbors: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    glove_top = task3_neighbors[task3_neighbors["model"] == "glove"]
    glove_top = glove_top[glove_top["rank"] <= top_k]

    rows: list[dict[str, object]] = []
    for word2vec_model in ["skipgram", "cbow"]:
        w2v_top = task2_neighbors[task2_neighbors["model"] == word2vec_model]
        w2v_top = w2v_top[w2v_top["rank"] <= top_k]

        query_words = sorted(set(glove_top["query_word"].astype(str)) | set(w2v_top["query_word"].astype(str)))
        for query_word in query_words:
            glove_set = set(
                glove_top[glove_top["query_word"] == query_word]["similar_word"].astype(str).tolist()
            )
            w2v_set = set(w2v_top[w2v_top["query_word"] == query_word]["similar_word"].astype(str).tolist())

            intersection = glove_set & w2v_set
            union = glove_set | w2v_set
            jaccard = (len(intersection) / len(union)) if union else 0.0

            rows.append(
                {
                    "word2vec_model": word2vec_model,
                    "query_word": query_word,
                    "top_k": top_k,
                    "glove_neighbors": len(glove_set),
                    "word2vec_neighbors": len(w2v_set),
                    "overlap_count": len(intersection),
                    "jaccard_similarity": float(jaccard),
                    "overlap_words": "; ".join(sorted(intersection)),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "word2vec_model",
            "query_word",
            "top_k",
            "glove_neighbors",
            "word2vec_neighbors",
            "overlap_count",
            "jaccard_similarity",
            "overlap_words",
        ],
    )


def _compare_equations(task2_equations: pd.DataFrame, task3_equations: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    w2v_top = task2_equations[task2_equations["rank"] <= top_k]
    glove_top = task3_equations[(task3_equations["model"] == "glove") & (task3_equations["rank"] <= top_k)]

    rows: list[dict[str, object]] = []
    for word2vec_model in ["skipgram", "cbow"]:
        model_eq = w2v_top[w2v_top["model"] == word2vec_model]
        query_words = sorted(set(model_eq["add_word"].astype(str)) | set(glove_top["add_word"].astype(str)))

        for query_word in query_words:
            glove_set = set(glove_top[glove_top["add_word"] == query_word]["result_word"].astype(str).tolist())
            w2v_set = set(model_eq[model_eq["add_word"] == query_word]["result_word"].astype(str).tolist())

            intersection = glove_set & w2v_set
            union = glove_set | w2v_set
            jaccard = (len(intersection) / len(union)) if union else 0.0

            rows.append(
                {
                    "word2vec_model": word2vec_model,
                    "query_word": query_word,
                    "top_k": top_k,
                    "glove_equation_results": len(glove_set),
                    "word2vec_equation_results": len(w2v_set),
                    "overlap_count": len(intersection),
                    "jaccard_similarity": float(jaccard),
                    "overlap_words": "; ".join(sorted(intersection)),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "word2vec_model",
            "query_word",
            "top_k",
            "glove_equation_results",
            "word2vec_equation_results",
            "overlap_count",
            "jaccard_similarity",
            "overlap_words",
        ],
    )


def _build_model_comparison(
    task2_config: dict[str, object],
    task3_config: dict[str, object],
    task2_neighbors: pd.DataFrame,
    task3_neighbors: pd.DataFrame,
    task2_equations: pd.DataFrame,
    task3_equations: pd.DataFrame,
    final_losses: pd.DataFrame,
) -> pd.DataFrame:
    loss_map = {
        str(row["model"]): {
            "final_loss": float(row["final_loss"]),
            "objective": str(row["objective"]),
            "rows_seen": int(row["rows_seen"]),
        }
        for _, row in final_losses.iterrows()
    }

    rows = []
    for model_name in ["skipgram", "cbow", "glove"]:
        if model_name in {"skipgram", "cbow"}:
            cfg = task2_config
            neighbors_count = int((task2_neighbors["model"] == model_name).sum())
            equations_count = int((task2_equations["model"] == model_name).sum())
        else:
            cfg = task3_config
            neighbors_count = int((task3_neighbors["model"] == "glove").sum())
            equations_count = int((task3_equations["model"] == "glove").sum())

        loss_entry = loss_map.get(model_name, {})
        rows.append(
            {
                "model": model_name,
                "embedding_dim": int(cfg.get("embedding_dim", 0)),
                "epochs": int(cfg.get("epochs", 0)),
                "vocab_size": int(cfg.get("vocab_size", 0)),
                "neighbor_rows": neighbors_count,
                "equation_rows": equations_count,
                "objective": loss_entry.get("objective", "unknown"),
                "final_loss": float(loss_entry.get("final_loss", 0.0)),
                "rows_seen": int(loss_entry.get("rows_seen", 0)),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "embedding_dim",
            "epochs",
            "vocab_size",
            "neighbor_rows",
            "equation_rows",
            "objective",
            "final_loss",
            "rows_seen",
        ],
    )


def _write_summary_markdown(
    path: Path,
    *,
    config: dict[str, object],
    model_comparison: pd.DataFrame,
    neighbor_overlap: pd.DataFrame,
    equation_overlap: pd.DataFrame,
) -> None:
    lines = [
        "# Task 4 Comparison Summary",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
    ]

    for key in [
        "task2_results_dir",
        "task3_results_dir",
        "neighbor_top_k",
        "equation_top_k",
        "query_word_count",
    ]:
        lines.append(f"| {key} | {config[key]} |")

    lines.extend(
        [
            "",
            "## Model Snapshot",
            "",
            "| Model | Objective | Embedding Dim | Epochs | Vocab Size | Final Loss |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in model_comparison.iterrows():
        lines.append(
            f"| {row['model']} | {row['objective']} | {int(row['embedding_dim'])} | {int(row['epochs'])} | {int(row['vocab_size'])} | {float(row['final_loss']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Neighbor Overlap (Word2Vec vs GloVe)",
            "",
            "| Word2Vec Model | Avg Jaccard | Avg Overlap Count |",
            "| --- | ---: | ---: |",
        ]
    )
    if not neighbor_overlap.empty:
        grouped = (
            neighbor_overlap.groupby("word2vec_model", as_index=False)
            .agg(avg_jaccard=("jaccard_similarity", "mean"), avg_overlap=("overlap_count", "mean"))
            .sort_values("word2vec_model")
        )
        for _, row in grouped.iterrows():
            lines.append(
                f"| {row['word2vec_model']} | {float(row['avg_jaccard']):.4f} | {float(row['avg_overlap']):.2f} |"
            )

    lines.extend(
        [
            "",
            "## Vector Equation Overlap (Word2Vec vs GloVe)",
            "",
            "| Word2Vec Model | Avg Jaccard | Avg Overlap Count |",
            "| --- | ---: | ---: |",
        ]
    )
    if not equation_overlap.empty:
        grouped = (
            equation_overlap.groupby("word2vec_model", as_index=False)
            .agg(avg_jaccard=("jaccard_similarity", "mean"), avg_overlap=("overlap_count", "mean"))
            .sort_values("word2vec_model")
        )
        for _, row in grouped.iterrows():
            lines.append(
                f"| {row['word2vec_model']} | {float(row['avg_jaccard']):.4f} | {float(row['avg_overlap']):.2f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Task 4 compares overlap patterns between learned neighborhoods and vector-arithmetic outputs.",
            "Higher overlap indicates stronger agreement between models for the same query words.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    if args.neighbor_top_k < 1:
        raise ValueError("--neighbor-top-k must be >= 1.")
    if args.equation_top_k < 1:
        raise ValueError("--equation-top-k must be >= 1.")

    task2_results_dir = Path(args.task2_results_dir)
    task3_results_dir = Path(args.task3_results_dir)
    if not task2_results_dir.exists():
        raise FileNotFoundError(f"Task 2 results directory not found: {task2_results_dir}")
    if not task3_results_dir.exists():
        raise FileNotFoundError(f"Task 3 results directory not found: {task3_results_dir}")

    task2_config = _load_json(task2_results_dir / "task2_config.json")
    task3_config = _load_json(task3_results_dir / "task3_config.json")

    task2_neighbors = _load_csv(
        task2_results_dir / "task2_neighbors.csv",
        required_columns=["model", "query_word", "rank", "similar_word", "cosine_similarity"],
    )
    task3_neighbors = _load_csv(
        task3_results_dir / "task3_neighbors.csv",
        required_columns=["model", "query_word", "rank", "similar_word", "cosine_similarity"],
    )
    task2_equations = _load_csv(
        task2_results_dir / "task2_equations.csv",
        required_columns=["model", "equation_id", "add_word", "rank", "result_word", "cosine_similarity"],
    )
    task3_equations = _load_csv(
        task3_results_dir / "task3_equations.csv",
        required_columns=["model", "equation_id", "add_word", "rank", "result_word", "cosine_similarity"],
    )
    task2_metrics = _load_csv(
        task2_results_dir / "task2_training_metrics.csv",
        required_columns=["model", "epoch", "avg_loss", "examples_seen"],
    )
    task3_metrics = _load_csv(
        task3_results_dir / "task3_training_metrics.csv",
        required_columns=["model", "epoch", "avg_weighted_loss", "pairs_seen"],
    )

    final_losses = _final_training_rows(task2_metrics, task3_metrics)
    neighbor_overlap = _compare_neighbors(task2_neighbors, task3_neighbors, top_k=args.neighbor_top_k)
    equation_overlap = _compare_equations(task2_equations, task3_equations, top_k=args.equation_top_k)
    model_comparison = _build_model_comparison(
        task2_config=task2_config,
        task3_config=task3_config,
        task2_neighbors=task2_neighbors,
        task3_neighbors=task3_neighbors,
        task2_equations=task2_equations,
        task3_equations=task3_equations,
        final_losses=final_losses,
    )

    query_word_count = int(
        len(set(task2_neighbors["query_word"].astype(str).tolist()) | set(task3_neighbors["query_word"].astype(str).tolist()))
    )

    config = {
        "task2_results_dir": str(task2_results_dir),
        "task3_results_dir": str(task3_results_dir),
        "neighbor_top_k": args.neighbor_top_k,
        "equation_top_k": args.equation_top_k,
        "query_word_count": query_word_count,
        "task2_vocab_size": int(task2_config.get("vocab_size", 0)),
        "task3_vocab_size": int(task3_config.get("vocab_size", 0)),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "task4_config.json"
    model_comparison_path = out_dir / "task4_model_comparison.csv"
    neighbor_overlap_path = out_dir / "task4_neighbors_overlap.csv"
    equation_overlap_path = out_dir / "task4_equations_overlap.csv"
    summary_path = out_dir / "task4_summary.md"

    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model_comparison.to_csv(model_comparison_path, index=False, encoding="utf-8")
    neighbor_overlap.to_csv(neighbor_overlap_path, index=False, encoding="utf-8")
    equation_overlap.to_csv(equation_overlap_path, index=False, encoding="utf-8")
    _write_summary_markdown(
        summary_path,
        config=config,
        model_comparison=model_comparison,
        neighbor_overlap=neighbor_overlap,
        equation_overlap=equation_overlap,
    )

    return {
        "config": config,
        "artifact_paths": {
            "config": config_path,
            "model_comparison": model_comparison_path,
            "neighbors_overlap": neighbor_overlap_path,
            "equations_overlap": equation_overlap_path,
            "summary": summary_path,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Task 3 GloVe outputs against Task 2 Word2Vec outputs.")
    parser.add_argument(
        "--task2-results-dir",
        default="project3/task2/results",
        help="Directory containing Task 2 result artifacts.",
    )
    parser.add_argument(
        "--task3-results-dir",
        default="project3/task3/results",
        help="Directory containing Task 3 result artifacts.",
    )
    parser.add_argument("--neighbor-top-k", type=int, default=10, help="Top-k neighbors used in overlap comparison.")
    parser.add_argument("--equation-top-k", type=int, default=5, help="Top-k equation outputs used in overlap comparison.")
    parser.add_argument("--out-dir", default="project3/task4/results", help="Output directory for Task 4 artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_pipeline(args)
    config = result["config"]

    print(f"Wrote artifacts to: {Path(args.out_dir)}")
    print(f"Compared query words: {config['query_word_count']}")
    print(f"Neighbor top-k: {config['neighbor_top_k']}")
    print(f"Equation top-k: {config['equation_top_k']}")


if __name__ == "__main__":
    main()
