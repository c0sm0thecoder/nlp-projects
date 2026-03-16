from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    from task4.task4_compare_embeddings import (
        _compare_equations,
        _compare_neighbors,
        main,
    )
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from task4.task4_compare_embeddings import (
        _compare_equations,
        _compare_neighbors,
        main,
    )


class Task4CompareEmbeddingsTests(unittest.TestCase):
    def test_compare_neighbors_overlap(self) -> None:
        task2_neighbors = pd.DataFrame(
            [
                {"model": "skipgram", "query_word": "can", "rank": 1, "similar_word": "yar", "cosine_similarity": 0.8},
                {"model": "skipgram", "query_word": "can", "rank": 2, "similar_word": "eşq", "cosine_similarity": 0.7},
                {"model": "cbow", "query_word": "can", "rank": 1, "similar_word": "dil", "cosine_similarity": 0.6},
                {"model": "cbow", "query_word": "can", "rank": 2, "similar_word": "yar", "cosine_similarity": 0.5},
            ]
        )
        task3_neighbors = pd.DataFrame(
            [
                {"model": "glove", "query_word": "can", "rank": 1, "similar_word": "yar", "cosine_similarity": 0.9},
                {"model": "glove", "query_word": "can", "rank": 2, "similar_word": "eşq", "cosine_similarity": 0.85},
            ]
        )

        overlap = _compare_neighbors(task2_neighbors, task3_neighbors, top_k=2)
        skipgram_row = overlap[(overlap["word2vec_model"] == "skipgram") & (overlap["query_word"] == "can")].iloc[0]
        cbow_row = overlap[(overlap["word2vec_model"] == "cbow") & (overlap["query_word"] == "can")].iloc[0]

        self.assertEqual(int(skipgram_row["overlap_count"]), 2)
        self.assertAlmostEqual(float(skipgram_row["jaccard_similarity"]), 1.0, places=6)
        self.assertEqual(int(cbow_row["overlap_count"]), 1)
        self.assertAlmostEqual(float(cbow_row["jaccard_similarity"]), 1.0 / 3.0, places=6)

    def test_compare_equations_overlap(self) -> None:
        task2_equations = pd.DataFrame(
            [
                {"model": "skipgram", "equation_id": "skipgram_1", "add_word": "can", "rank": 1, "result_word": "yar", "cosine_similarity": 0.7},
                {"model": "skipgram", "equation_id": "skipgram_1", "add_word": "can", "rank": 2, "result_word": "eşq", "cosine_similarity": 0.6},
                {"model": "cbow", "equation_id": "cbow_1", "add_word": "can", "rank": 1, "result_word": "ruh", "cosine_similarity": 0.5},
            ]
        )
        task3_equations = pd.DataFrame(
            [
                {"model": "glove", "equation_id": "glove_1", "add_word": "can", "rank": 1, "result_word": "yar", "cosine_similarity": 0.8},
                {"model": "glove", "equation_id": "glove_1", "add_word": "can", "rank": 2, "result_word": "eşq", "cosine_similarity": 0.7},
            ]
        )

        overlap = _compare_equations(task2_equations, task3_equations, top_k=2)
        skipgram_row = overlap[(overlap["word2vec_model"] == "skipgram") & (overlap["query_word"] == "can")].iloc[0]
        cbow_row = overlap[(overlap["word2vec_model"] == "cbow") & (overlap["query_word"] == "can")].iloc[0]

        self.assertEqual(int(skipgram_row["overlap_count"]), 2)
        self.assertAlmostEqual(float(skipgram_row["jaccard_similarity"]), 1.0, places=6)
        self.assertEqual(int(cbow_row["overlap_count"]), 0)
        self.assertAlmostEqual(float(cbow_row["jaccard_similarity"]), 0.0, places=6)

    def test_cli_smoke_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            task2_results = tmp_path / "task2" / "results"
            task3_results = tmp_path / "task3" / "results"
            out_dir = tmp_path / "task4" / "results"
            task2_results.mkdir(parents=True, exist_ok=True)
            task3_results.mkdir(parents=True, exist_ok=True)

            (task2_results / "task2_config.json").write_text(
                json.dumps({"embedding_dim": 8, "epochs": 2, "vocab_size": 10}),
                encoding="utf-8",
            )
            (task3_results / "task3_config.json").write_text(
                json.dumps({"embedding_dim": 8, "epochs": 3, "vocab_size": 10}),
                encoding="utf-8",
            )

            pd.DataFrame(
                [
                    {"model": "skipgram", "query_word": "can", "rank": 1, "similar_word": "yar", "cosine_similarity": 0.7},
                    {"model": "cbow", "query_word": "can", "rank": 1, "similar_word": "dil", "cosine_similarity": 0.6},
                ]
            ).to_csv(task2_results / "task2_neighbors.csv", index=False, encoding="utf-8")
            pd.DataFrame(
                [{"model": "glove", "query_word": "can", "rank": 1, "similar_word": "yar", "cosine_similarity": 0.8}]
            ).to_csv(task3_results / "task3_neighbors.csv", index=False, encoding="utf-8")

            pd.DataFrame(
                [
                    {"model": "skipgram", "equation_id": "skipgram_1", "add_word": "can", "rank": 1, "result_word": "yar", "cosine_similarity": 0.6},
                    {"model": "cbow", "equation_id": "cbow_1", "add_word": "can", "rank": 1, "result_word": "ruh", "cosine_similarity": 0.5},
                ]
            ).to_csv(task2_results / "task2_equations.csv", index=False, encoding="utf-8")
            pd.DataFrame(
                [{"model": "glove", "equation_id": "glove_1", "add_word": "can", "rank": 1, "result_word": "yar", "cosine_similarity": 0.75}]
            ).to_csv(task3_results / "task3_equations.csv", index=False, encoding="utf-8")

            pd.DataFrame(
                [
                    {"model": "skipgram", "epoch": 1, "avg_loss": 2.0, "examples_seen": 10},
                    {"model": "skipgram", "epoch": 2, "avg_loss": 1.5, "examples_seen": 10},
                    {"model": "cbow", "epoch": 1, "avg_loss": 1.8, "examples_seen": 8},
                    {"model": "cbow", "epoch": 2, "avg_loss": 1.4, "examples_seen": 8},
                ]
            ).to_csv(task2_results / "task2_training_metrics.csv", index=False, encoding="utf-8")
            pd.DataFrame(
                [
                    {"model": "glove", "epoch": 1, "avg_weighted_loss": 0.3, "pairs_seen": 12},
                    {"model": "glove", "epoch": 2, "avg_weighted_loss": 0.2, "pairs_seen": 12},
                ]
            ).to_csv(task3_results / "task3_training_metrics.csv", index=False, encoding="utf-8")

            main(
                [
                    "--task2-results-dir",
                    str(task2_results),
                    "--task3-results-dir",
                    str(task3_results),
                    "--neighbor-top-k",
                    "1",
                    "--equation-top-k",
                    "1",
                    "--out-dir",
                    str(out_dir),
                ]
            )

            expected_files = [
                "task4_config.json",
                "task4_model_comparison.csv",
                "task4_neighbors_overlap.csv",
                "task4_equations_overlap.csv",
                "task4_summary.md",
            ]
            for file_name in expected_files:
                self.assertTrue((out_dir / file_name).exists(), file_name)

            config = json.loads((out_dir / "task4_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["neighbor_top_k"], 1)
            self.assertEqual(config["equation_top_k"], 1)
            self.assertEqual(config["query_word_count"], 1)


if __name__ == "__main__":
    unittest.main()
