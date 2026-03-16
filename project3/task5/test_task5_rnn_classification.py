from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from task5.task5_rnn_classification import (
        _compute_class_word_ppmi,
        average_embedding_features,
        main,
    )
except ModuleNotFoundError:
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from task5.task5_rnn_classification import (
        _compute_class_word_ppmi,
        average_embedding_features,
        main,
    )


class Task5RnnClassificationTests(unittest.TestCase):
    def test_ppmi_matrix_is_non_negative(self) -> None:
        binary_matrix = np.asarray(
            [
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 1], dtype=np.int64)

        ppmi = _compute_class_word_ppmi(binary_matrix, labels, num_classes=2)

        self.assertEqual(ppmi.shape, (3, 2))
        self.assertTrue(np.all(ppmi >= 0.0))

    def test_average_embedding_features_shape(self) -> None:
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        )
        token_to_id = {"a": 0, "b": 1, "c": 2}
        token_lists = [["a", "c"], ["b"], ["x"]]

        features = average_embedding_features(token_lists, embeddings, token_to_id)

        self.assertEqual(features.shape, (3, 2))
        self.assertTrue(np.allclose(features[0], np.asarray([0.75, 0.25], dtype=np.float32)))
        self.assertTrue(np.allclose(features[2], np.asarray([0.0, 0.0], dtype=np.float32)))

    def test_cli_smoke_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "toy.parquet"
            task2_dir = tmp_path / "task2_results"
            task3_dir = tmp_path / "task3_results"
            out_dir = tmp_path / "task5_results"
            task2_dir.mkdir(parents=True, exist_ok=True)
            task3_dir.mkdir(parents=True, exist_ok=True)

            authors = ["A", "B", "C"]
            rows = []
            for author_index, author in enumerate(authors):
                for doc_index in range(6):
                    rows.append(
                        {
                            "author": author,
                            "title": f"{author}_{doc_index}",
                            "text": f"{author.lower()} söz {author_index} sevgi ürək şeir",
                        }
                    )
            pd.DataFrame(rows).to_parquet(input_path)

            vocab = ["a", "b", "c", "söz", "sevgi", "ürək", "şeir", "0", "1", "2"]
            vocab_df = pd.DataFrame(
                {
                    "token_id": list(range(len(vocab))),
                    "token": vocab,
                    "corpus_frequency": [5] * len(vocab),
                }
            )
            vocab_df.to_csv(task2_dir / "task2_vocab.csv", index=False, encoding="utf-8")
            vocab_df.to_csv(task3_dir / "task3_vocab.csv", index=False, encoding="utf-8")

            rng = np.random.default_rng(9)
            np.save(task2_dir / "task2_skipgram_embeddings.npy", rng.normal(size=(len(vocab), 8)).astype(np.float32))
            np.save(task3_dir / "task3_glove_embeddings.npy", rng.normal(size=(len(vocab), 8)).astype(np.float32))

            main(
                [
                    "--input",
                    str(input_path),
                    "--label-col",
                    "author",
                    "--min-docs-per-class",
                    "2",
                    "--test-size",
                    "0.25",
                    "--max-vectorizer-features",
                    "100",
                    "--hidden-dim",
                    "16",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--learning-rate",
                    "0.01",
                    "--seed",
                    "13",
                    "--task2-embeddings",
                    str(task2_dir / "task2_skipgram_embeddings.npy"),
                    "--task2-vocab",
                    str(task2_dir / "task2_vocab.csv"),
                    "--task3-embeddings",
                    str(task3_dir / "task3_glove_embeddings.npy"),
                    "--task3-vocab",
                    str(task3_dir / "task3_vocab.csv"),
                    "--out-dir",
                    str(out_dir),
                ]
            )

            expected_files = [
                "task5_config.json",
                "task5_results.csv",
                "task5_training_metrics.csv",
                "task5_summary.md",
            ]
            for file_name in expected_files:
                self.assertTrue((out_dir / file_name).exists(), file_name)

            config = json.loads((out_dir / "task5_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["num_classes"], 3)
            self.assertEqual(len(config["features"]), 5)
            self.assertEqual(len(config["models"]), 3)

            results = pd.read_csv(out_dir / "task5_results.csv")
            self.assertEqual(len(results), 15)
            self.assertEqual(sorted(results["feature"].unique().tolist()), ["count", "glove", "pmi", "tfidf", "word2vec"])
            self.assertEqual(sorted(results["model"].unique().tolist()), ["birnn", "lstm", "rnn"])


if __name__ == "__main__":
    unittest.main()
