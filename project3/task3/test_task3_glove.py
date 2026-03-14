from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project3.task3.task3_glove import (
    build_cooccurrence_pairs,
    build_vocabulary,
    evaluate_vector_equations,
    glove_weighting,
    initialize_glove_parameters,
    most_similar,
    prepare_corpus,
    train_glove_epoch,
)


class Task3GloveTests(unittest.TestCase):
    def test_build_vocabulary_is_deterministic(self) -> None:
        vocabulary = build_vocabulary({"beta": 2, "alpha": 2, "gamma": 1}, min_count=2)
        self.assertEqual([(item.token_id, item.token) for item in vocabulary], [(0, "alpha"), (1, "beta")])

    def test_prepare_corpus_reuses_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "toy.parquet"
            pd.DataFrame({"text": ["A\u015f\u0131q-\u018fl\u0259sg\u0259r g\u00f6z\u0259ldir"]}).to_parquet(input_path)

            bundle = prepare_corpus(input_path, text_col="text", min_count=1)

            self.assertEqual(bundle.tokenized_documents, [["aşıq", "ələsgər", "gözəldir"]])
            self.assertEqual(bundle.id_to_token, ["aşıq", "gözəldir", "ələsgər"])

    def test_build_cooccurrence_pairs_uses_inverse_distance(self) -> None:
        cooccurrence = build_cooccurrence_pairs([[0, 1, 2]], window_size=2)
        observed = {
            (int(row), int(col)): float(count)
            for row, col, count in zip(cooccurrence.row_ids, cooccurrence.col_ids, cooccurrence.counts)
        }
        expected = {
            (0, 1): 1.0,
            (0, 2): 0.5,
            (1, 0): 1.0,
            (1, 2): 1.0,
            (2, 0): 0.5,
            (2, 1): 1.0,
        }
        self.assertEqual(cooccurrence.directed_nonzero_pairs, 6)
        self.assertEqual(cooccurrence.total_pair_events, 6)
        self.assertEqual(observed, expected)

    def test_glove_weighting_matches_expected_behavior(self) -> None:
        counts = np.asarray([1.0, 10.0, 200.0], dtype=np.float32)
        weights = glove_weighting(counts, x_max=100.0, alpha=0.75)
        self.assertAlmostEqual(float(weights[0]), float((1.0 / 100.0) ** 0.75), places=6)
        self.assertAlmostEqual(float(weights[1]), float((10.0 / 100.0) ** 0.75), places=6)
        self.assertEqual(float(weights[2]), 1.0)

    def test_train_glove_epoch_updates_parameters(self) -> None:
        params = initialize_glove_parameters(vocab_size=3, embedding_dim=4, seed=5)
        original = params.word_vectors.copy()
        avg_loss, pairs_seen = train_glove_epoch(
            params=params,
            row_ids=np.asarray([0, 0, 1, 2], dtype=np.int32),
            col_ids=np.asarray([1, 2, 2, 0], dtype=np.int32),
            counts=np.asarray([2.0, 1.0, 3.0, 4.0], dtype=np.float32),
            x_max=10.0,
            alpha=0.75,
            batch_size=2,
            learning_rate=0.05,
            rng=np.random.default_rng(3),
        )
        self.assertTrue(np.isfinite(avg_loss))
        self.assertEqual(pairs_seen, 4)
        self.assertFalse(np.allclose(original, params.word_vectors))

    def test_most_similar_orders_by_cosine_similarity(self) -> None:
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.5, 0.5],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        token_to_id = {"a": 0, "b": 1, "c": 2, "d": 3}
        id_to_token = ["a", "b", "c", "d"]

        results = most_similar(embeddings, token_to_id, id_to_token, "a", top_k=3)

        self.assertEqual([row["similar_word"] for row in results], ["b", "c", "d"])

    def test_evaluate_vector_equations_is_deterministic(self) -> None:
        id_to_token = ["aşıq", "saz", "şeir", "can", "ruh", "dərman"]
        token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
        embeddings = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.8, 0.2],
            ],
            dtype=np.float32,
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        rows = evaluate_vector_equations(
            embeddings=embeddings,
            query_words=["aşıq", "can"],
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            exclusion_set={"bir", "ki"},
            neighbor_top_k=3,
            equation_count=1,
            equation_result_top_k=2,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["equation_id"], "glove_1")

    def test_cli_smoke_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "toy.parquet"
            out_dir = tmp_path / "results"
            stopword_path = tmp_path / "task1_tokens.csv"

            pd.DataFrame(
                {
                    "text": [
                        "aşıq can dil yar gözəl könül gül göz dərd sultan",
                        "aşıq könül gül göz can dil yar sultan dərd gözəl",
                        "yar gözəl könül can dil gül göz dərd sultan aşıq",
                        "göz dərd sultan aşıq can dil yar gözəl könül gül",
                    ]
                }
            ).to_parquet(input_path)
            pd.DataFrame(
                {"token": ["bir", "ki", "bu"], "term_frequency": [10, 9, 8], "document_frequency": [4, 4, 4]}
            ).to_csv(stopword_path, index=False, encoding="utf-8")

            from project3.task3.task3_glove import main

            main(
                [
                    "--input",
                    str(input_path),
                    "--min-count",
                    "1",
                    "--embedding-dim",
                    "8",
                    "--window-size",
                    "2",
                    "--x-max",
                    "10",
                    "--alpha",
                    "0.75",
                    "--epochs",
                    "2",
                    "--batch-size",
                    "8",
                    "--learning-rate",
                    "0.05",
                    "--seed",
                    "9",
                    "--task1-token-frequencies",
                    str(stopword_path),
                    "--neighbor-top-k",
                    "5",
                    "--equation-count",
                    "2",
                    "--equation-result-top-k",
                    "3",
                    "--out-dir",
                    str(out_dir),
                ]
            )

            expected_files = [
                "task3_config.json",
                "task3_summary.md",
                "task3_vocab.csv",
                "task3_training_metrics.csv",
                "task3_neighbors.csv",
                "task3_equations.csv",
                "task3_glove_embeddings.npy",
                "task3_cooccurrence_pairs.npz",
                "task3_cooccurrence_stats.json",
            ]
            for file_name in expected_files:
                self.assertTrue((out_dir / file_name).exists(), file_name)

            config = json.loads((out_dir / "task3_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["vocab_size"], 10)
            self.assertEqual(config["epochs"], 2)


if __name__ == "__main__":
    unittest.main()
