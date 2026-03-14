from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from project3.task2.task2_word2vec import (
    build_vocabulary,
    generate_cbow_examples,
    generate_skipgram_pairs,
    most_similar,
    prepare_corpus,
    sample_negative_ids,
)


class Task2Word2VecTests(unittest.TestCase):
    def test_prepare_corpus_reuses_tokenizer_for_azerbaijani_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "toy.parquet"
            pd.DataFrame({"text": ["A\u015f\u0131q-\u018fl\u0259sg\u0259r g\u00f6z\u0259ldir"]}).to_parquet(input_path)

            bundle = prepare_corpus(input_path, text_col="text", min_count=1)

            self.assertEqual(bundle.tokenized_documents, [["aşıq", "ələsgər", "gözəldir"]])
            self.assertEqual(bundle.id_to_token, ["aşıq", "gözəldir", "ələsgər"])

    def test_build_vocabulary_is_deterministic(self) -> None:
        vocabulary = build_vocabulary({"beta": 2, "alpha": 2, "gamma": 1}, min_count=2)
        self.assertEqual([(item.token_id, item.token) for item in vocabulary], [(0, "alpha"), (1, "beta")])

    def test_generate_skipgram_pairs(self) -> None:
        centers, targets = generate_skipgram_pairs([[1, 2, 3]], window_size=1)
        self.assertEqual(centers.tolist(), [1, 2, 2, 3])
        self.assertEqual(targets.tolist(), [2, 1, 3, 2])

    def test_generate_cbow_examples_variable_context(self) -> None:
        contexts, targets = generate_cbow_examples([[1, 2, 3, 4]], window_size=1)
        self.assertEqual(contexts, [[2], [1, 3], [2, 4], [3]])
        self.assertEqual(targets.tolist(), [1, 2, 3, 4])

    def test_sample_negative_ids_shape_and_exclusion(self) -> None:
        distribution = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(7)
        forbidden = torch.tensor([0, 1, 2], dtype=torch.long)

        sampled = sample_negative_ids(
            distribution,
            batch_size=3,
            negative_samples=4,
            forbidden_ids=forbidden,
            generator=generator,
        )

        self.assertEqual(tuple(sampled.shape), (3, 4))
        for row_index, forbidden_id in enumerate(forbidden.tolist()):
            self.assertNotIn(forbidden_id, sampled[row_index].tolist())

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

    def test_cli_smoke_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "toy.parquet"
            out_dir = tmp_path / "results"
            pd.DataFrame(
                {
                    "text": [
                        "aşıq can yar könül gül göz gözəl dərd sultan dil",
                        "aşıq can yar könül gül göz gözəl dərd sultan dil",
                        "aşıq can yar könül gül göz gözəl dərd sultan dil",
                    ]
                }
            ).to_parquet(input_path)

            from project3.task2.task2_word2vec import main

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
                    "--negative-samples",
                    "2",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--seed",
                    "11",
                    "--neighbor-top-k",
                    "3",
                    "--equation-count",
                    "1",
                    "--equation-result-top-k",
                    "2",
                    "--task1-token-frequencies",
                    "",
                    "--out-dir",
                    str(out_dir),
                ]
            )

            expected_files = [
                "task2_config.json",
                "task2_summary.md",
                "task2_vocab.csv",
                "task2_training_metrics.csv",
                "task2_neighbors.csv",
                "task2_equations.csv",
                "task2_skipgram_embeddings.npy",
                "task2_cbow_embeddings.npy",
            ]
            for file_name in expected_files:
                self.assertTrue((out_dir / file_name).exists(), file_name)

            config = json.loads((out_dir / "task2_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["vocab_size"], 10)
            self.assertEqual(config["epochs"], 1)


if __name__ == "__main__":
    unittest.main()
