from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from project2.task2.smoothing_core import (
    METHODS,
    ORDERS,
    build_model_counts,
    build_smoothed_model,
    evaluate_manual_params,
    evaluate_model,
    iter_events,
    model_probability,
    prepare_task2_data,
    rank_methods,
    run_tuning,
    write_task2_outputs,
)


class Task2SmoothingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train_sequences = [["a", "b"], ["a", "a"], ["b", "a"], ["a", "c"]]
        self.val_sequences = [["a", "b"], ["b", "c"]]
        self.test_sequences = [["a", "b"], ["c", "a"]]

    def _params_for(self, method: str, order: int) -> dict[str, float]:
        if method == "laplace":
            return {"alpha": 1.0}
        if method == "interpolation" and order == 2:
            return {"lambda1": 0.3, "lambda2": 0.7}
        if method == "interpolation" and order == 3:
            return {"lambda1": 0.2, "lambda2": 0.3, "lambda3": 0.5}
        return {"discount": 0.75}

    def test_all_methods_produce_positive_probabilities(self) -> None:
        for method in METHODS:
            for order in ORDERS:
                model = build_smoothed_model(
                    sequences=self.train_sequences,
                    order=order,
                    method=method,
                    params=self._params_for(method, order),
                )
                for context, word in iter_events(self.val_sequences, order):
                    prob = model_probability(model, context, word)
                    self.assertGreater(prob, 0.0)
                diag = evaluate_model(model, self.val_sequences)
                self.assertEqual(diag["zero_prob_events"], 0)
                self.assertTrue(math.isfinite(float(diag["perplexity"])))

    def test_backoff_alpha_non_negative(self) -> None:
        model = build_smoothed_model(
            sequences=self.train_sequences,
            order=3,
            method="backoff",
            params={"discount": 0.75},
        )
        self.assertTrue(all(v >= 0 for v in model.backoff_alpha_bigram.values()))
        self.assertTrue(all(v >= 0 for v in model.backoff_alpha_trigram.values()))

    def test_kneser_ney_continuation_counts(self) -> None:
        counts = build_model_counts([["a", "b"], ["c", "b"]])
        self.assertEqual(counts.continuation_counts.get("b"), 2)
        self.assertGreater(counts.continuation_total, 0)

    def test_tuning_is_deterministic(self) -> None:
        best1, _ = run_tuning(
            inner_train_sequences=self.train_sequences,
            val_sequences=self.val_sequences,
            interp_bigram_grid=[0.5, 0.7],
            interp_trigram_step=0.5,
            discount_grid=[0.5, 0.75],
        )
        best2, _ = run_tuning(
            inner_train_sequences=self.train_sequences,
            val_sequences=self.val_sequences,
            interp_bigram_grid=[0.5, 0.7],
            interp_trigram_step=0.5,
            discount_grid=[0.5, 0.75],
        )
        self.assertEqual(
            {(k[0], k[1]): v["params_json"] for k, v in best1.items()},
            {(k[0], k[1]): v["params_json"] for k, v in best2.items()},
        )

    def test_ranking_utility_schema(self) -> None:
        method_rows = [
            {"method": "laplace", "order": 2, "test_perplexity": 100.0},
            {"method": "laplace", "order": 3, "test_perplexity": 200.0},
            {"method": "interpolation", "order": 2, "test_perplexity": 70.0},
            {"method": "interpolation", "order": 3, "test_perplexity": 120.0},
            {"method": "backoff", "order": 2, "test_perplexity": 80.0},
            {"method": "backoff", "order": 3, "test_perplexity": 110.0},
            {"method": "kneser_ney", "order": 2, "test_perplexity": 60.0},
            {"method": "kneser_ney", "order": 3, "test_perplexity": 90.0},
        ]
        ranked = rank_methods(method_rows, "trigram_test_ppl")
        self.assertEqual(ranked[0]["method"], "kneser_ney")
        self.assertIn("score", ranked[0])
        self.assertIn("rank", ranked[0])

    def test_manual_params_schema(self) -> None:
        result = evaluate_manual_params(
            method="interpolation",
            order=2,
            params={"lambda1": 0.3, "lambda2": 0.7},
            inner_train_sequences=self.train_sequences,
            val_sequences=self.val_sequences,
            final_train_sequences=self.train_sequences,
            test_sequences=self.test_sequences,
        )
        self.assertIn("train_perplexity", result)
        self.assertIn("val_perplexity", result)
        self.assertIn("test_perplexity", result)
        self.assertIn("zero_prob_events_test", result)

    def test_write_outputs_contract(self) -> None:
        artifacts = {
            "metrics": {"config": {}, "corpus": {}, "split": {}, "methods": [], "best_method_defaults": {}},
            "tuning_rows": [
                {
                    "method": "laplace",
                    "order": 2,
                    "params_json": '{"alpha": 1.0}',
                    "train_perplexity": 1.0,
                    "val_perplexity": 2.0,
                    "val_zero_prob_events": 0,
                    "val_unseen_rate": 0.0,
                }
            ],
            "method_rows": [
                {
                    "method": "laplace",
                    "order": 2,
                    "best_params": {"alpha": 1.0},
                    "train_perplexity": 1.0,
                    "val_perplexity": 2.0,
                    "test_perplexity": 3.0,
                    "zero_prob_events_test": 0,
                    "unseen_rate_test": 0.0,
                }
            ],
            "ranking_rows": [
                {
                    "rule": "trigram_test_ppl",
                    "rank": 1,
                    "method": "laplace",
                    "score": 3.0,
                    "bigram_test_perplexity": 2.0,
                    "trigram_test_perplexity": 3.0,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_task2_outputs(artifacts, tmpdir)
            self.assertTrue(Path(paths["metrics"]).exists())
            self.assertTrue(Path(paths["tuning"]).exists())
            self.assertTrue(Path(paths["comparison"]).exists())
            self.assertTrue(Path(paths["ranking"]).exists())

    def test_prepare_task2_data_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            prepare_task2_data(
                input_path="project2/not_found.parquet",
                text_col="modern_text",
                author_col="author",
                test_size=0.2,
                val_size=0.1,
                seed=42,
                min_freq=2,
            )


if __name__ == "__main__":
    unittest.main()
