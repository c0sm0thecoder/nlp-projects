from __future__ import annotations

import math
import unittest

import pandas as pd

from project2.task1.task1_ngram import (
    apply_unk_policy,
    build_ngram_model,
    perplexity_diagnostics,
    stratified_split,
    tokenize,
    validate_columns,
)


class Task1NgramTests(unittest.TestCase):
    def test_tokenize_keeps_azerbaijani_diacritics(self) -> None:
        text = "Yenə bir təzə gözələ aşiq oldum"
        tokens = tokenize(text)
        self.assertIn("yenə", tokens)
        self.assertIn("təzə", tokens)
        self.assertIn("gözələ", tokens)

    def test_validate_columns_raises_on_missing(self) -> None:
        df = pd.DataFrame({"author": ["a"], "text": ["x"]})
        with self.assertRaises(ValueError):
            validate_columns(df, ["author", "modern_text"])

    def test_stratified_split_is_reproducible(self) -> None:
        df = pd.DataFrame(
            {
                "row_id": list(range(10)),
                "author": ["a"] * 5 + ["b"] * 5,
                "tokens": [["x", "y"]] * 10,
            }
        )
        train1, test1 = stratified_split(df, "author", test_size=0.2, seed=42)
        train2, test2 = stratified_split(df, "author", test_size=0.2, seed=42)

        self.assertEqual(train1["row_id"].tolist(), train2["row_id"].tolist())
        self.assertEqual(test1["row_id"].tolist(), test2["row_id"].tolist())

    def test_unk_mapping_maps_oov_test_tokens(self) -> None:
        train = [["a", "a", "b"], ["a", "c"]]
        test = [["a", "z"]]
        train_mapped, test_mapped, stats = apply_unk_policy(train, test, min_freq=2)

        self.assertEqual(train_mapped, [["a", "a", "<unk>"], ["a", "<unk>"]])
        self.assertEqual(test_mapped, [["a", "<unk>"]])
        self.assertEqual(stats["test_oov_before_unk_count"], 1)

    def test_ngram_count_totals_and_perplexity_behavior(self) -> None:
        train = [["a", "b"], ["a"]]
        unigrams, _, uni_total = build_ngram_model(train, 1)
        bigrams, bigram_ctx, bigram_total = build_ngram_model(train, 2)
        trigrams, trigram_ctx, trigram_total = build_ngram_model(train, 3)

        self.assertEqual(uni_total, 3)
        self.assertEqual(bigram_total, 5)
        self.assertEqual(trigram_total, 5)
        self.assertEqual(unigrams[("a",)], 2)

        train_diag = perplexity_diagnostics(train, 2, bigrams, bigram_ctx, uni_total)
        seen_test_diag = perplexity_diagnostics([["a", "b"]], 2, bigrams, bigram_ctx, uni_total)
        unseen_test_diag = perplexity_diagnostics([["b", "a"]], 2, bigrams, bigram_ctx, uni_total)

        self.assertFalse(math.isinf(float(train_diag["perplexity"])))
        self.assertFalse(math.isinf(float(seen_test_diag["perplexity"])))
        self.assertTrue(math.isinf(float(unseen_test_diag["perplexity"])))
        self.assertGreater(unseen_test_diag["zero_prob_events"], 0)


if __name__ == "__main__":
    unittest.main()
