from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from project3.task1.task1_corpus_matrices import (
    build_term_document_matrix,
    build_vocabulary,
    build_word_word_matrix,
    compute_corpus_statistics,
    compute_frequency_tables,
    main,
    prepare_documents,
    tokenize,
)


class Task1CorpusMatricesTests(unittest.TestCase):
    def test_tokenize_preserves_azerbaijani_letters_and_splits_punctuation(self) -> None:
        text = "A\u015f\u0131q-\u018fl\u0259sg\u0259r, g\u00f6z\u0259ls\u0259n! M\u0259n'im dostum."
        self.assertEqual(tokenize(text), ["aşıq", "ələsgər", "gözəlsən", "mən", "im", "dostum"])

    def test_compute_corpus_statistics_counts_rare_and_frequent_words(self) -> None:
        df = pd.DataFrame(
            {
                "row_id": [0, 1, 2],
                "author": ["a", "a", "b"],
                "title": ["t1", "t2", "t3"],
                "tokens": [["alma", "alma", "armud"], ["armud", "heyva"], ["nar"]],
            }
        )

        summary, _, _ = compute_corpus_statistics(df, author_col="author", rare_threshold=1, frequent_threshold=2)

        self.assertEqual(summary["documents"], 3)
        self.assertEqual(summary["authors"], 2)
        self.assertEqual(summary["total_tokens"], 6)
        self.assertEqual(summary["unique_tokens"], 4)
        self.assertEqual(summary["rare_words_freq_eq_1"], 2)
        self.assertEqual(summary["frequent_words_freq_ge_2"], 2)

    def test_build_term_document_matrix_exact_counts(self) -> None:
        docs = [["alma", "alma", "armud"], ["armud", "heyva"]]
        term_frequency, document_frequency = compute_frequency_tables(docs)
        vocabulary = build_vocabulary(term_frequency, document_frequency)

        matrix = build_term_document_matrix(docs, vocabulary).toarray().tolist()
        self.assertEqual([entry["token"] for entry in vocabulary], ["alma", "armud", "heyva"])
        self.assertEqual(matrix, [[2, 1, 0], [0, 1, 1]])

    def test_build_word_word_matrix_is_symmetric_with_zero_diagonal(self) -> None:
        docs = [["a", "b", "c"], ["a", "c"]]
        term_frequency, document_frequency = compute_frequency_tables(docs)
        vocabulary = build_vocabulary(term_frequency, document_frequency, min_frequency=1)

        matrix = build_word_word_matrix(docs, vocabulary, window_size=2).toarray()
        self.assertEqual([entry["token"] for entry in vocabulary], ["a", "c", "b"])
        self.assertEqual(matrix.diagonal().tolist(), [0, 0, 0])
        self.assertEqual(matrix.tolist(), [[0, 2, 1], [2, 0, 1], [1, 1, 0]])

    def test_cli_smoke_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "toy.parquet"
            out_dir = tmp_path / "results"

            df = pd.DataFrame(
                {
                    "author": ["a", "b", "a"],
                    "title": ["t1", "t2", "t3"],
                    "text": ["alma alma armud", "armud heyva", "alma heyva alma"],
                }
            )
            df.to_parquet(input_path)

            main(
                [
                    "--input",
                    str(input_path),
                    "--out-dir",
                    str(out_dir),
                    "--word-min-freq",
                    "1",
                    "--viz-top-n",
                    "3",
                    "--viz-doc-limit",
                    "2",
                ]
            )

            expected_files = [
                "task1_summary.json",
                "task1_summary.md",
                "task1_token_frequencies.csv",
                "task1_documents.csv",
                "task1_term_document_terms.csv",
                "task1_term_document_matrix.npz",
                "task1_term_document_dense.csv",
                "task1_term_document_heatmap.png",
                "task1_word_word_terms.csv",
                "task1_word_word_matrix.npz",
                "task1_word_word_dense.csv",
                "task1_word_word_heatmap.png",
            ]
            for name in expected_files:
                self.assertTrue((out_dir / name).exists(), name)

            summary = json.loads((out_dir / "task1_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["documents"], 3)
            self.assertEqual(summary["authors"], 2)

    def test_prepare_documents_preserves_row_order(self) -> None:
        df = pd.DataFrame(
            {
                    "author": ["x", "y"],
                    "title": ["first", "second"],
                    "text": ["Bir dost", "ikinci dost"],
                }
            )

        prepared = prepare_documents(df, text_col="text", author_col="author", title_col="title")

        self.assertEqual(prepared["row_id"].tolist(), [0, 1])
        self.assertEqual(prepared["tokens"].tolist(), [["bir", "dost"], ["ikinci", "dost"]])


if __name__ == "__main__":
    unittest.main()
