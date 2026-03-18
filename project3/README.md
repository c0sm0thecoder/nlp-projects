# NLP Project 3 Report: Distributional Semantics, Embedding Comparison, and Sequence Classification

**Authors:** Kamal Aghazada, Fatulla Bashirov  
**Date:** March 2026

## Abstract
This report summarizes Project 3 deliverables for Task 1-Task 5 and the Extra Task UI. The project studies corpus-level matrix representations, neural word embeddings (Word2Vec and GloVe), cross-model agreement, and author classification with recurrent models over multiple feature spaces. Experiments are reproducible from saved artifacts in each task's results directory. The UI provides an integrated dashboard for comparative analysis.

## Team Member Contributions
- Fatulla Bashirov: Task 1 (corpus and matrix analysis), Task 2 (Word2Vec training and probing), Task 3 (GloVe training and probing).
- Kamal Aghazada: Task 4 (embedding comparison), Task 5 (RNN/BiRNN/LSTM classification), Extra Task UI integration.
- Both members: code review, artifact validation, and final report preparation.

## 1. Introduction and Dataset
The project uses cleaned poem data from [poems_cleaned.parquet](poems_cleaned.parquet). Compared to the previous project's dataset ([poems_translated.parquet](../project2/poems_translated.parquet)), which contained 846 poems from 9 authors, the current dataset has been expanded to 1058 poems from 10 authors by adding 211 poems by a new author, **Aşıq Ələsgər**. Core experiments operate on the text field and preserve deterministic settings (seed = 42 in training/config artifacts).

High-level scope:
- Task 1: corpus statistics + term-document and word-word matrices.
- Task 2: Word2Vec (Skip-gram and CBOW) training, neighbors, and vector arithmetic probes.
- Task 3: GloVe training, neighbors, and vector arithmetic probes.
- Task 4: overlap-based comparison of Word2Vec and GloVe outputs.
- Task 5: multi-class author classification using RNN, BiRNN, and LSTM with five feature families.
- Extra Task: FastAPI dashboard for interactive exploration.

## 2. Task 1: Corpus Matrices
Implementation: [task1/task1_corpus_matrices.py](task1/task1_corpus_matrices.py)  
Primary summary: [task1/results/task1_summary.md](task1/results/task1_summary.md)

### 2.1 Corpus Statistics
| Metric | Value |
| --- | ---: |
| Documents | 1058 |
| Authors | 10 |
| Total tokens | 209091 |
| Unique tokens | 44113 |
| Average document length | 197.628544 |
| Median document length | 100 |
| Rare words (freq = 1) | 26289 |
| Frequent words (freq >= 10) | 3218 |

### 2.2 Matrix Setup
- Term-document vocabulary: full corpus vocabulary (44113 terms).
- Word-word vocabulary: tokens with corpus frequency >= 10.
- Word-word context window: 2.

Produced artifacts include dense CSV and sparse NPZ outputs:
- [task1/results/task1_term_document_dense.csv](task1/results/task1_term_document_dense.csv)
- [task1/results/task1_term_document_matrix.npz](task1/results/task1_term_document_matrix.npz)
- [task1/results/task1_word_word_dense.csv](task1/results/task1_word_word_dense.csv)
- [task1/results/task1_word_word_matrix.npz](task1/results/task1_word_word_matrix.npz)

### 2.3 Interpretation
The corpus exhibits a classic long-tail lexical pattern (many rare words, relatively small frequent core), which motivates frequency-thresholded representations for stable co-occurrence modeling.

## 3. Task 2: Word2Vec
Implementation: [task2/task2_word2vec.py](task2/task2_word2vec.py)  
Primary summary: [task2/results/task2_summary.md](task2/results/task2_summary.md)

### 3.1 Configuration Snapshot
| Parameter | Value |
| --- | --- |
| min_count | 5 |
| embedding_dim | 100 |
| window_size | 5 |
| negative_samples | 5 |
| subsample_t | 0.0001 |
| epochs | 8 |
| batch_size | 2048 |
| vocab_size | 6575 |

### 3.2 Training Outcome
| Model | Final Average Loss |
| --- | ---: |
| skipgram | 1.576596 |
| cbow | 1.323117 |

### 3.3 Qualitative Findings
Nearest neighbors are generally semantically related in poetic context but often reflect stylistic/topical association rather than strict synonymy. Vector arithmetic probes produce coherent but noisy analogical behavior, consistent with domain-specific sparse contexts.

Detailed outputs:
- [task2/results/task2_neighbors.csv](task2/results/task2_neighbors.csv)
- [task2/results/task2_equations.csv](task2/results/task2_equations.csv)
- [task2/results/task2_training_metrics.csv](task2/results/task2_training_metrics.csv)

## 4. Task 3: GloVe
Implementation: [task3/task3_glove.py](task3/task3_glove.py)  
Primary summary: [task3/results/task3_summary.md](task3/results/task3_summary.md)

### 4.1 Configuration Snapshot
| Parameter | Value |
| --- | --- |
| min_count | 5 |
| embedding_dim | 100 |
| window_size | 5 |
| x_max | 100.0 |
| alpha | 0.75 |
| epochs | 25 |
| batch_size | 4096 |
| learning_rate | 0.05 |
| vocab_size | 6575 |
| directed_nonzero_pairs | 876452 |

### 4.2 Training Outcome
| Model | Final Average Weighted Loss |
| --- | ---: |
| glove | 0.013351 |

### 4.3 Qualitative Findings
GloVe neighbors capture global co-occurrence structure effectively, with visible topical cohesion in poetry vocabulary. Arithmetic probes show limited but interpretable regularities.

Detailed outputs:
- [task3/results/task3_neighbors.csv](task3/results/task3_neighbors.csv)
- [task3/results/task3_equations.csv](task3/results/task3_equations.csv)
- [task3/results/task3_training_metrics.csv](task3/results/task3_training_metrics.csv)

## 5. Task 4: Word2Vec vs GloVe Comparison
Implementation: [task4/task4_compare_embeddings.py](task4/task4_compare_embeddings.py)  
Primary summary: [task4/results/task4_summary.md](task4/results/task4_summary.md)

### 5.1 Neighbor Overlap (Jaccard)
| Word2Vec Model | Avg Jaccard | Avg Overlap Count |
| --- | ---: | ---: |
| cbow | 0.0269 | 0.50 |
| skipgram | 0.0334 | 0.60 |

### 5.2 Vector Equation Overlap (Jaccard)
| Word2Vec Model | Avg Jaccard | Avg Overlap Count |
| --- | ---: | ---: |
| cbow | 0.0222 | 0.20 |
| skipgram | 0.0000 | 0.00 |

### 5.3 Interpretation
Cross-model overlap is low, indicating that Word2Vec and GloVe encode partially different neighborhood geometry for the same vocabulary. Skip-gram aligns slightly better than CBOW on neighbor overlap, while equation-level agreement is very weak.

## 6. Task 5: Author Classification with RNN Family
Implementation: [task5/task5_rnn_classification.py](task5/task5_rnn_classification.py)  
Primary summary: [task5/results/task5_summary.md](task5/results/task5_summary.md)

### 6.1 Setup
- Train documents: 844
- Test documents: 211
- Classes: 9 authors
- Models: RNN, BiRNN, LSTM
- Features: Count, TF-IDF, PMI, Word2Vec, GloVe

### 6.2 Test Performance (Accuracy / Macro-F1)
Best rows from [task5/results/task5_results.csv](task5/results/task5_results.csv):

| Feature | Best Model | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| count | rnn | 0.9005 | 0.7921 |
| tfidf | birnn | 0.9005 | 0.7925 |
| pmi | birnn | 0.8720 | 0.7516 |
| word2vec | birnn | 0.7915 | 0.5739 |
| glove | birnn | 0.5071 | 0.2763 |

### 6.3 Interpretation
Sparse lexical features (Count and TF-IDF) outperform dense embedding-only features in this task setup. Sequence models trained over embedding feature variants underperform, suggesting either information loss in the current feature pipeline or insufficient supervision for robust author-style discrimination in embedding space.

## 7. Extra Task: Interactive UI
UI implementation:
- [extra_task_ui/app.py](extra_task_ui/app.py)
- [extra_task_ui/templates/index.html](extra_task_ui/templates/index.html)
- [extra_task_ui/static/app.js](extra_task_ui/static/app.js)
- [extra_task_ui/static/style.css](extra_task_ui/static/style.css)

Run instructions are documented in [extra_task_ui/README.md](extra_task_ui/README.md).

Implemented capabilities include:
- task cards and metric highlights,
- Task 5 bubble chart and sortable performance table,
- animated training-race view for average loss,
- Task 4 tabbed comparison,
- user-input filtering for Task 4 query words and model selection.

## 8. Reproducibility
From project root, run each task script to regenerate artifacts:

```bash
python task1/task1_corpus_matrices.py
python task2/task2_word2vec.py
python task3/task3_glove.py
python task4/task4_compare_embeddings.py
python task5/task5_rnn_classification.py
```

Run tests:

```bash
python -m unittest task1/test_task1_corpus_matrices.py
python -m unittest task2/test_task2_word2vec.py
python -m unittest task3/test_task3_glove.py
python -m unittest task4/test_task4_compare_embeddings.py
python -m unittest task5/test_task5_rnn_classification.py
```

Start UI:

```bash
.venv/bin/uvicorn extra_task_ui.app:app --reload
```

## 9. Conclusion
Project 3 delivers a full pipeline from count-based corpus structure to neural embedding learning and downstream sequence classification. Main findings are:
- corpus statistics show strong lexical sparsity,
- Word2Vec and GloVe capture useful but only partially overlapping semantic structure,
- in author classification, TF-IDF/Count features with recurrent architectures currently provide the strongest macro-level performance,
- the UI consolidates outputs into an interpretable, interactive analysis layer.

Future improvements can focus on stronger contextual encoders, better sequence input construction from dense embeddings, and expanded intrinsic/extrinsic evaluation of semantic spaces.
