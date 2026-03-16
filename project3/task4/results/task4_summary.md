# Task 4 Comparison Summary

## Configuration

| Parameter | Value |
| --- | --- |
| task2_results_dir | task2/results |
| task3_results_dir | task3/results |
| neighbor_top_k | 10 |
| equation_top_k | 5 |
| query_word_count | 10 |

## Model Snapshot

| Model | Objective | Embedding Dim | Epochs | Vocab Size | Final Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| skipgram | negative_sampling | 100 | 8 | 6575 | 1.576596 |
| cbow | negative_sampling | 100 | 8 | 6575 | 1.323117 |
| glove | global_cooccurrence | 100 | 25 | 6575 | 0.013351 |

## Neighbor Overlap (Word2Vec vs GloVe)

| Word2Vec Model | Avg Jaccard | Avg Overlap Count |
| --- | ---: | ---: |
| cbow | 0.0269 | 0.50 |
| skipgram | 0.0334 | 0.60 |

## Vector Equation Overlap (Word2Vec vs GloVe)

| Word2Vec Model | Avg Jaccard | Avg Overlap Count |
| --- | ---: | ---: |
| cbow | 0.0222 | 0.20 |
| skipgram | 0.0000 | 0.00 |

## Interpretation

Task 4 compares overlap patterns between learned neighborhoods and vector-arithmetic outputs.
Higher overlap indicates stronger agreement between models for the same query words.
