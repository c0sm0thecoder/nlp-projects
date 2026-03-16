# Task 5 Classification Summary

## Configuration

| Parameter | Value |
| --- | --- |
| input | poems_cleaned.parquet |
| text_col | text |
| label_col | author |
| min_docs_per_class | 5 |
| test_size | 0.2 |
| seed | 42 |
| epochs | 10 |
| batch_size | 32 |
| learning_rate | 0.001 |
| hidden_dim | 128 |
| max_vectorizer_features | 3000 |
| train_docs | 844 |
| test_docs | 211 |
| num_classes | 9 |

## Performance Table

| Feature | Model | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| count | birnn | 0.8910 | 0.7832 |
| count | lstm | 0.8815 | 0.7257 |
| count | rnn | 0.9005 | 0.7921 |
| glove | birnn | 0.5071 | 0.2763 |
| glove | lstm | 0.4123 | 0.1721 |
| glove | rnn | 0.4455 | 0.2005 |
| pmi | birnn | 0.8720 | 0.7516 |
| pmi | lstm | 0.7251 | 0.4748 |
| pmi | rnn | 0.8294 | 0.6421 |
| tfidf | birnn | 0.9005 | 0.7925 |
| tfidf | lstm | 0.8768 | 0.6818 |
| tfidf | rnn | 0.8957 | 0.7904 |
| word2vec | birnn | 0.7915 | 0.5739 |
| word2vec | lstm | 0.6303 | 0.3364 |
| word2vec | rnn | 0.6967 | 0.4092 |
