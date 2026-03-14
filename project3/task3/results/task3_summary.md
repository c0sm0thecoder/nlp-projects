# Task 3 GloVe Summary

## Configuration

| Parameter | Value |
| --- | --- |
| input | project3\poems_cleaned.parquet |
| text_col | text |
| min_count | 5 |
| embedding_dim | 100 |
| window_size | 5 |
| x_max | 100.0 |
| alpha | 0.75 |
| epochs | 25 |
| batch_size | 4096 |
| learning_rate | 0.05 |
| seed | 42 |
| vocab_size | 6575 |
| tokens_after_min_count | 154027 |
| directed_nonzero_pairs | 876452 |
| total_pair_events | 1508590 |

## Training Loss

| Model | Final Average Weighted Loss | Pairs Seen |
| --- | ---: | ---: |
| glove | 0.013351 | 876452 |

## Similarity Notes

The nearest-neighbor results mostly capture topical and poetic co-occurrence structure rather than strict dictionary synonymy.

| Query Word | Top 3 Similar Words |
| --- | --- |
| aşıq | ələsgəri, ələsgərin, yanına |
| can | yar, eşq, eşqi |
| dil | könül, zülfü, et |
| yar | can, eşqi, eşq |
| gözəl | durub, yoxdu, dana |
| könül | dil, əşk, sənəm |
| gül | xar, tazə, naz |
| göz | yaşı, olan, görüb |
| dərd | aləm, könlümü, dərdi |
| sultan | artıq, qəbul, onlar |

## Vector Arithmetic Notes

Each probe uses `neighbor_1 - neighbor_2 + query_word` with neighbors chosen deterministically from the learned top-10 results after excluding very common tokens.

| Equation | Top Result |
| --- | --- |
| ələsgəri - ələsgərin + aşıq | ələsgər |
| yar - eşq + can | eşqi |
| könül - zülfü + dil | sənəm |
| can - eşqi + yar | eşq |
| durub - yoxdu + gözəl | od |
