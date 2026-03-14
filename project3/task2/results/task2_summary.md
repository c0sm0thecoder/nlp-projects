# Task 2 Word2Vec Summary

## Configuration

| Parameter | Value |
| --- | --- |
| input | project3\poems_cleaned.parquet |
| text_col | text |
| min_count | 5 |
| embedding_dim | 100 |
| window_size | 5 |
| negative_samples | 5 |
| subsample_t | 0.0001 |
| epochs | 8 |
| batch_size | 2048 |
| seed | 42 |
| vocab_size | 6575 |
| tokens_after_min_count | 154027 |
| tokens_after_subsampling | 102497 |

## Training Loss

| Model | Final Average Loss | Examples Seen |
| --- | ---: | ---: |
| skipgram | 1.576596 | 993290 |
| cbow | 1.323117 | 102497 |

## Similarity Notes

The nearest-neighbor results mostly capture poetic association, style, and lexical relatedness rather than strict dictionary synonymy.

### Skipgram

| Query Word | Top 3 Similar Words |
| --- | --- |
| aşıq | ələsgər, dəyirmanın, ələsgərə |
| can | yetǝr, sanır, tor |
| dil | qeyd, edǝr, mahi |
| yar | fǝda, bivǝfa, xandadır |
| gözəl | kimsənin, çaşır, nizamı |
| könül | bǝnzǝr, yüzün, sayru |
| gül | rəfiqim, ayrılır, gülşəndə |
| göz | ruzigarım, soyuq, dəyməsin |
| dərd | ürəkdən, tüf, dərdi |
| sultan | geysin, cavad, mehman |

### Cbow

| Query Word | Top 3 Similar Words |
| --- | --- |
| aşıq | ələsgərə, bunlara, qurtarandan |
| can | sənindir, qandasan, təndə |
| dil | çöhreyi, xun, əşk |
| yar | etmǝk, baxtın, gizli |
| gözəl | buxağın, əndam, büllur |
| könül | yandı, qoyma, axdı |
| gül | lalə, gülşəndə, gülzar |
| göz | gözə, axdı, yaş |
| dərd | dərman, bilmir, еyləmişəm |
| sultan | yeksan, xan, sərbəsər |

## Vector Arithmetic Notes

Each probe uses `neighbor_1 - neighbor_2 + query_word` with neighbors chosen deterministically from the learned top-10 results after excluding very common tokens.

| Model | Equation | Top Result |
| --- | --- | --- |
| skipgram | dəyirmanın - ələsgərə + aşıq | qabağında |
| skipgram | yetǝr - sanır + can | yüzündǝ |
| skipgram | qeyd - edǝr + dil | ətlaz |
| skipgram | fǝda - bivǝfa + yar | çəkən |
| skipgram | kimsənin - çaşır + gözəl | elmə |
| cbow | ələsgərə - bunlara + aşıq | oxumağa |
| cbow | sənindir - qandasan + can | ahi |
| cbow | çöhreyi - xun + dil | əşk |
| cbow | etmǝk - baxtın + yar | nǝdir |
| cbow | buxağın - əndam + gözəl | hayıf |
