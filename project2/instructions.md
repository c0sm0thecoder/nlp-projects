# Project 2 – Language Modelling and Text Classification

## Dataset:
- **Instances**: 846 rows
- **Fields**:
  - `author` (string)
  - `title` (string)
  - `url` (string, Wikisource page)
  - `text` (string, original poem text)
  - `modern_text` (string, modernized Azerbaijani; fully filled)
- **Authors (9)**:
  - İmadəddin Nəsimi
  - Qasım bəy Zakir
  - Xaqani Şirvani
  - Molla Pənah Vaqif
  - Seyid Əzim Şirvani
  - Xurşidbanu Natəvan
  - Qətran Təbrizi
  - Şah İsmayıl Xətai
  - Məhəmməd Füzuli

## Tasks
Task 1. Calculate unigram, bigram and trigram models on your dataset. Then calculate their perplexity. (20%)

Task 2. Apply Laplace, Interpolation, Backoff and Kneser-Ney smoothing to language model and define which smoothing method is best for your DataSet. (20%)

Task 3. Apply Naive Bayes, Binary Naïve Bayes and Logistic algorithm to one of the Sentiment DataSets. Use Bag of word and sentiment lexicon for feature extraction. Use statistical significance testing and analyze which classifier is better? (30%).

Task 4. Apply logistic regression to determine whether a dot signifies the end of a sentence and, based on that, detect sentences. Use L1 and L2 regularization and compare results. (10%)

Task 5. Write a report. (20%)

Extra Task . Create UI for program results (20%)
