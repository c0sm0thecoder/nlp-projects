# Datasheet: Azerbaijani Classical Poems (Modernized)

**Dataset name**: az-poems-modernized (project working name)

**Date**: 2026-02-03

## 1. Motivation
- **Purpose**: Provide Azerbaijani classical poetry with a modernized paraphrase (`modern_text`) for training and evaluation, including CPT-style pretraining for Azerbaijani LLMs.
- **Language**: Azerbaijani (original + modernized).
- **Source**: Collected via Wikimedia API from Wikisource pages.

## 2. Composition
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
- **Text size (current snapshot)**:
  - Original text characters: 909,630
  - Modernized text characters: 935,784
  - Translation coverage: 846 / 846 rows filled in `modern_text`

## 3. Collection Process
- **Method**: Wikimedia API was used to collect poems from Wikisource pages for the listed authors.
- **Source URLs**: Stored in the `url` column for traceability.

## 4. Preprocessing & Cleaning
- **Metadata removal**: Removed leading template blocks (e.g., `| vikipediya_ke?idi =` ... `}}`).
- **TOC removal**: Removed table-of-contents style bullet lists (lines beginning with `*`/`**`) when they dominate the text.
- **Output**: Cleaned input saved as `data_cleaning/poems_cleaned.parquet`.

## 5. Labeling / Annotation
- **Modernization**: `modern_text` is generated using a generative model prompt that preserves line breaks and poem structure, while modernizing wording.
- **Model**: Gemini 2.5 Flash (project setting for translation runs).
- **Prompt**: Stored in `data_translation/prompts.py` and used by `data_translation/translate_poems.py`.

## 6. Uses
- **Intended**:
  - Training Azerbaijani LLMs (CPT-style pretraining).
  - Research on modernization / paraphrase generation for classical Azerbaijani poetry.
- **Out of scope**:
  - High-stakes applications (legal, medical, etc.).

## 7. Quality & Limitations
- **Coverage**: All poems are modernized in `modern_text`.
- **Model risk**: Generative output may contain paraphrase errors, omissions, or hallucinations.
- **Genre/style**: Texts are classical poetry; not representative of modern everyday Azerbaijani.

## 8. Ethics & Bias
- **Cultural bias**: Corpus is limited to canonical poets and may underrepresent other voices or dialects.
- **Religious/historical content**: Some poems include religious themes; use with care for downstream tasks.

## 9. Licensing & Rights
- **Source content**: Wikisource/Wikimedia texts are often public domain or under CC licenses; exact licensing may vary by poem.
- **Recommendation**: Verify licensing using the `url` field for each poem before redistribution or commercial use.

## 10. Maintenance
- **Maintainer**: Project team.
- **Updates**: Add translations to `modern_text` and re-run cleaning as needed.

