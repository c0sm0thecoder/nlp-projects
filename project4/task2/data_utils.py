"""
Data utilities for SQuAD v1.1 loading, vocabulary building,
GloVe embedding loading, and dataset/dataloader creation for
both baseline BiDAF and BiDAF-BERT models.
"""

from __future__ import annotations

import os
import re
import json
import zipfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1
MAX_CONTEXT_LEN = 400
MAX_QUERY_LEN = 60
MAX_WORD_LEN = 16

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DIM = 100
GLOVE_FILENAME = "glove.6B.100d.txt"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower().strip()
    # split on whitespace, then separate punctuation
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens


def char_tokenize(word: str, max_len: int = MAX_WORD_LEN) -> list[str]:
    """Character-level tokenization of a word."""
    return list(word[:max_len])


# ---------------------------------------------------------------------------
# SQuAD loading via HuggingFace datasets
# ---------------------------------------------------------------------------
def load_squad(num_train: int = 5000, num_val: int = 1000) -> tuple[list, list]:
    """Load SQuAD v1.1 from HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad", split={"train": "train", "validation": "validation"})

    train_data = []
    for item in list(ds["train"])[:num_train]:
        context = item["context"]
        question = item["question"]
        answer_text = item["answers"]["text"][0]
        answer_start = item["answers"]["answer_start"][0]
        train_data.append({
            "context": context,
            "question": question,
            "answer_text": answer_text,
            "answer_start_char": answer_start,
        })

    val_data = []
    for item in list(ds["validation"])[:num_val]:
        context = item["context"]
        question = item["question"]
        answer_text = item["answers"]["text"][0]
        answer_start = item["answers"]["answer_start"][0]
        val_data.append({
            "context": context,
            "question": question,
            "answer_text": answer_text,
            "answer_start_char": answer_start,
        })

    print(f"Loaded {len(train_data)} train, {len(val_data)} val examples from SQuAD v1.1")
    return train_data, val_data


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
class Vocabulary:
    """Word and character vocabularies."""

    def __init__(self):
        self.word2idx: dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word: dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        self.char2idx: dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.word_counter: Counter = Counter()

    def build_from_data(self, data: list[dict], min_freq: int = 1):
        """Build vocab from SQuAD data."""
        for item in data:
            for text in [item["context"], item["question"]]:
                tokens = tokenize(text)
                self.word_counter.update(tokens)
                for tok in tokens:
                    for ch in tok:
                        if ch not in self.char2idx:
                            self.char2idx[ch] = len(self.char2idx)

        for word, freq in self.word_counter.items():
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"Vocabulary: {len(self.word2idx)} words, {len(self.char2idx)} chars")

    @property
    def word_vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def char_vocab_size(self) -> int:
        return len(self.char2idx)

    def encode_word(self, word: str) -> int:
        return self.word2idx.get(word, UNK_IDX)

    def encode_char(self, char: str) -> int:
        return self.char2idx.get(char, UNK_IDX)


# ---------------------------------------------------------------------------
# GloVe
# ---------------------------------------------------------------------------
def download_glove(cache_dir: str = ".glove_cache") -> str:
    """Download GloVe embeddings if not cached."""
    os.makedirs(cache_dir, exist_ok=True)
    glove_path = os.path.join(cache_dir, GLOVE_FILENAME)

    if os.path.exists(glove_path):
        print(f"GloVe already cached at {glove_path}")
        return glove_path

    zip_path = os.path.join(cache_dir, "glove.6B.zip")
    if not os.path.exists(zip_path):
        print("Downloading GloVe embeddings (862 MB)...")
        urllib.request.urlretrieve(GLOVE_URL, zip_path)
        print("Download complete.")

    print("Extracting GloVe...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(GLOVE_FILENAME, cache_dir)
    print(f"GloVe extracted to {glove_path}")
    return glove_path


def load_glove_embeddings(
    vocab: Vocabulary, glove_path: str, embed_dim: int = GLOVE_DIM
) -> torch.Tensor:
    """Load GloVe vectors for words in vocabulary."""
    embeddings = torch.randn(vocab.word_vocab_size, embed_dim) * 0.1
    embeddings[PAD_IDX] = 0.0

    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab.word2idx:
                idx = vocab.word2idx[word]
                vec = torch.tensor([float(x) for x in parts[1:]])
                if vec.size(0) == embed_dim:
                    embeddings[idx] = vec
                    found += 1

    print(f"Loaded GloVe vectors for {found}/{vocab.word_vocab_size} words")
    return embeddings


# ---------------------------------------------------------------------------
# Answer span alignment (character offset -> token index)
# ---------------------------------------------------------------------------
def find_answer_span(context_tokens: list[str], context: str, answer_start_char: int, answer_text: str):
    """Map character-level answer span to token indices."""
    answer_end_char = answer_start_char + len(answer_text)

    # Build character-to-token mapping
    char_to_token = [None] * len(context)
    current_pos = 0
    context_lower = context.lower()
    for tok_idx, token in enumerate(context_tokens):
        # Find token in context starting from current_pos
        start = context_lower.find(token, current_pos)
        if start == -1:
            start = current_pos
        for i in range(start, min(start + len(token), len(context))):
            char_to_token[i] = tok_idx
        current_pos = start + len(token)

    # Find start and end token indices
    start_token = None
    end_token = None
    for i in range(answer_start_char, min(answer_end_char, len(context))):
        if char_to_token[i] is not None:
            if start_token is None:
                start_token = char_to_token[i]
            end_token = char_to_token[i]

    if start_token is None or end_token is None:
        return 0, 0  # fallback

    return start_token, end_token


# ---------------------------------------------------------------------------
# Baseline BiDAF Dataset
# ---------------------------------------------------------------------------
class SQuADBaselineDataset(Dataset):
    """Dataset for baseline BiDAF (word + char ids)."""

    def __init__(self, data: list[dict], vocab: Vocabulary):
        self.data = data
        self.vocab = vocab
        self.examples = self._process()

    def _process(self) -> list[dict]:
        examples = []
        for item in self.data:
            context_tokens = tokenize(item["context"])[:MAX_CONTEXT_LEN]
            query_tokens = tokenize(item["question"])[:MAX_QUERY_LEN]

            start_tok, end_tok = find_answer_span(
                context_tokens, item["context"],
                item["answer_start_char"], item["answer_text"],
            )

            # Clamp to valid range
            start_tok = min(start_tok, len(context_tokens) - 1)
            end_tok = min(end_tok, len(context_tokens) - 1)

            examples.append({
                "context_tokens": context_tokens,
                "query_tokens": query_tokens,
                "answer_start": start_tok,
                "answer_end": end_tok,
                "answer_text": item["answer_text"],
                "context_text": item["context"],
            })
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        v = self.vocab

        # Word ids
        ctx_word = [v.encode_word(w) for w in ex["context_tokens"]]
        qry_word = [v.encode_word(w) for w in ex["query_tokens"]]

        # Char ids
        ctx_char = [
            [v.encode_char(c) for c in char_tokenize(w)]
            for w in ex["context_tokens"]
        ]
        qry_char = [
            [v.encode_char(c) for c in char_tokenize(w)]
            for w in ex["query_tokens"]
        ]

        return {
            "context_word": ctx_word,
            "context_char": ctx_char,
            "query_word": qry_word,
            "query_char": qry_char,
            "answer_start": ex["answer_start"],
            "answer_end": ex["answer_end"],
            "context_tokens": ex["context_tokens"],
            "answer_text": ex["answer_text"],
        }


def collate_baseline(batch: list[dict]) -> dict:
    """Collate function for baseline BiDAF with padding."""
    max_ctx = max(len(b["context_word"]) for b in batch)
    max_qry = max(len(b["query_word"]) for b in batch)

    ctx_word = torch.zeros(len(batch), max_ctx, dtype=torch.long)
    ctx_char = torch.zeros(len(batch), max_ctx, MAX_WORD_LEN, dtype=torch.long)
    qry_word = torch.zeros(len(batch), max_qry, dtype=torch.long)
    qry_char = torch.zeros(len(batch), max_qry, MAX_WORD_LEN, dtype=torch.long)
    ctx_mask = torch.zeros(len(batch), max_ctx)
    qry_mask = torch.zeros(len(batch), max_qry)
    starts = torch.zeros(len(batch), dtype=torch.long)
    ends = torch.zeros(len(batch), dtype=torch.long)

    context_tokens_list = []
    answer_texts = []

    for i, b in enumerate(batch):
        cw = b["context_word"]
        ctx_word[i, :len(cw)] = torch.tensor(cw)
        ctx_mask[i, :len(cw)] = 1.0

        for j, chars in enumerate(b["context_char"]):
            ctx_char[i, j, :len(chars)] = torch.tensor(chars[:MAX_WORD_LEN])

        qw = b["query_word"]
        qry_word[i, :len(qw)] = torch.tensor(qw)
        qry_mask[i, :len(qw)] = 1.0

        for j, chars in enumerate(b["query_char"]):
            qry_char[i, j, :len(chars)] = torch.tensor(chars[:MAX_WORD_LEN])

        starts[i] = b["answer_start"]
        ends[i] = b["answer_end"]
        context_tokens_list.append(b["context_tokens"])
        answer_texts.append(b["answer_text"])

    return {
        "context_word": ctx_word,
        "context_char": ctx_char,
        "query_word": qry_word,
        "query_char": qry_char,
        "context_mask": ctx_mask,
        "query_mask": qry_mask,
        "answer_start": starts,
        "answer_end": ends,
        "context_tokens": context_tokens_list,
        "answer_texts": answer_texts,
    }


# ---------------------------------------------------------------------------
# BERT BiDAF Dataset
# ---------------------------------------------------------------------------
class SQuADBertDataset(Dataset):
    """Dataset for BiDAF-BERT (uses BERT tokenizer)."""

    def __init__(
        self,
        data: list[dict],
        tokenizer: BertTokenizer,
        max_context_len: int = 384,
        max_query_len: int = 64,
    ):
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.max_query_len = max_query_len
        self.examples = self._process(data)

    def _process(self, data: list[dict]) -> list[dict]:
        examples = []
        for item in data:
            context = item["context"]
            question = item["question"]

            # Tokenize context
            ctx_enc = self.tokenizer(
                context,
                max_length=self.max_context_len,
                truncation=True,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            ctx_ids = ctx_enc["input_ids"]
            ctx_offsets = ctx_enc["offset_mapping"]

            # Tokenize query
            qry_enc = self.tokenizer(
                question,
                max_length=self.max_query_len,
                truncation=True,
                add_special_tokens=False,
            )
            qry_ids = qry_enc["input_ids"]

            # Find answer span in BERT tokens
            answer_start_char = item["answer_start_char"]
            answer_end_char = answer_start_char + len(item["answer_text"])

            start_token = 0
            end_token = 0
            for idx, (s, e) in enumerate(ctx_offsets):
                if s <= answer_start_char < e:
                    start_token = idx
                if s < answer_end_char <= e:
                    end_token = idx

            start_token = min(start_token, len(ctx_ids) - 1)
            end_token = min(end_token, len(ctx_ids) - 1)

            examples.append({
                "context_ids": ctx_ids,
                "query_ids": qry_ids,
                "answer_start": start_token,
                "answer_end": end_token,
                "answer_text": item["answer_text"],
                "context_text": context,
                "context_tokens": self.tokenizer.convert_ids_to_tokens(ctx_ids),
            })
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_bert(batch: list[dict]) -> dict:
    """Collate function for BERT BiDAF with padding."""
    max_ctx = max(len(b["context_ids"]) for b in batch)
    max_qry = max(len(b["query_ids"]) for b in batch)

    ctx_ids = torch.zeros(len(batch), max_ctx, dtype=torch.long)
    ctx_mask = torch.zeros(len(batch), max_ctx)
    qry_ids = torch.zeros(len(batch), max_qry, dtype=torch.long)
    qry_mask = torch.zeros(len(batch), max_qry)
    starts = torch.zeros(len(batch), dtype=torch.long)
    ends = torch.zeros(len(batch), dtype=torch.long)

    context_tokens_list = []
    answer_texts = []

    for i, b in enumerate(batch):
        ci = b["context_ids"]
        ctx_ids[i, :len(ci)] = torch.tensor(ci)
        ctx_mask[i, :len(ci)] = 1.0

        qi = b["query_ids"]
        qry_ids[i, :len(qi)] = torch.tensor(qi)
        qry_mask[i, :len(qi)] = 1.0

        starts[i] = b["answer_start"]
        ends[i] = b["answer_end"]
        context_tokens_list.append(b["context_tokens"])
        answer_texts.append(b["answer_text"])

    return {
        "context_ids": ctx_ids,
        "context_mask": ctx_mask,
        "query_ids": qry_ids,
        "query_mask": qry_mask,
        "answer_start": starts,
        "answer_end": ends,
        "context_tokens": context_tokens_list,
        "answer_texts": answer_texts,
    }
