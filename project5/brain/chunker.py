"""
chunker.py — Semantic chunking using local GPU embeddings.

Splits documents at natural semantic boundaries by comparing
sentence embeddings and finding breakpoints where similarity drops.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document

from core.clients import get_embeddings
from core.logger import get_logger

logger = get_logger(__name__)

_MIN_CHUNK_SIZE = 100  # chars
_MAX_CHUNK_SIZE = 1500  # chars
_SIMILARITY_THRESHOLD = 0.75  # lower = more aggressive splitting


@dataclass
class Chunk:
    text: str
    start_idx: int
    end_idx: int


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter empty
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def semantic_chunk(text: str, min_size: int = _MIN_CHUNK_SIZE, max_size: int = _MAX_CHUNK_SIZE) -> list[str]:
    """
    Split text into semantic chunks using embedding similarity.

    Finds natural breakpoints where semantic similarity between
    adjacent sentences drops below threshold.
    """
    if len(text) <= max_size:
        return [text]

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    # Get embeddings for all sentences (batched, GPU-accelerated)
    embeddings = get_embeddings()
    sentence_embeddings = np.array(embeddings.embed_documents(sentences))

    # Find breakpoints where similarity drops
    breakpoints = []
    for i in range(len(sentences) - 1):
        sim = _cosine_similarity(sentence_embeddings[i], sentence_embeddings[i + 1])
        if sim < _SIMILARITY_THRESHOLD:
            breakpoints.append(i + 1)

    # Build chunks from breakpoints
    chunks = []
    current_chunk = []
    current_size = 0

    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        current_size += len(sentence)

        # Split if at breakpoint and chunk is big enough, or if too big
        should_split = (
            (i + 1 in breakpoints and current_size >= min_size) or
            current_size >= max_size
        )

        if should_split and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0

    # Don't forget last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.info("Split %d chars into %d chunks", len(text), len(chunks))
    return chunks


def chunk_document(doc: Document) -> list[Document]:
    """
    Split a document into semantically chunked documents.
    Preserves metadata and adds chunk index.
    """
    chunks = semantic_chunk(doc.page_content)

    chunked_docs = []
    for i, chunk_text in enumerate(chunks):
        chunked_doc = Document(
            page_content=chunk_text,
            metadata={
                **doc.metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
        )
        chunked_docs.append(chunked_doc)

    return chunked_docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Chunk multiple documents."""
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    logger.info("Chunked %d docs into %d chunks", len(docs), len(all_chunks))
    return all_chunks
