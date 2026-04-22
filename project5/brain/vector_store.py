from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from brain.chunker import chunk_documents
from core.clients import get_embeddings, get_pinecone_index
from core.logger import get_logger

logger = get_logger(__name__)

_EMBED_BATCH_SIZE = 64  # larger batches since local GPU is fast


def _doc_id(doc: Document) -> str:
    chunk_idx = doc.metadata.get("chunk_index", 0)
    raw = f"{doc.metadata.get('source', '')}::{doc.metadata.get('url', '')}::{chunk_idx}::{doc.page_content[:64]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def upsert_documents(documents: list[Document], namespace: str, use_chunking: bool = True) -> None:
    """Embed and upsert documents into Pinecone with semantic chunking."""
    if not documents:
        logger.info("No documents to upsert for namespace '%s'.", namespace)
        return

    # Semantic chunking (GPU-accelerated)
    if use_chunking:
        documents = chunk_documents(documents)

    embeddings = get_embeddings()
    index = get_pinecone_index()

    logger.info("Upserting %d chunks → namespace '%s'...", len(documents), namespace)

    for offset in range(0, len(documents), _EMBED_BATCH_SIZE):
        batch = documents[offset : offset + _EMBED_BATCH_SIZE]
        texts = [d.page_content for d in batch]

        vectors = embeddings.embed_documents(texts)

        payload: list[dict[str, Any]] = [
            {
                "id": _doc_id(doc),
                "values": vec,
                "metadata": {**doc.metadata, "text": doc.page_content},
            }
            for doc, vec in zip(batch, vectors)
        ]
        index.upsert(vectors=payload, namespace=namespace)
        logger.info("  batch %d–%d done.", offset, offset + len(batch) - 1)

    logger.info("Upsert complete for namespace '%s'.", namespace)


def get_vector_store(namespace: str) -> PineconeVectorStore:
    return PineconeVectorStore(
        index=get_pinecone_index(),
        embedding=get_embeddings(),
        namespace=namespace,
        text_key="text",
    )
