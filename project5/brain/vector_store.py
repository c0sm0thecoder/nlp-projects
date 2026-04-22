from __future__ import annotations

import hashlib
import time
from typing import Any

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from core.clients import get_embeddings, get_pinecone_index
from core.logger import get_logger

logger = get_logger(__name__)

_EMBED_BATCH_SIZE = 20
_RATE_LIMIT_DELAY = 1.5


def _doc_id(doc: Document) -> str:
    raw = f"{doc.metadata.get('source', '')}::{doc.metadata.get('url', '')}::{doc.page_content[:128]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def upsert_documents(documents: list[Document], namespace: str) -> None:
    """Embed and upsert documents into Pinecone under the given namespace (idempotent)."""
    if not documents:
        logger.info("No documents to upsert for namespace '%s'.", namespace)
        return

    embeddings = get_embeddings()
    index = get_pinecone_index()

    logger.info("Upserting %d docs → namespace '%s'...", len(documents), namespace)

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

        if offset + _EMBED_BATCH_SIZE < len(documents):
            time.sleep(_RATE_LIMIT_DELAY)

    logger.info("Upsert complete for namespace '%s'.", namespace)


def get_vector_store(namespace: str) -> PineconeVectorStore:
    return PineconeVectorStore(
        index=get_pinecone_index(),
        embedding=get_embeddings(),
        namespace=namespace,
        text_key="text",
    )
