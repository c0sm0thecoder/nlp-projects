from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from neo4j import GraphDatabase
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


class LocalEmbeddings(Embeddings):
    """Local embeddings using sentence-transformers on Apple Silicon GPU (MPS)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Local embeddings loaded: %s on %s (%d dims)", model_name, device, self.dimension)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


@lru_cache(maxsize=1)
def get_pinecone_index():
    s = get_settings()
    pc = Pinecone(api_key=s.pinecone_api_key)
    index = pc.Index(s.pinecone_index_name)
    logger.info("Pinecone index '%s' connected.", s.pinecone_index_name)
    return index


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    s = get_settings()
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=s.google_api_key,
        temperature=1,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> LocalEmbeddings:
    """Get local GPU-accelerated embeddings (768 dims, MPS on Apple Silicon)."""
    return LocalEmbeddings("BAAI/bge-base-en-v1.5")


@lru_cache(maxsize=1)
def get_neo4j_driver():
    s = get_settings()
    driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    logger.info("Neo4j connected at %s", s.neo4j_uri)
    return driver
