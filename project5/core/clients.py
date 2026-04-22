from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import GraphDatabase
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


class LocalEmbeddings(Embeddings):
    """Local embeddings using sentence-transformers on Apple Silicon GPU (MPS)."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            cls._instance = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
            logger.info("Local chunking embeddings loaded on %s", device)
        return cls._instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = self.get_instance()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        model = self.get_instance()
        embedding = model.encode(text, normalize_embeddings=True)
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
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Gemini embeddings for Pinecone storage (768 dims)."""
    s = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=s.google_api_key,
        output_dimensionality=768,
    )


@lru_cache(maxsize=1)
def get_local_embeddings() -> LocalEmbeddings:
    """Local GPU embeddings for fast chunking decisions."""
    return LocalEmbeddings()


@lru_cache(maxsize=1)
def get_neo4j_driver():
    s = get_settings()
    driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    logger.info("Neo4j connected at %s", s.neo4j_uri)
    return driver
