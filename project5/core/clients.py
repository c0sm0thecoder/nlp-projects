from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import GraphDatabase
from pinecone import Pinecone

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)


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
    s = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=s.google_api_key,
        output_dimensionality=768,
    )


@lru_cache(maxsize=1)
def get_neo4j_driver():
    s = get_settings()
    driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    logger.info("Neo4j connected at %s", s.neo4j_uri)
    return driver
