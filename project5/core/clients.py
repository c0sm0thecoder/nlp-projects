from __future__ import annotations

from functools import lru_cache

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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
    genai.configure(api_key=s.google_api_key)
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=s.google_api_key,
        temperature=1,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    s = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=s.google_api_key,
    )
