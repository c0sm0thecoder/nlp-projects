"""
resolver.py — Graph-first RAG resolver for Athena.

Query flow:
1. Extract entities from question using LLM
2. Query knowledge graph for related entities (2-hop)
3. Get document IDs connected to graph entities
4. Vector search with graph context
5. Merge and rank results
6. LLM synthesis with enriched context
"""
from __future__ import annotations

import time
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from brain.vector_store import get_vector_store
from brain.entity_extractor import extract_entities_from_question
from brain import knowledge_graph as kg
from core.clients import get_llm
from core.logger import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0


def _retriever_with_retry(store, question: str, k: int = 4) -> list:
    """Retrieve documents with retry on rate limit."""
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    for attempt in range(_MAX_RETRIES):
        try:
            return retriever.invoke(question)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < _MAX_RETRIES - 1:
                    wait = _RETRY_DELAY * (attempt + 1)
                    logger.warning("Rate limited, retrying in %.1fs...", wait)
                    time.sleep(wait)
                else:
                    logger.error("Rate limit exceeded after %d retries", _MAX_RETRIES)
                    return []
            else:
                raise
    return []

_SYSTEM_PROMPT = """\
You are Athena, the Wise Company Historian. You have access to both a knowledge graph (structured relationships between people, departments, projects, and services) and document search.

When answering:
1. Use the graph context to understand relationships and organizational structure.
2. Compare timestamps of all retrieved documents.
3. Weigh information from users with higher 'authority_score' more heavily.
4. If a Lead's Slack message (newer) contradicts a Confluence page (older), prioritize the Slack message but explicitly state: 'The Official Wiki suggests X, but [Name] (Lead) updated this in Slack on [Date] to Y'.
5. Use the conversation history to understand context from previous questions.

IMPORTANT: Format your response as plain text without any markdown formatting (no asterisks, no bullet points with *, no **bold**). Use simple dashes (-) for lists and plain text for emphasis.

{history_section}
Graph Context (relationships and entities):
{graph_context}

Document Context (from Slack and Confluence):
{doc_context}

Question: {question}"""

_NAMESPACES = ["slack", "confluence"]
_TOP_K = 4


def _format_timestamp(ts: str) -> str:
    if not ts or ts == "?":
        return "Unknown date"
    try:
        ts_float = float(ts)
        return datetime.fromtimestamp(ts_float).strftime("%B %d, %Y at %H:%M")
    except (ValueError, TypeError):
        if "T" in str(ts):
            return ts.split("T")[0]
        return str(ts)


def _format_docs(docs: list) -> str:
    if not docs:
        return "No documents found."
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        author = m.get("author_name") or m.get("last_modified_by") or "Unknown"
        timestamp = _format_timestamp(m.get("timestamp", "?"))
        graph_boost = " [Graph-connected]" if m.get("graph_boost") else ""
        parts.append(
            f"[{i}] Source: {m.get('source', '?')} | "
            f"Author: {author} ({m.get('author_role', '?')}) | "
            f"Authority: {m.get('authority_score', 0)} | "
            f"Date: {timestamp}{graph_boost}\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def _format_graph_context(entities: list[dict]) -> str:
    if not entities:
        return "No graph entities found for this query."

    parts = []
    for e in entities[:15]:
        label = e.get("label", "Entity")
        name = e.get("name", "Unknown")
        props = e.get("props", {})

        details = []
        if props.get("role"):
            details.append(f"Role: {props['role']}")
        if props.get("department"):
            details.append(f"Dept: {props['department']}")
        if props.get("email"):
            details.append(f"Email: {props['email']}")
        if props.get("authority_score"):
            details.append(f"Authority: {props['authority_score']}")
        if props.get("status"):
            details.append(f"Status: {props['status']}")
        if props.get("owner_dept"):
            details.append(f"Owner: {props['owner_dept']}")
        if props.get("tech_stack"):
            details.append(f"Tech: {', '.join(props['tech_stack'][:3])}")

        detail_str = f" ({', '.join(details)})" if details else ""
        parts.append(f"- {label}: {name}{detail_str}")

    return "\n".join(parts)


def _ts_float(doc) -> float:
    raw = doc.metadata.get("timestamp", "0")
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0.0


def _format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    parts = ["Conversation History:"]
    for msg in history:
        if msg["role"] == "system":
            parts.append(msg["content"])
        elif msg["role"] == "user":
            parts.append(f"User: {msg['content'][:500]}")
        else:
            parts.append(f"Athena: {msg['content'][:500]}")
    return "\n".join(parts) + "\n\n"


def ask(question: str, history: list[dict[str, str]] | None = None) -> str:
    """Graph-first RAG: extract entities, query graph, then vector search."""

    # Step 1: Extract entities from question (include recent history for context)
    search_context = question
    if history:
        recent_user_msgs = [m["content"] for m in history[-4:] if m["role"] == "user"]
        search_context = " ".join(recent_user_msgs + [question])

    entity_names = extract_entities_from_question(search_context)
    logger.info("Extracted entities: %s", entity_names)

    # Step 2: Query knowledge graph for related entities
    graph_entities = []
    graph_doc_ids = set()

    if entity_names:
        try:
            graph_entities = kg.query_related_entities(entity_names, hops=2)
            entity_ids = [e["id"] for e in graph_entities if e.get("id")]
            graph_doc_ids = set(kg.get_documents_for_entities(entity_ids))
            logger.info("Graph found %d entities, %d connected docs",
                       len(graph_entities), len(graph_doc_ids))
        except Exception as e:
            logger.warning("Graph query failed (Neo4j may not be running): %s", e)

    # Step 3: Vector search with retry
    all_docs = []
    for ns in _NAMESPACES:
        store = get_vector_store(ns)
        docs = _retriever_with_retry(store, question, k=_TOP_K)

        for doc in docs:
            doc_id = doc.metadata.get("url", "")
            if doc_id in graph_doc_ids:
                doc.metadata["graph_boost"] = True
        all_docs.extend(docs)

    # Step 4: Rank documents
    def score(doc):
        authority = int(doc.metadata.get("authority_score", 0))
        recency = _ts_float(doc)
        graph_boost = 100 if doc.metadata.get("graph_boost") else 0
        return -(authority + graph_boost), -recency

    all_docs.sort(key=score)

    if not all_docs and not graph_entities:
        return "I could not find any relevant information in the knowledge base."

    # Step 5: Build context and synthesize
    graph_context = _format_graph_context(graph_entities)
    doc_context = _format_docs(all_docs)
    history_section = _format_history(history or [])

    prompt = ChatPromptTemplate.from_template(_SYSTEM_PROMPT)
    chain = (
        {
            "graph_context": lambda _: graph_context,
            "doc_context": lambda _: doc_context,
            "history_section": lambda _: history_section,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    logger.info("Athena answering with graph context: %s", question[:80])
    return chain.invoke(question)


def ask_simple(question: str, history: list[dict[str, str]] | None = None) -> str:
    """Fallback: Simple vector-only RAG (no graph)."""
    all_docs = []
    for ns in _NAMESPACES:
        store = get_vector_store(ns)
        retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": _TOP_K})
        all_docs.extend(retriever.invoke(question))

    if not all_docs:
        return "I could not find any relevant information in the knowledge base."

    all_docs.sort(key=lambda d: (-int(d.metadata.get("authority_score", 0)), -_ts_float(d)))

    simple_prompt = """\
You are Athena, the Wise Company Historian. Answer based on the context.
Format your response as plain text without markdown.

Context:
{context}

Question: {question}"""

    prompt = ChatPromptTemplate.from_template(simple_prompt)
    chain = (
        {"context": lambda _: _format_docs(all_docs), "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    return chain.invoke(question)
