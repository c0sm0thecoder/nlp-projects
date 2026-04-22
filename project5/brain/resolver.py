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

import re
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

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

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _extract_time_filter(question: str) -> tuple[float | None, float | None, str | None]:
    """Extract time constraints from question. Returns (start_ts, end_ts, description)."""
    q = question.lower()
    now = datetime.now()

    # "last January", "in January"
    for month_name, month_num in _MONTH_MAP.items():
        if month_name in q:
            # Determine year
            if "last" in q or "previous" in q:
                if now.month <= month_num:
                    year = now.year - 1
                else:
                    year = now.year - 1 if "last year" in q else now.year
            elif match := re.search(r"(20\d{2})", q):
                year = int(match.group(1))
            else:
                year = now.year if now.month > month_num else now.year - 1

            start = datetime(year, month_num, 1)
            end = start + relativedelta(months=1)
            return start.timestamp(), end.timestamp(), f"{month_name.title()} {year}"

    # "last year", "in 2023"
    if match := re.search(r"(?:in |during )(20\d{2})", q):
        year = int(match.group(1))
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59, 59)
        return start.timestamp(), end.timestamp(), str(year)

    if "last year" in q:
        year = now.year - 1
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59, 59)
        return start.timestamp(), end.timestamp(), str(year)

    # "last month"
    if "last month" in q:
        end = now.replace(day=1)
        start = end - relativedelta(months=1)
        return start.timestamp(), end.timestamp(), start.strftime("%B %Y")

    # "last week"
    if "last week" in q:
        end = now - relativedelta(days=now.weekday())
        start = end - relativedelta(weeks=1)
        return start.timestamp(), end.timestamp(), "last week"

    # "X months ago", "X weeks ago"
    if match := re.search(r"(\d+)\s+months?\s+ago", q):
        months = int(match.group(1))
        target = now - relativedelta(months=months)
        start = target.replace(day=1)
        end = start + relativedelta(months=1)
        return start.timestamp(), end.timestamp(), target.strftime("%B %Y")

    return None, None, None


_TOPIC_CHANNEL_MAP = {
    "deploy": ("engineering", "Engineering team handles deployments"),
    "deployment": ("engineering", "Engineering team handles deployments"),
    "jenkins": ("engineering", "Engineering team manages CI/CD"),
    "github": ("engineering", "Engineering team manages code repos"),
    "code": ("engineering", "Engineering team handles code questions"),
    "api": ("engineering", "Engineering team owns APIs"),
    "bug": ("engineering", "Engineering team fixes bugs"),
    "pto": ("hr-updates", "HR manages PTO policies"),
    "vacation": ("hr-updates", "HR manages vacation policies"),
    "leave": ("hr-updates", "HR manages leave policies"),
    "benefits": ("hr-updates", "HR manages benefits"),
    "onboarding": ("hr-updates", "HR handles onboarding"),
    "salary": ("hr-updates", "HR handles compensation"),
    "sales": ("sales", "Sales team handles customer inquiries"),
    "customer": ("sales", "Sales team handles customer relations"),
    "pricing": ("sales", "Sales team handles pricing"),
    "contract": ("legal", "Legal team handles contracts"),
    "compliance": ("legal", "Legal team handles compliance"),
    "budget": ("finance", "Finance team handles budgets"),
    "expense": ("finance", "Finance team handles expenses"),
    "invoice": ("finance", "Finance team handles invoicing"),
}


def _suggest_channel_for_gap(question: str, entity_names: list[str]) -> str | None:
    """Suggest a Slack channel to ask when no info is found."""
    q = question.lower()

    # Check topic keywords
    for keyword, (channel, reason) in _TOPIC_CHANNEL_MAP.items():
        if keyword in q:
            return f"#{channel} - {reason}"

    # Check entity names against departments
    for entity in entity_names:
        entity_lower = entity.lower()
        if "engineer" in entity_lower or "dev" in entity_lower:
            return "#engineering - Engineering team might know"
        if "hr" in entity_lower or "human" in entity_lower:
            return "#hr-updates - HR team might know"
        if "sales" in entity_lower:
            return "#sales - Sales team might know"
        if "finance" in entity_lower or "budget" in entity_lower:
            return "#finance - Finance team might know"

    return None


def _filter_docs_by_time(docs: list, start_ts: float | None, end_ts: float | None) -> list:
    """Filter documents to those within the time range."""
    if start_ts is None:
        return docs

    filtered = []
    for doc in docs:
        ts = doc.metadata.get("timestamp")
        if not ts:
            continue
        try:
            ts_float = float(ts)
            if start_ts <= ts_float <= end_ts:
                filtered.append(doc)
        except (ValueError, TypeError):
            # Try ISO format
            try:
                if "T" in str(ts):
                    doc_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    ts_float = doc_dt.timestamp()
                    if start_ts <= ts_float <= end_ts:
                        filtered.append(doc)
            except Exception:
                pass
    return filtered


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


def ask_stream(question: str, history: list[dict[str, str]] | None = None):
    """Graph-first RAG with streaming response. Yields chunks of text."""
    for item in ask_stream_with_status(question, history):
        if not isinstance(item, dict):  # Skip status updates
            yield item


def ask_stream_with_status(question: str, history: list[dict[str, str]] | None = None):
    """Graph-first RAG with streaming. Yields status dicts and text chunks."""

    # Step 0: Check for time-travel query
    start_ts, end_ts, time_desc = _extract_time_filter(question)
    if time_desc:
        yield {"status": "time", "message": f"⏰ Time-travel: {time_desc}..."}
        logger.info("Time filter detected: %s (%s - %s)", time_desc, start_ts, end_ts)

    # Step 1: Extract entities
    yield {"status": "extracting", "message": "🔍 Analyzing question..."}

    search_context = question
    if history:
        recent_user_msgs = [m["content"] for m in history[-4:] if m["role"] == "user"]
        search_context = " ".join(recent_user_msgs + [question])

    entity_names = extract_entities_from_question(search_context)
    logger.info("Extracted entities: %s", entity_names)

    # Step 2: Query knowledge graph
    yield {"status": "graph", "message": "🕸️ Querying knowledge graph..."}

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
            logger.warning("Graph query failed: %s", e)

    # Step 3: Vector search with retry
    yield {"status": "vector", "message": "📚 Searching documents..."}

    all_docs = []
    # Fetch more docs if filtering by time (some will be filtered out)
    fetch_k = _TOP_K * 3 if start_ts else _TOP_K
    for ns in _NAMESPACES:
        store = get_vector_store(ns)
        docs = _retriever_with_retry(store, question, k=fetch_k)

        for doc in docs:
            doc_id = doc.metadata.get("url", "")
            if doc_id in graph_doc_ids:
                doc.metadata["graph_boost"] = True
        all_docs.extend(docs)

    # Apply time filter if specified
    if start_ts:
        all_docs = _filter_docs_by_time(all_docs, start_ts, end_ts)
        logger.info("After time filter: %d docs remain", len(all_docs))

    # Step 4: Rank documents
    def score(doc):
        authority = int(doc.metadata.get("authority_score", 0))
        recency = _ts_float(doc)
        graph_boost = 100 if doc.metadata.get("graph_boost") else 0
        return -(authority + graph_boost), -recency

    all_docs.sort(key=score)

    if not all_docs and not graph_entities:
        # Knowledge gap detected - suggest who to ask
        suggestion = _suggest_channel_for_gap(question, entity_names)
        if suggestion:
            yield f"I don't have information on this topic in the knowledge base.\n\n💡 Try asking in {suggestion}"
        else:
            yield "I could not find any relevant information in the knowledge base.\n\n💡 Consider posting your question in a relevant Slack channel."
        return

    # Step 5: Build context and stream response
    yield {"status": "generating", "message": "✨ Generating response..."}

    graph_context = _format_graph_context(graph_entities)
    doc_context = _format_docs(all_docs)
    history_section = _format_history(history or [])

    # Add time context if filtering
    time_context = ""
    if time_desc:
        time_context = f"\n\nIMPORTANT: The user is asking about {time_desc}. Only use information from that time period. Make it clear you're answering based on historical data."

    full_prompt = _SYSTEM_PROMPT.format(
        graph_context=graph_context,
        doc_context=doc_context,
        history_section=history_section,
        question=question + time_context,
    )

    llm = get_llm()
    logger.info("Athena streaming answer: %s", question[:80])

    for chunk in llm.stream(full_prompt):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content


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
