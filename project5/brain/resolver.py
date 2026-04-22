from __future__ import annotations

from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from brain.vector_store import get_vector_store
from core.clients import get_llm
from core.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are Athena, the Wise Company Historian. When answering:
1. Compare the timestamps of all retrieved context.
2. Weigh information from users with higher 'authority_score' more heavily.
3. If a Lead's Slack message (newer) contradicts a Confluence page (older), prioritize the Slack message but explicitly state: 'The Official Wiki suggests X, but [Name] (Lead) updated this in Slack on [Date] to Y'.

IMPORTANT: Format your response as plain text without any markdown formatting (no asterisks, no bullet points with *, no **bold**). Use simple dashes (-) for lists and plain text for emphasis.

Context:
{context}

Question: {question}"""

_NAMESPACES = ["slack", "confluence"]
_TOP_K = 3


def _format_timestamp(ts: str) -> str:
    """Convert Unix timestamp or ISO string to human-readable date."""
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
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        author = m.get("author_name") or m.get("last_modified_by") or "Unknown"
        timestamp = _format_timestamp(m.get("timestamp", "?"))
        parts.append(
            f"[{i}] Source: {m.get('source', '?')} | "
            f"Author: {author} ({m.get('author_role', '?')}) | "
            f"Authority: {m.get('authority_score', 0)} | "
            f"Date: {timestamp}\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def ask(question: str) -> str:
    """Retrieve from all namespaces, rank by authority + recency, and run the LCEL chain."""
    all_docs: list = []
    for ns in _NAMESPACES:
        store = get_vector_store(ns)
        retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": _TOP_K})
        all_docs.extend(retriever.invoke(question))

    if not all_docs:
        return "I could not find any relevant information in the knowledge base."

    def _ts_float(doc) -> float:
        raw = doc.metadata.get("timestamp", "0")
        try:
            return float(raw)
        except (ValueError, TypeError):
            return 0.0

    all_docs.sort(key=lambda d: (-int(d.metadata.get("authority_score", 0)), -_ts_float(d)))

    prompt = ChatPromptTemplate.from_template(_SYSTEM_PROMPT)
    chain = (
        {"context": lambda _: _format_docs(all_docs), "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    logger.info("Athena answering: %s", question[:80])
    return chain.invoke(question)
