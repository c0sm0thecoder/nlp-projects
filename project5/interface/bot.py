"""
bot.py — Async Telegram bot interface for Athena with persistent conversation memory.

Run from inside project5/:  python interface/bot.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from brain import resolver, knowledge_graph as kg
from core.clients import get_llm, get_neo4j_driver
from core.config import get_settings
from core.logger import get_logger

logger = get_logger("athena_bot")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _whois(name: str) -> str:
    """Look up a person by name."""
    results = kg.find_entity_by_name(name)
    persons = [r for r in results if r["label"] == "Person"]

    if not persons:
        return f"No person found matching '{name}'."

    lines = []
    for p in persons[:3]:
        props = p["props"]
        lines.append(
            f"Name: {props.get('name', 'Unknown')}\n"
            f"Role: {props.get('role', 'Unknown')}\n"
            f"Department: {props.get('department', 'Unknown')}\n"
            f"Email: {props.get('email', 'N/A')}\n"
            f"Authority: {props.get('authority_score', 'N/A')}"
        )
    return "\n\n".join(lines)


def _team(department: str) -> str:
    """List all people in a department."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Person)
            WHERE toLower(p.department) CONTAINS toLower($dept)
            RETURN p.name AS name, p.role AS role, p.email AS email
            ORDER BY p.authority_score DESC
            """,
            dept=department,
        )
        members = list(result)

    if not members:
        return f"No team members found in '{department}'."

    lines = [f"Team: {department}\n"]
    for m in members:
        lines.append(f"- {m['name']} ({m['role']}) - {m['email']}")
    return "\n".join(lines)


def _summarize_channel(channel_name: str) -> str:
    """Summarize recent Slack channel activity."""
    from slack_sdk import WebClient
    settings = get_settings()
    client = WebClient(token=settings.slack_bot_token)

    # Find channel ID
    channel_name = channel_name.lstrip("#")
    channel_id = None
    cursor = None
    while True:
        resp = client.conversations_list(types="public_channel", limit=200, cursor=cursor)
        for ch in resp.get("channels", []):
            if ch.get("name") == channel_name:
                channel_id = ch["id"]
                break
        if channel_id:
            break
        cursor = resp.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            break

    if not channel_id:
        return f"Channel '#{channel_name}' not found."

    # Fetch recent messages
    resp = client.conversations_history(channel=channel_id, limit=30)
    messages = resp.get("messages", [])

    if not messages:
        return f"No recent messages in #{channel_name}."

    # Format messages for summarization
    msg_texts = []
    for msg in reversed(messages):
        text = msg.get("text", "")[:200]
        if text:
            msg_texts.append(text)

    prompt = f"""Summarize the following Slack channel activity from #{channel_name} in 3-5 bullet points. Focus on key discussions, decisions, and updates.

Messages:
{chr(10).join(msg_texts[:20])}

Summary:"""

    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def _diff_page(page_title: str) -> str:
    """Show what changed in a Confluence page."""
    from atlassian import Confluence
    settings = get_settings()
    cf = Confluence(
        url=settings.confluence_url,
        username=settings.confluence_user,
        password=settings.confluence_api_token,
        cloud=True,
    )

    # Search for the page
    pages = cf.cql(f'title ~ "{page_title}" AND type = page', limit=1).get("results", [])
    if not pages:
        return f"Page '{page_title}' not found."

    page_id = pages[0]["content"]["id"]
    page = cf.get_page_by_id(page_id, expand="version,history.lastUpdated")

    version = page.get("version", {})
    current_version = version.get("number", 1)
    modified_by = version.get("by", {}).get("displayName", "Unknown")
    modified_at = version.get("when", "Unknown")
    message = version.get("message", "No version message")

    result = [
        f"Page: {page.get('title')}",
        f"Current Version: {current_version}",
        f"Last Modified: {modified_at}",
        f"Modified By: {modified_by}",
        f"Change Note: {message}",
    ]

    if current_version > 1:
        # Get previous version content for comparison summary
        try:
            history = cf.get_page_by_id(page_id, expand="body.storage", version=current_version - 1)
            old_length = len(history.get("body", {}).get("storage", {}).get("value", ""))
            current = cf.get_page_by_id(page_id, expand="body.storage")
            new_length = len(current.get("body", {}).get("storage", {}).get("value", ""))
            diff = new_length - old_length
            result.append(f"Content Change: {'+' if diff > 0 else ''}{diff} characters")
        except Exception:
            pass

    return "\n".join(result)


def _find_expert(topic: str) -> str:
    """Find who to talk to about a topic."""
    prompt = f"""Based on this topic: "{topic}"

Given a typical tech company structure with these departments and roles:
- Engineering: CTO, VP Engineering, Lead Architect, Senior Engineers, DevOps
- Product: Director of Product, Product Managers, UX Lead
- HR: HR Lead
- Sales: VP Sales
- Marketing: Director of Marketing
- Finance: CFO
- Legal: General Counsel
- IT Operations: Director of IT

Who would be the best person(s) to talk to about this topic? Be specific about the role and why.
Keep response to 2-3 sentences."""

    llm = get_llm()
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    # Also search the knowledge graph
    results = kg.find_entity_by_name(topic)
    if results:
        people = [r for r in results if r["label"] == "Person"]
        if people:
            names = [p["props"].get("name") for p in people[:2]]
            answer += f"\n\nRelated people in knowledge graph: {', '.join(names)}"

    return answer


def _sync_now() -> str:
    """Trigger manual sync."""
    import requests
    try:
        # Try to call sync service if running
        requests.post("http://localhost:8000/sync/slack", timeout=2)
        requests.post("http://localhost:8000/sync/confluence", timeout=2)
        return "Sync triggered. Check sync service logs for progress."
    except Exception:
        return "Sync service not running. Start it with: python interface/sync_service.py"

_GREETING = (
    "Hello! I am *Athena*, your Wise Company Historian.\n\n"
    "I have indexed your Slack history and Confluence wiki and can resolve "
    "conflicts between them using source authority and recency.\n\n"
    "Ask me anything — policies, deployments, team decisions.\n\n"
    "Use /clear to reset our conversation history."
)

_HISTORY_DIR = Path(__file__).resolve().parents[1] / "chat_history"
_MAX_RECENT = 10  # Keep last 10 message pairs before summarizing

_SUMMARY_PROMPT = """\
Summarize this conversation in 2-3 sentences, focusing on the main topics discussed and any key facts established. Be concise.

Conversation:
{conversation}

Summary:"""


def _get_history(chat_id: int) -> FileChatMessageHistory:
    _HISTORY_DIR.mkdir(exist_ok=True)
    return FileChatMessageHistory(str(_HISTORY_DIR / f"{chat_id}.json"))


def _get_summary_path(chat_id: int) -> Path:
    _HISTORY_DIR.mkdir(exist_ok=True)
    return _HISTORY_DIR / f"{chat_id}_summary.json"


def _load_summary(chat_id: int) -> str:
    path = _get_summary_path(chat_id)
    if path.exists():
        data = json.loads(path.read_text())
        return data.get("summary", "")
    return ""


def _save_summary(chat_id: int, summary: str) -> None:
    path = _get_summary_path(chat_id)
    path.write_text(json.dumps({"summary": summary}))


def _summarize_conversation(messages: list) -> str:
    """Use LLM to summarize older messages."""
    conversation_text = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            conversation_text.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            conversation_text.append(f"Athena: {msg.content[:300]}")

    prompt = _SUMMARY_PROMPT.format(conversation="\n".join(conversation_text))
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def _history_to_list(chat_history: FileChatMessageHistory, chat_id: int) -> list[dict[str, str]]:
    messages = chat_history.messages
    result = []

    # Add existing summary if present
    summary = _load_summary(chat_id)
    if summary:
        result.append({"role": "system", "content": f"Previous conversation summary: {summary}"})

    # Add recent messages
    recent = messages[-_MAX_RECENT * 2:]
    for msg in recent:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})

    return result


def _maybe_summarize(chat_history: FileChatMessageHistory, chat_id: int) -> None:
    """Summarize older messages if history exceeds threshold."""
    messages = chat_history.messages

    if len(messages) > _MAX_RECENT * 2 + 2:
        # Messages to summarize (everything except recent)
        to_summarize = messages[:-_MAX_RECENT * 2]

        # Get existing summary
        existing_summary = _load_summary(chat_id)
        if existing_summary:
            # Prepend existing summary as context
            to_summarize = [SystemMessage(content=f"Previous summary: {existing_summary}")] + list(to_summarize)

        # Generate new summary
        new_summary = _summarize_conversation(to_summarize)
        _save_summary(chat_id, new_summary)

        # Keep only recent messages
        recent = messages[-_MAX_RECENT * 2:]
        chat_history.clear()
        for msg in recent:
            if isinstance(msg, HumanMessage):
                chat_history.add_user_message(msg.content)
            elif isinstance(msg, AIMessage):
                chat_history.add_ai_message(msg.content)

        logger.info("Summarized conversation for chat %d", chat_id)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    history = _get_history(chat_id)
    history.clear()
    summary_path = _get_summary_path(chat_id)
    if summary_path.exists():
        summary_path.unlink()
    await update.message.reply_text(_GREETING, parse_mode="Markdown")


async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    history = _get_history(chat_id)
    history.clear()
    summary_path = _get_summary_path(chat_id)
    if summary_path.exists():
        summary_path.unlink()
    await update.message.reply_text("Conversation history cleared.")


async def whois_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /whois <name>")
        return

    name = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _whois, name)
    await update.message.reply_text(result)


async def team_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /team <department>")
        return

    department = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _team, department)
    await update.message.reply_text(result)


async def summarize_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /summarize <#channel>")
        return

    channel = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _summarize_channel, channel)
        await update.message.reply_text(result)
    except Exception as e:
        logger.error("Summarize error: %s", e)
        await update.message.reply_text(f"Error summarizing channel: {e}")


async def diff_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /diff <page title>")
        return

    page_title = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _diff_page, page_title)
        await update.message.reply_text(result)
    except Exception as e:
        logger.error("Diff error: %s", e)
        await update.message.reply_text(f"Error getting page diff: {e}")


async def expert_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /expert <topic>")
        return

    topic = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _find_expert, topic)
        await update.message.reply_text(result)
    except Exception as e:
        logger.error("Expert error: %s", e)
        await update.message.reply_text(f"Error finding expert: {e}")


async def sync_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    result = _sync_now()
    await update.message.reply_text(result)


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = update.message.text
    if not question or not question.strip():
        return

    chat_id = update.effective_chat.id
    chat_history = _get_history(chat_id)

    await context.bot.send_chat_action(
        chat_id=chat_id,
        action=ChatAction.TYPING,
    )

    try:
        history_list = _history_to_list(chat_history, chat_id)
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None, resolver.ask, question, history_list
        )
    except Exception as exc:
        logger.error("Resolver error: %s", exc, exc_info=True)
        answer = "I'm sorry, I encountered an error while searching the knowledge base. Please try again."

    chat_history.add_user_message(question)
    chat_history.add_ai_message(answer)

    # Summarize if history is getting long
    await asyncio.get_event_loop().run_in_executor(
        None, _maybe_summarize, chat_history, chat_id
    )

    await update.message.reply_text(answer)


async def post_init(application) -> None:
    """Register bot commands with Telegram."""
    from telegram import BotCommand
    commands = [
        BotCommand("start", "Start conversation with Athena"),
        BotCommand("clear", "Clear conversation history"),
        BotCommand("whois", "Look up a person: /whois Alex Chen"),
        BotCommand("team", "List department members: /team Engineering"),
        BotCommand("summarize", "Summarize Slack channel: /summarize #general"),
        BotCommand("diff", "Page changes: /diff PTO Policy"),
        BotCommand("expert", "Find expert: /expert kubernetes"),
        BotCommand("sync", "Trigger manual sync"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Bot commands registered.")


def main() -> None:
    settings = get_settings()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    app = Application.builder().token(settings.telegram_bot_token).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("clear", clear_handler))
    app.add_handler(CommandHandler("whois", whois_handler))
    app.add_handler(CommandHandler("team", team_handler))
    app.add_handler(CommandHandler("summarize", summarize_handler))
    app.add_handler(CommandHandler("diff", diff_handler))
    app.add_handler(CommandHandler("expert", expert_handler))
    app.add_handler(CommandHandler("sync", sync_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.info("Athena bot is running. Waiting for messages...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
