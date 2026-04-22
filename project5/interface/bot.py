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

from brain import resolver
from core.clients import get_llm
from core.config import get_settings
from core.logger import get_logger

logger = get_logger("athena_bot")

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


def main() -> None:
    settings = get_settings()
    logging.getLogger("httpx").setLevel(logging.WARNING)

    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("clear", clear_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.info("Athena bot is running. Waiting for messages...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
