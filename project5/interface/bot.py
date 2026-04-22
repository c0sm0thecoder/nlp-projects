"""
bot.py — Async Telegram bot interface for Athena with persistent conversation memory.

Run from inside project5/:  python interface/bot.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
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
_MAX_HISTORY = 10


def _get_history(chat_id: int) -> FileChatMessageHistory:
    _HISTORY_DIR.mkdir(exist_ok=True)
    return FileChatMessageHistory(str(_HISTORY_DIR / f"{chat_id}.json"))


def _history_to_list(chat_history: FileChatMessageHistory) -> list[dict[str, str]]:
    messages = chat_history.messages[-_MAX_HISTORY * 2:]
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    history = _get_history(chat_id)
    history.clear()
    await update.message.reply_text(_GREETING, parse_mode="Markdown")


async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    history = _get_history(chat_id)
    history.clear()
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
        history_list = _history_to_list(chat_history)
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None, resolver.ask, question, history_list
        )
    except Exception as exc:
        logger.error("Resolver error: %s", exc, exc_info=True)
        answer = "I'm sorry, I encountered an error while searching the knowledge base. Please try again."

    chat_history.add_user_message(question)
    chat_history.add_ai_message(answer)

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
