from __future__ import annotations

import time
from typing import Any

from langchain.schema import Document
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

_AUTHORITY_KEYWORDS: dict[str, int] = {
    "lead": 10,
    "manager": 10,
    "architect": 10,
    "senior": 7,
}


def _score_title(title: str) -> int:
    lower = title.lower()
    for kw, score in _AUTHORITY_KEYWORDS.items():
        if kw in lower:
            return score
    return 3


def _get_user_info(
    client: WebClient, user_id: str, cache: dict[str, dict[str, str]]
) -> dict[str, str]:
    if user_id in cache:
        return cache[user_id]
    try:
        resp = client.users_info(user=user_id)
        profile = resp["user"].get("profile", {})
        cache[user_id] = {
            "display_name": profile.get("display_name") or profile.get("real_name", user_id),
            "title": profile.get("title", ""),
        }
    except SlackApiError as exc:
        logger.warning("Could not fetch user info for %s: %s", user_id, exc)
        cache[user_id] = {"display_name": user_id, "title": ""}
    return cache[user_id]


def fetch_slack_documents(oldest: str | None = None) -> list[Document]:
    """Fetch all messages from configured channels, aggregating threads into single Documents."""
    settings = get_settings()
    client = WebClient(token=settings.slack_bot_token)
    documents: list[Document] = []
    user_cache: dict[str, dict[str, str]] = {}

    for channel_id in settings.slack_channel_list:
        logger.info("Fetching channel %s...", channel_id)
        cursor: str | None = None

        while True:
            kwargs: dict[str, Any] = {"channel": channel_id, "limit": 200}
            if oldest:
                kwargs["oldest"] = oldest
            if cursor:
                kwargs["cursor"] = cursor

            try:
                resp = client.conversations_history(**kwargs)
            except SlackApiError as exc:
                logger.error("Slack API error on channel %s: %s", channel_id, exc)
                break

            for msg in resp.get("messages", []):
                if msg.get("subtype"):
                    continue

                user_id = msg.get("user", "UNKNOWN")
                info = _get_user_info(client, user_id, user_cache)
                authority = _score_title(info["title"])
                ts = msg.get("ts", "0")
                thread_ts = msg.get("thread_ts")

                text_parts: list[str] = [msg.get("text", "")]

                if thread_ts and thread_ts == ts:
                    try:
                        thread_resp = client.conversations_replies(
                            channel=channel_id, ts=thread_ts, limit=100
                        )
                        for reply in thread_resp.get("messages", [])[1:]:
                            reply_id = reply.get("user", "UNKNOWN")
                            reply_info = _get_user_info(client, reply_id, user_cache)
                            text_parts.append(
                                f"[{reply_info['display_name']}]: {reply.get('text', '')}"
                            )
                    except SlackApiError as exc:
                        logger.warning("Could not fetch thread %s: %s", thread_ts, exc)

                full_text = "\n".join(filter(None, text_parts))
                if not full_text.strip():
                    continue

                documents.append(
                    Document(
                        page_content=full_text,
                        metadata={
                            "source": "slack",
                            "url": f"https://slack.com/archives/{channel_id}/p{ts.replace('.', '')}",
                            "author_role": info["title"] or "Unknown",
                            "authority_score": authority,
                            "timestamp": ts,
                            "namespace": "slack",
                            "channel_id": channel_id,
                            "author_name": info["display_name"],
                        },
                    )
                )

            next_cursor = resp.get("response_metadata", {}).get("next_cursor", "")
            if not next_cursor:
                break
            cursor = next_cursor
            time.sleep(0.5)

    logger.info("Slack ingestion complete: %d documents.", len(documents))
    return documents
