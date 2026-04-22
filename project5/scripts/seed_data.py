"""
seed_data.py — Seed Confluence pages and Slack messages, then upsert both into Pinecone.

Slack messages use chat:write.customize to simulate named personas.
Run from inside project5/:  python scripts/seed_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlassian import Confluence
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from brain.vector_store import upsert_documents
from core.config import get_settings
from core.logger import get_logger
from ingestion.confluence_engine import _strip_html

logger = get_logger("seed_data")

# ── Confluence seed ───────────────────────────────────────────────────────────

_CONFLUENCE_PAGES = [
    {
        "space": "WIKI",
        "title": "PTO Policy",
        "body": (
            "<h1>PTO Policy</h1>"
            "<p>All employees are entitled to <strong>20 days</strong> of paid time off per year. "
            "PTO accrues monthly at a rate of 1.67 days per month. "
            "Unused days may be rolled over up to a maximum of 5 days.</p>"
        ),
    },
    {
        "space": "ENG",
        "title": "Deployment Guide",
        "body": (
            "<h1>Deployment Guide</h1>"
            "<p>All services are deployed via <strong>Jenkins</strong>. "
            "Pipelines are defined in a <code>Jenkinsfile</code> at the repository root. "
            "Trigger a production deployment by pushing a tag matching <code>v*.*.*</code>.</p>"
        ),
    },
]


def _seed_confluence(cf: Confluence, settings) -> list[Document]:
    docs: list[Document] = []
    for page in _CONFLUENCE_PAGES:
        existing = cf.get_page_by_title(space=page["space"], title=page["title"])
        if existing:
            logger.info("Page '%s' already exists (id=%s), skipping create.", page["title"], existing["id"])
            page_id = existing["id"]
        else:
            created = cf.create_page(space=page["space"], title=page["title"], body=page["body"])
            page_id = created.get("id", "")
            logger.info("Created Confluence page '%s' (id=%s).", page["title"], page_id)

        docs.append(
            Document(
                page_content=f"{page['title']}\n\n{_strip_html(page['body'])}",
                metadata={
                    "source": "confluence",
                    "url": f"{settings.confluence_url}/pages/{page_id}",
                    "author_role": f"Confluence Author ({page['space']})",
                    "authority_score": 5,
                    "timestamp": "2024-01-15T09:00:00.000Z",
                    "namespace": "confluence",
                    "space": page["space"],
                    "page_title": page["title"],
                    "page_id": page_id,
                    "last_modified_by": "System",
                },
            )
        )
    return docs


# ── Slack seed ────────────────────────────────────────────────────────────────

_SLACK_MESSAGES = [
    {
        "channel_name": "general",
        "username": "Alex Chen",
        "icon_emoji": ":hammer_and_wrench:",
        "text": (
            "Team announcement: Jenkins is dead. We use GitHub Actions now. "
            "The wiki is wrong — I've migrated all service pipelines this week. "
            "Update your local workflows accordingly. Reach out if you need help."
        ),
        "author_role": "Lead Architect",
        "authority_score": 10,
        "timestamp_hint": "2024-06-15",
    },
    {
        "channel_name": "hr-updates",
        "username": "Sarah Mitchell",
        "icon_emoji": ":memo:",
        "text": (
            "HR Update: New update effective immediately — all Lead roles now receive 25 days PTO annually. "
            "This supersedes the current 20-day policy listed in the Company Wiki. "
            "Please update your leave tracking records."
        ),
        "author_role": "HR Lead",
        "authority_score": 10,
        "timestamp_hint": "2024-07-01",
    },
    {
        "channel_name": "engineering",
        "username": "Jordan Kim",
        "icon_emoji": ":technologist:",
        "text": (
            "Hey team — how do I deploy the Auth-Service? "
            "The wiki says Jenkins but I heard we moved to something else?"
        ),
        "author_role": "Junior Developer",
        "authority_score": 3,
        "timestamp_hint": "2024-07-10",
    },
]


def _get_channel_id(client: WebClient, name: str) -> str | None:
    cursor = None
    while True:
        kwargs: dict = {"types": "public_channel", "limit": 200}
        if cursor:
            kwargs["cursor"] = cursor
        resp = client.conversations_list(**kwargs)
        for ch in resp.get("channels", []):
            if ch.get("name") == name:
                return ch["id"]
        cursor = resp.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            return None


def _seed_slack(client: WebClient) -> list[Document]:
    docs: list[Document] = []
    for msg in _SLACK_MESSAGES:
        channel_id = _get_channel_id(client, msg["channel_name"])
        if not channel_id:
            logger.warning("Channel '#%s' not found — skipping.", msg["channel_name"])
            continue

        ts = "0"
        try:
            resp = client.chat_postMessage(
                channel=channel_id,
                text=msg["text"],
                username=msg["username"],
                icon_emoji=msg["icon_emoji"],
            )
            ts = resp["ts"]
            logger.info("Posted as '%s' in #%s (ts=%s).", msg["username"], msg["channel_name"], ts)
        except SlackApiError as exc:
            logger.error("Failed to post to #%s: %s", msg["channel_name"], exc)

        url = f"https://slack.com/archives/{channel_id}/p{ts.replace('.', '')}"
        docs.append(
            Document(
                page_content=msg["text"],
                metadata={
                    "source": "slack",
                    "url": url,
                    "author_role": msg["author_role"],
                    "authority_score": msg["authority_score"],
                    "timestamp": ts,
                    "namespace": "slack",
                    "channel_id": channel_id,
                    "author_name": msg["username"],
                },
            )
        )
    return docs


# ── Index bootstrap ───────────────────────────────────────────────────────────

def _ensure_index(settings) -> None:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = [i.name for i in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        logger.info("Creating Pinecone index '%s'...", settings.pinecone_index_name)
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_region),
        )
        logger.info("Index created.")
    else:
        logger.info("Pinecone index '%s' already exists.", settings.pinecone_index_name)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    settings = get_settings()

    logger.info("=== Athena seed starting ===")
    _ensure_index(settings)

    logger.info("--- Seeding Confluence ---")
    cf = Confluence(
        url=settings.confluence_url,
        username=settings.confluence_user,
        password=settings.confluence_api_token,
        cloud=True,
    )
    cf_docs = _seed_confluence(cf, settings)
    upsert_documents(cf_docs, namespace="confluence")

    logger.info("--- Seeding Slack ---")
    slack_client = WebClient(token=settings.slack_bot_token)
    slack_docs = _seed_slack(slack_client)
    upsert_documents(slack_docs, namespace="slack")

    logger.info("=== Seed complete. Confluence: %d, Slack: %d ===", len(cf_docs), len(slack_docs))
