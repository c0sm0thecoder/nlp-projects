"""
sync_service.py — FastAPI service for automated syncing.

- Confluence webhook: POST /webhook/confluence
- Slack cron: Runs daily at midnight

Run: uvicorn interface.sync_service:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from pydantic import BaseModel

from datetime import datetime, timedelta

from brain.vector_store import upsert_documents
from ingestion.confluence_engine import fetch_confluence_documents
from ingestion.slack_engine import fetch_slack_documents
from core.clients import get_llm
from core.config import get_settings
from core.logger import get_logger

logger = get_logger("sync_service")

scheduler = AsyncIOScheduler()


# ══════════════════════════════════════════════════════════════════════════════
# SYNC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def sync_slack() -> None:
    """Fetch and index new Slack messages."""
    logger.info("Starting Slack sync...")
    try:
        docs = fetch_slack_documents(oldest=None)
        if docs:
            upsert_documents(docs, namespace="slack")
            logger.info("Slack sync complete: %d documents indexed.", len(docs))
        else:
            logger.info("Slack sync: No new documents.")
    except Exception as e:
        logger.error("Slack sync failed: %s", e)


def sync_confluence_page(page_id: str) -> None:
    """Fetch and index a specific Confluence page."""
    logger.info("Syncing Confluence page %s...", page_id)
    try:
        from atlassian import Confluence
        from langchain_core.documents import Document
        from ingestion.confluence_engine import _strip_html

        settings = get_settings()
        cf = Confluence(
            url=settings.confluence_url,
            username=settings.confluence_user,
            password=settings.confluence_api_token,
            cloud=True,
        )

        page = cf.get_page_by_id(page_id, expand="body.storage,version,space")
        if not page:
            logger.warning("Page %s not found.", page_id)
            return

        body = page.get("body", {}).get("storage", {}).get("value", "")
        title = page.get("title", "")
        space_key = page.get("space", {}).get("key", "")
        version = page.get("version", {})
        modified_by = version.get("by", {}).get("displayName", "Unknown")
        modified_at = version.get("when", "")

        content = f"{title}\n\n{_strip_html(body)}"

        doc = Document(
            page_content=content,
            metadata={
                "source": "confluence",
                "url": f"{settings.confluence_url}/pages/{page_id}",
                "author_role": f"Confluence Author ({space_key})",
                "authority_score": 5,
                "timestamp": modified_at,
                "namespace": "confluence",
                "space": space_key,
                "page_title": title,
                "page_id": page_id,
                "last_modified_by": modified_by,
            },
        )

        upsert_documents([doc], namespace="confluence")
        logger.info("Confluence page '%s' synced.", title)

    except Exception as e:
        logger.error("Confluence sync failed for page %s: %s", page_id, e)


def sync_all_confluence() -> None:
    """Fetch and index all recent Confluence changes."""
    logger.info("Starting full Confluence sync...")
    try:
        docs = fetch_confluence_documents(last_modified_date=None)
        if docs:
            upsert_documents(docs, namespace="confluence")
            logger.info("Confluence sync complete: %d documents indexed.", len(docs))
        else:
            logger.info("Confluence sync: No documents.")
    except Exception as e:
        logger.error("Confluence sync failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# WEEKLY DIGEST
# ══════════════════════════════════════════════════════════════════════════════

def generate_weekly_digest() -> None:
    """Generate and post weekly knowledge digest to Slack."""
    from slack_sdk import WebClient

    logger.info("Generating weekly digest...")
    settings = get_settings()
    client = WebClient(token=settings.slack_bot_token)

    # Get date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    start_ts = str(start_date.timestamp())

    # Collect changes
    changes = {"slack": [], "confluence": []}

    # Fetch Slack messages from last week
    try:
        slack_docs = fetch_slack_documents(oldest=start_ts)
        for doc in slack_docs[:20]:  # Limit to top 20
            author = doc.metadata.get("author_name", "Unknown")
            channel = doc.metadata.get("channel_id", "")
            content = doc.page_content[:100]
            changes["slack"].append(f"- {author}: {content}...")
    except Exception as e:
        logger.error("Failed to fetch Slack for digest: %s", e)

    # Fetch Confluence changes from last week
    try:
        conf_docs = fetch_confluence_documents(last_modified_date=start_date.strftime("%Y-%m-%d"))
        for doc in conf_docs[:10]:  # Limit to top 10
            title = doc.metadata.get("page_title", "Unknown")
            author = doc.metadata.get("last_modified_by", "Unknown")
            changes["confluence"].append(f"- {title} (by {author})")
    except Exception as e:
        logger.error("Failed to fetch Confluence for digest: %s", e)

    if not changes["slack"] and not changes["confluence"]:
        logger.info("No changes to report in weekly digest.")
        return

    # Generate summary with LLM
    prompt = f"""Generate a brief weekly knowledge digest (5-7 bullet points) summarizing the key updates, decisions, and discussions. Be concise and highlight what's important.

Slack Activity:
{chr(10).join(changes['slack'][:15]) if changes['slack'] else 'No activity'}

Confluence Updates:
{chr(10).join(changes['confluence']) if changes['confluence'] else 'No updates'}

Weekly Digest:"""

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        summary = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error("LLM summary failed: %s", e)
        summary = "Could not generate summary."

    # Format digest message
    digest_message = f"""📊 *Weekly Knowledge Digest*
_{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}_

{summary}

---
_📈 Stats: {len(changes['slack'])} Slack messages, {len(changes['confluence'])} Confluence updates_
_🤖 Generated by Athena_"""

    # Post to first channel in list
    try:
        channel_ids = settings.slack_channel_list
        if channel_ids:
            client.chat_postMessage(
                channel=channel_ids[0],
                text=digest_message,
                mrkdwn=True,
            )
            logger.info("Weekly digest posted to %s", channel_ids[0])
    except Exception as e:
        logger.error("Failed to post weekly digest: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start scheduler on startup
    scheduler.add_job(
        sync_slack,
        CronTrigger(hour=0, minute=0),  # Midnight daily
        id="slack_sync",
        replace_existing=True,
    )
    scheduler.add_job(
        generate_weekly_digest,
        CronTrigger(day_of_week="mon", hour=9, minute=0),  # Monday 9 AM
        id="weekly_digest",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started. Slack sync: midnight daily. Weekly digest: Monday 9 AM.")
    yield
    # Shutdown scheduler
    scheduler.shutdown()
    logger.info("Scheduler stopped.")


app = FastAPI(
    title="Athena Sync Service",
    description="Webhook and cron-based sync for Athena knowledge base",
    lifespan=lifespan,
)


class ConfluenceWebhookPayload(BaseModel):
    webhookEvent: str | None = None
    page: dict | None = None
    comment: dict | None = None


@app.get("/health")
async def health():
    return {"status": "ok", "scheduler_running": scheduler.running}


@app.post("/webhook/confluence")
async def confluence_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle Confluence webhook events.

    Supported events:
    - page_created
    - page_updated
    - page_restored
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event = payload.get("webhookEvent", "")
    logger.info("Received Confluence webhook: %s", event)

    # Extract page ID from various event types
    page_id = None

    if "page" in payload and payload["page"]:
        page_id = str(payload["page"].get("id", ""))
    elif "content" in payload and payload["content"]:
        page_id = str(payload["content"].get("id", ""))

    if not page_id:
        logger.warning("No page ID found in webhook payload")
        return {"status": "ignored", "reason": "no page_id"}

    # Handle relevant events
    if event in ("page_created", "page_updated", "page_restored", "content_created", "content_updated"):
        background_tasks.add_task(sync_confluence_page, page_id)
        return {"status": "accepted", "page_id": page_id, "event": event}

    return {"status": "ignored", "event": event}


@app.post("/sync/slack")
async def trigger_slack_sync(background_tasks: BackgroundTasks):
    """Manually trigger Slack sync."""
    background_tasks.add_task(sync_slack)
    return {"status": "accepted", "message": "Slack sync started"}


@app.post("/sync/confluence")
async def trigger_confluence_sync(background_tasks: BackgroundTasks):
    """Manually trigger full Confluence sync."""
    background_tasks.add_task(sync_all_confluence)
    return {"status": "accepted", "message": "Confluence sync started"}


@app.post("/digest")
async def trigger_weekly_digest(background_tasks: BackgroundTasks):
    """Manually trigger weekly digest generation."""
    background_tasks.add_task(generate_weekly_digest)
    return {"status": "accepted", "message": "Weekly digest generation started"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
