"""
sync_all.py — Incremental sync from Slack and Confluence into Pinecone.

Tracks per-source timestamps in sync_state.json to avoid re-indexing unchanged data.
Run from inside project5/:  python scripts/sync_all.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.vector_store import upsert_documents
from core.config import get_settings
from core.logger import get_logger
from ingestion.confluence_engine import fetch_confluence_documents
from ingestion.slack_engine import fetch_slack_documents

logger = get_logger("sync_all")


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_sync() -> None:
    settings = get_settings()
    state_path = Path(settings.sync_state_path)
    state = _load_state(state_path)

    now_ts = str(time.time())
    now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── Slack ─────────────────────────────────────────────────────────────
    last_slack_ts = state.get("slack_last_sync_ts")
    logger.info("Slack sync (oldest=%s)...", last_slack_ts or "all time")
    slack_docs = fetch_slack_documents(oldest=last_slack_ts)
    if slack_docs:
        upsert_documents(slack_docs, namespace="slack")
    state["slack_last_sync_ts"] = now_ts

    # ── Confluence ────────────────────────────────────────────────────────
    last_cf_date = state.get("confluence_last_sync_date")
    logger.info("Confluence sync (lastModified > %s)...", last_cf_date or "all time")
    cf_docs = fetch_confluence_documents(last_modified_date=last_cf_date)
    if cf_docs:
        upsert_documents(cf_docs, namespace="confluence")
    state["confluence_last_sync_date"] = now_date

    _save_state(state_path, state)
    logger.info(
        "Sync complete — Slack: %d docs, Confluence: %d docs. State saved to %s.",
        len(slack_docs),
        len(cf_docs),
        state_path,
    )


if __name__ == "__main__":
    run_sync()
