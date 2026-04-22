"""
import_slack_users.py — Import real Slack users into the knowledge graph.

Run: python scripts/import_slack_users.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slack_sdk import WebClient

from brain import knowledge_graph as kg
from core.config import get_settings
from core.logger import get_logger

logger = get_logger("import_slack_users")


def import_slack_users() -> None:
    """Fetch all Slack users and add to Neo4j."""
    settings = get_settings()
    client = WebClient(token=settings.slack_bot_token)

    logger.info("Fetching Slack users...")

    cursor = None
    users_added = 0

    while True:
        resp = client.users_list(cursor=cursor, limit=200)
        members = resp.get("members", [])

        for member in members:
            # Skip bots and deleted users
            if member.get("is_bot") or member.get("deleted"):
                continue

            profile = member.get("profile", {})
            user_id = member.get("id", "")
            name = profile.get("real_name") or profile.get("display_name") or member.get("name", "Unknown")
            email = profile.get("email", "")
            title = profile.get("title", "Team Member")

            if not name or name == "Unknown":
                continue

            # Determine authority from title
            title_lower = title.lower()
            if any(x in title_lower for x in ["ceo", "cto", "cfo", "vp", "president", "chief"]):
                authority = 10
            elif any(x in title_lower for x in ["director", "head", "lead", "principal", "manager"]):
                authority = 8
            elif any(x in title_lower for x in ["senior", "staff", "architect"]):
                authority = 6
            else:
                authority = 4

            # Guess department from title
            dept = "General"
            if any(x in title_lower for x in ["engineer", "develop", "software", "backend", "frontend", "devops"]):
                dept = "Engineering"
            elif any(x in title_lower for x in ["product", "pm"]):
                dept = "Product"
            elif any(x in title_lower for x in ["design", "ux", "ui"]):
                dept = "Design"
            elif any(x in title_lower for x in ["sales", "account"]):
                dept = "Sales"
            elif any(x in title_lower for x in ["market", "growth"]):
                dept = "Marketing"
            elif any(x in title_lower for x in ["hr", "people", "recruit"]):
                dept = "HR"
            elif any(x in title_lower for x in ["finance", "account", "billing"]):
                dept = "Finance"

            # Add to knowledge graph
            kg.upsert_person(
                person_id=f"slack_{user_id}",
                name=name,
                role=title or "Team Member",
                department=dept,
                authority_score=authority,
                email=email,
            )
            users_added += 1
            logger.info("Added: %s (%s) - %s", name, title, dept)

        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    logger.info("Done! Added %d users to knowledge graph.", users_added)


if __name__ == "__main__":
    import_slack_users()
