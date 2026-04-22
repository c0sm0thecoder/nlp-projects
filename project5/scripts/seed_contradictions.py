"""
seed_contradictions.py — Add Slack messages that contradict/update GitLab handbook policies.

Tests the authority-aware conflict resolution in Athena.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from brain.vector_store import upsert_documents
from core.config import get_settings
from core.logger import get_logger

logger = get_logger("seed_contradictions")

# Contradicting messages from high-authority users
CONTRADICTION_MESSAGES = [
    # Travel & Expenses - contradicts GitLab handbook
    {
        "channel": "general",
        "username": "Robert Chen",
        "icon_emoji": ":money_with_wings:",
        "text": "IMPORTANT UPDATE: Effective immediately, the travel expense limit has been increased from $150/day to $200/day for meals. Also, we're now reimbursing business class flights for trips over 6 hours (was 8 hours). Updated policy coming to Confluence next week.",
        "authority": 10,
        "role": "CFO",
    },
    # Security Policy - contradicts GitLab handbook
    {
        "channel": "general",
        "username": "Diana Park",
        "icon_emoji": ":shield:",
        "text": "Security policy change: We're moving from quarterly security reviews to MONTHLY reviews starting next month. Also, all employees must now complete security training within 30 days of joining (previously 90 days). This supersedes what's in the handbook.",
        "authority": 10,
        "role": "General Counsel",
    },
    # Remote Work - contradicts any hybrid policy
    {
        "channel": "hr-updates",
        "username": "Sarah Mitchell",
        "icon_emoji": ":house:",
        "text": "Big news! After reviewing employee feedback, we're shifting to a fully remote-first model. The 2 days/week in-office requirement is now OPTIONAL. You can come in whenever you want, but it's not mandatory. Confluence will be updated soon.",
        "authority": 10,
        "role": "HR Lead",
    },
    # Code Review - contradicts engineering handbook
    {
        "channel": "engineering",
        "username": "Michael Torres",
        "icon_emoji": ":computer:",
        "text": "Engineering process update: We're reducing required code reviewers from 2 to 1 for non-critical changes. Critical paths (auth, payments, security) still need 2 reviewers. Also, review SLA is now 24 hours, not 48. Let's move faster!",
        "authority": 10,
        "role": "VP Engineering",
    },
    # PTO Carryover - contradicts handbook
    {
        "channel": "hr-updates",
        "username": "Sarah Mitchell",
        "icon_emoji": ":palm_tree:",
        "text": "PTO update for end of year: We're increasing the carryover limit from 5 days to 10 days this year due to high workload. Use it or bank it! Also reminder: Leads now get 25 days PTO, not 20 as listed in the old handbook.",
        "authority": 10,
        "role": "HR Lead",
    },
    # Communication tools - contradicts handbook
    {
        "channel": "general",
        "username": "Ryan Patel",
        "icon_emoji": ":speech_balloon:",
        "text": "Heads up: We're migrating from Zoom to Google Meet for all video calls starting next week. Zoom licenses will be revoked on the 15th. Also, we're piloting async video updates via Loom instead of status meetings for some teams.",
        "authority": 9,
        "role": "IT Director",
    },
    # Engineering workflow - contradicts GitLab workflow
    {
        "channel": "engineering",
        "username": "Alex Chen",
        "icon_emoji": ":rocket:",
        "text": "Deployment freeze lifted! We can now deploy on Fridays again (with extra caution). The old 'no Friday deploys' rule is officially retired. Just make sure you have rollback ready and someone on-call. Also, hotfixes can skip staging for P0 issues.",
        "authority": 10,
        "role": "Lead Architect",
    },
    # Values - adds to GitLab values
    {
        "channel": "general",
        "username": "Lisa Nguyen",
        "icon_emoji": ":star:",
        "text": "We're adding a 7th company value: 'Customer Obsession'. This sits alongside our existing values (Collaboration, Results, Efficiency, Diversity, Iteration, Transparency). More details in next all-hands.",
        "authority": 9,
        "role": "Product Director",
    },
]


def post_slack_messages(client: WebClient, settings):
    """Post contradiction messages to Slack."""
    channels = {c.split(":")[0]: c.split(":")[1] if ":" in c else c
                for c in settings.slack_channel_list}

    # Get actual channel IDs
    try:
        resp = client.conversations_list(types="public_channel", limit=100)
        channel_map = {ch["name"]: ch["id"] for ch in resp.get("channels", [])}
    except SlackApiError as e:
        logger.error("Failed to list channels: %s", e)
        return []

    posted_docs = []

    for msg in CONTRADICTION_MESSAGES:
        channel_name = msg["channel"]
        channel_id = channel_map.get(channel_name)

        if not channel_id:
            logger.warning("Channel #%s not found, skipping", channel_name)
            continue

        try:
            result = client.chat_postMessage(
                channel=channel_id,
                text=msg["text"],
                username=msg["username"],
                icon_emoji=msg["icon_emoji"],
            )

            ts = result["ts"]
            logger.info("Posted to #%s as %s: %s...", channel_name, msg["username"], msg["text"][:50])

            # Create document for indexing
            doc = Document(
                page_content=f"{msg['username']} ({msg['role']}): {msg['text']}",
                metadata={
                    "source": "slack",
                    "url": f"https://slack.com/archives/{channel_id}/p{ts.replace('.', '')}",
                    "author_name": msg["username"],
                    "author_role": msg["role"],
                    "authority_score": msg["authority"],
                    "timestamp": ts,
                    "namespace": "slack",
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                }
            )
            posted_docs.append(doc)

            time.sleep(1)  # Rate limit

        except SlackApiError as e:
            logger.error("Failed to post message: %s", e)

    return posted_docs


def main():
    settings = get_settings()
    client = WebClient(token=settings.slack_bot_token)

    logger.info("=== Seeding Contradiction Messages ===")

    # Post to Slack
    docs = post_slack_messages(client, settings)
    logger.info("Posted %d messages to Slack", len(docs))

    # Index in Pinecone
    if docs:
        logger.info("Indexing in Pinecone...")
        upsert_documents(docs, namespace="slack")

    logger.info("=== Done! %d contradiction messages added ===", len(docs))

    print("\n" + "="*60)
    print("TEST QUESTIONS TO ASK ATHENA:")
    print("="*60)
    print("""
1. "What's the daily meal expense limit for travel?"
   - Handbook: $150/day → Slack: $200/day (Robert Chen, CFO)

2. "How often are security reviews conducted?"
   - Handbook: quarterly → Slack: monthly (Diana Park, Legal)

3. "Do I need to come to the office?"
   - Handbook: 2 days/week required → Slack: fully optional (Sarah Mitchell, HR)

4. "How many reviewers do I need for a code review?"
   - Handbook: 2 reviewers → Slack: 1 for non-critical (Michael Torres, VP Eng)

5. "Can I carry over PTO days?"
   - Handbook: 5 days max → Slack: 10 days this year (Sarah Mitchell, HR)

6. "What video conferencing tool should I use?"
   - Handbook: Zoom → Slack: Google Meet (Ryan Patel, IT)

7. "Can I deploy on Fridays?"
   - Handbook: no Friday deploys → Slack: yes with caution (Alex Chen, Architect)

8. "What are the company values?"
   - Handbook: 6 values → Slack: 7th added (Lisa Nguyen, Product)
""")


if __name__ == "__main__":
    main()
