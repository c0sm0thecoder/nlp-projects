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

def _whois(name: str) -> tuple[str, str]:
    """Look up a person by name. Returns (text, parse_mode)."""
    results = kg.find_entity_by_name(name)
    persons = [r for r in results if r["label"] == "Person"]

    if not persons:
        return f"No person found matching '{name}'.", "HTML"

    cards = []
    for p in persons[:3]:
        props = p["props"]
        authority = props.get('authority_score', 0)
        auth_stars = "★" * min(authority // 2, 5) + "☆" * (5 - min(authority // 2, 5))

        card = (
            f"<b>👤 {props.get('name', 'Unknown')}</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💼  <b>Role:</b> {props.get('role', 'Unknown')}\n"
            f"🏢  <b>Dept:</b> {props.get('department', 'Unknown')}\n"
            f"📧  <b>Email:</b> <code>{props.get('email', 'N/A')}</code>\n"
            f"⭐  <b>Authority:</b> {auth_stars}"
        )
        cards.append(card)
    return "\n\n".join(cards), "HTML"


def _team(department: str) -> tuple[str, str]:
    """List all people in a department. Returns (text, parse_mode)."""
    driver = get_neo4j_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Person)
            WHERE toLower(p.department) CONTAINS toLower($dept)
            RETURN p.name AS name, p.role AS role, p.email AS email, p.authority_score AS auth
            ORDER BY p.authority_score DESC
            """,
            dept=department,
        )
        members = list(result)

    if not members:
        return f"No team members found in '{department}'.", "HTML"

    lines = [
        f"<b>🏢 {department.title()} Team</b>",
        f"━━━━━━━━━━━━━━━",
        f"<i>{len(members)} members</i>\n",
    ]

    for m in members:
        auth = m['auth'] or 0
        if auth >= 10:
            icon = "👑"
        elif auth >= 7:
            icon = "🔹"
        else:
            icon = "▫️"
        lines.append(f"{icon} <b>{m['name']}</b>\n     {m['role']}\n     <code>{m['email']}</code>")

    return "\n".join(lines), "HTML"


def _summarize_channel(channel_name: str) -> tuple[str, str]:
    """Summarize recent Slack channel activity with chunked processing. Returns (text, parse_mode)."""
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
        return f"Channel '#{channel_name}' not found.", "HTML"

    # Fetch recent messages
    resp = client.conversations_history(channel=channel_id, limit=30)
    messages = resp.get("messages", [])

    if not messages:
        return f"No recent messages in #{channel_name}.", "HTML"

    # Extract and filter messages (prioritize those with reactions or replies)
    processed = []
    for msg in reversed(messages):
        text = msg.get("text", "")[:150]  # Shorter truncation
        if not text or text.startswith("<"):  # Skip empty or bot messages
            continue

        # Prioritize messages with engagement
        reactions = len(msg.get("reactions", []))
        replies = msg.get("reply_count", 0)
        score = reactions + replies

        processed.append({"text": text, "score": score})

    # Sort by engagement, take top 15
    processed.sort(key=lambda x: x["score"], reverse=True)
    top_messages = [m["text"] for m in processed[:15]]

    if len(top_messages) <= 8:
        # Small batch - single summarization
        prompt = f"""Summarize in 3-4 bullet points with emojis. Key topics and decisions only. Plain text, no markdown.

Messages:
{chr(10).join(top_messages)}

Summary:"""
        llm = get_llm()
        response = llm.invoke(prompt)
        summary = response.content if hasattr(response, "content") else str(response)
    else:
        # Chunk into 2 batches, summarize each, then combine
        llm = get_llm()
        mid = len(top_messages) // 2
        chunk_summaries = []

        for chunk in [top_messages[:mid], top_messages[mid:]]:
            prompt = f"""List 2-3 key points from these messages. Very brief, plain text:

{chr(10).join(chunk)}

Points:"""
            resp = llm.invoke(prompt)
            chunk_summaries.append(resp.content if hasattr(resp, "content") else str(resp))

        # Combine chunk summaries
        combine_prompt = f"""Combine into 4-5 bullet points with emojis. No markdown:

{chr(10).join(chunk_summaries)}

Summary:"""
        response = llm.invoke(combine_prompt)
        summary = response.content if hasattr(response, "content") else str(response)

    result = (
        f"<b>💬 #{channel_name} Summary</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"<i>{len(messages)} messages analyzed</i>\n\n"
        f"{summary}"
    )
    return result, "HTML"


def _diff_page(page_title: str) -> tuple[str, str]:
    """Show what changed in a Confluence page. Returns (text, parse_mode)."""
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
        return f"Page '{page_title}' not found.", "HTML"

    page_id = pages[0]["content"]["id"]
    page = cf.get_page_by_id(page_id, expand="version,history.lastUpdated")

    version = page.get("version", {})
    current_version = version.get("number", 1)
    modified_by = version.get("by", {}).get("displayName", "Unknown")
    modified_at = version.get("when", "Unknown")[:10] if version.get("when") else "Unknown"
    message = version.get("message", "—")

    diff_info = ""
    if current_version > 1:
        try:
            history = cf.get_page_by_id(page_id, expand="body.storage", version=current_version - 1)
            old_length = len(history.get("body", {}).get("storage", {}).get("value", ""))
            current_page = cf.get_page_by_id(page_id, expand="body.storage")
            new_length = len(current_page.get("body", {}).get("storage", {}).get("value", ""))
            diff = new_length - old_length
            if diff > 0:
                diff_info = f"📈  <b>Change:</b> +{diff} chars"
            elif diff < 0:
                diff_info = f"📉  <b>Change:</b> {diff} chars"
            else:
                diff_info = f"📊  <b>Change:</b> Modified (same size)"
        except Exception:
            diff_info = ""

    result = (
        f"<b>📄 {page.get('title')}</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"📌  <b>Version:</b> {current_version}\n"
        f"📅  <b>Modified:</b> {modified_at}\n"
        f"👤  <b>Author:</b> {modified_by}\n"
        f"💬  <b>Note:</b> {message}\n"
        f"{diff_info}"
    )
    return result.strip(), "HTML"


def _find_expert(topic: str) -> tuple[str, str]:
    """Find specific people to talk to about a topic. Returns (text, parse_mode)."""
    driver = get_neo4j_driver()
    experts = []

    with driver.session() as session:
        # Search for people related to the topic via projects, technologies, or departments
        result = list(session.run("""
            // Direct name match
            OPTIONAL MATCH (p1:Person)
            WHERE toLower(p1.name) CONTAINS toLower($topic)
               OR toLower(p1.role) CONTAINS toLower($topic)

            // People who lead projects matching topic
            OPTIONAL MATCH (p2:Person)-[:LEADS]->(proj:Project)
            WHERE toLower(proj.name) CONTAINS toLower($topic)

            // People in departments matching topic
            OPTIONAL MATCH (p3:Person)-[:WORKS_IN]->(d:Department)
            WHERE toLower(d.name) CONTAINS toLower($topic)

            // People who lead departments matching topic
            OPTIONAL MATCH (p4:Person)-[:LEADS]->(d2:Department)
            WHERE toLower(d2.name) CONTAINS toLower($topic)

            // People working on projects using technology matching topic
            OPTIONAL MATCH (p5:Person)-[:LEADS]->(proj2:Project)-[:USES]->(t:Technology)
            WHERE toLower(t.name) CONTAINS toLower($topic)

            WITH collect(DISTINCT p1) + collect(DISTINCT p2) + collect(DISTINCT p3) + collect(DISTINCT p4) + collect(DISTINCT p5) AS all_people
            UNWIND all_people AS person
            WITH DISTINCT person
            WHERE person IS NOT NULL
            RETURN person.name AS name, person.role AS role, person.email AS email,
                   person.department AS dept, person.authority_score AS auth
            ORDER BY person.authority_score DESC
            LIMIT 5
        """, topic=topic))

        for row in result:
            if row['name']:
                experts.append({
                    'name': row['name'],
                    'role': row['role'] or 'Unknown',
                    'email': row['email'] or 'N/A',
                    'dept': row['dept'] or 'Unknown',
                    'auth': row['auth'] or 0,
                })

    if not experts:
        # Fallback: get highest authority people
        with driver.session() as session:
            fallback = list(session.run("""
                MATCH (p:Person)
                WHERE p.authority_score >= 7
                RETURN p.name AS name, p.role AS role, p.email AS email,
                       p.department AS dept, p.authority_score AS auth
                ORDER BY p.authority_score DESC
                LIMIT 3
            """))
            for row in fallback:
                experts.append({
                    'name': row['name'],
                    'role': row['role'],
                    'email': row['email'],
                    'dept': row['dept'],
                    'auth': row['auth'],
                })

    result_parts = [
        f"<b>🎯 Expert Finder: {topic}</b>",
        "━━━━━━━━━━━━━━━\n",
    ]

    if experts:
        for e in experts:
            auth = e['auth']
            if auth >= 10:
                icon = "👑"
            elif auth >= 7:
                icon = "🔹"
            else:
                icon = "▫️"

            result_parts.append(
                f"{icon} <b>{e['name']}</b>\n"
                f"     {e['role']} • {e['dept']}\n"
                f"     📧 <code>{e['email']}</code>\n"
            )
    else:
        result_parts.append("No specific experts found for this topic.")

    return "\n".join(result_parts), "HTML"


def _sync_now() -> tuple[str, str]:
    """Trigger manual sync. Returns (text, parse_mode)."""
    import requests
    results = []

    try:
        resp = requests.post("http://localhost:8000/sync/slack", timeout=2)
        if resp.status_code == 200:
            results.append("✅ Slack sync triggered")
        else:
            results.append("⚠️ Slack sync failed")
    except Exception:
        results.append("❌ Slack sync: service unavailable")

    try:
        resp = requests.post("http://localhost:8000/sync/confluence", timeout=2)
        if resp.status_code == 200:
            results.append("✅ Confluence sync triggered")
        else:
            results.append("⚠️ Confluence sync failed")
    except Exception:
        results.append("❌ Confluence sync: service unavailable")

    if all("❌" in r for r in results):
        results.append("\n<i>Start sync service:</i>\n<code>python interface/sync_service.py</code>")

    return (
        "<b>🔄 Sync Status</b>\n"
        "━━━━━━━━━━━━━━━\n" +
        "\n".join(results)
    ), "HTML"


def _generate_org_graph() -> bytes | None:
    """Generate org chart as PNG image."""
    import graphviz
    import tempfile

    driver = get_neo4j_driver()

    # Get all people with their departments and roles
    with driver.session() as session:
        people = list(session.run("""
            MATCH (p:Person)
            RETURN p.name AS name, p.role AS role, p.department AS dept, p.authority_score AS auth
            ORDER BY p.authority_score DESC
        """))

        # Get LEADS relationships
        leads = list(session.run("""
            MATCH (p:Person)-[:LEADS]->(d:Department)
            RETURN p.name AS person, d.name AS dept
        """))

        # Get department info
        depts = list(session.run("""
            MATCH (d:Department)
            RETURN d.name AS name, d.head AS head
        """))

    if not people:
        return None

    # Create graph
    dot = graphviz.Digraph(
        'org_chart',
        format='png',
        graph_attr={
            'rankdir': 'TB',
            'splines': 'ortho',
            'nodesep': '0.5',
            'ranksep': '0.8',
            'bgcolor': '#1a1a2e',
            'pad': '0.5',
        },
        node_attr={
            'shape': 'box',
            'style': 'filled,rounded',
            'fontname': 'Arial',
            'fontsize': '11',
            'margin': '0.2,0.1',
        },
        edge_attr={
            'color': '#4a4a6a',
            'arrowsize': '0.7',
        }
    )

    # Color scheme by authority
    def get_color(auth):
        if auth >= 10:
            return '#e94560', '#ffffff'  # Red for executives
        elif auth >= 7:
            return '#0f3460', '#ffffff'  # Blue for seniors
        else:
            return '#16213e', '#cccccc'  # Dark for juniors

    # Create department subgraphs
    dept_people = {}
    for p in people:
        dept = p['dept'] or 'Other'
        if dept not in dept_people:
            dept_people[dept] = []
        dept_people[dept].append(p)

    # Find department heads
    dept_heads = {d['name']: d['head'] for d in depts}
    lead_relations = {l['person']: l['dept'] for l in leads}

    # Add nodes by department
    for dept, members in dept_people.items():
        with dot.subgraph(name=f'cluster_{dept.replace(" ", "_")}') as c:
            c.attr(
                label=dept,
                style='filled,rounded',
                color='#2a2a4a',
                fillcolor='#0f0f23',
                fontcolor='#ffffff',
                fontname='Arial Bold',
                fontsize='12',
            )

            for p in members:
                bg, fg = get_color(p['auth'] or 0)
                label = f"{p['name']}\\n{p['role']}"
                c.node(
                    p['name'],
                    label=label,
                    fillcolor=bg,
                    fontcolor=fg,
                )

    # Add reporting edges (heads to their reports)
    for dept, head_name in dept_heads.items():
        if head_name:
            # Connect head to other members in same dept
            if dept in dept_people:
                for p in dept_people[dept]:
                    if p['name'] != head_name and (p['auth'] or 0) < 10:
                        dot.edge(head_name, p['name'])

    # Add cross-department leadership
    for person, dept in lead_relations.items():
        head = dept_heads.get(dept)
        if head and head != person:
            dot.edge(person, head, style='dashed', color='#e94560')

    # Render to bytes
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = dot.render(directory=tmpdir, cleanup=True)
        with open(filepath, 'rb') as f:
            return f.read()


def _generate_entity_graph(entity_name: str) -> bytes | None:
    """Generate graph centered on an entity (people and dependencies only)."""
    import graphviz
    import tempfile

    driver = get_neo4j_driver()

    with driver.session() as session:
        # Find entity and get people + department dependencies with correct direction
        result = list(session.run("""
            MATCH (center)
            WHERE toLower(center.name) CONTAINS toLower($name)
            WITH center, labels(center)[0] AS center_label

            // Get people who work in or lead this (if center is Department)
            OPTIONAL MATCH (person:Person)-[r1:WORKS_IN|LEADS]->(center)
            WHERE center_label = 'Department'

            // Get departments this depends on
            OPTIONAL MATCH (center)-[r2:DEPENDS_ON]->(dep:Department)

            // Get departments that depend on this
            OPTIONAL MATCH (dep2:Department)-[r3:DEPENDS_ON]->(center)

            // If center is a Person, get their department
            OPTIONAL MATCH (center)-[r4:WORKS_IN|LEADS]->(dept:Department)
            WHERE center_label = 'Person'

            RETURN DISTINCT
                center_label,
                center.name AS center_name,
                person.name AS person_name,
                person.role AS person_role,
                person.authority_score AS person_auth,
                type(r1) AS person_rel,
                dep.name AS depends_on,
                dep2.name AS depended_by,
                dept.name AS person_dept,
                type(r4) AS person_dept_rel
        """, name=entity_name))

    if not result or not result[0]['center_name']:
        return None

    dot = graphviz.Digraph(
        'entity_graph',
        format='png',
        graph_attr={
            'rankdir': 'TB',
            'bgcolor': '#1a1a2e',
            'pad': '0.5',
            'nodesep': '0.4',
            'ranksep': '0.6',
        },
        node_attr={
            'shape': 'box',
            'style': 'filled,rounded',
            'fontname': 'Arial',
            'fontsize': '10',
            'margin': '0.15,0.08',
        },
        edge_attr={
            'fontname': 'Arial',
            'fontsize': '9',
            'fontcolor': '#aaaaaa',
            'color': '#4a4a6a',
        }
    )

    center_name = result[0]['center_name']
    center_label = result[0]['center_label']

    # Color scheme
    def get_person_color(auth):
        auth = auth or 0
        if auth >= 10:
            return '#e94560', '#ffffff'  # Red for leaders
        elif auth >= 7:
            return '#0f3460', '#ffffff'  # Blue for seniors
        else:
            return '#16213e', '#cccccc'  # Dark for others

    dept_color = ('#00b894', '#ffffff')  # Green for departments

    # Add center node
    if center_label == 'Department':
        dot.node(center_name, label=f"🏢 {center_name}", fillcolor=dept_color[0], fontcolor=dept_color[1], penwidth='2')
    else:
        bg, fg = get_person_color(result[0].get('person_auth'))
        dot.node(center_name, label=f"👤 {center_name}", fillcolor=bg, fontcolor=fg, penwidth='2')

    added = set([center_name])

    for row in result:
        # Add people
        if row['person_name'] and row['person_name'] not in added:
            bg, fg = get_person_color(row['person_auth'])
            role = row['person_role'] or ''
            dot.node(row['person_name'], label=f"👤 {row['person_name']}\\n{role}", fillcolor=bg, fontcolor=fg)
            added.add(row['person_name'])

            rel = row['person_rel']
            if rel == 'LEADS':
                dot.edge(row['person_name'], center_name, label='leads', color='#e94560', penwidth='1.5')
            else:
                dot.edge(row['person_name'], center_name, label='works in', style='dashed')

        # Add dependencies (this dept depends on)
        if row['depends_on'] and row['depends_on'] not in added:
            dot.node(row['depends_on'], label=f"🏢 {row['depends_on']}", fillcolor=dept_color[0], fontcolor=dept_color[1])
            added.add(row['depends_on'])
            dot.edge(center_name, row['depends_on'], label='depends on', color='#fdcb6e')

        # Add dependents (depts that depend on this)
        if row['depended_by'] and row['depended_by'] not in added:
            dot.node(row['depended_by'], label=f"🏢 {row['depended_by']}", fillcolor=dept_color[0], fontcolor=dept_color[1])
            added.add(row['depended_by'])
            dot.edge(row['depended_by'], center_name, label='depends on', color='#fdcb6e')

        # If center is a person, show their department
        if row['person_dept'] and row['person_dept'] not in added:
            dot.node(row['person_dept'], label=f"🏢 {row['person_dept']}", fillcolor=dept_color[0], fontcolor=dept_color[1])
            added.add(row['person_dept'])
            rel = row['person_dept_rel']
            if rel == 'LEADS':
                dot.edge(center_name, row['person_dept'], label='leads', color='#e94560', penwidth='1.5')
            else:
                dot.edge(center_name, row['person_dept'], label='works in', style='dashed')

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = dot.render(directory=tmpdir, cleanup=True)
        with open(filepath, 'rb') as f:
            return f.read()

_GREETING = (
    "<b>🏛 Welcome to Athena</b>\n"
    "━━━━━━━━━━━━━━━\n\n"
    "I am your <b>Wise Company Historian</b>.\n\n"
    "I have indexed your Slack and Confluence, "
    "resolving conflicts using authority scores and timestamps.\n\n"
    "<b>📝 Ask me anything:</b>\n"
    "• Policies, deployments, decisions\n"
    "• Who to contact, team info\n\n"
    "<b>🛠 Commands:</b>\n"
    "/whois — Look up a person\n"
    "/team — List department members\n"
    "/graph — Visual org chart or entity graph\n"
    "/summarize — Summarize a channel\n"
    "/diff — Page change history\n"
    "/expert — Find who to ask\n"
    "/sync — Trigger data sync\n"
    "/clear — Reset conversation"
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
    await update.message.reply_text(_GREETING, parse_mode="HTML")


async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    history = _get_history(chat_id)
    history.clear()
    summary_path = _get_summary_path(chat_id)
    if summary_path.exists():
        summary_path.unlink()
    await update.message.reply_text("✨ <b>Conversation cleared</b>\n\nStarting fresh!", parse_mode="HTML")


async def whois_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /whois &lt;name&gt;", parse_mode="HTML")
        return

    name = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    loop = asyncio.get_event_loop()
    result, parse_mode = await loop.run_in_executor(None, _whois, name)
    await update.message.reply_text(result, parse_mode=parse_mode)


async def team_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /team &lt;department&gt;", parse_mode="HTML")
        return

    department = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    loop = asyncio.get_event_loop()
    result, parse_mode = await loop.run_in_executor(None, _team, department)
    await update.message.reply_text(result, parse_mode=parse_mode)


async def summarize_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /summarize &lt;#channel&gt;", parse_mode="HTML")
        return

    channel = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        loop = asyncio.get_event_loop()
        result, parse_mode = await loop.run_in_executor(None, _summarize_channel, channel)
        await update.message.reply_text(result, parse_mode=parse_mode)
    except Exception as e:
        logger.error("Summarize error: %s", e)
        await update.message.reply_text(f"❌ Error: {e}")


async def diff_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /diff &lt;page title&gt;", parse_mode="HTML")
        return

    page_title = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        loop = asyncio.get_event_loop()
        result, parse_mode = await loop.run_in_executor(None, _diff_page, page_title)
        await update.message.reply_text(result, parse_mode=parse_mode)
    except Exception as e:
        logger.error("Diff error: %s", e)
        await update.message.reply_text(f"❌ Error: {e}")


async def expert_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /expert &lt;topic&gt;", parse_mode="HTML")
        return

    topic = " ".join(context.args)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        loop = asyncio.get_event_loop()
        result, parse_mode = await loop.run_in_executor(None, _find_expert, topic)
        await update.message.reply_text(result, parse_mode=parse_mode)
    except Exception as e:
        logger.error("Expert error: %s", e)
        await update.message.reply_text(f"❌ Error: {e}")


async def sync_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    result, parse_mode = _sync_now()
    await update.message.reply_text(result, parse_mode=parse_mode)


async def graph_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)

    try:
        loop = asyncio.get_event_loop()

        if context.args:
            # Graph centered on specific entity
            entity = " ".join(context.args)
            image_bytes = await loop.run_in_executor(None, _generate_entity_graph, entity)
            caption = f"🔗 <b>Graph: {entity}</b>"
        else:
            # Full org chart
            image_bytes = await loop.run_in_executor(None, _generate_org_graph)
            caption = "📊 <b>Organization Chart</b>"

        if image_bytes:
            from io import BytesIO
            await update.message.reply_photo(
                photo=BytesIO(image_bytes),
                caption=caption,
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text("❌ No data found for graph.", parse_mode="HTML")

    except Exception as e:
        logger.error("Graph error: %s", e, exc_info=True)
        await update.message.reply_text(
            f"❌ Error generating graph.\n\n<i>Make sure graphviz is installed:</i>\n<code>brew install graphviz</code>",
            parse_mode="HTML"
        )


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
        BotCommand("graph", "Visual graph: /graph or /graph Auth Service"),
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
    app.add_handler(CommandHandler("graph", graph_handler))
    app.add_handler(CommandHandler("summarize", summarize_handler))
    app.add_handler(CommandHandler("diff", diff_handler))
    app.add_handler(CommandHandler("expert", expert_handler))
    app.add_handler(CommandHandler("sync", sync_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    logger.info("Athena bot is running. Waiting for messages...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
