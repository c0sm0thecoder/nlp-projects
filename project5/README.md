# Athena — Graph RAG Knowledge Assistant

A Graph RAG system that indexes Slack and Confluence, resolving conflicts using authority scores and timestamps. Features a Telegram bot interface with voice support.

## Architecture

```
User Question (Text/Voice)
         │
         ▼
┌─────────────────┐
│ Entity Extractor│ ← Gemini extracts entities
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Neo4j Graph     │ ← Query related entities (2-hop)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pinecone Vector │ ← Semantic search, boosted by graph
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Synthesis   │ ← Gemini with graph + vector context
└─────────────────┘
```

## Features

### Core RAG
- **Graph-first retrieval**: Extracts entities → queries Neo4j → boosts connected docs
- **Authority-aware**: Weighs information by author's authority score (Lead=10, Senior=7, etc.)
- **Conflict resolution**: Newer high-authority sources override older docs
- **Time-travel queries**: "What was our PTO policy last January?"

### Telegram Bot Commands
| Command | Description |
|---------|-------------|
| `/start` | Start conversation |
| `/clear` | Clear conversation history |
| `/whois <name>` | Look up a person |
| `/team <dept>` | List department members |
| `/graph [entity]` | Visual org chart or entity graph |
| `/summarize <#channel>` | Summarize Slack channel |
| `/diff <page>` | Confluence page change history |
| `/expert <topic>` | Find who to ask |
| `/route <question>` | Post question to relevant Slack channel |
| `/voice` | Toggle voice responses on/off |
| `/sync` | Trigger manual sync |

### Voice Interface
- Send voice messages → transcribed via local Whisper
- Toggle `/voice` for voice-only responses (edge-tts)

### Automated Features
- **Weekly Digest**: Auto-posts summary to Slack every Monday 9 AM
- **Confluence Webhook**: Real-time sync on page updates
- **Slack Cron**: Daily sync at midnight

## Setup

### 1. Environment

```bash
cd project5
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For voice (optional)
brew install ffmpeg
```

### 2. Configuration

Copy `.env.example` to `.env` and fill in:

```env
# LLM & Embeddings
GOOGLE_API_KEY=your_gemini_api_key

# Vector Store
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=athena-knowledge

# Slack
SLACK_BOT_TOKEN=xoxb-...
SLACK_CHANNEL_IDS=C01234567,C07654321

# Confluence
CONFLUENCE_URL=https://your-domain.atlassian.net/wiki
CONFLUENCE_USER=your@email.com
CONFLUENCE_API_TOKEN=your_api_token
CONFLUENCE_SPACES=WIKI,ENG

# Telegram
TELEGRAM_BOT_TOKEN=123456:ABC...

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=athena123
```

### 3. Start Services

```bash
# Start Neo4j
docker compose up -d

# Seed data (fake corporate data for testing)
python scripts/seed_data.py

# Build knowledge graph
python scripts/build_graph.py

# Import real Slack users (optional)
python scripts/import_slack_users.py

# Run bot
python interface/bot.py

# Run sync service (separate terminal)
python interface/sync_service.py
```

## Project Structure

```
project5/
├── core/
│   ├── config.py          # Pydantic settings
│   ├── clients.py         # LLM, embeddings, Neo4j, Pinecone
│   └── logger.py          # Logging
├── brain/
│   ├── resolver.py        # Graph-first RAG resolver
│   ├── vector_store.py    # Pinecone operations
│   ├── knowledge_graph.py # Neo4j operations
│   └── entity_extractor.py # LLM entity extraction
├── ingestion/
│   ├── slack_engine.py    # Fetch Slack messages
│   └── confluence_engine.py # Fetch Confluence pages
├── interface/
│   ├── bot.py             # Telegram bot
│   └── sync_service.py    # FastAPI webhook + cron
├── scripts/
│   ├── seed_data.py       # Seed fake corporate data
│   ├── build_graph.py     # Build Neo4j graph
│   └── import_slack_users.py # Import real Slack users
├── docker-compose.yml     # Neo4j service
└── requirements.txt
```

## Neo4j Graph Schema

### Nodes
- `Person`: id, name, role, department, authority_score, email
- `Department`: id, name, head, description
- `Project`: id, name, status, tech_stack[], owner_dept
- `Service`: id, name, status, owner_dept
- `Technology`: id, name, category

### Relationships
- `(Person)-[:WORKS_IN]->(Department)`
- `(Person)-[:LEADS]->(Department|Project)`
- `(Department)-[:DEPENDS_ON]->(Department)`
- `(Department)-[:OWNS]->(Project|Service)`
- `(Project)-[:USES]->(Technology)`

## API Endpoints (sync_service.py)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/webhook/confluence` | POST | Confluence webhook |
| `/sync/slack` | POST | Trigger Slack sync |
| `/sync/confluence` | POST | Trigger Confluence sync |
| `/digest` | POST | Trigger weekly digest |

## Tech Stack

- **LLM**: Gemini 2.5 Flash
- **Embeddings**: text-embedding-004 (768 dims)
- **Vector DB**: Pinecone
- **Graph DB**: Neo4j
- **Bot**: python-telegram-bot
- **Voice STT**: OpenAI Whisper (local)
- **Voice TTS**: edge-tts (Microsoft)
