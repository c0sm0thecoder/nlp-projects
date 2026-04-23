"""
dashboard.py — FastAPI monitoring dashboard for Athena.

Shows query history, retrieved chunks, latency, and performance metrics.
Run: uvicorn interface.dashboard:app --port 8080
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from brain.resolver import ask, _format_docs
from brain.vector_store import get_vector_store
from brain.entity_extractor import extract_entities_from_question
from brain.knowledge_graph import query_related_entities, get_graph_stats
from core.logger import get_logger

logger = get_logger("dashboard")

app = FastAPI(title="Athena Dashboard", version="1.0")

# In-memory query log (last 100 queries)
QUERY_LOG: deque = deque(maxlen=100)


@dataclass
class QueryRecord:
    id: int
    timestamp: str
    question: str
    answer: str
    latency_ms: int
    chunks_used: list[dict]
    entities_extracted: list[str]
    graph_entities: int
    source_breakdown: dict


def log_query(record: QueryRecord):
    QUERY_LOG.appendleft(asdict(record))


# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/ask")
async def api_ask(request: Request):
    """Ask a question and log the full trace."""
    data = await request.json()
    question = data.get("question", "")

    if not question:
        return JSONResponse({"error": "No question provided"}, status_code=400)

    start = time.time()

    # Extract entities
    entities = extract_entities_from_question(question)

    # Get graph context
    graph_results = query_related_entities(entities, hops=2) if entities else []

    # Get chunks from vector stores
    chunks_used = []
    source_breakdown = {"slack": 0, "confluence": 0}

    for ns in ["slack", "confluence"]:
        store = get_vector_store(ns)
        docs = store.similarity_search_with_score(question, k=4)
        for doc, score in docs:
            source_breakdown[ns] += 1
            chunks_used.append({
                "source": doc.metadata.get("source", ns),
                "author": doc.metadata.get("author_name", "Unknown"),
                "role": doc.metadata.get("author_role", ""),
                "authority": doc.metadata.get("authority_score", 0),
                "score": round(score, 4),
                "timestamp": doc.metadata.get("timestamp", ""),
                "text": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "url": doc.metadata.get("url", ""),
            })

    # Sort by score
    chunks_used.sort(key=lambda x: x["score"], reverse=True)

    # Get answer
    answer = ask(question)

    latency_ms = int((time.time() - start) * 1000)

    # Log the query
    record = QueryRecord(
        id=len(QUERY_LOG) + 1,
        timestamp=datetime.now().isoformat(),
        question=question,
        answer=answer,
        latency_ms=latency_ms,
        chunks_used=chunks_used[:8],  # Top 8 chunks
        entities_extracted=entities,
        graph_entities=len(graph_results),
        source_breakdown=source_breakdown,
    )
    log_query(record)

    return JSONResponse(asdict(record))


@app.get("/api/queries")
async def get_queries(limit: int = 20):
    """Get recent query history."""
    return JSONResponse(list(QUERY_LOG)[:limit])


@app.get("/api/stats")
async def get_stats():
    """Get aggregate statistics."""
    if not QUERY_LOG:
        return JSONResponse({
            "total_queries": 0,
            "avg_latency_ms": 0,
            "avg_chunks_per_query": 0,
            "source_distribution": {},
            "graph_stats": get_graph_stats(),
        })

    queries = list(QUERY_LOG)
    total = len(queries)
    avg_latency = sum(q["latency_ms"] for q in queries) / total
    avg_chunks = sum(len(q["chunks_used"]) for q in queries) / total

    source_dist = {"slack": 0, "confluence": 0}
    for q in queries:
        for src, count in q["source_breakdown"].items():
            source_dist[src] = source_dist.get(src, 0) + count

    return JSONResponse({
        "total_queries": total,
        "avg_latency_ms": round(avg_latency),
        "avg_chunks_per_query": round(avg_chunks, 1),
        "source_distribution": source_dist,
        "graph_stats": get_graph_stats(),
    })


@app.get("/api/graph-stats")
async def api_graph_stats():
    """Get Neo4j graph statistics."""
    return JSONResponse(get_graph_stats())


# ══════════════════════════════════════════════════════════════════════════════
# HTML DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Athena Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        .chunk-card { transition: all 0.2s; }
        .chunk-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .fade-in { animation: fadeIn 0.3s ease-in; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .source-slack { border-left: 4px solid #4A154B; }
        .source-confluence { border-left: 4px solid #0052CC; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <nav class="bg-indigo-600 text-white p-4 shadow-lg">
        <div class="max-w-7xl mx-auto flex justify-between items-center">
            <h1 class="text-2xl font-bold flex items-center gap-2">
                <span>🦉</span> Athena Dashboard
            </h1>
            <div id="stats-bar" class="text-sm flex gap-6">
                <span>Queries: <strong id="stat-total">0</strong></span>
                <span>Avg Latency: <strong id="stat-latency">0ms</strong></span>
                <span>Graph Nodes: <strong id="stat-nodes">0</strong></span>
            </div>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto p-6">
        <!-- Ask Section -->
        <div class="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 class="text-lg font-semibold mb-4">Ask Athena</h2>
            <div class="flex gap-4">
                <input type="text" id="question-input"
                    class="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    placeholder="Enter your question..."
                    onkeypress="if(event.key==='Enter') askQuestion()">
                <button onclick="askQuestion()"
                    class="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 transition">
                    Ask
                </button>
            </div>
        </div>

        <!-- Current Query Details -->
        <div id="current-query" class="hidden mb-6 fade-in">
            <div class="bg-white rounded-lg shadow-md p-6">
                <div class="flex justify-between items-start mb-4">
                    <h2 class="text-lg font-semibold">Query Details</h2>
                    <span id="current-latency" class="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm"></span>
                </div>

                <div class="mb-4">
                    <h3 class="font-medium text-gray-700 mb-2">Question</h3>
                    <p id="current-question" class="text-gray-900 bg-gray-50 p-3 rounded"></p>
                </div>

                <div class="mb-4">
                    <h3 class="font-medium text-gray-700 mb-2">Answer</h3>
                    <p id="current-answer" class="text-gray-900 bg-blue-50 p-3 rounded whitespace-pre-wrap"></p>
                </div>

                <div class="mb-4">
                    <h3 class="font-medium text-gray-700 mb-2">Entities Extracted</h3>
                    <div id="current-entities" class="flex flex-wrap gap-2"></div>
                </div>

                <div>
                    <h3 class="font-medium text-gray-700 mb-2">Retrieved Chunks (<span id="chunk-count">0</span>)</h3>
                    <div id="chunks-grid" class="grid grid-cols-1 md:grid-cols-2 gap-4"></div>
                </div>
            </div>
        </div>

        <!-- Query History -->
        <div class="bg-white rounded-lg shadow-md p-6">
            <h2 class="text-lg font-semibold mb-4">Query History</h2>
            <div id="query-history" class="space-y-3"></div>
        </div>
    </main>

    <script>
        async function loadStats() {
            const res = await fetch('/api/stats');
            const data = await res.json();
            document.getElementById('stat-total').textContent = data.total_queries;
            document.getElementById('stat-latency').textContent = data.avg_latency_ms + 'ms';
            const totalNodes = data.graph_stats.nodes.reduce((sum, n) => sum + n.count, 0);
            document.getElementById('stat-nodes').textContent = totalNodes;
        }

        async function loadHistory() {
            const res = await fetch('/api/queries?limit=10');
            const queries = await res.json();
            const container = document.getElementById('query-history');

            if (queries.length === 0) {
                container.innerHTML = '<p class="text-gray-500">No queries yet. Ask a question above!</p>';
                return;
            }

            container.innerHTML = queries.map(q => `
                <div class="border rounded-lg p-4 hover:bg-gray-50 cursor-pointer transition"
                     onclick="showQueryDetails(${JSON.stringify(q).replace(/"/g, '&quot;')})">
                    <div class="flex justify-between items-start">
                        <div class="flex-1">
                            <p class="font-medium text-gray-900">${escapeHtml(q.question)}</p>
                            <p class="text-sm text-gray-500 mt-1">${q.timestamp.split('T')[1].split('.')[0]} · ${q.latency_ms}ms · ${q.chunks_used.length} chunks</p>
                        </div>
                        <div class="flex gap-2">
                            <span class="bg-purple-100 text-purple-800 px-2 py-1 rounded text-xs">${q.source_breakdown.slack || 0} Slack</span>
                            <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">${q.source_breakdown.confluence || 0} Conf</span>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        async function askQuestion() {
            const input = document.getElementById('question-input');
            const question = input.value.trim();
            if (!question) return;

            input.disabled = true;
            const btn = document.querySelector('button');
            btn.textContent = 'Thinking...';
            btn.disabled = true;

            try {
                const res = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                const data = await res.json();
                showQueryDetails(data);
                loadHistory();
                loadStats();
                input.value = '';
            } catch (e) {
                alert('Error: ' + e.message);
            } finally {
                input.disabled = false;
                btn.textContent = 'Ask';
                btn.disabled = false;
            }
        }

        function showQueryDetails(query) {
            const container = document.getElementById('current-query');
            container.classList.remove('hidden');

            document.getElementById('current-question').textContent = query.question;
            document.getElementById('current-answer').textContent = query.answer;
            document.getElementById('current-latency').textContent = query.latency_ms + 'ms';
            document.getElementById('chunk-count').textContent = query.chunks_used.length;

            // Entities
            const entitiesContainer = document.getElementById('current-entities');
            entitiesContainer.innerHTML = query.entities_extracted.length > 0
                ? query.entities_extracted.map(e => `<span class="bg-yellow-100 text-yellow-800 px-2 py-1 rounded text-sm">${escapeHtml(e)}</span>`).join('')
                : '<span class="text-gray-500 text-sm">None extracted</span>';

            // Chunks
            const chunksGrid = document.getElementById('chunks-grid');
            chunksGrid.innerHTML = query.chunks_used.map((chunk, i) => `
                <div class="chunk-card border rounded-lg p-4 bg-white source-${chunk.source}">
                    <div class="flex justify-between items-start mb-2">
                        <span class="font-medium text-sm">${chunk.author}</span>
                        <span class="text-xs ${chunk.source === 'slack' ? 'bg-purple-100 text-purple-800' : 'bg-blue-100 text-blue-800'} px-2 py-0.5 rounded">
                            ${chunk.source}
                        </span>
                    </div>
                    <p class="text-xs text-gray-500 mb-2">${chunk.role} · Authority: ${chunk.authority} · Score: ${chunk.score}</p>
                    <p class="text-sm text-gray-700">${escapeHtml(chunk.text)}</p>
                </div>
            `).join('');

            container.scrollIntoView({ behavior: 'smooth' });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Initial load
        loadStats();
        loadHistory();
        setInterval(loadStats, 30000);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML."""
    return DASHBOARD_HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
