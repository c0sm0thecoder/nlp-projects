"""
Extra Task UI – Unified dashboard for Project 4 (Task 1 + Task 2).

Serves pre-computed results as a single-page FastAPI application
with the same glass-card visual style as project3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent  # project4/

app = FastAPI(title="Project 4 Dashboard")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return {}


def load_dashboard_data() -> dict[str, Any]:
    task1 = _safe_read_json(BASE_DIR / "task1" / "sentiment_analysis_results.json")
    task2 = _safe_read_json(BASE_DIR / "task2" / "results" / "task2_training_results.json")
    return {"task1": task1, "task2": task2}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    data = load_dashboard_data()
    data_json = json.dumps(data, ensure_ascii=False, default=str)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"data_json": data_json},
    )


@app.get("/api/dashboard")
def dashboard_api() -> JSONResponse:
    return JSONResponse(load_dashboard_data())
