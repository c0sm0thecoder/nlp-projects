# FastAPI UI (No Streamlit)

This UI is a single-page FastAPI + HTML/CSS/JS demo for Tasks 1–4.

## What you get

- Left-side navigation between all tasks
- Per-task views for **Input**, **Output**, and **Implementation**
- Structured outputs (metric cards + tables) and raw JSON output
- Lightweight frontend (no heavy UI framework)

## Run

From project root:

```bash
source /home/kamal/testjson/.venv/bin/activate
pip install fastapi uvicorn jinja2
uvicorn fastapi_ui.main:app --reload --port 8001
```

Open:

- http://127.0.0.1:8001

## Architecture

The UI calls FastAPI endpoints that reuse core logic from:

- `task1/task1_ngram.py`
- `task2/smoothing_core.py`
- `task3/classification_core.py`
- `task4/sentence_boundary_core.py`

Task 3 defaults to `twitter_samples`, but can run on parquet input as well.
