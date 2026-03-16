# Extra Task UI (FastAPI)

Modern colorful dashboard for Tasks 1–5 results with animated glassmorphism cards and bubble-style visualization.

## Run

From project root:

```bash
/home/kamal/NLP/nlp-projects/project3/.venv/bin/uvicorn extra_task_ui.app:app --reload
```

Open:

- http://127.0.0.1:8000
- JSON API: http://127.0.0.1:8000/api/dashboard

## Data sources

The UI reads existing artifacts only:

- `task1/results/task1_summary.json`
- `task2/results/task2_config.json`
- `task3/results/task3_config.json`
- `task4/results/task4_config.json`, `task4_neighbors_overlap.csv`, `task4_equations_overlap.csv`
- `task5/results/task5_config.json`, `task5_results.csv`

If any file is missing, the UI degrades gracefully and still renders.
