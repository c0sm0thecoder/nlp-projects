FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY extra_task_ui/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

COPY extra_task_ui /app/extra_task_ui
COPY task1/results /app/task1/results
COPY task2/results /app/task2/results
COPY task3/results /app/task3/results
COPY task4/results /app/task4/results
COPY task5/results /app/task5/results

EXPOSE 8000

CMD ["uvicorn", "extra_task_ui.app:app", "--host", "0.0.0.0", "--port", "8000"]
