"""
Extra Task UI – Unified dashboard for Project 4 (Task 1 + Task 2).

Serves pre-computed results as a single-page FastAPI application
with live Azerbaijani sentiment testing using the fine-tuned model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent  # project4/

app = FastAPI(title="Project 4 Dashboard")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# --- Load sentiment models ---
ORIGINAL_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
FINETUNED_MODEL_PATH = APP_DIR / "az_sentiment_model"

# Load tokenizer (shared)
tokenizer = None
original_model = None
finetuned_model = None

def load_models():
    global tokenizer, original_model, finetuned_model
    if tokenizer is None:
        print("Loading sentiment models...")
        tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_NAME)
        original_model = AutoModelForSequenceClassification.from_pretrained(ORIGINAL_MODEL_NAME)
        original_model.eval()
        if FINETUNED_MODEL_PATH.exists():
            finetuned_model = AutoModelForSequenceClassification.from_pretrained(str(FINETUNED_MODEL_PATH))
            finetuned_model.eval()
            print("Fine-tuned Azerbaijani model loaded!")
        else:
            print("Fine-tuned model not found, using original only")
            finetuned_model = original_model

load_models()


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return {}


def load_dashboard_data() -> dict[str, Any]:
    task1 = _safe_read_json(BASE_DIR / "task1" / "sentiment_analysis_results.json")
    task2 = _safe_read_json(BASE_DIR / "task2" / "results" / "task2_training_results.json")
    finetune = _safe_read_json(BASE_DIR / "task1" / "azerbaijani_finetune_results.json")
    return {"task1": task1, "task2": task2, "finetune": finetune}


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


# --- Sentiment Prediction API ---
class SentimentRequest(BaseModel):
    text: str
    use_finetuned: bool = True


class SentimentResponse(BaseModel):
    text: str
    prediction: int  # 1-5 stars
    confidence: float
    all_probabilities: list[float]
    model_used: str


@app.post("/api/sentiment", response_model=SentimentResponse)
def predict_sentiment(req: SentimentRequest):
    model = finetuned_model if req.use_finetuned else original_model
    model_name = "Fine-tuned (Azerbaijani)" if req.use_finetuned else "Original (mBERT)"

    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    pred = torch.argmax(probs).item()
    conf = probs[pred].item()

    return SentimentResponse(
        text=req.text,
        prediction=pred + 1,  # 1-5 stars
        confidence=round(conf * 100, 1),
        all_probabilities=[round(p.item() * 100, 1) for p in probs],
        model_used=model_name,
    )


@app.get("/api/sentiment/compare")
def compare_sentiment(text: str):
    """Compare original vs fine-tuned model predictions."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    results = {}
    for name, model in [("original", original_model), ("finetuned", finetuned_model)]:
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred = torch.argmax(probs).item()
        results[name] = {
            "prediction": pred + 1,
            "confidence": round(probs[pred].item() * 100, 1),
            "probabilities": [round(p.item() * 100, 1) for p in probs],
        }

    return JSONResponse({
        "text": text,
        "original": results["original"],
        "finetuned": results["finetuned"],
        "improvement": round(results["finetuned"]["confidence"] - results["original"]["confidence"], 1),
    })
