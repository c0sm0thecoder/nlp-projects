"""
Extra Task UI – Unified dashboard for Project 4 (Task 1 + Task 2).

Serves pre-computed results + live inference for sentiment analysis
and question answering (BiDAF-BERT from saved checkpoint).
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import BertTokenizer

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent  # project4/

# Add task2 to path so we can import BiDAF_BERT and its dependencies
sys.path.insert(0, str(BASE_DIR / "task2"))

app = FastAPI(title="Project 4 Dashboard")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# ---------------------------------------------------------------------------
# Lazy-loaded models
# ---------------------------------------------------------------------------
_sentiment_pipeline = None
_qa_model = None
_qa_tokenizer = None

BERT_MODEL_NAME = "bert-base-multilingual-uncased"
QA_CHECKPOINT = BASE_DIR / "models" / "bert_bidaf_best.pt"


def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        print("Loading sentiment model...")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            top_k=5,
        )
        print("Sentiment model loaded.")
    return _sentiment_pipeline


def get_qa_model():
    global _qa_model, _qa_tokenizer
    if _qa_model is None:
        from bidaf_bert_model import BiDAF_BERT
        print(f"Loading BiDAF-BERT from {QA_CHECKPOINT}...")
        ckpt = torch.load(QA_CHECKPOINT, map_location="cpu", weights_only=False)

        bert_name = ckpt.get("bert_model_name", BERT_MODEL_NAME)
        hidden_dim = ckpt.get("hidden_dim", 100)

        model = BiDAF_BERT(
            bert_model_name=bert_name,
            hidden_dim=hidden_dim,
            num_modeling_layers=2,
            dropout=0.0,  # no dropout at inference
        )

        # Load saved non-BERT weights
        state = ckpt["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model.eval()
        _qa_model = model

        _qa_tokenizer = BertTokenizer.from_pretrained(bert_name)
        print(f"BiDAF-BERT loaded (hidden_dim={hidden_dim}, best_f1={ckpt.get('best_val_f1', '?')}%)")
    return _qa_model, _qa_tokenizer


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class SentimentRequest(BaseModel):
    text: str


class QARequest(BaseModel):
    question: str
    context: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
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


@app.post("/api/sentiment")
def sentiment_api(req: SentimentRequest) -> JSONResponse:
    if not req.text.strip():
        return JSONResponse({"error": "Text is empty"}, status_code=400)
    pipe = get_sentiment_pipeline()
    results = pipe(req.text[:512])[0]
    formatted = [{"label": r["label"], "score": round(r["score"], 4)} for r in results]
    best = max(formatted, key=lambda x: x["score"])
    return JSONResponse({
        "text": req.text[:512],
        "prediction": best["label"],
        "confidence": best["score"],
        "all_scores": formatted,
    })


@app.post("/api/qa")
def qa_api(req: QARequest) -> JSONResponse:
    if not req.question.strip() or not req.context.strip():
        return JSONResponse({"error": "Question and context are required"}, status_code=400)

    model, tokenizer = get_qa_model()

    # Tokenize context and question separately (no special tokens)
    ctx_enc = tokenizer(
        req.context[:2048],
        max_length=384,
        truncation=True,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    qry_enc = tokenizer(
        req.question,
        max_length=64,
        truncation=True,
        add_special_tokens=False,
    )

    ctx_ids = torch.tensor([ctx_enc["input_ids"]])
    ctx_mask = torch.ones(1, len(ctx_enc["input_ids"]))
    qry_ids = torch.tensor([qry_enc["input_ids"]])
    qry_mask = torch.ones(1, len(qry_enc["input_ids"]))

    with torch.no_grad():
        start_logits, end_logits = model(
            context_ids=ctx_ids,
            context_mask=ctx_mask,
            query_ids=qry_ids,
            query_mask=qry_mask,
        )

    # Get best span
    start_probs = F.softmax(start_logits, dim=-1)
    end_probs = F.softmax(end_logits, dim=-1)
    start_idx = torch.argmax(start_probs, dim=-1).item()
    end_idx = torch.argmax(end_probs, dim=-1).item()
    if end_idx < start_idx:
        end_idx = start_idx

    # Decode answer from token ids
    answer_ids = ctx_enc["input_ids"][start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    # Confidence
    score = (start_probs[0, start_idx].item() + end_probs[0, end_idx].item()) / 2

    # Map back to character offsets if available
    offsets = ctx_enc.get("offset_mapping", [])
    char_start = offsets[start_idx][0] if start_idx < len(offsets) else -1
    char_end = offsets[end_idx][1] if end_idx < len(offsets) else -1

    return JSONResponse({
        "question": req.question,
        "answer": answer,
        "score": round(score, 4),
        "start_token": start_idx,
        "end_token": end_idx,
        "char_start": char_start,
        "char_end": char_end,
    })
