"""
Task 2.3: Training and Evaluation for BiDAF (baseline) and BiDAF-BERT.

Trains both models on SQuAD v1.1 (subset), evaluates with EM and F1,
and saves a comparison report.
"""

from __future__ import annotations

import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bidaf_model import BiDAF, compute_loss
from bidaf_bert_model import BiDAF_BERT
from data_utils import (
    Vocabulary,
    load_squad,
    download_glove,
    load_glove_embeddings,
    SQuADBaselineDataset,
    SQuADBertDataset,
    collate_baseline,
    collate_bert,
    GLOVE_DIM,
)
from transformers import BertTokenizer


# ---------------------------------------------------------------------------
# Evaluation metrics (official SQuAD-style)
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    # remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # collapse whitespace
    s = " ".join(s.split())
    return s


def compute_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_type: str = "baseline",
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        if model_type == "baseline":
            start_logits, end_logits = model(
                context_word_ids=batch["context_word"].to(device),
                context_char_ids=batch["context_char"].to(device),
                query_word_ids=batch["query_word"].to(device),
                query_char_ids=batch["query_char"].to(device),
                context_mask=batch["context_mask"].to(device),
                query_mask=batch["query_mask"].to(device),
            )
        else:  # bert
            start_logits, end_logits = model(
                context_ids=batch["context_ids"].to(device),
                context_mask=batch["context_mask"].to(device),
                query_ids=batch["query_ids"].to(device),
                query_mask=batch["query_mask"].to(device),
            )

        loss = compute_loss(
            start_logits,
            end_logits,
            batch["answer_start"].to(device),
            batch["answer_end"].to(device),
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_type: str = "baseline",
) -> dict:
    model.eval()
    total_em = 0.0
    total_f1 = 0.0
    total_loss = 0.0
    count = 0

    for batch in loader:
        if model_type == "baseline":
            start_logits, end_logits = model(
                context_word_ids=batch["context_word"].to(device),
                context_char_ids=batch["context_char"].to(device),
                query_word_ids=batch["query_word"].to(device),
                query_char_ids=batch["query_char"].to(device),
                context_mask=batch["context_mask"].to(device),
                query_mask=batch["query_mask"].to(device),
            )
        else:
            start_logits, end_logits = model(
                context_ids=batch["context_ids"].to(device),
                context_mask=batch["context_mask"].to(device),
                query_ids=batch["query_ids"].to(device),
                query_mask=batch["query_mask"].to(device),
            )

        loss = compute_loss(
            start_logits,
            end_logits,
            batch["answer_start"].to(device),
            batch["answer_end"].to(device),
        )
        total_loss += loss.item()

        # Get predictions
        start_preds = torch.argmax(start_logits, dim=-1)
        end_preds = torch.argmax(end_logits, dim=-1)

        # Ensure end >= start
        end_preds = torch.max(end_preds, start_preds)

        for i in range(len(batch["answer_texts"])):
            ctx_tokens = batch["context_tokens"][i]
            s = start_preds[i].item()
            e = end_preds[i].item()

            # Extract predicted answer from tokens
            pred_tokens = ctx_tokens[s : e + 1]
            pred_answer = " ".join(pred_tokens)

            gt_answer = batch["answer_texts"][i]

            total_em += compute_em(pred_answer, gt_answer)
            total_f1 += compute_f1(pred_answer, gt_answer)
            count += 1

    return {
        "loss": total_loss / max(len(loader), 1),
        "em": total_em / max(count, 1) * 100,
        "f1": total_f1 / max(count, 1) * 100,
        "num_examples": count,
    }


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def train_baseline(
    train_data: list,
    val_data: list,
    device: torch.device,
    num_epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 100,
    glove_cache: str = ".glove_cache",
) -> dict:
    """Train baseline BiDAF with GloVe embeddings."""
    print("\n" + "=" * 60)
    print("TRAINING BASELINE BiDAF (GloVe + Char CNN)")
    print("=" * 60)

    # Build vocabulary
    vocab = Vocabulary()
    vocab.build_from_data(train_data + val_data)

    # Load GloVe
    glove_path = download_glove(glove_cache)
    glove_embeddings = load_glove_embeddings(vocab, glove_path, GLOVE_DIM)

    # Create datasets
    train_ds = SQuADBaselineDataset(train_data, vocab)
    val_ds = SQuADBaselineDataset(val_data, vocab)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_baseline)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_baseline)

    # Create model
    model = BiDAF(
        word_vocab_size=vocab.word_vocab_size,
        char_vocab_size=vocab.char_vocab_size,
        word_embed_dim=GLOVE_DIM,
        char_embed_dim=8,
        char_num_filters=100,
        hidden_dim=hidden_dim,
        num_highway_layers=2,
        num_modeling_layers=2,
        dropout=0.2,
        pretrained_word_embeddings=glove_embeddings,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,} total, {trainable:,} trainable")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # Prepare save directory
    save_dir = Path(__file__).resolve().parent / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_f1 = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, "baseline")
        val_metrics = evaluate(model, val_loader, device, "baseline")
        elapsed = time.time() - t0

        scheduler.step(val_metrics["f1"])

        epoch_info = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_em": round(val_metrics["em"], 2),
            "val_f1": round(val_metrics["f1"], 2),
            "time_sec": round(elapsed, 1),
        }
        history.append(epoch_info)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"  Epoch {epoch:2d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"EM={val_metrics['em']:.2f}% | "
            f"F1={val_metrics['f1']:.2f}% | "
            f"{elapsed:.1f}s"
        )

    # Save best model checkpoint + vocabulary
    ckpt_path = save_dir / "baseline_bidaf_best.pt"
    torch.save({
        "model_state_dict": best_state,
        "word_vocab_size": vocab.word_vocab_size,
        "char_vocab_size": vocab.char_vocab_size,
        "word_embed_dim": GLOVE_DIM,
        "hidden_dim": hidden_dim,
        "best_val_f1": best_f1,
    }, ckpt_path)
    print(f"  Saved best baseline checkpoint to {ckpt_path}")

    # Save vocabulary for inference
    vocab_path = save_dir / "baseline_vocab.json"
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump({
            "word2idx": vocab.word2idx,
            "char2idx": vocab.char2idx,
        }, f, ensure_ascii=False)
    print(f"  Saved vocabulary to {vocab_path}")

    # Final evaluation
    final = evaluate(model, val_loader, device, "baseline")

    return {
        "model": "BiDAF (GloVe + Char CNN)",
        "embedding": "GloVe 6B 100d + Character CNN",
        "total_parameters": total,
        "trainable_parameters": trainable,
        "best_val_f1": round(best_f1, 2),
        "final_val_em": round(final["em"], 2),
        "final_val_f1": round(final["f1"], 2),
        "final_val_loss": round(final["loss"], 4),
        "training_history": history,
        "hyperparameters": {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "hidden_dim": hidden_dim,
            "optimizer": "Adam",
            "grad_clip": 5.0,
        },
    }


def train_bert_bidaf(
    train_data: list,
    val_data: list,
    device: torch.device,
    num_epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    hidden_dim: int = 100,
    bert_model_name: str = "bert-base-multilingual-uncased",
) -> dict:
    """Train BiDAF-BERT with frozen BERT embeddings."""
    print("\n" + "=" * 60)
    print("TRAINING BiDAF-BERT (Frozen mBERT Embeddings)")
    print("=" * 60)

    # BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # Create datasets
    train_ds = SQuADBertDataset(train_data, tokenizer)
    val_ds = SQuADBertDataset(val_data, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_bert)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_bert)

    # Create model
    model = BiDAF_BERT(
        bert_model_name=bert_model_name,
        hidden_dim=hidden_dim,
        num_modeling_layers=2,
        dropout=0.2,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,} total, {trainable:,} trainable (BERT frozen)")

    # Only optimize non-BERT parameters
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # Prepare save directory
    save_dir = Path(__file__).resolve().parent / "results"
    save_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_f1 = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, "bert")
        val_metrics = evaluate(model, val_loader, device, "bert")
        elapsed = time.time() - t0

        scheduler.step(val_metrics["f1"])

        epoch_info = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_em": round(val_metrics["em"], 2),
            "val_f1": round(val_metrics["f1"], 2),
            "time_sec": round(elapsed, 1),
        }
        history.append(epoch_info)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            # Only save non-BERT parameters (BERT is frozen and loaded from HF)
            non_bert_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
                if not k.startswith("bert.")
            }
            best_state = non_bert_state

        print(
            f"  Epoch {epoch:2d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"EM={val_metrics['em']:.2f}% | "
            f"F1={val_metrics['f1']:.2f}% | "
            f"{elapsed:.1f}s"
        )

    # Save best model checkpoint (non-BERT weights only, much smaller)
    ckpt_path = save_dir / "bert_bidaf_best.pt"
    torch.save({
        "model_state_dict": best_state,
        "bert_model_name": bert_model_name,
        "hidden_dim": hidden_dim,
        "best_val_f1": best_f1,
    }, ckpt_path)
    print(f"  Saved best BERT-BiDAF checkpoint to {ckpt_path}")

    # Final evaluation
    final = evaluate(model, val_loader, device, "bert")

    return {
        "model": "BiDAF-BERT (Frozen mBERT)",
        "embedding": f"Frozen {bert_model_name} (768-dim)",
        "total_parameters": total,
        "trainable_parameters": trainable,
        "best_val_f1": round(best_f1, 2),
        "final_val_em": round(final["em"], 2),
        "final_val_f1": round(final["f1"], 2),
        "final_val_loss": round(final["loss"], 4),
        "training_history": history,
        "hyperparameters": {
            "epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "hidden_dim": hidden_dim,
            "optimizer": "Adam",
            "grad_clip": 5.0,
            "bert_frozen": True,
        },
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(
    num_train: int = 5000,
    num_val: int = 1000,
    num_epochs: int = 15,
    baseline_batch: int = 32,
    bert_batch: int = 16,
    lr: float = 1e-3,
    hidden_dim: int = 100,
):
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_data, val_data = load_squad(num_train=num_train, num_val=num_val)

    # Train baseline
    baseline_results = train_baseline(
        train_data, val_data, device,
        num_epochs=num_epochs, batch_size=baseline_batch,
        lr=lr, hidden_dim=hidden_dim,
    )

    # Train BERT variant
    bert_results = train_bert_bidaf(
        train_data, val_data, device,
        num_epochs=num_epochs, batch_size=bert_batch,
        lr=lr, hidden_dim=hidden_dim,
    )

    # Comparison summary
    comparison = {
        "dataset": "SQuAD v1.1",
        "train_size": num_train,
        "val_size": num_val,
        "device": str(device),
        "baseline_bidaf": baseline_results,
        "bert_bidaf": bert_results,
        "comparison": {
            "em_improvement": round(
                bert_results["final_val_em"] - baseline_results["final_val_em"], 2
            ),
            "f1_improvement": round(
                bert_results["final_val_f1"] - baseline_results["final_val_f1"], 2
            ),
            "analysis": (
                "BERT embeddings provide richer contextual representations compared to "
                "static GloVe embeddings. The pre-trained language model captures semantic "
                "nuances, polysemy, and long-range dependencies that static word vectors "
                "cannot. This typically results in improved EM and F1 scores, especially "
                "for questions requiring deeper language understanding."
            ),
        },
    }

    # Save results
    results_path = out_dir / "task2_training_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\nResults saved to: {results_path}")

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Baseline BiDAF':>15} {'BiDAF-BERT':>15} {'Δ':>10}")
    print("-" * 65)
    print(
        f"{'Exact Match (%)':.<25} "
        f"{baseline_results['final_val_em']:>15.2f} "
        f"{bert_results['final_val_em']:>15.2f} "
        f"{bert_results['final_val_em'] - baseline_results['final_val_em']:>+10.2f}"
    )
    print(
        f"{'F1 Score (%)':.<25} "
        f"{baseline_results['final_val_f1']:>15.2f} "
        f"{bert_results['final_val_f1']:>15.2f} "
        f"{bert_results['final_val_f1'] - baseline_results['final_val_f1']:>+10.2f}"
    )
    print(
        f"{'Trainable Params':.<25} "
        f"{baseline_results['trainable_parameters']:>15,} "
        f"{bert_results['trainable_parameters']:>15,}"
    )
    print(
        f"{'Total Params':.<25} "
        f"{baseline_results['total_parameters']:>15,} "
        f"{bert_results['total_parameters']:>15,}"
    )

    return comparison


if __name__ == "__main__":
    main()
