"""
evaluate.py — Evaluate RAG system vs baseline LLM (no retrieval).

Metrics:
- Answer relevance (LLM-as-judge)
- Factual grounding (citations present)
- Response latency
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from brain.resolver import ask
from core.clients import get_llm
from core.logger import get_logger

logger = get_logger("evaluate")

# Test questions with expected topics/keywords
TEST_QUESTIONS = [
    {
        "question": "What is GitLab's policy on travel expenses?",
        "expected_topics": ["travel", "expense", "reimbursement", "policy"],
        "category": "policy"
    },
    {
        "question": "How many PTO days do employees get?",
        "expected_topics": ["pto", "days", "vacation", "time off", "leave"],
        "category": "hr"
    },
    {
        "question": "What are GitLab's core values?",
        "expected_topics": ["values", "collaboration", "results", "efficiency", "transparency"],
        "category": "culture"
    },
    {
        "question": "What is the deployment process?",
        "expected_topics": ["deploy", "github", "actions", "jenkins", "ci/cd"],
        "category": "engineering"
    },
    {
        "question": "What security measures does the company have?",
        "expected_topics": ["security", "assurance", "compliance", "protection"],
        "category": "security"
    },
    {
        "question": "How does the engineering team handle code reviews?",
        "expected_topics": ["review", "code", "merge", "approval"],
        "category": "engineering"
    },
    {
        "question": "What is the company's privacy policy?",
        "expected_topics": ["privacy", "data", "gdpr", "personal"],
        "category": "legal"
    },
    {
        "question": "How is the product team structured?",
        "expected_topics": ["product", "team", "manager", "structure"],
        "category": "organization"
    },
    {
        "question": "What communication tools does the company use?",
        "expected_topics": ["slack", "communication", "tools", "async"],
        "category": "tools"
    },
    {
        "question": "What is the process for employee benefits enrollment?",
        "expected_topics": ["benefits", "enrollment", "health", "insurance"],
        "category": "hr"
    },
]

JUDGE_PROMPT = """You are an impartial judge evaluating answer quality.

Question: {question}
Expected topics: {expected_topics}

Answer to evaluate:
{answer}

Rate the answer on these criteria (1-5 scale):
1. Relevance: Does it address the question directly?
2. Specificity: Does it provide concrete details, not vague generalities?
3. Accuracy: Does it mention expected topics appropriately?

Return ONLY a JSON object:
{{"relevance": X, "specificity": X, "accuracy": X, "explanation": "brief reason"}}
"""


def baseline_ask(question: str) -> str:
    """Ask LLM directly without retrieval (baseline)."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful corporate assistant. Answer this question:\n\n{question}"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})


def judge_answer(question: str, answer: str, expected_topics: list[str]) -> dict:
    """Use LLM to judge answer quality."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "question": question,
        "expected_topics": ", ".join(expected_topics),
        "answer": answer
    })

    try:
        # Extract JSON from response
        import re
        json_match = re.search(r'\{[^}]+\}', result, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass

    return {"relevance": 3, "specificity": 3, "accuracy": 3, "explanation": "parse error"}


def has_citations(answer: str) -> bool:
    """Check if answer contains source citations."""
    citation_patterns = ["according to", "states that", "mentions", "in the",
                         "slack", "confluence", "gitlab", "updated", "says"]
    return any(p in answer.lower() for p in citation_patterns)


def run_evaluation():
    """Run full evaluation comparing RAG vs baseline."""
    results = {
        "rag": {"scores": [], "latencies": [], "citations": []},
        "baseline": {"scores": [], "latencies": [], "citations": []},
        "questions": []
    }

    logger.info("=== Starting Evaluation ===")
    logger.info("Testing %d questions", len(TEST_QUESTIONS))

    for i, test in enumerate(TEST_QUESTIONS):
        q = test["question"]
        expected = test["expected_topics"]
        logger.info("\n[%d/%d] %s", i+1, len(TEST_QUESTIONS), q)

        # RAG system
        start = time.time()
        rag_answer = ask(q)
        rag_latency = time.time() - start
        rag_scores = judge_answer(q, rag_answer, expected)
        rag_cited = has_citations(rag_answer)

        # Baseline (no retrieval)
        start = time.time()
        baseline_answer = baseline_ask(q)
        baseline_latency = time.time() - start
        baseline_scores = judge_answer(q, baseline_answer, expected)
        baseline_cited = has_citations(baseline_answer)

        # Store results
        results["rag"]["scores"].append(rag_scores)
        results["rag"]["latencies"].append(rag_latency)
        results["rag"]["citations"].append(rag_cited)

        results["baseline"]["scores"].append(baseline_scores)
        results["baseline"]["latencies"].append(baseline_latency)
        results["baseline"]["citations"].append(baseline_cited)

        results["questions"].append({
            "question": q,
            "category": test["category"],
            "rag_answer": rag_answer[:500],
            "baseline_answer": baseline_answer[:500]
        })

        logger.info("  RAG: rel=%d, spec=%d, acc=%d, latency=%.2fs, cited=%s",
                   rag_scores.get("relevance", 0), rag_scores.get("specificity", 0),
                   rag_scores.get("accuracy", 0), rag_latency, rag_cited)
        logger.info("  Baseline: rel=%d, spec=%d, acc=%d, latency=%.2fs",
                   baseline_scores.get("relevance", 0), baseline_scores.get("specificity", 0),
                   baseline_scores.get("accuracy", 0), baseline_latency)

    return results


def generate_graphs(results: dict, output_dir: Path):
    """Generate comparison graphs."""
    output_dir.mkdir(exist_ok=True)

    # Extract metrics
    rag_rel = [s.get("relevance", 0) for s in results["rag"]["scores"]]
    rag_spec = [s.get("specificity", 0) for s in results["rag"]["scores"]]
    rag_acc = [s.get("accuracy", 0) for s in results["rag"]["scores"]]

    base_rel = [s.get("relevance", 0) for s in results["baseline"]["scores"]]
    base_spec = [s.get("specificity", 0) for s in results["baseline"]["scores"]]
    base_acc = [s.get("accuracy", 0) for s in results["baseline"]["scores"]]

    # 1. Overall scores comparison (bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["Relevance", "Specificity", "Accuracy"]
    rag_means = [np.mean(rag_rel), np.mean(rag_spec), np.mean(rag_acc)]
    base_means = [np.mean(base_rel), np.mean(base_spec), np.mean(base_acc)]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, rag_means, width, label="RAG (with retrieval)", color="#2ecc71")
    bars2 = ax.bar(x + width/2, base_means, width, label="Baseline (no retrieval)", color="#e74c3c")

    ax.set_ylabel("Score (1-5)")
    ax.set_title("RAG vs Baseline: Answer Quality Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 5.5)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / "quality_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved: quality_comparison.png")

    # 2. Per-question comparison (grouped bar)
    fig, ax = plt.subplots(figsize=(14, 6))
    questions_short = [f"Q{i+1}" for i in range(len(rag_rel))]

    x = np.arange(len(questions_short))
    width = 0.35

    ax.bar(x - width/2, rag_acc, width, label="RAG", color="#2ecc71")
    ax.bar(x + width/2, base_acc, width, label="Baseline", color="#e74c3c")

    ax.set_ylabel("Accuracy Score")
    ax.set_title("Per-Question Accuracy: RAG vs Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(questions_short)
    ax.legend()
    ax.set_ylim(0, 5.5)

    plt.tight_layout()
    plt.savefig(output_dir / "per_question_accuracy.png", dpi=150)
    plt.close()
    logger.info("Saved: per_question_accuracy.png")

    # 3. Latency comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    rag_lat = results["rag"]["latencies"]
    base_lat = results["baseline"]["latencies"]

    ax.bar(x - width/2, rag_lat, width, label="RAG", color="#3498db")
    ax.bar(x + width/2, base_lat, width, label="Baseline", color="#9b59b6")

    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Response Latency: RAG vs Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(questions_short)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "latency_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved: latency_comparison.png")

    # 4. Citation rate (pie charts)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    rag_cited = sum(results["rag"]["citations"])
    rag_not_cited = len(results["rag"]["citations"]) - rag_cited

    base_cited = sum(results["baseline"]["citations"])
    base_not_cited = len(results["baseline"]["citations"]) - base_cited

    ax1.pie([rag_cited, rag_not_cited], labels=["With Citations", "No Citations"],
            autopct='%1.0f%%', colors=["#2ecc71", "#bdc3c7"])
    ax1.set_title("RAG: Source Citations")

    ax2.pie([base_cited, base_not_cited], labels=["With Citations", "No Citations"],
            autopct='%1.0f%%', colors=["#e74c3c", "#bdc3c7"])
    ax2.set_title("Baseline: Source Citations")

    plt.tight_layout()
    plt.savefig(output_dir / "citation_rate.png", dpi=150)
    plt.close()
    logger.info("Saved: citation_rate.png")

    # 5. Radar chart - overall comparison
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    categories = ["Relevance", "Specificity", "Accuracy", "Citations\n(normalized)", "Speed\n(inverted)"]

    # Normalize metrics to 0-5 scale
    rag_citation_norm = (sum(results["rag"]["citations"]) / len(results["rag"]["citations"])) * 5
    base_citation_norm = (sum(results["baseline"]["citations"]) / len(results["baseline"]["citations"])) * 5

    # Invert latency (lower is better) - normalize to 0-5
    max_lat = max(max(rag_lat), max(base_lat))
    rag_speed = 5 - (np.mean(rag_lat) / max_lat * 5)
    base_speed = 5 - (np.mean(base_lat) / max_lat * 5)

    rag_values = [np.mean(rag_rel), np.mean(rag_spec), np.mean(rag_acc), rag_citation_norm, rag_speed]
    base_values = [np.mean(base_rel), np.mean(base_spec), np.mean(base_acc), base_citation_norm, base_speed]

    # Close the polygon
    rag_values += rag_values[:1]
    base_values += base_values[:1]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax.plot(angles, rag_values, 'o-', linewidth=2, label="RAG", color="#2ecc71")
    ax.fill(angles, rag_values, alpha=0.25, color="#2ecc71")
    ax.plot(angles, base_values, 'o-', linewidth=2, label="Baseline", color="#e74c3c")
    ax.fill(angles, base_values, alpha=0.25, color="#e74c3c")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.set_title("Overall Performance Comparison", y=1.08)

    plt.tight_layout()
    plt.savefig(output_dir / "radar_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved: radar_comparison.png")

    # 6. Summary statistics
    summary = {
        "rag": {
            "avg_relevance": np.mean(rag_rel),
            "avg_specificity": np.mean(rag_spec),
            "avg_accuracy": np.mean(rag_acc),
            "avg_latency": np.mean(rag_lat),
            "citation_rate": sum(results["rag"]["citations"]) / len(results["rag"]["citations"])
        },
        "baseline": {
            "avg_relevance": np.mean(base_rel),
            "avg_specificity": np.mean(base_spec),
            "avg_accuracy": np.mean(base_acc),
            "avg_latency": np.mean(base_lat),
            "citation_rate": sum(results["baseline"]["citations"]) / len(results["baseline"]["citations"])
        },
        "improvement": {
            "relevance": np.mean(rag_rel) - np.mean(base_rel),
            "specificity": np.mean(rag_spec) - np.mean(base_spec),
            "accuracy": np.mean(rag_acc) - np.mean(base_acc)
        }
    }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump({"summary": summary, "detailed": results}, f, indent=2, default=str)
    logger.info("Saved: evaluation_results.json")

    return summary


def main():
    output_dir = Path(__file__).parent.parent / "evaluation_results"

    logger.info("Running evaluation...")
    results = run_evaluation()

    logger.info("\nGenerating graphs...")
    summary = generate_graphs(results, output_dir)

    logger.info("\n=== Evaluation Complete ===")
    logger.info("RAG avg scores: rel=%.2f, spec=%.2f, acc=%.2f",
               summary["rag"]["avg_relevance"],
               summary["rag"]["avg_specificity"],
               summary["rag"]["avg_accuracy"])
    logger.info("Baseline avg scores: rel=%.2f, spec=%.2f, acc=%.2f",
               summary["baseline"]["avg_relevance"],
               summary["baseline"]["avg_specificity"],
               summary["baseline"]["avg_accuracy"])
    logger.info("Improvement: rel=%+.2f, spec=%+.2f, acc=%+.2f",
               summary["improvement"]["relevance"],
               summary["improvement"]["specificity"],
               summary["improvement"]["accuracy"])
    logger.info("\nResults saved to: %s", output_dir)


if __name__ == "__main__":
    main()
