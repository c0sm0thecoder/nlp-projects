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
    # Company-specific questions (RAG should win - info from internal Slack/Confluence)
    {
        "question": "How many PTO days do Lead roles get?",
        "expected_topics": ["25", "days", "lead", "pto"],
        "category": "internal-hr",
        "rag_advantage": True,  # This info is from internal Slack
    },
    {
        "question": "What CI/CD tool does the company currently use for deployments?",
        "expected_topics": ["github actions", "deploy", "ci/cd"],
        "category": "internal-eng",
        "rag_advantage": True,  # Alex Chen announced switch from Jenkins
    },
    {
        "question": "Who announced the change from Jenkins to GitHub Actions?",
        "expected_topics": ["alex", "chen", "lead", "architect"],
        "category": "internal-eng",
        "rag_advantage": True,
    },
    {
        "question": "What did Sarah Mitchell announce about PTO policy?",
        "expected_topics": ["sarah", "mitchell", "25", "days", "lead"],
        "category": "internal-hr",
        "rag_advantage": True,
    },
    # General policy questions (from GitLab handbook - both may know)
    {
        "question": "What is the policy on travel expenses?",
        "expected_topics": ["travel", "expense", "reimbursement", "policy"],
        "category": "policy",
        "rag_advantage": False,
    },
    {
        "question": "What are the company's core values?",
        "expected_topics": ["values", "collaboration", "results", "efficiency", "transparency"],
        "category": "culture",
        "rag_advantage": False,
    },
    {
        "question": "What security policies are in place?",
        "expected_topics": ["security", "assurance", "compliance", "protection"],
        "category": "security",
        "rag_advantage": False,
    },
    {
        "question": "How does the engineering team handle code reviews?",
        "expected_topics": ["review", "code", "merge", "approval"],
        "category": "engineering",
        "rag_advantage": False,
    },
    {
        "question": "What is the company's privacy policy?",
        "expected_topics": ["privacy", "data", "gdpr", "personal"],
        "category": "legal",
        "rag_advantage": False,
    },
    {
        "question": "What communication guidelines does the company follow?",
        "expected_topics": ["communication", "async", "slack", "meetings"],
        "category": "tools",
        "rag_advantage": False,
    },
]

JUDGE_PROMPT = """You are an impartial judge evaluating answer quality for a company Q&A system.

Question: {question}
Expected topics/keywords that should appear: {expected_topics}

Answer to evaluate:
{answer}

Rate the answer on these criteria (1-5 scale):
1. Relevance: Does it directly answer the question asked?
2. Specificity: Does it give concrete details (names, dates, numbers) rather than vague generalities?
3. Accuracy: Does it mention the expected topics/keywords? (Check if {expected_topics} appear)
4. Grounding: Does it cite sources (names, dates, "according to", "in Slack/Confluence")?

Scoring guide:
- 5: Excellent - directly answers with specific details and citations
- 4: Good - answers correctly with some specifics
- 3: Adequate - general answer without specifics
- 2: Poor - vague or partially wrong
- 1: Bad - doesn't answer or wrong

Return ONLY a JSON object:
{{"relevance": X, "specificity": X, "accuracy": X, "grounding": X, "explanation": "brief reason"}}
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
            "rag_advantage": test.get("rag_advantage", False),
            "rag_answer": rag_answer[:500],
            "baseline_answer": baseline_answer[:500]
        })

        logger.info("  RAG: rel=%d, spec=%d, acc=%d, ground=%d, latency=%.2fs",
                   rag_scores.get("relevance", 0), rag_scores.get("specificity", 0),
                   rag_scores.get("accuracy", 0), rag_scores.get("grounding", 0), rag_latency)
        logger.info("  Baseline: rel=%d, spec=%d, acc=%d, ground=%d, latency=%.2fs",
                   baseline_scores.get("relevance", 0), baseline_scores.get("specificity", 0),
                   baseline_scores.get("accuracy", 0), baseline_scores.get("grounding", 0), baseline_latency)

    return results


def generate_graphs(results: dict, output_dir: Path):
    """Generate comparison graphs."""
    output_dir.mkdir(exist_ok=True)

    # Extract metrics
    rag_rel = [s.get("relevance", 0) for s in results["rag"]["scores"]]
    rag_spec = [s.get("specificity", 0) for s in results["rag"]["scores"]]
    rag_acc = [s.get("accuracy", 0) for s in results["rag"]["scores"]]
    rag_ground = [s.get("grounding", 0) for s in results["rag"]["scores"]]

    base_rel = [s.get("relevance", 0) for s in results["baseline"]["scores"]]
    base_spec = [s.get("specificity", 0) for s in results["baseline"]["scores"]]
    base_acc = [s.get("accuracy", 0) for s in results["baseline"]["scores"]]
    base_ground = [s.get("grounding", 0) for s in results["baseline"]["scores"]]

    # 1. Overall scores comparison (bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ["Relevance", "Specificity", "Accuracy", "Grounding"]
    rag_means = [np.mean(rag_rel), np.mean(rag_spec), np.mean(rag_acc), np.mean(rag_ground)]
    base_means = [np.mean(base_rel), np.mean(base_spec), np.mean(base_acc), np.mean(base_ground)]

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

    # 5. Radar chart - overall comparison (with grounding)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    categories = ["Relevance", "Specificity", "Accuracy", "Grounding", "Speed\n(inverted)"]

    max_lat = max(max(rag_lat), max(base_lat))
    rag_speed = 5 - (np.mean(rag_lat) / max_lat * 5)
    base_speed = 5 - (np.mean(base_lat) / max_lat * 5)

    rag_values = [np.mean(rag_rel), np.mean(rag_spec), np.mean(rag_acc), np.mean(rag_ground), rag_speed]
    base_values = [np.mean(base_rel), np.mean(base_spec), np.mean(base_acc), np.mean(base_ground), base_speed]

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

    # 6. Internal vs External questions comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    internal_idx = [i for i, q in enumerate(results["questions"]) if q.get("rag_advantage")]
    external_idx = [i for i, q in enumerate(results["questions"]) if not q.get("rag_advantage")]

    rag_internal_acc = np.mean([rag_acc[i] for i in internal_idx]) if internal_idx else 0
    rag_external_acc = np.mean([rag_acc[i] for i in external_idx]) if external_idx else 0
    base_internal_acc = np.mean([base_acc[i] for i in internal_idx]) if internal_idx else 0
    base_external_acc = np.mean([base_acc[i] for i in external_idx]) if external_idx else 0

    x = np.arange(2)
    width = 0.35

    bars1 = ax.bar(x - width/2, [rag_internal_acc, rag_external_acc], width, label="RAG", color="#2ecc71")
    bars2 = ax.bar(x + width/2, [base_internal_acc, base_external_acc], width, label="Baseline", color="#e74c3c")

    ax.set_ylabel("Accuracy Score")
    ax.set_title("Internal (Private) vs External (Public) Questions")
    ax.set_xticks(x)
    ax.set_xticklabels(["Internal Questions\n(from Slack/Confluence)", "External Questions\n(public knowledge)"])
    ax.legend()
    ax.set_ylim(0, 5.5)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / "internal_vs_external.png", dpi=150)
    plt.close()
    logger.info("Saved: internal_vs_external.png")

    # 7. Summary statistics
    summary = {
        "rag": {
            "avg_relevance": np.mean(rag_rel),
            "avg_specificity": np.mean(rag_spec),
            "avg_accuracy": np.mean(rag_acc),
            "avg_grounding": np.mean(rag_ground),
            "avg_latency": np.mean(rag_lat),
            "citation_rate": sum(results["rag"]["citations"]) / len(results["rag"]["citations"]),
            "internal_accuracy": rag_internal_acc,
            "external_accuracy": rag_external_acc,
        },
        "baseline": {
            "avg_relevance": np.mean(base_rel),
            "avg_specificity": np.mean(base_spec),
            "avg_accuracy": np.mean(base_acc),
            "avg_grounding": np.mean(base_ground),
            "avg_latency": np.mean(base_lat),
            "citation_rate": sum(results["baseline"]["citations"]) / len(results["baseline"]["citations"]),
            "internal_accuracy": base_internal_acc,
            "external_accuracy": base_external_acc,
        },
        "improvement": {
            "relevance": np.mean(rag_rel) - np.mean(base_rel),
            "specificity": np.mean(rag_spec) - np.mean(base_spec),
            "accuracy": np.mean(rag_acc) - np.mean(base_acc),
            "grounding": np.mean(rag_ground) - np.mean(base_ground),
            "internal_accuracy": rag_internal_acc - base_internal_acc,
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
    logger.info("RAG avg scores: rel=%.2f, spec=%.2f, acc=%.2f, ground=%.2f",
               summary["rag"]["avg_relevance"],
               summary["rag"]["avg_specificity"],
               summary["rag"]["avg_accuracy"],
               summary["rag"]["avg_grounding"])
    logger.info("Baseline avg scores: rel=%.2f, spec=%.2f, acc=%.2f, ground=%.2f",
               summary["baseline"]["avg_relevance"],
               summary["baseline"]["avg_specificity"],
               summary["baseline"]["avg_accuracy"],
               summary["baseline"]["avg_grounding"])
    logger.info("Overall improvement: rel=%+.2f, spec=%+.2f, acc=%+.2f, ground=%+.2f",
               summary["improvement"]["relevance"],
               summary["improvement"]["specificity"],
               summary["improvement"]["accuracy"],
               summary["improvement"]["grounding"])
    logger.info("Internal questions (RAG advantage): RAG=%.2f, Baseline=%.2f, Improvement=%+.2f",
               summary["rag"]["internal_accuracy"],
               summary["baseline"]["internal_accuracy"],
               summary["improvement"]["internal_accuracy"])
    logger.info("\nResults saved to: %s", output_dir)


if __name__ == "__main__":
    main()
