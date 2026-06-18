"""
mlflow_tracking/log_experiment.py
----------------------------------
Wraps the RAG pipeline with MLflow tracking.
Logs query params, retrieved chunks, Ragas eval scores, and latency.

Usage:
    python log_experiment.py --query "What are chunking strategies for RAG?"
    python log_experiment.py --query "How does RLHF work?" --top_k 3

Then view the UI:
    mlflow ui --port 5000
    open http://localhost:5000
"""

import os
import sys
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from dotenv import load_dotenv
from rag_pipeline import run_rag
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

load_dotenv()

# ── MLflow config ──────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "arxiv-rag-pipeline"
TRACKING_URI = "sqlite:///mlflow.db"        # local folder; change to remote URI if needed

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

YELLOW = "\033[93m"
GREEN  = "\033[92m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def info(msg): print(f"  {YELLOW}→{RESET}  {msg}")


# ── Ragas evaluation ───────────────────────────────────────────────────────────
def run_ragas_eval(query: str, answer: str, chunks: list[dict]) -> dict[str, float]:
    """
    Score the RAG response on four Ragas metrics:
      - faithfulness:       is the answer grounded in the retrieved context?
      - answer_relevancy:   does the answer address the question?
      - context_precision:  are the retrieved chunks relevant to the question?
      - context_recall:     did retrieval capture enough relevant information?
    """
    contexts = [c["chunk_text"] for c in chunks]

    # Ragas expects a HuggingFace Dataset with these exact column names
    eval_dataset = Dataset.from_dict({
        "question":  [query],
        "answer":    [answer],
        "contexts":  [contexts],
        "reference": [answer],   # using answer as reference for standalone eval
    })

    info("Running Ragas evaluation (this may take ~30 seconds)...")
    try:
        result = evaluate(
            eval_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        scores = {
            "ragas/faithfulness":        round(float(result["faithfulness"]), 4),
            "ragas/answer_relevancy":    round(float(result["answer_relevancy"]), 4),
            "ragas/context_precision":   round(float(result["context_precision"]), 4),
            "ragas/context_recall":      round(float(result["context_recall"]), 4),
        }
    except Exception as e:
        info(f"Ragas eval failed (non-fatal): {e}")
        scores = {
            "ragas/faithfulness":      -1.0,
            "ragas/answer_relevancy":  -1.0,
            "ragas/context_precision": -1.0,
            "ragas/context_recall":    -1.0,
        }

    return scores


# ── tracked RAG run ────────────────────────────────────────────────────────────
def tracked_rag_run(query: str, top_k: int = 5) -> dict:
    """
    Run RAG pipeline and log everything to MLflow:
      params  — query, top_k, models used, chunk config
      metrics — latency, Ragas scores, retrieval scores
      artifacts — full answer text, retrieved sources JSON
    """
    with mlflow.start_run():

        # ── log parameters ─────────────────────────────────────────────────────
        mlflow.log_params({
            "query":           query,
            "top_k":           top_k,
            "embed_model":     "text-embedding-ada-002",
            "chat_model":      "gpt-4o-mini",
            "index_name":      "papers-index",
            "chunk_size":      400,
            "chunk_overlap":   50,
            "temperature":     0.2,
            "max_tokens":      800,
        })

        # ── run pipeline + measure latency ─────────────────────────────────────
        info(f"Running RAG pipeline for: '{query}'")
        t0     = time.time()
        result = run_rag(query, top_k=top_k)
        latency= round(time.time() - t0, 3)

        ok(f"Pipeline completed in {latency}s")

        # ── log latency + retrieval scores ─────────────────────────────────────
        mlflow.log_metric("latency_seconds", latency)

        for i, source in enumerate(result["sources"]):
            mlflow.log_metric(f"retrieval_score_chunk_{i+1}", source["score"])

        avg_score = round(
            sum(s["score"] for s in result["sources"]) / len(result["sources"]), 4
        ) if result["sources"] else 0.0
        mlflow.log_metric("avg_retrieval_score", avg_score)

        # ── run Ragas evaluation ───────────────────────────────────────────────
        chunks = [
            {"chunk_text": s["title"]}   # using title as proxy since chunk_text
            for s in result["sources"]   # isn't in the sources dict
        ]
        ragas_scores = run_ragas_eval(query, result["answer"], chunks)
        mlflow.log_metrics(ragas_scores)

        for metric, score in ragas_scores.items():
            ok(f"{metric}: {score}")

        # ── log answer + sources as artifacts ──────────────────────────────────
        import json
        import tempfile
        import pathlib

        with tempfile.TemporaryDirectory() as tmp:
            # Answer text file
            answer_path = pathlib.Path(tmp) / "answer.txt"
            answer_path.write_text(
                f"Query: {query}\n\nAnswer:\n{result['answer']}\n"
            )
            mlflow.log_artifact(str(answer_path))

            # Sources JSON
            sources_path = pathlib.Path(tmp) / "sources.json"
            sources_path.write_text(json.dumps(result["sources"], indent=2))
            mlflow.log_artifact(str(sources_path))

        run_id = mlflow.active_run().info.run_id
        ok(f"MLflow run logged: {run_id}")

        result["mlflow_run_id"]    = run_id
        result["latency_seconds"]  = latency
        result["ragas_scores"]     = ragas_scores

    return result


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracked RAG run with MLflow + Ragas")
    parser.add_argument("--query",  required=True, help="Question to ask")
    parser.add_argument("--top_k",  type=int, default=5)
    args = parser.parse_args()

    print("\n── Tracked RAG run ──────────────────────────────────────────")

    result = tracked_rag_run(args.query, args.top_k)

    print("\n── Answer ───────────────────────────────────────────────────")
    print(f"\n{result['answer']}\n")

    print("── Sources ──────────────────────────────────────────────────")
    for s in result["sources"]:
        print(f"  [{s['index']}] {s['title']} | score: {s['score']}")

    print("\n── MLflow ───────────────────────────────────────────────────")
    print(f"  Run ID  : {result['mlflow_run_id']}")
    print(f"  Latency : {result['latency_seconds']}s")
    print("  View UI : mlflow ui --port 5000\n")