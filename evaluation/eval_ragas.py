"""
evaluation/eval_ragas.py
------------------------
Batch evaluation of the RAG pipeline using Ragas.
Runs a test set of queries, scores each one, and logs a
summary report to MLflow for comparison across experiments.

Usage:
    python eval_ragas.py
    python eval_ragas.py --queries_file my_test_queries.json
"""

import os
import sys
import json
import argparse
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from rag_pipeline import run_rag

load_dotenv()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("arxiv-rag-evaluation")

# ── default test query set ─────────────────────────────────────────────────────
DEFAULT_QUERIES = [
    "What are the best chunking strategies for RAG?",
    "How does retrieval augmented generation improve LLM accuracy?",
    "What evaluation metrics are used for RAG pipelines?",
    "How does vector similarity search work in information retrieval?",
    "What are the limitations of current RAG approaches?",
]

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def info(msg): print(f"  {YELLOW}→{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")


def evaluate_queries(queries: list[str], top_k: int = 5) -> dict:
    """Run all queries through RAG and evaluate with Ragas."""

    questions, answers, contexts_list = [], [], []
    latencies, retrieval_scores       = [], []

    print(f"\n── Running {len(queries)} queries ───────────────────────────────────")

    for i, query in enumerate(queries, 1):
        info(f"[{i}/{len(queries)}] {query[:60]}...")
        try:
            t0     = time.time()
            result = run_rag(query, top_k=top_k)
            latency= round(time.time() - t0, 3)

            questions.append(query)
            answers.append(result["answer"])
            contexts_list.append([s["title"] for s in result["sources"]])
            latencies.append(latency)

            avg_score = sum(s["score"] for s in result["sources"]) / len(result["sources"])
            retrieval_scores.append(round(avg_score, 4))

            ok(f"Done in {latency}s | avg retrieval score: {round(avg_score, 4)}")
            time.sleep(1)   # avoid rate limits

        except Exception as e:
            fail(f"Query failed: {e}")

    # ── Ragas batch evaluation ─────────────────────────────────────────────────
    print(f"\n── Ragas batch evaluation ───────────────────────────────────")
    info(f"Evaluating {len(questions)} responses...")

    eval_dataset = Dataset.from_dict({
        "question":  questions,
        "answer":    answers,
        "contexts":  contexts_list,
        "reference": answers,
    })

    try:
        ragas_result = evaluate(
            eval_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        ragas_scores = {
            "ragas/faithfulness":      round(float(ragas_result["faithfulness"]), 4),
            "ragas/answer_relevancy":  round(float(ragas_result["answer_relevancy"]), 4),
            "ragas/context_precision": round(float(ragas_result["context_precision"]), 4),
        }
    except Exception as e:
        fail(f"Ragas evaluation failed: {e}")
        ragas_scores = {}

    return {
        "queries":          questions,
        "answers":          answers,
        "latencies":        latencies,
        "retrieval_scores": retrieval_scores,
        "ragas_scores":     ragas_scores,
        "top_k":            top_k,
    }


def log_to_mlflow(eval_result: dict):
    """Log batch evaluation results to MLflow."""
    with mlflow.start_run(run_name="batch-eval"):

        # params
        mlflow.log_params({
            "num_queries":   len(eval_result["queries"]),
            "top_k":         eval_result["top_k"],
            "embed_model":   "text-embedding-ada-002",
            "chat_model":    "gpt-4o-mini",
            "chunk_size":    400,
            "chunk_overlap": 50,
        })

        # aggregate metrics
        latencies = eval_result["latencies"]
        ret_scores= eval_result["retrieval_scores"]

        mlflow.log_metrics({
            "avg_latency_seconds":    round(sum(latencies) / len(latencies), 3),
            "max_latency_seconds":    round(max(latencies), 3),
            "avg_retrieval_score":    round(sum(ret_scores) / len(ret_scores), 4),
            **eval_result["ragas_scores"],
        })

        # artifact: full results JSON
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            results_path = pathlib.Path(tmp) / "eval_results.json"
            results_path.write_text(json.dumps({
                "queries":  eval_result["queries"],
                "answers":  eval_result["answers"],
                "latencies":eval_result["latencies"],
                "retrieval_scores": eval_result["retrieval_scores"],
                "ragas_scores":     eval_result["ragas_scores"],
            }, indent=2))
            mlflow.log_artifact(str(results_path))

        run_id = mlflow.active_run().info.run_id
        ok(f"Batch eval logged to MLflow run: {run_id}")
        return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Ragas evaluation")
    parser.add_argument("--queries_file", help="JSON file with list of query strings")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    if args.queries_file:
        with open(args.queries_file) as f:
            queries = json.load(f)
    else:
        queries = DEFAULT_QUERIES
        info(f"Using default test set of {len(queries)} queries")

    eval_result = evaluate_queries(queries, top_k=args.top_k)

    print(f"\n── Ragas scores ─────────────────────────────────────────────")
    for metric, score in eval_result["ragas_scores"].items():
        ok(f"{metric}: {score}")

    run_id = log_to_mlflow(eval_result)

    print(f"\n── Summary ──────────────────────────────────────────────────")
    ok(f"Queries evaluated : {len(eval_result['queries'])}")
    ok(f"Avg latency       : {round(sum(eval_result['latencies'])/len(eval_result['latencies']), 2)}s")
    ok(f"MLflow run        : {run_id}")
    print(f"\n  View results: mlflow ui --port 5000\n")