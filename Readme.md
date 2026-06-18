# arXiv RAG Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline that lets you ask natural language questions over arXiv research papers and get cited, grounded answers — built entirely on the Azure free tier.

> *"What are the best chunking strategies for RAG?"*
> → Retrieves relevant paper excerpts → Generates a cited answer with GPT-4o-mini

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Data Ingestion Layer                                           │
│                                                                 │
│  arXiv API → Python Parser → Azure Blob Storage → AI Search    │
│  (free)       (chunk+clean)   (raw + chunked)     (vector idx) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  RAG Inference Layer                                            │
│                                                                 │
│  User Query → Embed (ada-002) → Top-k Retrieval → GPT-4o-mini │
│                                  (Azure AI Search)  (cited ans) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  MLOps Layer                                                    │
│                                                                 │
│  MLflow Tracking → GitHub Actions CI/CD → Docker → ACR         │
│  (params+metrics)  (lint+test+build+push)  (FastAPI)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| Data source | arXiv public API | Free |
| Storage | Azure Blob Storage | Free (5 GB / 12 mo) |
| Vector store | Azure AI Search | Free (always) |
| Embeddings | OpenAI `text-embedding-ada-002` | ~$0.10 |
| LLM | GPT-4o-mini | ~$3–5 |
| API | FastAPI + Uvicorn | Free |
| Experiment tracking | MLflow (SQLite backend) | Free |
| Evaluation | Ragas (faithfulness, relevancy) | Free |
| CI/CD | GitHub Actions | Free |
| Container registry | Azure Container Registry | Free (12 mo) |
| **Total project cost** | | **< $10** |

---

## Project Structure

```
arxiv-rag/
├── ingestion/
│   ├── fetch_arxiv.py        # Fetch papers from arXiv API
│   ├── chunk_docs.py         # Sliding window chunking (400w, 50w overlap)
│   └── upload_to_azure.py    # Embed + index into Azure AI Search
├── mlflow_tracking/
│   └── log_experiment.py     # MLflow tracked RAG runs
├── evaluation/
│   └── eval_ragas.py         # Batch Ragas evaluation
├── tests/
│   └── test_pipeline.py      # Pytest unit tests
├── .github/workflows/
│   └── ci.yml                # GitHub Actions CI/CD pipeline
├── rag_pipeline.py           # Core RAG logic (embed → retrieve → generate)
├── main.py                   # FastAPI app (/ask and /search endpoints)
├── Dockerfile                # Multi-stage Docker build
├── requirements.txt
└── .env.example
```

---

## Setup

### Prerequisites
- Azure free account ([azure.microsoft.com/free](https://azure.microsoft.com/free))
- OpenAI API account with credits ([platform.openai.com](https://platform.openai.com))
- Python 3.11+
- Docker (optional, for containerized deployment)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/arxiv-rag.git
cd arxiv-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Fill in your actual keys in .env
```

Required variables:

```
AZURE_STORAGE_CONN_STR=your-connection-string
AZURE_SEARCH_ENDPOINT=https://your-service.search.windows.net
AZURE_SEARCH_KEY=your-admin-key
OPENAI_API_KEY=sk-...
```

### 4. Verify Azure connections

```bash
python verify_conn.py
```

---

## Usage

### Ingest papers

```bash
# Fetch 50 papers on RAG from arXiv
python ingestion/fetch_arxiv.py --query "retrieval augmented generation" --max 50

# Chunk into 400-word sliding windows
python ingestion/chunk_docs.py --all

# Embed with ada-002 and index into Azure AI Search
python ingestion/upload_to_azure.py --all
```

### Query the pipeline

```bash
python rag_pipeline.py --query "What are the best chunking strategies for RAG?"
```

**Sample output:**
```
── Answer ──────────────────────────────────────────────────
Chunking strategies for RAG primarily involve dividing documents
into smaller segments. Two advanced techniques are late chunking
and contextual retrieval [1]. A cost-constrained framework using
Monte Carlo Tree Search has also been proposed [2].

── Sources ─────────────────────────────────────────────────
[1] Reconstructing Context: Evaluating Advanced Chunking Strategies
    Carlo Merola, Jaspinder Singh | score: 0.8946
[2] CARROT: A Learned Cost-Constrained Retrieval Optimization System
    Ziting Wang et al. | score: 0.8617
```

### Start the API

```bash
uvicorn main:app --reload --port 8000
```

Endpoints:
- `POST /ask` — ask a question, get a cited answer
- `POST /search` — semantic search without generation
- `GET /docs` — interactive Swagger UI

### Track experiments with MLflow

```bash
python mlflow_tracking/log_experiment.py --query "How does RLHF work?"
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Run evaluation

```bash
python evaluation/eval_ragas.py
```

Scores each response on faithfulness, answer relevancy, and context precision.

### Run tests

```bash
pytest tests/ -v
```

### Docker

```bash
docker build -t arxiv-rag-api .
docker run -p 8000:8000 --env-file .env arxiv-rag-api
```

---

## CI/CD Pipeline

Every push to `main` triggers the GitHub Actions workflow:

```
push to main
    ↓
Lint (ruff) + Unit tests (pytest)
    ↓ (if passing)
Docker build
    ↓
Push to Azure Container Registry
```

---

## MLOps Design Decisions

**Why Azure AI Search over ChromaDB?**
Azure AI Search integrates natively with the rest of the Azure stack, supports HNSW vector search on the free tier, and mirrors production-grade vector store patterns used in enterprise MLOps pipelines.

**Why sliding window chunking?**
Fixed 400-word chunks with 50-word overlap preserve cross-sentence context at chunk boundaries — a key failure mode of naive fixed-size chunking identified in recent RAG literature.

**Why MLflow with SQLite backend?**
MLflow's file store backend is deprecated in recent versions. SQLite provides a lightweight, portable tracking backend that works locally and can be swapped for a remote MLflow server without code changes.

**Why GPT-4o-mini over GPT-4o?**
GPT-4o-mini delivers 90%+ of GPT-4o quality at ~15x lower cost for RAG tasks, where the answer quality is primarily determined by retrieval quality rather than model size.

---

## Author

**Parimala** — MLOps Engineer  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/Parimala99)