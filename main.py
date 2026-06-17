"""
api/main.py
-----------
FastAPI app exposing the RAG pipeline as a REST API.

Endpoints:
    GET  /              health check
    POST /ask           ask a question, get a cited answer
    GET  /search        keyword search (no generation)

Run locally:
    uvicorn main:app --reload --port 8000

Then test:
    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"question": "What are chunking strategies for RAG?"}'
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from rag_pipeline import run_rag, embed_query, retrieve_chunks, get_clients

# ── app setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="arXiv RAG Assistant",
    description="Ask questions over arXiv research papers with cited answers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ──────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=5,
                          example="What are the best chunking strategies for RAG?")
    top_k:    Optional[int] = Field(default=5, ge=1, le=10,
                                    description="Number of chunks to retrieve")

class Source(BaseModel):
    index:   int
    title:   str
    authors: str
    score:   float

class AskResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[Source]
    top_k:    int

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, example="transformer attention")
    top_k: Optional[int] = Field(default=5, ge=1, le=10)

class SearchResult(BaseModel):
    title:      str
    authors:    str
    chunk_text: str
    score:      float


# ── endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {
        "status":  "ok",
        "service": "arXiv RAG Assistant",
        "version": "1.0.0",
    }


@app.post("/ask", response_model=AskResponse, summary="Ask a question")
def ask(request: AskRequest):
    """
    Ask a question and get an answer grounded in arXiv paper chunks,
    with citations back to the source papers.
    """
    try:
        result = run_rag(request.question, top_k=request.top_k)
        return AskResponse(
            question=result["query"],
            answer=result["answer"],
            sources=[Source(**s) for s in result["sources"]],
            top_k=result["top_k"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=list[SearchResult],
          summary="Semantic search without generation")
def search(request: SearchRequest):
    """
    Pure semantic search — returns the most relevant paper chunks
    without running them through the LLM. Useful for debugging retrieval.
    """
    try:
        openai_client, search_client = get_clients()
        query_embedding = embed_query(request.query, openai_client)
        chunks = retrieve_chunks(query_embedding, search_client, request.top_k)
        return [
            SearchResult(
                title=      c["title"],
                authors=    c["authors"],
                chunk_text= c["chunk_text"],
                score=      round(c["score"], 4),
            )
            for c in chunks
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))