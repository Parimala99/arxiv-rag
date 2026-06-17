"""
inference/rag_pipeline.py
--------------------------
Full RAG inference pipeline:
  1. Embed the user query with OpenAI ada-002
  2. Retrieve top-k chunks from Azure AI Search
  3. Build a prompt with retrieved context
  4. Generate a cited answer with GPT-4o-mini

Usage (standalone test):
    python rag_pipeline.py --query "What are the best chunking strategies for RAG?"
"""

import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# ── config ─────────────────────────────────────────────────────────────────────
INDEX_NAME   = "papers-index"
EMBED_MODEL  = "text-embedding-ada-002"
CHAT_MODEL   = "gpt-4o-mini"
TOP_K        = 5      # number of chunks to retrieve
MAX_TOKENS   = 800    # max tokens in generated answer

YELLOW = "\033[93m"
GREEN  = "\033[92m"
RESET  = "\033[0m"

def info(msg): print(f"  {YELLOW}→{RESET}  {msg}")
def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")


# ── clients ────────────────────────────────────────────────────────────────────
def get_clients():
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
    )
    return openai_client, search_client


# ── step 1: embed query ────────────────────────────────────────────────────────
def embed_query(query: str, openai_client: OpenAI) -> list[float]:
    response = openai_client.embeddings.create(
        input=query,
        model=EMBED_MODEL,
    )
    return response.data[0].embedding


# ── step 2: retrieve top-k chunks ─────────────────────────────────────────────
def retrieve_chunks(query_embedding: list[float], search_client: SearchClient,
                    top_k: int = TOP_K) -> list[dict]:
    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=top_k,
        fields="embedding",
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["id", "title", "authors", "chunk_text", "published", "pdf_url"],
        top=top_k,
    )

    chunks = []
    for r in results:
        chunks.append({
            "id":         r["id"],
            "title":      r["title"],
            "authors":    r.get("authors", ""),
            "chunk_text": r["chunk_text"],
            "published":  r.get("published", ""),
            "pdf_url":    r.get("pdf_url", ""),
            "score":      r["@search.score"],
        })

    return chunks


# ── step 3: build prompt ───────────────────────────────────────────────────────
def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[{i}] Title: {chunk['title']}\n"
            f"    Authors: {chunk['authors']}\n"
            f"    Published: {chunk['published']}\n"
            f"    PDF URL: {chunk['pdf_url']}\n"
            f"    Excerpt: {chunk['chunk_text']}"
        )

    context = "\n\n".join(context_blocks)

    system_prompt = """You are a research assistant that answers questions about \
academic papers. You answer ONLY using the provided context excerpts. \
For every claim you make, cite the source using its bracket number e.g. [1], [2]. \
If the context does not contain enough information to answer, say so clearly. \
Be concise and precise."""

    user_prompt = f"""Context excerpts from research papers:

{context}

---

Question: {query}

Answer (cite sources with bracket numbers):"""

    return [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_prompt},
    ]


# ── step 4: generate answer ────────────────────────────────────────────────────
def generate_answer(messages: list[dict], openai_client: OpenAI) -> str:
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.2,   # low temp = more factual, less hallucination
    )
    return response.choices[0].message.content.strip()


# ── full pipeline ──────────────────────────────────────────────────────────────
def run_rag(query: str, top_k: int = TOP_K) -> dict:
    """
    Run the full RAG pipeline for a query.
    Returns a dict with answer, sources, and debug info.
    """
    openai_client, search_client = get_clients()

    # 1. Embed
    query_embedding = embed_query(query, openai_client)

    # 2. Retrieve
    chunks = retrieve_chunks(query_embedding, search_client, top_k)

    # 3. Build prompt
    messages = build_prompt(query, chunks)

    # 4. Generate
    answer = generate_answer(messages, openai_client)

    # Build source list for the response
    sources = [
        {
            "index":   i + 1,
            "title":   c["title"],
            "authors": c["authors"],
            "score":   round(c["score"], 4),
        }
        for i, c in enumerate(chunks)
    ]

    return {
        "query":   query,
        "answer":  answer,
        "sources": sources,
        "top_k":   top_k,
    }


# ── standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG pipeline")
    parser.add_argument("--query", required=True, help="Your question")
    parser.add_argument("--top_k", type=int, default=TOP_K,
                        help=f"Chunks to retrieve (default {TOP_K})")
    args = parser.parse_args()

    print(f"\n── RAG query ────────────────────────────────────────────────")
    info(f"Query : {args.query}")
    info(f"Top-k : {args.top_k}")
    print()

    result = run_rag(args.query, args.top_k)

    print(f"\n── Answer ───────────────────────────────────────────────────")
    print(f"\n{result['answer']}\n")

    print(f"── Sources ──────────────────────────────────────────────────")
    for s in result["sources"]:
        print(f"  [{s['index']}] {s['title']}")
        print(f"       {s['authors']} | score: {s['score']}")
    print()