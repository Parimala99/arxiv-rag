"""
upload_to_azure.py
------------------
Loads chunked docs from Azure Blob Storage, generates embeddings
using OpenAI text-embedding-ada-002, and uploads to Azure AI Search.

Usage:
    python upload_to_azure.py --blob chunked-docs/my_chunks.json
    python upload_to_azure.py --all   # embed + index all chunked blobs
"""

import os
import json
import time
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import IndexingResult
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient

load_dotenv()

# ── config ─────────────────────────────────────────────────────────────────────
INDEX_NAME    = "papers-index"
EMBED_MODEL   = "text-embedding-ada-002"
BATCH_SIZE    = 16    # documents per embedding + indexing batch
EMBED_DELAY   = 0.5   # seconds between embedding batches (rate limit safety)

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  \033[92m✓\033[0m  {msg}")
def info(msg): print(f"  {YELLOW}→\033[0m  {msg}")
def fail(msg): print(f"  \033[91m✗\033[0m  {msg}")


# ── clients ────────────────────────────────────────────────────────────────────
def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")),
    )


def get_blob_client():
    return BlobServiceClient.from_connection_string(
        os.getenv("AZURE_STORAGE_CONN_STR")
    )


# ── embedding ──────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str], openai_client: OpenAI) -> list[list[float]]:
    """Embed a batch of texts using OpenAI ada-002."""
    response = openai_client.embeddings.create(
        input=texts,
        model=EMBED_MODEL,
    )
    return [item.embedding for item in response.data]


# ── blob helpers ───────────────────────────────────────────────────────────────
def load_chunked_blob(blob_name: str) -> dict:
    client = get_blob_client()
    data   = client.get_container_client("chunked-docs") \
                   .download_blob(blob_name).readall()
    return json.loads(data)


def list_chunked_blobs() -> list[str]:
    client = get_blob_client()
    return [b["name"] for b in
            client.get_container_client("chunked-docs").list_blobs()
            if b["name"].endswith("_chunks.json")]


# ── main upload logic ──────────────────────────────────────────────────────────
def upload_blob(blob_name: str):
    info(f"Loading chunks from: chunked-docs/{blob_name}")
    data   = load_chunked_blob(blob_name)
    chunks = data.get("chunks", [])
    info(f"Found {len(chunks)} chunks to embed and index")

    openai_client = get_openai_client()
    search_client = get_search_client()

    total_uploaded = 0
    total_failed   = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["chunk_text"] for c in batch]

        # Generate embeddings
        try:
            embeddings = embed_texts(texts, openai_client)
        except Exception as e:
            fail(f"Embedding error on batch {i//BATCH_SIZE + 1}: {e}")
            total_failed += len(batch)
            continue

        # Build search documents
        search_docs = []
        for chunk, embedding in zip(batch, embeddings):
            search_docs.append({
                "id":          chunk["id"],
                "title":       chunk["title"],
                "abstract":    chunk.get("chunk_text", ""),
                "authors":     chunk.get("authors", ""),
                "chunk_text":  chunk["chunk_text"],
                "embedding":   embedding,
            })

        # Upload to Azure AI Search
        try:
            results: list[IndexingResult] = search_client.upload_documents(search_docs)
            succeeded = sum(1 for r in results if r.succeeded)
            failed    = len(results) - succeeded
            total_uploaded += succeeded
            total_failed   += failed

            ok(f"Batch {i//BATCH_SIZE + 1}: {succeeded}/{len(batch)} indexed "
               f"(total: {total_uploaded})")

        except Exception as e:
            fail(f"Search upload error on batch {i//BATCH_SIZE + 1}: {e}")
            total_failed += len(batch)

        time.sleep(EMBED_DELAY)

    print()
    ok(f"Uploaded:  {total_uploaded} documents")
    if total_failed:
        fail(f"Failed:    {total_failed} documents")

    return total_uploaded, total_failed


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed + index chunked papers")
    parser.add_argument("--blob", help="Blob name inside chunked-docs/")
    parser.add_argument("--all",  action="store_true",
                        help="Process all blobs in chunked-docs/")
    args = parser.parse_args()

    print(f"\n── Embedding + indexing ─────────────────────────────────────")
    info(f"Model: {EMBED_MODEL} | Batch size: {BATCH_SIZE}")

    if args.all:
        blobs = list_chunked_blobs()
        info(f"Found {len(blobs)} chunked blobs")
        for blob in blobs:
            upload_blob(blob)
    elif args.blob:
        upload_blob(args.blob)
    else:
        blobs = list_chunked_blobs()
        if blobs:
            info(f"No blob specified — processing most recent: {blobs[-1]}")
            upload_blob(blobs[-1])
        else:
            print("  \033[91m✗\033[0m  No chunked blobs found. Run chunk_docs.py first.")

    print(f"\n\033[92mIndexing complete! Papers are ready to query.\033[0m\n")