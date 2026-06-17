"""
chunk_docs.py
-------------
Loads raw paper JSONs from Azure Blob Storage, chunks the abstract
+ title text using a sliding window, and saves chunked docs back
to the 'chunked-docs' container ready for embedding.

Usage:
    python chunk_docs.py --blob raw-papers/retrieval_augmented_20240101.json
    python chunk_docs.py --all   # process all blobs in raw-papers
"""

import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

# ── chunking config ────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 50    # overlapping words between chunks

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  \033[92m✓\033[0m  {msg}")
def info(msg): print(f"  {YELLOW}→\033[0m  {msg}")


# ── chunking logic ─────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Each chunk is chunk_size words, sliding forward by (chunk_size - overlap).
    """
    words  = text.split()
    stride = chunk_size - overlap
    chunks = []

    for i in range(0, len(words), stride):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break

    return chunks


def build_chunks_for_paper(paper: dict) -> list[dict]:
    """
    Build chunk documents for a single paper.
    Each chunk includes enough metadata for the retriever to cite the source.
    """
    # Combine title + abstract as the text to chunk
    full_text = f"{paper['title']}. {paper['abstract']}"
    text_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

    chunks = []
    for idx, chunk in enumerate(text_chunks):
        chunk_id = f"{paper['id'].replace('/', '_').replace('.', '_')}_chunk_{idx}"
        chunks.append({
            "id":          chunk_id,
            "paper_id":    paper["id"],
            "title":       paper["title"],
            "authors":     ", ".join(paper.get("authors", [])),
            "published":   paper.get("published", ""),
            "categories":  ", ".join(paper.get("categories", [])),
            "pdf_url":     paper.get("pdf_url", ""),
            "chunk_index": idx,
            "chunk_total": len(text_chunks),
            "chunk_text":  chunk,
            "chunked_at":  datetime.utcnow().isoformat(),
        })

    return chunks


# ── blob helpers ───────────────────────────────────────────────────────────────
def get_blob_client():
    conn_str = os.getenv("AZURE_STORAGE_CONN_STR")
    return BlobServiceClient.from_connection_string(conn_str)


def load_raw_blob(blob_name: str) -> dict:
    client   = get_blob_client()
    data     = client.get_container_client("raw-papers") \
                     .download_blob(blob_name).readall()
    return json.loads(data)


def list_raw_blobs() -> list[str]:
    client = get_blob_client()
    return [b["name"] for b in
            client.get_container_client("raw-papers").list_blobs()
            if b["name"].endswith(".json") and not b["name"].startswith("_")]


def save_chunks(chunks: list[dict], source_blob: str):
    client     = get_blob_client()
    container  = client.get_container_client("chunked-docs")

    safe_name  = source_blob.replace("/", "_").replace(".json", "")
    blob_name  = f"{safe_name}_chunks.json"
    payload    = json.dumps({
        "source_blob":  source_blob,
        "chunked_at":   datetime.utcnow().isoformat(),
        "chunk_count":  len(chunks),
        "chunks":       chunks,
    }, indent=2)

    container.upload_blob(blob_name, payload, overwrite=True)
    ok(f"Saved {len(chunks)} chunks → chunked-docs/{blob_name}")
    return blob_name


# ── main ───────────────────────────────────────────────────────────────────────
def process_blob(blob_name: str):
    info(f"Processing: {blob_name}")
    raw      = load_raw_blob(blob_name)
    papers   = raw.get("papers", [])
    info(f"Found {len(papers)} papers")

    all_chunks = []
    for paper in papers:
        chunks = build_chunks_for_paper(paper)
        all_chunks.extend(chunks)

    ok(f"Generated {len(all_chunks)} chunks from {len(papers)} papers")
    save_chunks(all_chunks, blob_name)
    return all_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk arXiv papers")
    parser.add_argument("--blob", help="Specific blob name in raw-papers/")
    parser.add_argument("--all",  action="store_true",
                        help="Process all blobs in raw-papers/")
    args = parser.parse_args()

    print(f"\n── Chunking papers ──────────────────────────────────────────")
    info(f"Chunk size: {CHUNK_SIZE} words | Overlap: {CHUNK_OVERLAP} words")

    if args.all:
        blobs = list_raw_blobs()
        info(f"Found {len(blobs)} blobs to process")
        for blob in blobs:
            process_blob(blob)
    elif args.blob:
        process_blob(args.blob)
    else:
        # Default: process the most recent blob
        blobs = list_raw_blobs()
        if blobs:
            info(f"No blob specified — processing most recent: {blobs[-1]}")
            process_blob(blobs[-1])
        else:
            print("  \033[91m✗\033[0m  No blobs found in raw-papers/. Run fetch_arxiv.py first.")

    print(f"\n\033[92mChunking complete!\033[0m\n")