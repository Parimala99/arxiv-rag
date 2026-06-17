"""
fix_index.py
------------
Deletes the existing papers-index and recreates it with all fields
correctly marked as retrievable, then re-uploads all chunked docs.

Run once:
    python fix_index.py
"""

import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField,
    SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
)
from azure.search.documents.models import IndexingResult
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient

load_dotenv()

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def info(msg): print(f"  {YELLOW}→{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")

INDEX_NAME  = "papers-index"
EMBED_MODEL = "text-embedding-ada-002"
BATCH_SIZE  = 16

endpoint   = os.getenv("AZURE_SEARCH_ENDPOINT")
key        = os.getenv("AZURE_SEARCH_KEY")
credential = AzureKeyCredential(key)

# ── Step 1: delete existing index ──────────────────────────────────────────────
print("\n── Step 1: delete existing index ────────────────────────────────")
index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

try:
    index_client.delete_index(INDEX_NAME)
    ok(f"Deleted index '{INDEX_NAME}'")
except Exception as e:
    info(f"Index didn't exist or already deleted: {e}")


# ── Step 2: recreate with all fields retrievable ───────────────────────────────
print("\n── Step 2: recreate index with correct schema ───────────────────")

fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
        retrievable=True,
    ),
    SearchableField(
        name="title",
        type=SearchFieldDataType.String,
        retrievable=True,
    ),
    SearchableField(
        name="authors",
        type=SearchFieldDataType.String,
        retrievable=True,
    ),
    SearchableField(
        name="chunk_text",
        type=SearchFieldDataType.String,
        retrievable=True,
    ),
    SimpleField(
        name="published",
        type=SearchFieldDataType.String,
        retrievable=True,
        filterable=True,
    ),
    SimpleField(
        name="pdf_url",
        type=SearchFieldDataType.String,
        retrievable=True,
    ),
    SearchField(
        name="embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        retrievable=False,      # embeddings don't need to be returned
        vector_search_dimensions=1536,
        vector_search_profile_name="hnsw-profile",
    ),
]

vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
    profiles=[VectorSearchProfile(
        name="hnsw-profile",
        algorithm_configuration_name="hnsw-config",
    )],
)

index = SearchIndex(
    name=INDEX_NAME,
    fields=fields,
    vector_search=vector_search,
)

index_client.create_index(index)
ok(f"Index '{INDEX_NAME}' recreated with all fields retrievable")


# ── Step 3: re-upload all chunks with embeddings ───────────────────────────────
print("\n── Step 3: re-embed and re-upload all chunks ────────────────────")

blob_client   = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONN_STR"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
search_client = SearchClient(endpoint=endpoint, index_name=INDEX_NAME, credential=credential)

chunked_blobs = [
    b["name"] for b in
    blob_client.get_container_client("chunked-docs").list_blobs()
    if b["name"].endswith("_chunks.json")
]

info(f"Found {len(chunked_blobs)} chunked blob(s) to reindex")

total_uploaded = 0

for blob_name in chunked_blobs:
    info(f"Processing: {blob_name}")
    raw    = blob_client.get_container_client("chunked-docs").download_blob(blob_name).readall()
    data   = json.loads(raw)
    chunks = data.get("chunks", [])
    info(f"  {len(chunks)} chunks found")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["chunk_text"] for c in batch]

        # Embed
        response   = openai_client.embeddings.create(input=texts, model=EMBED_MODEL)
        embeddings = [item.embedding for item in response.data]

        # Build search docs with all retrievable fields populated
        search_docs = []
        for chunk, embedding in zip(batch, embeddings):
            search_docs.append({
                "id":         chunk["id"],
                "title":      chunk.get("title", ""),
                "authors":    chunk.get("authors", ""),
                "chunk_text": chunk.get("chunk_text", ""),
                "published":  chunk.get("published", ""),
                "pdf_url":    chunk.get("pdf_url", ""),
                "embedding":  embedding,
            })

        results: list[IndexingResult] = search_client.upload_documents(search_docs)
        succeeded = sum(1 for r in results if r.succeeded)
        total_uploaded += succeeded
        ok(f"  Batch {i//BATCH_SIZE + 1}: {succeeded}/{len(batch)} indexed (total: {total_uploaded})")
        time.sleep(0.5)

print(f"\n{GREEN}All done! {total_uploaded} documents reindexed with correct schema.{RESET}")
print(f"{GREEN}Now re-run: python rag_pipeline.py --query \"What are chunking strategies for RAG?\"{RESET}\n")