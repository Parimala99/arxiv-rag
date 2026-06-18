"""
verify_azure_connections.py
----------------------------
Run this script to confirm your Azure Blob Storage and AI Search
connections are working before building the ingestion pipeline.

Usage:
    pip install azure-storage-blob azure-search-documents python-dotenv
    python verify_azure_connections.py
"""

import os
import json
import sys
from dotenv import load_dotenv

load_dotenv()

# ── colour helpers for terminal output ────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def info(msg): print(f"  {YELLOW}→{RESET}  {msg}")

# ── 1. Check required env vars ─────────────────────────────────────────────────
print("\n── Step 1: environment variables ────────────────────────────────")

REQUIRED = {
    "AZURE_STORAGE_CONN_STR": os.getenv("AZURE_STORAGE_CONN_STR"),
    "AZURE_SEARCH_ENDPOINT":  os.getenv("AZURE_SEARCH_ENDPOINT"),
    "AZURE_SEARCH_KEY":       os.getenv("AZURE_SEARCH_KEY"),
    "OPENAI_API_KEY":         os.getenv("OPENAI_API_KEY"),
}

env_ok = True
for var, val in REQUIRED.items():
    if val:
        masked = val[:6] + "..." + val[-4:] if len(val) > 12 else "***"
        ok(f"{var} = {masked}")
    else:
        fail(f"{var} is missing from .env")
        env_ok = False

if not env_ok:
    print(f"\n{RED}Fix missing env vars before continuing.{RESET}\n")
    sys.exit(1)


# ── 2. Azure Blob Storage ──────────────────────────────────────────────────────
print("\n── Step 2: Azure Blob Storage ───────────────────────────────────")

try:
    from azure.storage.blob import BlobServiceClient

    conn_str = os.getenv("AZURE_STORAGE_CONN_STR")
    client   = BlobServiceClient.from_connection_string(conn_str)

    # List existing containers
    containers = [c["name"] for c in client.list_containers()]
    ok("Connected to storage account")
    info(f"Containers found: {containers if containers else '(none yet)'}")

    # Check for expected containers
    for expected in ["raw-papers", "chunked-docs"]:
        if expected in containers:
            ok(f"Container '{expected}' exists")
        else:
            info(f"Container '{expected}' not found — creating it now")
            client.create_container(expected)
            ok(f"Container '{expected}' created")

    # Write + read a test blob
    container_client = client.get_container_client("raw-papers")
    test_data        = json.dumps({"test": True, "message": "arxiv-rag connection check"})
    blob_name        = "_connection_test.json"

    container_client.upload_blob(blob_name, test_data, overwrite=True)
    ok(f"Test blob uploaded to 'raw-papers/{blob_name}'")

    downloaded = container_client.download_blob(blob_name).readall()
    parsed     = json.loads(downloaded)
    assert parsed["test"] is True
    ok("Test blob downloaded and verified")

    # Clean up
    container_client.delete_blob(blob_name)
    ok("Test blob cleaned up")

    blob_ok = True

except Exception as e:
    fail(f"Blob Storage error: {e}")
    blob_ok = False


# ── 3. Azure AI Search ─────────────────────────────────────────────────────────
print("\n── Step 3: Azure AI Search ──────────────────────────────────────")

try:
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchIndex, SimpleField, SearchableField,
        SearchField, SearchFieldDataType, VectorSearch,
        HnswAlgorithmConfiguration, VectorSearchProfile,
    )
    from azure.core.credentials import AzureKeyCredential

    endpoint   = os.getenv("AZURE_SEARCH_ENDPOINT")
    key        = os.getenv("AZURE_SEARCH_KEY")
    credential = AzureKeyCredential(key)

    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

    # List existing indexes
    existing = [idx.name for idx in index_client.list_indexes()]
    ok("Connected to Azure AI Search")
    info(f"Existing indexes: {existing if existing else '(none yet)'}")

    # Create papers-index if it doesn't exist
    INDEX_NAME = "papers-index"

    if INDEX_NAME not in existing:
        info(f"Creating index '{INDEX_NAME}'...")

        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(name="title",     type=SearchFieldDataType.String),
            SearchableField(name="abstract",  type=SearchFieldDataType.String),
            SearchableField(name="authors",   type=SearchFieldDataType.String),
            SearchableField(name="chunk_text",type=SearchFieldDataType.String),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
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
        ok(f"Index '{INDEX_NAME}' created with vector search enabled")
    else:
        ok(f"Index '{INDEX_NAME}' already exists")

    # Upload + retrieve a test document
    from azure.search.documents import SearchClient

    search_client = SearchClient(
        endpoint=endpoint,
        index_name=INDEX_NAME,
        credential=credential,
    )

    test_doc = {
        "id":         "test-doc-001",
        "title":      "Connection Test Paper",
        "abstract":   "This is a connection verification document.",
        "authors":    "Test Author",
        "chunk_text": "Azure AI Search connection verified successfully.",
        "embedding":  [0.0] * 1536,
    }

    search_client.upload_documents([test_doc])
    ok(f"Test document uploaded to '{INDEX_NAME}'")

    # Simple keyword search to verify retrieval
    import time
    time.sleep(2)  # allow indexing to propagate

    results = list(search_client.search(
        search_text="connection verification",
        select=["id", "title"],
        top=1,
    ))

    if results:
        ok(f"Test document retrieved: '{results[0]['title']}'")
    else:
        info("Document uploaded but not yet searchable (indexing delay — this is normal)")

    # Clean up test doc
    search_client.delete_documents([{"id": "test-doc-001"}])
    ok("Test document cleaned up")

    search_ok = True

except Exception as e:
    fail(f"AI Search error: {e}")
    search_ok = False


# ── 4. Summary ─────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────────────")

results = {
    "Blob Storage":  blob_ok,
    "AI Search":     search_ok,
}

all_ok = True
for service, status in results.items():
    if status:
        ok(f"{service}: ready")
    else:
        fail(f"{service}: needs attention")
        all_ok = False

if all_ok:
    print(f"\n{GREEN}All connections verified. You're ready to build the ingestion pipeline!{RESET}\n")
else:
    print(f"\n{YELLOW}Fix the failing connections above, then re-run this script.{RESET}\n")
    sys.exit(1)