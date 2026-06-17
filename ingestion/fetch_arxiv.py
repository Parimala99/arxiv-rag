"""
fetch_arxiv.py
--------------
Fetches research papers from the arXiv API by topic/category
and saves raw results to Azure Blob Storage.

Usage:
    python fetch_arxiv.py --query "RAG retrieval augmented generation" --max 50
    python fetch_arxiv.py --query "MLOps machine learning" --max 100
"""

import os
import json
import time
import argparse
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

# ── arXiv API config ───────────────────────────────────────────────────────────
ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
NS = {
    "atom":   "http://www.w3.org/2005/Atom",
    "arxiv":  "http://arxiv.org/schemas/atom",
    "dc":     "http://purl.org/dc/elements/1.1/",
    "opensc": "http://a9.com/-/spec/opensearch/1.1/",
}

# ── helpers ────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):   print(f"  \033[92m✓\033[0m  {msg}")
def info(msg): print(f"  {YELLOW}→\033[0m  {msg}")


def fetch_batch(query: str, start: int, max_results: int) -> list[dict]:
    """Fetch one batch of papers from arXiv API."""
    params = urllib.parse.urlencode({
        "search_query": query,
        "start":        start,
        "max_results":  max_results,
        "sortBy":       "relevance",
        "sortOrder":    "descending",
    })
    url = f"{ARXIV_BASE_URL}?{params}"
    info(f"Fetching batch start={start}, size={max_results} ...")

    with urllib.request.urlopen(url) as resp:
        xml_data = resp.read()

    root    = ET.fromstring(xml_data)
    entries = root.findall("atom:entry", NS)
    papers  = []

    for entry in entries:
        arxiv_id = entry.find("atom:id", NS).text.split("/abs/")[-1]
        title    = entry.find("atom:title", NS).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", NS).text.strip().replace("\n", " ")
        published= entry.find("atom:published", NS).text[:10]

        authors = [
            a.find("atom:name", NS).text
            for a in entry.findall("atom:author", NS)
        ]

        categories = [
            c.attrib.get("term", "")
            for c in entry.findall("atom:category", NS)
        ]

        pdf_link = ""
        for link in entry.findall("atom:link", NS):
            if link.attrib.get("type") == "application/pdf":
                pdf_link = link.attrib.get("href", "")

        papers.append({
            "id":          arxiv_id,
            "title":       title,
            "abstract":    abstract,
            "authors":     authors,
            "published":   published,
            "categories":  categories,
            "pdf_url":     pdf_link,
            "source":      "arxiv",
            "fetched_at":  datetime.utcnow().isoformat(),
        })

    return papers


def fetch_all_papers(query: str, max_total: int, batch_size: int = 25) -> list[dict]:
    """Fetch papers in batches, respecting arXiv's rate limits."""
    all_papers = []
    start      = 0

    while len(all_papers) < max_total:
        remaining   = max_total - len(all_papers)
        this_batch  = min(batch_size, remaining)
        batch       = fetch_batch(query, start, this_batch)

        if not batch:
            info("No more results from arXiv.")
            break

        all_papers.extend(batch)
        start += this_batch
        ok(f"Fetched {len(all_papers)}/{max_total} papers")

        if len(all_papers) < max_total:
            time.sleep(3)   # arXiv asks for 3s between requests

    return all_papers


def upload_to_blob(papers: list[dict], query: str) -> str:
    """Upload raw papers JSON to Azure Blob Storage."""
    conn_str = os.getenv("AZURE_STORAGE_CONN_STR")
    client   = BlobServiceClient.from_connection_string(conn_str)
    container= client.get_container_client("raw-papers")

    safe_query = query.replace(" ", "_")[:40]
    timestamp  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    blob_name  = f"{safe_query}_{timestamp}.json"

    payload = json.dumps({
        "query":      query,
        "fetched_at": datetime.utcnow().isoformat(),
        "count":      len(papers),
        "papers":     papers,
    }, indent=2)

    container.upload_blob(blob_name, payload, overwrite=True)
    ok(f"Uploaded {len(papers)} papers → raw-papers/{blob_name}")
    return blob_name


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch arXiv papers")
    parser.add_argument("--query", default="retrieval augmented generation",
                        help="Search query for arXiv")
    parser.add_argument("--max",   type=int, default=50,
                        help="Max number of papers to fetch (default 50)")
    args = parser.parse_args()

    print(f"\n── Fetching arXiv papers ────────────────────────────────────")
    info(f"Query : {args.query}")
    info(f"Max   : {args.max} papers")

    papers    = fetch_all_papers(args.query, args.max)
    blob_name = upload_to_blob(papers, args.query)

    print(f"\n\033[92mDone! {len(papers)} papers saved to raw-papers/{blob_name}\033[0m\n")
