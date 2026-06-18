"""
tests/test_pipeline.py
-----------------------
Unit tests for the RAG pipeline.
Run with: pytest tests/ -v
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_pipeline import build_prompt


# ── helpers ────────────────────────────────────────────────────────────────────
SAMPLE_CHUNKS = [
    {
        "id":         "paper_001_chunk_0",
        "title":      "Attention Is All You Need",
        "authors":    "Vaswani et al.",
        "chunk_text": "The transformer model relies on attention mechanisms to process sequences.",
        "published":  "2017-06-12",
        "pdf_url":    "https://arxiv.org/pdf/1706.03762",
        "score":      0.91,
    },
    {
        "id":         "paper_002_chunk_0",
        "title":      "RAG for Knowledge-Intensive NLP",
        "authors":    "Lewis et al.",
        "chunk_text": "Retrieval augmented generation combines parametric and non-parametric memory.",
        "published":  "2020-05-22",
        "pdf_url":    "https://arxiv.org/pdf/2005.11401",
        "score":      0.87,
    },
]

# ── prompt building ────────────────────────────────────────────────────────────
def test_build_prompt_returns_two_messages():
    messages = build_prompt("What is attention?", SAMPLE_CHUNKS)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_prompt_includes_query():
    query    = "What is attention?"
    messages = build_prompt(query, SAMPLE_CHUNKS)
    assert query in messages[1]["content"]


def test_build_prompt_includes_chunk_titles():
    messages = build_prompt("test query", SAMPLE_CHUNKS)
    user_content = messages[1]["content"]
    assert "Attention Is All You Need" in user_content
    assert "RAG for Knowledge-Intensive NLP" in user_content


def test_build_prompt_includes_citation_markers():
    messages = build_prompt("test query", SAMPLE_CHUNKS)
    user_content = messages[1]["content"]
    assert "[1]" in user_content
    assert "[2]" in user_content


def test_build_prompt_empty_chunks():
    messages = build_prompt("test query", [])
    assert len(messages) == 2   # should still return valid structure


# ── source structure ───────────────────────────────────────────────────────────
def test_chunk_has_required_keys():
    required = {"id", "title", "chunk_text", "score", "published", "pdf_url"}
    for chunk in SAMPLE_CHUNKS:
        assert required.issubset(chunk.keys()), \
            f"Chunk missing keys: {required - chunk.keys()}"


def test_chunk_scores_in_valid_range():
    for chunk in SAMPLE_CHUNKS:
        assert 0.0 <= chunk["score"] <= 1.0, \
            f"Score out of range: {chunk['score']}"


def test_chunk_text_not_empty():
    for chunk in SAMPLE_CHUNKS:
        assert len(chunk["chunk_text"].strip()) > 0