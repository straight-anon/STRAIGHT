#!/usr/bin/env python3
"""Lightweight helper to fetch RAG context text for a given question."""

from __future__ import annotations

import argparse
import os
import sqlite3
from array import array
from dataclasses import dataclass
from math import dist
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from openai import OpenAI

DEFAULT_DB = Path("ragpdf.sqlite")
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    page_start: int
    page_end: int
    distance: float
    text: str


def connect_db(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    return sqlite3.connect(path)


def embed_query(client: OpenAI, model: str, question: str) -> List[float]:
    response = client.embeddings.create(model=model, input=[question])
    return response.data[0].embedding


def retrieve_chunks(conn: sqlite3.Connection, query_vec: Sequence[float], top_k: int) -> List[RetrievedChunk]:
    sql = """
        SELECT e.chunk_id, c.source, c.page_start, c.page_end, c.text, e.embedding
        FROM rag_embeddings AS e
        JOIN rag_chunks AS c ON c.id = e.chunk_id
    """
    rows = conn.execute(sql).fetchall()
    scored: List[RetrievedChunk] = []
    for chunk_id, source, page_start, page_end, text, blob in rows:
        vec = array("f", blob)
        distance = dist(query_vec, vec)
        scored.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                source=source,
                page_start=page_start,
                page_end=page_end,
                distance=distance,
                text=text,
            )
        )
    scored.sort(key=lambda item: item.distance)
    return scored[: top_k or 1]


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return "No context retrieved."
    parts = []
    for chunk in chunks:
        label = f"[{chunk.source} p.{chunk.page_start}-{chunk.page_end}]"
        parts.append(f"{label}\n{chunk.text.strip()}")
    return "\n\n".join(parts)


def get_rag_context(
    question: str,
    *,
    top_k: int = 5,
    db_path: Path = DEFAULT_DB,
    embed_model: str = DEFAULT_EMBED_MODEL,
    client: Optional[OpenAI] = None,
    api_key: Optional[str] = None,
) -> Tuple[str, List[RetrievedChunk]]:
    """Return the formatted context and raw chunks for the given question."""

    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    conn = connect_db(db_path)
    close_conn = True
    try:
        if client is None:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is required to fetch embeddings.")
            client = OpenAI(api_key=key)
        query_vec = embed_query(client, embed_model, question)
        chunks = retrieve_chunks(conn, query_vec, top_k)
    finally:
        if close_conn:
            conn.close()

    context = format_context(chunks)
    return context, chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Print retrieved RAG context for a question.")
    parser.add_argument("question", nargs="+", help="Question text to retrieve context for.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to ragpdf SQLite database.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL, help="Embedding model name.")
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    context, chunks = get_rag_context(
        question,
        top_k=args.top_k,
        db_path=args.db,
        embed_model=args.embed_model,
    )
    print("=== Retrieved Context ===\n")
    print(context)
    print("\n--- Chunks ---")
    for chunk in chunks:
        print(f"{chunk.chunk_id} | {chunk.source} p.{chunk.page_start}-{chunk.page_end} | dist={chunk.distance:.4f}")


if __name__ == "__main__":
    main()
