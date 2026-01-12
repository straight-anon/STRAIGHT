#!/usr/bin/env python3
"""
Embed RAG chunks with OpenAI and store vectors inside an SQLite database
backed by the sqlite-vec extension.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

from openai import OpenAI, OpenAIError
from sqlite_vec import (
    Connection as VecConnection,
    load as load_sqlite_vec,
    serialize_float32,
)
from tqdm import tqdm


DEFAULT_CHUNKS = Path("ragpdf_chunks.jsonl")
DEFAULT_DB = Path("ragpdf.sqlite")
DEFAULT_MODEL = "text-embedding-3-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate OpenAI embeddings for ragpdf chunk JSONL and store in SQLite."
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS, help="Path to chunk JSONL.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB, help="SQLite destination file.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI embedding model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Drop existing tables before inserting new data.",
    )
    return parser.parse_args()


def read_chunks(path: Path) -> List[dict]:
    if not path.exists():
        raise SystemExit(f"Chunk file not found: {path}")
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def connect_db(path: Path, overwrite: bool) -> sqlite3.Connection:
    conn = sqlite3.connect(path, factory=VecConnection)
    conn.enable_load_extension(True)
    load_sqlite_vec(conn)
    if overwrite:
        conn.execute("DROP TABLE IF EXISTS rag_embeddings")
        conn.execute("DROP TABLE IF EXISTS rag_chunks")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            text TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_embeddings (
            chunk_id TEXT PRIMARY KEY REFERENCES rag_chunks(id) ON DELETE CASCADE,
            embedding BLOB NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def ensure_vec_registration(conn: sqlite3.Connection, dimension: int) -> None:
    try:
        conn.execute("SELECT vec_store('rag_embeddings', 'embedding', ?)", (dimension,))
    except sqlite3.DatabaseError:
        # Assume already registered with correct dimension.
        pass


def existing_chunk_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT chunk_id FROM rag_embeddings").fetchall()
    return {row[0] for row in rows}


def batch_iter(seq: Sequence[dict], batch_size: int) -> Iterable[Sequence[dict]]:
    for idx in range(0, len(seq), batch_size):
        yield seq[idx : idx + batch_size]


def embed_batch(
    client: OpenAI,
    model: str,
    batch: Sequence[dict],
    max_retries: int = 3,
    backoff: float = 2.0,
) -> List[List[float]]:
    inputs = [item["text"] for item in batch]
    attempt = 0
    while True:
        try:
            response = client.embeddings.create(model=model, input=inputs)
            return [data.embedding for data in response.data]
        except OpenAIError as exc:
            attempt += 1
            if attempt >= max_retries:
                raise
            wait = backoff * attempt
            print(f"⚠️  OpenAI error ({exc}); retrying in {wait:.1f}s...", file=sys.stderr)
            time.sleep(wait)


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY environment variable is required.")

    records = read_chunks(args.chunks)
    print(f"Loaded {len(records)} chunks from {args.chunks}")

    conn = connect_db(args.database, args.overwrite)
    client = OpenAI(api_key=api_key)

    skip_ids = existing_chunk_ids(conn) if not args.overwrite else set()
    to_process = [rec for rec in records if rec["id"] not in skip_ids]
    print(f"{len(to_process)} chunks pending embedding ({len(records) - len(to_process)} skipped).")
    if not to_process:
        print("Nothing to do.")
        return

    dim_registered = False
    inserted = 0

    for batch in tqdm(list(batch_iter(to_process, args.batch_size)), desc="Embedding"):
        embeddings = embed_batch(client, args.model, batch)
        if not dim_registered:
            ensure_vec_registration(conn, len(embeddings[0]))
            dim_registered = True

        for item, vector in zip(batch, embeddings):
            conn.execute(
                """
                INSERT OR IGNORE INTO rag_chunks (id, source, page_start, page_end, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (item["id"], item["source"], item["page_start"], item["page_end"], item["text"]),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO rag_embeddings (chunk_id, embedding)
                VALUES (?, ?)
                """,
                (item["id"], serialize_float32(vector)),
            )
            inserted += 1
        conn.commit()

    print(f"✅ Inserted {inserted} embeddings into {args.database}")


if __name__ == "__main__":
    main()
