#!/usr/bin/env python3
"""
Utility helpers to convert PDFs inside ragpdf/ into text dumps and RAG-friendly
chunks. Each PDF produces a raw text file plus overlapping chunks serialized as
JSONL for downstream embedding.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - dependency missing at runtime
    PdfReader = None  # type: ignore[assignment]


DEFAULT_INPUT_DIR = Path("ragpdf")
DEFAULT_TEXT_DIR = Path("ragpdf_text")
DEFAULT_CHUNKS_PATH = Path("ragpdf_chunks.jsonl")
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDFs under ragpdf/ into text files and chunked JSONL."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory holding source PDFs (default: ragpdf/).",
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=DEFAULT_TEXT_DIR,
        help="Where to store extracted plain-text dumps (default: ragpdf_text/).",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Destination JSONL file for chunks (default: ragpdf_chunks.jsonl).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Approximate character budget per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Approximate overlap size in characters between consecutive chunks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional progress information while processing.",
    )
    return parser.parse_args()


def require_pypdf() -> None:
    if PdfReader is None:
        raise SystemExit(
            "pypdf is required for PDF extraction. Install it via `pip install pypdf`."
        )


def extract_pages(pdf_path: Path) -> List[str]:
    require_pypdf()
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - pdf quirks
            print(f"⚠️  Failed to extract text from {pdf_path.name}: {exc}", file=sys.stderr)
            text = ""
        pages.append(text)
    return pages


def normalize_page(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_paragraphs(pages: Sequence[str]) -> List[Tuple[str, int]]:
    paragraphs: List[Tuple[str, int]] = []
    for page_idx, raw in enumerate(pages, start=1):
        normalized = normalize_page(raw)
        if not normalized:
            continue
        for block in re.split(r"\n{2,}", normalized):
            block = block.strip()
            if not block:
                continue
            paragraphs.append((block, page_idx))
    return paragraphs


@dataclass
class Chunk:
    document: str
    chunk_id: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str


def build_chunks(
    paragraphs: Sequence[Tuple[str, int]],
    *,
    doc_name: str,
    chunk_size: int,
    overlap: int,
    progress_cb: Callable[[Chunk], None] | None = None,
) -> List[Chunk]:
    if not paragraphs:
        return []

    chunks: List[Chunk] = []
    cursor = 0
    chunk_counter = 0

    while cursor < len(paragraphs):
        assembled: List[str] = []
        pages: List[int] = []
        length = 0
        start_idx = cursor
        idx = cursor

        while idx < len(paragraphs) and length < chunk_size:
            paragraph, page = paragraphs[idx]
            assembled.append(paragraph)
            pages.append(page)
            length += len(paragraph) + 2  # add separator allowance
            idx += 1

        if not assembled:
            break

        chunk_counter += 1
        text = "\n\n".join(assembled).strip()
        chunk_id = f"{Path(doc_name).stem}-{chunk_counter:04d}"
        chunk = Chunk(
            document=doc_name,
            chunk_id=chunk_id,
            chunk_index=chunk_counter - 1,
            page_start=min(pages),
            page_end=max(pages),
            text=text,
        )
        chunks.append(chunk)
        if progress_cb:
            progress_cb(chunk)

        if idx >= len(paragraphs):
            break

        span = idx - start_idx
        if span <= 0:
            cursor = idx
            continue

        # Step back to create overlap
        overlap_chars = 0
        cursor = idx
        while cursor > start_idx and overlap_chars < overlap:
            prev_text, _ = paragraphs[cursor - 1]
            overlap_chars += len(prev_text) + 2
            cursor -= 1

        # Ensure forward progress: never restart before start_idx + 1
        cursor = max(cursor, start_idx + 1)
        if cursor >= len(paragraphs):
            break

    return chunks


def dump_text(text_dir: Path, pdf_path: Path, pages: Sequence[str]) -> Path:
    text_dir.mkdir(parents=True, exist_ok=True)
    target = text_dir / f"{pdf_path.stem}.txt"
    merged = "\n\n".join(page.strip() for page in pages if page.strip())
    target.write_text(merged, encoding="utf-8")
    return target


def write_chunks(chunks_path: Path, chunks: Sequence[Chunk]) -> None:
    if not chunks:
        chunks_path.write_text("", encoding="utf-8")
        return

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            payload = {
                "id": chunk.chunk_id,
                "source": chunk.document,
                "chunk_index": chunk.chunk_index,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "text": chunk.text,
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    all_chunks: List[Chunk] = []

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"⚠️  No PDFs found under {input_dir}", file=sys.stderr)
    else:
        print(f"🗂️  Found {len(pdf_paths)} PDF(s) under {input_dir}")

    for pdf_path in pdf_paths:
        print(f"📄 Processing {pdf_path.name}...")
        pages = extract_pages(pdf_path)
        print(f"   • Pages extracted: {len(pages)}")
        text_path = dump_text(args.text_dir, pdf_path, pages)
        print(f"   • Raw text saved to {text_path}")
        paragraphs = split_paragraphs(pages)
        print(f"   • Paragraphs detected: {len(paragraphs)}")
        print("   • Building chunk set...")
        progress_cb = None
        if args.verbose:
            def progress_cb(chunk: Chunk) -> None:
                preview = chunk.text[:120].replace("\n", " ")
                if len(chunk.text) > 120:
                    preview += "…"
                print(
                    f"     ↳ Chunk {chunk.chunk_id} "
                    f"[{chunk.page_start}-{chunk.page_end}, {len(chunk.text)} chars]"
                    f": {preview}"
                )

        chunks = build_chunks(
            paragraphs,
            doc_name=pdf_path.name,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            progress_cb=progress_cb,
        )
        all_chunks.extend(chunks)
        print(f"   • Chunks generated: {len(chunks)} (running total {len(all_chunks)})")
        if args.verbose and chunks:
            sample = chunks[0]
            preview = sample.text[:160].replace("\n", " ")
            if len(sample.text) > 160:
                preview += "…"
            print(
                f"     ↳ Sample chunk {sample.chunk_id} "
                f"[pages {sample.page_start}-{sample.page_end}, "
                f"{len(sample.text)} chars]: {preview}"
            )

    write_chunks(args.chunks_path, all_chunks)
    print(f"✅ Wrote {len(all_chunks)} chunks to {args.chunks_path}")


if __name__ == "__main__":
    main()
