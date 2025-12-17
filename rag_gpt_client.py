#!/usr/bin/env python3
"""Generic helper for sending RAG-augmented prompts (with optional images) to OpenAI."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

from rag_context import get_rag_context

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
CONFIG_DIR = Path("config")
PROMPT_DIR = CONFIG_DIR / "prompts"
MASTER_PROMPT_PATH = PROMPT_DIR / "master_prompt.txt"
PERSONAL_INFO_PATH = CONFIG_DIR / "personal_info.txt"


def _encode_image_to_data_url(image_path: Optional[Path]) -> Optional[str]:
    if image_path is None:
        return None
    path = image_path.expanduser()
    if not path.exists():
        return None
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _extract_response_text(response) -> str:
    if hasattr(response, "output_text"):
        text = "".join(response.output_text)
        if text:
            return text
    output = getattr(response, "output", None)
    if output:
        chunks = []
        for item in output:
            for content in getattr(item, "content", []):
                if getattr(content, "type", "") == "output_text":
                    chunks.append(content.text)
        if chunks:
            return " ".join(chunks)
    return ""


def _load_text(path: Path, *, required: bool = True) -> str:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text and required:
        raise ValueError(f"{path} is empty.")
    return text


def _build_master_prompt(question: str, context: str, notes: str) -> str:
    template = _load_text(MASTER_PROMPT_PATH)
    personal_info = _load_text(PERSONAL_INFO_PATH)
    return (
        template.replace("{personal info}", personal_info)
        .replace("{notes}", notes)
        .replace("{context}", context if context else "")
        .replace("{question}", question)
    )




def call_rag_assistant(
    question_text: str,
    *,
    image_path: Optional[Path] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    top_k: int = 1,
    notes_text: Optional[str] = None,
) -> Optional[str]:
    """
    Send a text (plus optional image) prompt to OpenAI after augmenting it with RAG context.

    Returns the assistant response text, or ``None`` if the call fails or is skipped.
    """

    question = question_text.strip()
    if not question:
        raise ValueError("question_text cannot be empty.")

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("⚠️  OPENAI_API_KEY not set; skipping assistant call.")
        return None
    client = OpenAI(api_key=key)

    try:
        context, _ = get_rag_context(question, client=client, api_key=key, top_k=top_k)
        context = context.strip()
    except Exception as exc:
        print(f"⚠️  Failed to retrieve RAG context: {exc}")
        context = ""

    try:
        final_prompt = _build_master_prompt(question, context, notes_text or "")
    except Exception as exc:
        print(f"⚠️  Failed to build master prompt: {exc}")
        return None

    print("=== RAG-enhanced prompt ===")
    print(final_prompt)
    print("=== End prompt ===")

    content_parts = [
        {
            "type": "input_text",
            "text": final_prompt,
        }
    ]
    encoded_image = _encode_image_to_data_url(image_path)
    if encoded_image:
        content_parts.append({"type": "input_image", "image_url": encoded_image})

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": system_prompt,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": content_parts,
                },
            ],
        )
    except Exception as exc:
        print(f"⚠️  OpenAI request failed: {exc}")
        return None

    reply_text = _extract_response_text(response).strip()
    if not reply_text:
        print("⚠️  Received empty response from OpenAI.")
        return None
    return reply_text
