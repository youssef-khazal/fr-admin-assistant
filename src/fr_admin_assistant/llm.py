"""Minimal client for a local LLM served by Ollama (http://localhost:11434)."""

from __future__ import annotations

import json
import os
from typing import Callable

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")


def chat(messages: list[dict], model: str = DEFAULT_MODEL, temperature: float = 0.0,
         max_tokens: int = 1024, json_mode: bool = False, timeout: int = 900, think: bool = False, num_ctx: int = 8192) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,  # Qwen3 can "think" before answering; off by default for speed
        "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": num_ctx},
    }
    if json_mode:
        payload["format"] = "json"
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def chat_stream(messages: list[dict], on_chunk: Callable[[str, str], None], model: str = DEFAULT_MODEL,
                temperature: float = 0.0, max_tokens: int = 1024, json_mode: bool = False, timeout: int = 900,
                think: bool = False, num_ctx: int = 8192) -> str:
    """Same as chat(), but calls on_chunk(kind, text) as the model writes; kind is "thinking" or "content".
    Returns the full content."""
    payload = {"model": model, "messages": messages, "stream": True, "think": think,
               "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": num_ctx}}
    if json_mode:
        payload["format"] = "json"
    out = []
    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            msg = json.loads(line).get("message", {})
            if msg.get("thinking"):
                on_chunk("thinking", msg["thinking"])
            if msg.get("content"):
                out.append(msg["content"])
                on_chunk("content", msg["content"])
    return "".join(out).strip()


def unload(model: str = DEFAULT_MODEL) -> None:
    """Free the GPU memory held by Ollama (it keeps a model loaded for a few minutes after use)."""
    try:
        requests.post(f"{OLLAMA_URL}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    except requests.RequestException:
        pass
