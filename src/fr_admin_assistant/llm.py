"""Minimal client for a local LLM: served by Ollama (http://localhost:11434), or, with LLM_BACKEND=hf, a
Hugging Face model loaded in this process (used to evaluate the fine-tuned model: HF_MODEL + HF_ADAPTER)."""

from __future__ import annotations

import json
import os
from typing import Callable

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
BACKEND = os.environ.get("LLM_BACKEND", "ollama")
_hf: dict = {}


def _hf_load():
    """Load HF_MODEL (default Qwen/Qwen3-4B) in 4-bit, with the LoRA adapter HF_ADAPTER if set."""
    if not _hf:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers.integrations import sdpa_attention
        # Qwen3 has 32 query heads for 8 key/value heads; handed over as they are, the Windows build of PyTorch falls
        # back to the plain attention kernel (a 4500 x 4500 table per head: GPU memory overflows into the much
        # slower PC memory). Copying the key/value heads up to 32 first lets the memory-saving kernel run.
        sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False
        name = os.environ.get("HF_MODEL", "Qwen/Qwen3-4B")
        tok = AutoTokenizer.from_pretrained(name)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)  # same as in training
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant, device_map="cuda", dtype=torch.bfloat16)
        if os.environ.get("HF_ADAPTER"):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, os.environ["HF_ADAPTER"])
        model.eval()
        _hf.update(tok=tok, model=model)
    return _hf["tok"], _hf["model"]


def _hf_chat(messages: list[dict], max_tokens: int = 1024, think: bool = False) -> str:
    import time
    import torch
    tok, model = _hf_load()
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    stop = [t for t in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id) if t is not None]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=False, use_cache=True,
                             eos_token_id=stop, pad_token_id=tok.eos_token_id)
    new = out[0, ids["input_ids"].shape[1]:]
    if os.environ.get("HF_TIMING"):
        print(f"    [hf] prompt {ids['input_ids'].shape[1]} tokens, wrote {len(new)} tokens in {time.time() - t0:.0f} s")
    text = tok.decode(new, skip_special_tokens=True)
    if "</think>" in text:  # keep only the answer after the reasoning
        text = text.split("</think>", 1)[1]
    return text.strip()


def chat(messages: list[dict], model: str = DEFAULT_MODEL, temperature: float = 0.0,
         max_tokens: int = 1024, json_mode: bool = False, timeout: int = 900, think: bool = False, num_ctx: int = 8192) -> str:
    if BACKEND == "hf":
        return _hf_chat(messages, max_tokens=max_tokens, think=think)
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
    if BACKEND == "hf":  # no live streaming: the whole answer at once
        text = _hf_chat(messages, max_tokens=max_tokens, think=think)
        on_chunk("content", text)
        return text
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
    if BACKEND == "hf":
        return
    try:
        requests.post(f"{OLLAMA_URL}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    except requests.RequestException:
        pass
