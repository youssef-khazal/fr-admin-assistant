"""Compute embeddings (meaning vectors) for the chunks and the test questions, on the GPU.

    python scripts/embed.py                       # default models and strategies
    python scripts/embed.py --models bge-m3 --strategies section

Writes, for each chunking strategy and model:
  data/index/<strategy>/chunks.jsonl         the chunks (same order as the vectors)
  data/index/<strategy>/<model>.npy          one normalized vector per chunk (float16)
and for the questions:
  data/index/questions/ids.json, <model>.npy

Searching is then a matrix product, done on any machine without a GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from fr_admin_assistant.chunking import build_chunks, load_sheets, save_chunks

MODELS = {
    # name: (Hugging Face id, query prefix, passage prefix)
    "bge-m3": ("BAAI/bge-m3", "", ""),
    "e5-large": ("intfloat/multilingual-e5-large", "query: ", "passage: "),
}
DEFAULT_PLAN = {"bge-m3": ["section", "bare", "chapter", "fixed"], "e5-large": ["section"]}


def question_text(q: dict) -> str:
    """The text used to search for a question: for a conversation, all the user's turns."""
    return " ".join(q["turns"]) if "turns" in q else q["q"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sheets", type=Path, default=Path("data/processed/sheets.jsonl"))
    p.add_argument("--questions", type=Path, default=Path("data/eval/questions.yaml"))
    p.add_argument("--out", type=Path, default=Path("data/index"))
    p.add_argument("--models", nargs="*", default=list(DEFAULT_PLAN))
    p.add_argument("--strategies", nargs="*", default=None, help="default: the plan in DEFAULT_PLAN")
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " - this will be slow"))
    sheets = load_sheets(a.sheets)
    questions = yaml.safe_load(open(a.questions, encoding="utf-8"))
    qdir = a.out / "questions"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "ids.json").write_text(json.dumps([q["id"] for q in questions]), encoding="utf-8")

    for name in a.models:
        hf_id, q_prefix, p_prefix = MODELS[name]
        model = SentenceTransformer(hf_id, device=device)
        model.max_seq_length = 512
        if device == "cuda":
            model.half()
        qv = model.encode([q_prefix + question_text(q) for q in questions], batch_size=a.batch_size,
                          normalize_embeddings=True, show_progress_bar=False)
        np.save(qdir / f"{name}.npy", qv.astype(np.float16))

        for strategy in a.strategies or DEFAULT_PLAN[name]:
            chunks = build_chunks(sheets, strategy)
            sdir = a.out / strategy
            save_chunks(chunks, sdir / "chunks.jsonl")
            t = time.time()
            # longest texts first: batches of similar length waste less padding
            order = sorted(range(len(chunks)), key=lambda i: -len(chunks[i].text))
            vecs = model.encode([p_prefix + chunks[i].text for i in order], batch_size=a.batch_size,
                                normalize_embeddings=True, show_progress_bar=True)
            out = np.empty_like(vecs)
            out[order] = vecs
            np.save(sdir / f"{name}.npy", out.astype(np.float16))
            print(f"{name} / {strategy}: {len(chunks)} chunks in {time.time() - t:.0f} s")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
