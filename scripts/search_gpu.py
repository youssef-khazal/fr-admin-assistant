"""Search experiments that need the GPU and the local LLM (run on the Windows machine).

    python scripts/search_gpu.py

For every test question:
  1. the LLM (Ollama, qwen3:8b) rewrites it into a clean French search query
  2. dense search (bge-m3) and keyword search (BM25) are run with the original and the rewritten text
  3. a cross-encoder re-ranker (bge-reranker-v2-m3) re-reads the top candidates together with the
     question and re-orders them; it is slower than embeddings but much more precise

Writes:
  results/rewrites.json            the rewritten queries (cached: delete to recompute)
  results/runs_gpu.json            ranked sheets per setup and question, scored by eval_retrieval.py
  results/candidates_gpu.json      the top passages of each setup (used later to generate answers)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from fr_admin_assistant.chunking import load_chunks
from fr_admin_assistant.retrieval import BM25Searcher, DenseSearcher, Hit, rrf, to_sheets
from fr_admin_assistant.llm import unload
from fr_admin_assistant.rewrite import rewrite


def turns_of(q: dict) -> list[str]:
    return q["turns"] if "turns" in q else [q["q"]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, default=Path("data/index"))
    p.add_argument("--questions", type=Path, default=Path("data/eval/questions.yaml"))
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--candidates", type=int, default=30, help="passages per list sent to the re-ranker")
    a = p.parse_args()

    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    questions = yaml.safe_load(open(a.questions, encoding="utf-8"))
    a.results.mkdir(parents=True, exist_ok=True)

    # 1. rewriting (cached)
    rw_path = a.results / "rewrites.json"
    rewrites = json.loads(rw_path.read_text(encoding="utf-8")) if rw_path.exists() else {}
    t = time.time()
    for i, q in enumerate(questions):
        if q["id"] not in rewrites:
            rewrites[q["id"]] = rewrite(turns_of(q))
            print(f"[{i + 1}/{len(questions)}] {q['id']}: {rewrites[q['id']]}")
            rw_path.write_text(json.dumps(rewrites, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rewriting done in {time.time() - t:.0f} s")
    unload()  # free the GPU for the embedding model and the re-ranker

    originals = {q["id"]: " ".join(turns_of(q)) for q in questions}
    queries = {"orig": originals, "rewritten": {q["id"]: rewrites[q["id"]] for q in questions}}

    # 2. first-stage search
    chunks = load_chunks(a.index / "section" / "chunks.jsonl")
    dense = DenseSearcher(np.load(a.index / "section" / "bge-m3.npy"))
    embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
    embedder.half()
    print("building BM25 index...")
    bm25 = BM25Searcher(chunks)
    first: dict[str, dict[str, list[Hit]]] = {}
    for qname, texts in queries.items():
        ids = list(texts)
        vecs = embedder.encode([texts[i] for i in ids], normalize_embeddings=True, batch_size=32)
        d = {qid: dense.search_vector(v, 100) for qid, v in zip(ids, vecs)}
        b = {qid: bm25.search(texts[qid], 100) for qid in ids}
        first[f"dense / {qname}"] = d
        first[f"hybrid / {qname}"] = {qid: rrf(b[qid], d[qid]) for qid in ids}
    del embedder
    torch.cuda.empty_cache()

    # 3. re-ranking
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512)
    reranker.model.half()
    reranked: dict[str, dict[str, list[Hit]]] = {}
    plans = [("rerank dense / orig", "dense / orig", "orig"),
             ("rerank hybrid / orig", "hybrid / orig", "orig"),
             ("rerank hybrid / rewritten", "hybrid / rewritten", "rewritten"),
             ("rerank dense+hybrid / both", None, None)]
    t = time.time()
    for name, source, qname in plans:
        out = {}
        for q in questions:
            qid = q["id"]
            if source:
                cands = [h.chunk for h in first[source][qid][:a.candidates]]
                text = queries[qname][qid]
            else:  # union of the original and rewritten hybrid lists, scored with the rewritten query
                cands = list(dict.fromkeys([h.chunk for h in first["hybrid / orig"][qid][:a.candidates]] +
                                           [h.chunk for h in first["hybrid / rewritten"][qid][:a.candidates]]))
                text = queries["rewritten"][qid]
            scores = reranker.predict([(text, chunks[c].text) for c in cands], batch_size=32, show_progress_bar=False)
            out[qid] = [Hit(c, float(s)) for c, s in sorted(zip(cands, scores), key=lambda x: -x[1])]
        reranked[name] = out
        print(f"{name}: done")
    print(f"re-ranking done in {time.time() - t:.0f} s")

    runs, candidates = {}, {}
    for name, per_q in {**first, **reranked}.items():
        runs[name] = {qid: [s for s, _ in to_sheets(hits, chunks)][:10] for qid, hits in per_q.items()}
        candidates[name] = {qid: [[chunks[h.chunk].id, round(h.score, 4)] for h in hits[:15]] for qid, hits in per_q.items()}
    (a.results / "runs_gpu.json").write_text(json.dumps(runs, ensure_ascii=False), encoding="utf-8")
    (a.results / "candidates_gpu.json").write_text(json.dumps(candidates, ensure_ascii=False), encoding="utf-8")
    print("wrote results/runs_gpu.json and results/candidates_gpu.json")


if __name__ == "__main__":
    main()
