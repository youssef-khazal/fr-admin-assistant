"""Compare the search of v4 and v5 on the test questions and on the reworded questions (GPU).

    python scripts/eval_search.py

v4 search: the question as typed. v5 search: abbreviations expanded (RECE -> carte de séjour recherche
d'emploi...) and multi-part questions searched part by part. For every question with an expected sheet:
is it in the top 1 / top 5 sheets? Also measured on data/eval/paraphrases.yaml (each rewording keeps the
expected sheets of its base question), split by writing style. Writes results/search_v5.json.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import yaml

from fr_admin_assistant import pipeline
from fr_admin_assistant.pipeline import Assistant

STYLES = ["formal", "spoken", "indirect", "keywords", "misspelled"]


def sheets_ranked(a: Assistant, query: str) -> list[str]:
    out = []
    for h in a.search(query)[:30]:
        s = a.chunks[h.chunk].sheet
        if s not in out:
            out.append(s)
    return out


def score(rows: list[dict]) -> dict:
    n = len(rows)
    return {"n": n, "hit@1": round(sum(r["rank"] == 1 for r in rows) / n, 3) if n else None,
            "hit@5": round(sum(0 < r["rank"] <= 5 for r in rows) / n, 3) if n else None}


def run(a: Assistant, items: list[dict]) -> list[dict]:
    rows = []
    for it in items:
        ranked = sheets_ranked(a, it["query"])
        rank = next((i + 1 for i, s in enumerate(ranked) if s in it["gold"]), 0)
        rows.append({**it, "rank": rank})
    return rows


def main() -> None:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("BAAI/bge-m3", device=device)
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=512)
    if device == "cuda":
        embedder.half()
        reranker.model.half()
    a = Assistant(embedder=embedder, reranker=reranker)

    qs = yaml.safe_load(Path("data/eval/questions.yaml").read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in qs}
    items = []
    for q in qs:
        if q.get("gold"):
            turns = q.get("turns") or [q["q"]]
            items.append({"id": q["id"], "set": "dev" if int(q["id"][1:]) % 3 == 0 else "test", "category": q["category"],
                          "query": " ".join(turns[-2:]), "gold": q["gold"]})
    para = []
    for p in yaml.safe_load(Path("data/eval/paraphrases.yaml").read_text(encoding="utf-8")):
        for style, v in zip(STYLES, p["variants"]):
            para.append({"id": p["base"], "style": style, "query": v, "gold": by_id[p["base"]]["gold"]})

    expand, parts = pipeline.expand, pipeline.parts
    report = {}
    for name, (e, s) in {"v4": (lambda q: q, lambda q: [q]), "v5": (expand, parts)}.items():
        pipeline.expand, pipeline.parts = e, s
        t = time.time()
        rows, prow = run(a, items), run(a, para)
        report[name] = {
            "questions": {k: score([r for r in rows if k == "all" or r["set"] == k]) for k in ("all", "dev", "test")},
            "rewordings": {"all": score(prow), **{st: score([r for r in prow if r["style"] == st]) for st in STYLES}},
            "by_category": {c: score([r for r in rows if r["category"] == c]) for c in sorted({r["category"] for r in rows})},
            "misses_hit5": [r["id"] for r in rows if not 0 < r["rank"] <= 5],
            "rewording_misses_hit5": [f"{r['id']}/{r['style']}" for r in prow if not 0 < r["rank"] <= 5],
        }
        print(f"{name}: questions {report[name]['questions']}  rewordings {report[name]['rewordings']['all']}  ({time.time() - t:.0f} s)")
        for st in STYLES:
            print(f"   {st:10} {report[name]['rewordings'][st]}")
    pipeline.expand, pipeline.parts = expand, parts
    Path("results/search_v5.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
