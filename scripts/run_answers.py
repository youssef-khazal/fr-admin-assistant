"""Answer every test question with the full assistant, then grade the answers (Windows, GPU + Ollama).

    python scripts/run_answers.py                  # search, answer, judge
    python scripts/run_answers.py --skip-judge     # only produce the answers

Phases (each one frees the GPU for the next):
  1. search for all questions (bge-m3 + BM25 + re-ranker on the GPU)
  2. answers from the LLM (Ollama)
  3. grading by the LLM judge (Ollama, with reasoning on)
Writes results/answers.jsonl and results/judgements.jsonl (both resumable: finished questions are skipped).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from fr_admin_assistant.judge import judge
from fr_admin_assistant.llm import unload
from fr_admin_assistant.pipeline import Assistant


def load_done(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {d["id"]: d for d in (json.loads(line) for line in f if line.strip())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", type=Path, default=Path("data/eval/questions.yaml"))
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--think", action="store_true", help="let the LLM reason before answering (slower)")
    p.add_argument("--tag", default="", help="suffix for the output files, to compare setups")
    p.add_argument("--judge-only", action="store_true", help="grade existing answers<tag>.jsonl")
    p.add_argument("--judge-out", default=None, help="judgements file name (default judgements<tag>.jsonl)")
    a = p.parse_args()

    questions = yaml.safe_load(open(a.questions, encoding="utf-8"))
    a.results.mkdir(parents=True, exist_ok=True)
    ans_path = a.results / f"answers{a.tag}.jsonl"
    jud_path = a.results / (a.judge_out or f"judgements{a.tag}.jsonl")
    answers = load_done(ans_path)
    turns = {q["id"]: q.get("turns") or [q["q"]] for q in questions}
    if not a.judge_only:
        produce_answers(a, questions, answers, ans_path, turns)
    if a.skip_judge:
        return
    run_judge(questions, answers, jud_path, turns)


def produce_answers(a, questions, answers, ans_path, turns) -> None:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    # 1. search
    todo = [q for q in questions if q["id"] not in answers]
    embedder = SentenceTransformer("BAAI/bge-m3", device="cuda").half()
    reranker = None
    if not a.no_rerank:
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512)
        reranker.model.half()
    assistant = Assistant(embedder=embedder, reranker=reranker, think=a.think)
    t = time.time()
    retrieved = {}
    for q in todo:
        tq = turns[q["id"]]
        # for a conversation, the earlier turns are answered too (their answers become the history)
        # test conversations are follow-ups by design (the follow-up detector is tested separately)
        retrieved[q["id"]] = [assistant.retrieve(tq[:i + 1], followup=i > 0) for i in range(len(tq))]
    print(f"search: {len(todo)} questions in {time.time() - t:.0f} s")
    del embedder, reranker
    assistant.embedder = assistant.reranker = None
    torch.cuda.empty_cache()

    # 2. answers
    t = time.time()
    with open(ans_path, "a", encoding="utf-8") as f:
        for i, q in enumerate(todo):
            tq, history = turns[q["id"]], []
            try:
                for k in range(len(tq)):
                    r = assistant.ask(tq[:k + 1], history, passages=retrieved[q["id"]][k], followup=k > 0)
                    history.append({"role": "assistant", "content": r.answer})
            except Exception as e:  # one failing question must not stop a night run; it is retried on the next run
                print(f"[{i + 1}/{len(todo)}] {q['id']} FAILED, skipped: {type(e).__name__}: {e}")
                continue
            rec = {"id": q["id"], "behaviour": r.behaviour, "answer": r.answer, "citations": r.citations,
                   "sources": r.sources, "contacts": r.contacts, "raw": r.raw, "checks": r.checks,
                   "history": history[:-1],
                   "passages": [{"n": p.n, "chunk": p.chunk.id, "sheet": p.chunk.sheet, "title": p.title,
                                 "path": p.chunk.path, "url": p.url, "modified": p.modified,
                                 "score": round(p.score, 4), "text": p.chunk.text} for p in r.passages]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            answers[q["id"]] = rec
            print(f"[{i + 1}/{len(todo)}] {q['id']} {r.behaviour}: {r.answer[:110]!r}")
    print(f"answers: {time.time() - t:.0f} s")


def run_judge(questions, answers, jud_path, turns) -> None:
    # 3. judge
    judged = load_done(jud_path)
    t = time.time()
    with open(jud_path, "a", encoding="utf-8") as f:
        for i, q in enumerate(questions):
            if q["id"] in judged or q["id"] not in answers:
                continue
            rec = answers[q["id"]]
            convo = "\n".join(f"Utilisateur : {u}" + (f"\nAssistant : {h['content']}" if j < len(rec["history"]) else "")
                              for j, (u, h) in enumerate(zip(turns[q["id"]], rec["history"] + [{}])))
            # passages cited in the answer first (in full), then the others; contacts in full: a detail taken
            # from a contact (opening hours, phone number) must not look invented to the judge
            cited = {int(m) for m in re.findall(r"\[(\d+)\]", rec["answer"])} | set(rec.get("citations") or [])
            order = sorted(rec["passages"], key=lambda p: (p["n"] not in cited, p["n"]))
            passages = "\n\n".join(f"[{p['n']}] {p['title']} ({p['modified']})\n{p['text']}" for p in order)
            contacts = "\n".join(f"- {c['title']}: {c['details']}" for c in rec["contacts"])
            passages = "CONTACTS :\n" + contacts + "\n\nEXTRAITS :\n" + passages
            try:
                g = judge(convo, q["behaviour"], q["facts"], passages, rec["answer"])
            except Exception as e:
                print(f"[judge {i + 1}/{len(questions)}] {q['id']} FAILED, skipped: {type(e).__name__}: {e}")
                continue
            g["id"] = q["id"]
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[judge {i + 1}/{len(questions)}] {q['id']} behaviour_ok={g.get('behaviour_ok')} facts={g.get('facts')}")
    unload()
    print(f"judging: {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
