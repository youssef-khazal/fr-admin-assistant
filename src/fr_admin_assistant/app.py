"""Web app: a chat page in front of the assistant.

    uvicorn fr_admin_assistant.app:app --port 8000      (Ollama must be running)

Endpoints:
  GET  /            the chat page
  GET  /health      status, model, data date
  POST /api/ask     {"turns": ["question 1", "question 2"...], "history": ["answer 1"...]}
                    -> behaviour, answer, sources (cited official pages), contacts, choices, timing
Settings (environment variables): LLM_MODEL (default qwen3:8b), THINK (1 = reasoning mode, slower and
more careful), DEVICE (cuda / cpu for the search models), DATA_DIR (default data).
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from fr_admin_assistant.pipeline import Assistant, _plain

STATIC = Path(__file__).parent / "static"
DATA = Path(os.environ.get("DATA_DIR", "data"))
MAX_TURNS, MAX_CHARS = 6, 1500
# local log of real questions and what the assistant saw (to debug and to build new test cases);
# stays on this computer, set UI_LOG= (empty) to turn it off
UI_LOG = os.environ.get("UI_LOG", "results/ui_log.jsonl")

app = FastAPI(title="Démarches FR", description="Questions on French administrative procedures, answered from Service-Public.fr")
_assistant: Assistant | None = None


def assistant() -> Assistant:
    global _assistant
    if _assistant is None:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        device = os.environ.get("DEVICE") or ("cuda" if _cuda() else "cpu")
        embedder = SentenceTransformer("BAAI/bge-m3", device=device)
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=512)
        if device == "cuda":
            embedder.half()
            reranker.model.half()
        _assistant = Assistant(data=DATA, embedder=embedder, reranker=reranker,
                               think=os.environ.get("THINK", "0") == "1")
    return _assistant


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class Ask(BaseModel):
    turns: list[str] = Field(..., min_length=1, description="the user's messages, the last one is answered")
    history: list[str] = Field(default_factory=list, description="the assistant's previous answers")
    think: bool | None = Field(None, description="careful mode (reasoning, slower); default: the THINK setting")


class Feedback(BaseModel):
    turns: list[str]
    answer: str
    vote: str = Field(..., pattern="^(up|down)$")
    reason: str = ""  # for a 👎: wrong / incomplete / unclear / other
    comment: str = ""


def choices_from(a: Assistant, passages, cited: list[int]) -> list[str]:
    """Official situation labels ("Logement vide", "Logement meublé"...) offered as buttons when the assistant asks.
    They are the top-level situations of the sheet most used in the answer; if it has none, the cases
    ("Cas : ...") of the passages from that sheet."""
    from collections import Counter

    weight = Counter(p.chunk.sheet for p in passages)
    for n in cited:
        if 1 <= n <= len(passages):
            weight[passages[n - 1].chunk.sheet] += 2
    main_sheet = weight.most_common(1)[0][0]
    out = list(a.sheets[main_sheet].get("situations", []))
    if not out:
        for p in passages:
            if p.chunk.sheet == main_sheet:
                for part in p.chunk.path:
                    if part.startswith("Cas : ") and part[6:] not in out:
                        out.append(part[6:])
                        break
    return [c for c in out if len(c) <= 80][:5]


def suggestions(a: Assistant, r, passages, turns: list[str], limit: int = 3) -> list[str]:
    """Next questions to offer under an answer: the other situations of the main sheet when the user named none
    (shown as "Et si : Logement meublé ?"), then official sheets found by the search but not used in the answer."""
    if r.behaviour in ("offtopic", "refuse", "emergency") or not passages:
        return []
    out = []
    cited_urls = {s["url"] for s in r.sources}
    main = next((p.chunk.sheet for p in passages if p.url in cited_urls), passages[0].chunk.sheet)
    situations = a.sheets.get(main, {}).get("situations", [])
    if r.behaviour != "clarify" and len(situations) > 1:
        # a situation is "named" when one of its distinctive words is in the conversation ("studio meublé" names
        # "Logement meublé"; "logement" is shared by all the situations of the sheet, so it does not count)
        words = {x: {w for w in re.findall(r"\w+", _plain(x)) if len(w) > 3} for x in situations}
        common = set.intersection(*words.values()) if words else set()
        asked = set(re.findall(r"\w+", _plain(" ".join(turns[-2:]))))
        named = [x for x in situations if (words[x] - common) & asked]
        if not named:
            lead = {"en": "What if: {}?", "ar": "ماذا لو: {}؟"}.get(r.lang, "Et si : {} ?")
            out += [lead.format(x) for x in situations[:2] if len(x) <= 60]
        else:
            lead = {"en": "And for: {}?", "ar": "وماذا عن: {}؟"}.get(r.lang, "Et pour : {} ?")
            out += [lead.format(x) for x in situations if x not in named and len(x) <= 60][:1]
    for p in passages:
        if p.url not in cited_urls and p.title not in out and len(p.title) <= 90:
            out.append(p.title)
        if len(out) >= limit:
            break
    return out[:limit]


def short(details: str, lines: int = 3, chars: int = 220) -> str:
    out = "\n".join(details.splitlines()[:lines])
    return out if len(out) <= chars else out[:chars].rsplit(" ", 1)[0] + "…"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    manifest = json.loads((DATA / "processed" / "manifest.json").read_text(encoding="utf-8"))
    return {"status": "ok", "model": os.environ.get("LLM_MODEL", "qwen3:8b"), "think": os.environ.get("THINK", "0") == "1",
            "data_export_date": manifest.get("export_date"), "sheets": manifest.get("sheets")}


def _prepare(req: Ask) -> tuple[list[str], list[dict]]:
    turns = [t.strip()[:MAX_CHARS] for t in req.turns][-MAX_TURNS:]
    if not turns[-1]:
        raise HTTPException(400, "Empty question")
    history = [{"role": "assistant", "content": h} for h in req.history][-(len(turns) - 1):] if len(turns) > 1 else []
    return turns, history


def _append(path: str, rec: dict) -> None:
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _response(a: Assistant, r, passages, turns: list[str], timings: dict) -> dict:
    # every cited passage number -> number of its official sheet in the source list (several passages
    # often come from the same sheet; the page shows one footnote per sheet, numbered 1, 2, 3...)
    by_url = {src["url"]: k + 1 for k, src in enumerate(r.sources)}
    refs = {str(n): by_url[passages[n - 1].url] for n in range(1, len(passages) + 1) if passages[n - 1].url in by_url}
    cited = {int(n) for n in refs} & ({int(m) for m in re.findall(r"\[(\d+)\]", r.answer)} | set(r.citations))
    sources = []
    for src in r.sources:  # each sheet with the sections actually cited from it
        paths = []
        for n in sorted(cited):
            if passages[n - 1].url == src["url"]:
                path = " > ".join(passages[n - 1].chunk.path)
                if path and path not in paths:
                    paths.append(path)
        # the exact official text the answer used ("voir les extraits")
        excerpts = []
        for n in sorted(cited):
            p = passages[n - 1]
            if p.url == src["url"]:
                body = p.chunk.text.split("\n", 2)[-1] if p.chunk.path else p.chunk.text.split("\n", 1)[-1]
                excerpts.append({"path": " > ".join(p.chunk.path), "text": body[:1500]})
        sources.append({**src, "path": paths[0] if paths else "", "paths": paths[:3], "excerpts": excerpts[:4]})
    _append(UI_LOG, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "turns": turns, "followup": r.followup,
                     "behaviour": r.behaviour, "answer": r.answer, "raw": r.raw, "checks": r.checks, "timing_ms": timings,
                     "passages": [{"n": p.n, "sheet": p.chunk.sheet, "path": p.chunk.path, "score": round(p.score, 3),
                                   "text": p.chunk.text} for p in passages]})
    return {
        "behaviour": r.behaviour,
        "refs": refs,
        "suggestions": suggestions(a, r, passages, turns),
        "answer": r.answer,
        "lang": r.lang,
        "sources": sources,
        "contacts": [{**c, "details": short(c["details"])} for c in r.contacts] if r.behaviour == "fallback" else [],
        "choices": choices_from(a, passages, [s["n"] for s in r.sources]) if r.behaviour == "clarify" else [],
        "followup": r.followup,
        "checks": r.checks,
        "timing_ms": timings,
    }


@app.post("/api/ask")
def ask(req: Ask) -> dict:
    turns, history = _prepare(req)
    t0 = time.perf_counter()
    a = assistant()
    # a new, unrelated question is answered on its own; a follow-up ("et pour un meublé ?") with the previous message
    followup = a.is_followup(turns) if len(turns) > 1 else False
    passages = a.retrieve(turns, followup=followup)
    t1 = time.perf_counter()
    r = a.ask(turns, history, passages=passages, followup=followup, think=req.think)
    t2 = time.perf_counter()
    return _response(a, r, passages, turns, {"search": round((t1 - t0) * 1000), "answer": round((t2 - t1) * 1000)})


@app.post("/api/ask/stream")
def ask_stream(req: Ask) -> StreamingResponse:
    """The same answer, with live progress (one JSON object per line):
    {"type": "stage", "stage": "search" | "writing" | "checking"}, {"type": "found", "sheets": [...]},
    {"type": "thinking", "chars": n}, {"type": "answer", "text": "..."} (the answer as it is written),
    {"type": "done", ...same fields as /api/ask...} or {"type": "error", "detail": "..."}."""
    turns, history = _prepare(req)
    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            t0 = time.perf_counter()
            events.put({"type": "stage", "stage": "search"})
            a = assistant()
            followup = a.is_followup(turns) if len(turns) > 1 else False
            passages = a.retrieve(turns, followup=followup)
            t1 = time.perf_counter()
            events.put({"type": "found", "followup": followup, "passages": len(passages),
                        "sheets": list(dict.fromkeys(p.title for p in passages))[:4]})
            events.put({"type": "stage", "stage": "writing"})
            state = {"thinking": 0, "last": 0.0}

            def on_event(kind: str, data) -> None:
                now = time.perf_counter()
                if kind == "thinking":
                    state["thinking"] += len(data)
                    if now - state["last"] > 0.4:
                        state["last"] = now
                        events.put({"type": "thinking", "chars": state["thinking"]})
                elif kind == "answer":
                    events.put({"type": "answer", "text": data})
                elif kind == "regenerating":
                    events.put({"type": "stage", "stage": "checking", "numbers": data})

            r = a.ask(turns, history, passages=passages, followup=followup, think=req.think, on_event=on_event)
            t2 = time.perf_counter()
            events.put({"type": "done", **_response(a, r, passages, turns,
                                                    {"search": round((t1 - t0) * 1000), "answer": round((t2 - t1) * 1000)})})
        except Exception as e:  # shown in the page instead of a silent failure
            events.put({"type": "error", "detail": f"{type(e).__name__}: {e}"})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (ev := events.get()) is not None:
            yield json.dumps(ev, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/feedback")
def feedback(fb: Feedback) -> dict:
    """👍 / 👎 on an answer, saved on this computer (results/ui_feedback.jsonl) to become new test cases."""
    _append(os.environ.get("UI_FEEDBACK", "results/ui_feedback.jsonl"),
            {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **fb.model_dump()})
    return {"ok": True}
