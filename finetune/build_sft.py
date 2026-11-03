"""Keep the teacher's answers that pass the checks and write the training files (no GPU needed).

    python finetune/build_sft.py

Checks per kind of example:
  answer    behaviour "answer"; cites the passage the question was made from (or another one of its sheet);
            every cited passage of that sheet has the same situation as the question; no number left
            unsupported; answer in the language of the question
  clarify   behaviour "clarify", or an answer that gives at least two situations of the sheet, each cited
  fallback  behaviour "fallback" (the sheet that answers was removed from the passages)
  special   behaviour "offtopic" or "refuse"
  all       no stray characters from another script (the 8B model sometimes slips Cyrillic or Chinese into Arabic)
Writes finetune/data/train.jsonl and val.jsonl (chat format: the prompt messages + the expected JSON answer),
split by sheet so that no sheet is in both, and finetune/data/stats.json.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from fr_admin_assistant.pipeline import Assistant

OUT = Path("finetune/data")
LANG_OF_STYLE = {"fr": "fr", "fr_oral": "fr", "fr_sms": "fr", "en": "en", "ar": "ar"}


def situation_of(path: list[str]) -> str | None:
    for part in path:
        if part.startswith("Situation : "):
            return part[len("Situation : "):]
    return None


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


_ODD_SCRIPT = re.compile(r"[\u0400-\u04ff\u3040-\u30ff\u4e00-\u9fff]")  # Cyrillic, Japanese, Chinese


def check(chunks, q: dict, t: dict) -> str:
    """'' if the example is kept, else the reason it is dropped."""
    if t.get("skipped"):
        return "skipped before the teacher"
    if _ODD_SCRIPT.search(q["question"]) or _ODD_SCRIPT.search(t["answer"]):
        return "stray characters from another script"
    if q["kind"] != "special" and t["lang"] != LANG_OF_STYLE[q["style"]]:
        return "wrong language"
    if t["checks"].get("still_unsupported") or (t["checks"].get("unsupported_numbers") and not t["checks"].get("regenerated")):
        return "unsupported number"
    ids = [i for i, _ in q["passages"]]
    cited = [ids[n - 1] for n in t["citations"] if 1 <= n <= len(ids)]
    kind, b = q["kind"], t["behaviour"]
    if kind == "special":
        return "" if b in ("offtopic", "refuse") else f"special -> {b}"
    if kind == "fallback":
        return "" if b == "fallback" else f"fallback -> {b}"
    same_sheet = [i for i in cited if chunks[i].sheet == q["sheet"]]
    if kind == "clarify":
        if b == "clarify":
            return ""
        sits = {situation_of(chunks[i].path) for i in same_sheet} - {None}
        return "" if b == "answer" and len(sits) >= 2 else f"clarify -> {b}, {len(sits)} situations cited"
    if b != "answer":
        return f"answer -> {b}"
    if not same_sheet:
        return "does not cite the sheet that answers"
    wrong = [i for i in same_sheet if situation_of(chunks[i].path) not in (None, q["situation"])]
    return "cites another situation" if wrong else ""


def main() -> None:
    assistant = Assistant()
    chunks = assistant.chunks
    questions = {r["id"]: r for r in load(OUT / "searched.jsonl")}
    kept, reasons = [], Counter()
    for t in load(OUT / "teacher.jsonl"):
        q = questions.get(t["id"])
        if not q:
            continue
        why = check(chunks, q, t)
        reasons[f"{q['kind']}: " + (why or "kept")] += 1
        if why:
            continue
        target = json.dumps({"behaviour": t["behaviour"], "answer": t["answer"], "citations": t["citations"]}, ensure_ascii=False)
        kept.append({"id": t["id"], "sheet": q["sheet"], "kind": q["kind"], "style": q["style"],
                     "messages": t["messages"] + [{"role": "assistant", "content": target}]})
    # split by sheet (10% of sheets for validation); the special requests are spread over both
    rng = random.Random(0)
    sheets = sorted({k["sheet"] for k in kept if k["sheet"]})
    rng.shuffle(sheets)
    val_sheets = set(sheets[: max(1, len(sheets) // 10)])
    train = [k for k in kept if k["sheet"] not in val_sheets and not (k["kind"] == "special" and rng.random() < 0.1)]
    val = [k for k in kept if k not in train]
    for name, rows in (("train", train), ("val", val)):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"kept": len(kept), "train": len(train), "val": len(val),
             "by_kind": Counter(k["kind"] for k in kept), "by_style": Counter(k["style"] for k in kept),
             "by_behaviour": Counter(json.loads(k["messages"][-1]["content"])["behaviour"] for k in kept),
             "reasons": dict(sorted(reasons.items()))}
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
