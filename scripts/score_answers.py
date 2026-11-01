"""Score judged answers: right behaviour, expected facts found, answers with invented or wrong content.

    python scripts/score_answers.py v1=results/judgements_v1.jsonl v2=results/judgements_v2.jsonl \
        v4=results/judgements_v4.jsonl --manual results/manual_grades_v4.jsonl --out results/scores.json

Each run is NAME=judgements file (written by run_answers.py). Numbers are given on all the questions
judged in each run, and on the questions judged in every run ("common"), which is the fair comparison
when one run is incomplete.

Correction for the LLM judge: it sometimes lists an expected fact that the answer *left out* under
"unsupported" or "wrong". A missing fact is already counted in fact recall, so such an item is not
counted as invented content when it overlaps strongly with one of the question's expected facts.
Both the raw and the corrected rates are reported.

--manual: grades written by hand in the same format (behaviour_ok, facts, invented_or_wrong) for
part of a run, to measure how far the LLM judge can be trusted.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml

QUESTIONS = Path("data/eval/questions.yaml")
STOP = set("le la les un une des de du d l et ou a au aux en dans pour par sur avec sans ne pas plus est sont "
           "vous votre vos il elle ce cette que qui the of to and or in for is are you your be on".split())


def words(text: str) -> set[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return {w for w in re.findall(r"[a-z0-9]+", t) if w not in STOP and len(w) > 1}


def is_missing_fact(item: str, facts: list[str], threshold: float = 0.5) -> bool:
    """True if a flagged item is mostly the text of an expected fact (a fact left out, not an invention)."""
    w = words(item)
    if not w:
        return False
    return any(len(w & words(f)) / min(len(w), len(words(f)) or 1) >= threshold for f in facts)


def load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def grade(j: dict, facts: list[str]) -> dict:
    """One judgement -> behaviour ok, facts found / expected, invented (raw and corrected)."""
    if "invented_or_wrong" in j:  # hand grading
        flagged = j["invented_or_wrong"]
        raw = corrected = bool(flagged)
    else:
        flagged = list(j.get("unsupported") or []) + list(j.get("wrong") or [])
        raw = bool(flagged)
        corrected = any(not is_missing_fact(str(x), facts) for x in flagged)
    found = [bool(x) for x in (j.get("facts") or [])][:len(facts)]
    found += [False] * (len(facts) - len(found))
    return {"behaviour_ok": bool(j.get("behaviour_ok")), "found": sum(found), "expected": len(facts),
            "facts": found, "invented_raw": raw, "invented": corrected}


def summary(grades: list[dict]) -> dict:
    n = len(grades)
    if not n:
        return {"n": 0}
    exp = sum(g["expected"] for g in grades)
    return {"n": n,
            "behaviour": round(sum(g["behaviour_ok"] for g in grades) / n, 3),
            "fact_recall": round(sum(g["found"] for g in grades) / exp, 3) if exp else None,
            "invented_raw": round(sum(g["invented_raw"] for g in grades) / n, 3),
            "invented": round(sum(g["invented"] for g in grades) / n, 3)}


def breakdown(grades: dict[str, dict], questions: dict[str, dict], key: str) -> dict:
    groups = defaultdict(list)
    for qid, g in grades.items():
        groups[questions[qid].get(key, "?")].append(g)
    return {k: summary(v) for k, v in sorted(groups.items())}


def agreement(judge: dict[str, dict], manual: dict[str, dict]) -> dict:
    ids = sorted(set(judge) & set(manual))
    if not ids:
        return {"n": 0}
    beh = sum(judge[i]["behaviour_ok"] == manual[i]["behaviour_ok"] for i in ids)
    pairs = [(a, b) for i in ids for a, b in zip(judge[i]["facts"], manual[i]["facts"])]
    inv_raw = sum(judge[i]["invented_raw"] == manual[i]["invented"] for i in ids)
    inv = sum(judge[i]["invented"] == manual[i]["invented"] for i in ids)
    return {"n": len(ids), "behaviour": round(beh / len(ids), 3),
            "facts": round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None,
            "invented_raw": round(inv_raw / len(ids), 3), "invented": round(inv / len(ids), 3),
            "behaviour_disagree": [i for i in ids if judge[i]["behaviour_ok"] != manual[i]["behaviour_ok"]],
            "invented_disagree": [i for i in ids if judge[i]["invented"] != manual[i]["invented"]]}


def table(rows: dict[str, dict], title: str) -> str:
    out = [f"| {title} | n | right behaviour | facts found | invented (judge raw) | invented (corrected) |",
           "|---|---|---|---|---|---|"]
    pct = lambda x: "–" if x is None else f"{100 * x:.0f}%"
    for name, s in rows.items():
        if s.get("n"):
            out.append(f"| {name} | {s['n']} | {pct(s['behaviour'])} | {pct(s['fact_recall'])} | "
                       f"{pct(s['invented_raw'])} | {pct(s['invented'])} |")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="NAME=path/to/judgements.jsonl")
    p.add_argument("--questions", type=Path, default=QUESTIONS)
    p.add_argument("--manual", type=Path, help="hand grades for part of the last run")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    qs = yaml.safe_load(a.questions.read_text(encoding="utf-8"))
    qs = qs["questions"] if isinstance(qs, dict) else qs
    questions = {q["id"]: q for q in qs}
    runs = {}
    for spec in a.runs:
        name, path = spec.split("=", 1)
        runs[name] = {i: grade(j, questions[i].get("facts") or []) for i, j in load_jsonl(Path(path)).items()
                      if i in questions and "behaviour_ok" in j}
    common = set.intersection(*(set(g) for g in runs.values()))

    report = {"all": {n: summary(list(g.values())) for n, g in runs.items()},
              "common": {n: summary([g[i] for i in common]) for n, g in runs.items()},
              "by_category": {n: breakdown(g, questions, "category") for n, g in runs.items()},
              "by_language": {n: breakdown(g, questions, "lang") for n, g in runs.items()}}
    print(table(report["all"], "run (all judged)"), end="\n\n")
    if len(runs) > 1:
        print(table(report["common"], f"run ({len(common)} common questions)"), end="\n\n")
    for n in runs:
        print(table(report["by_language"][n], f"{n} by language"), end="\n\n")
    if a.manual:
        last = list(runs)[-1]
        manual = {i: grade(j, questions[i].get("facts") or []) for i, j in load_jsonl(a.manual).items()}
        report["manual"] = summary(list(manual.values()))
        report["judge_vs_manual"] = agreement(runs[last], manual)
        print(table({f"{last} by hand": report["manual"],
                     f"{last} by judge, same questions": summary([runs[last][i] for i in manual if i in runs[last]])},
                    "hand vs judge"), end="\n\n")
        print("agreement judge/hand:", json.dumps(report["judge_vs_manual"]))
    if a.out:
        a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
