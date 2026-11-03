"""Scores of several graded runs on the same questions, with 95% bootstrap intervals and paired differences.

    python scripts/compare_runs.py v7=results/manual_grades_v7.jsonl ft4b=results/manual_grades_ft4b.jsonl \
        --out results/finetune_scores.json

Each run is NAME=grades file: hand grades (manual_grades_*.jsonl) or LLM-judge verdicts (judgements2_*.jsonl),
graded the same way as in scripts/score_answers.py. The intervals resample the questions (2,000 resamples);
the differences are paired (the same resampled questions for both runs), each run against the first one.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from score_answers import grade, load_jsonl


def rates(grades: dict[str, dict], ids: list[str]) -> tuple[float, float, float]:
    """(right behaviour, facts found, answers with invented content) on the questions ids."""
    expected = sum(grades[i]["expected"] for i in ids)
    return (sum(grades[i]["behaviour_ok"] for i in ids) / len(ids),
            sum(grades[i]["found"] for i in ids) / expected,
            sum(grades[i]["invented"] for i in ids) / len(ids))


def interval(values: list[float]) -> list[float]:
    v = sorted(values)
    return [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v)) - 1], 4)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="NAME=path/to/grades.jsonl (the first one is the reference)")
    p.add_argument("--questions", type=Path, default=Path("data/eval/questions.yaml"))
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    questions = {q["id"]: q for q in yaml.safe_load(a.questions.read_text(encoding="utf-8"))}
    runs = {}
    for spec in a.runs:
        name, path = spec.split("=", 1)
        runs[name] = {i: grade(j, questions[i].get("facts") or []) for i, j in load_jsonl(Path(path)).items()
                      if i in questions and "behaviour_ok" in j}
    ids = sorted(set.intersection(*(set(g) for g in runs.values())))
    rng = random.Random(0)
    samples = [[rng.choice(ids) for _ in ids] for _ in range(a.resamples)]
    keys = ("behaviour", "facts", "invented")

    report, ref = {"n": len(ids)}, list(runs)[0]
    boot = {n: [rates(g, s) for s in samples] for n, g in runs.items()}
    for n, g in runs.items():
        point = rates(g, ids)
        report[n] = {k: round(point[m], 4) for m, k in enumerate(keys)}
        report[n]["ci95"] = {k: interval([b[m] for b in boot[n]]) for m, k in enumerate(keys)}
        print(f"{n:12} " + "  ".join(f"{k} {100 * point[m]:.1f}% {[round(100 * x, 1) for x in report[n]['ci95'][k]]}"
                                     for m, k in enumerate(keys)))
    for n in list(runs)[1:]:
        point = [x - y for x, y in zip(rates(runs[n], ids), rates(runs[ref], ids))]
        diff = {k: interval([b[m] - r[m] for b, r in zip(boot[n], boot[ref])]) for m, k in enumerate(keys)}
        report[f"{n}-{ref}"] = {**{k: round(point[m], 4) for m, k in enumerate(keys)}, "ci95": diff}
        print(f"{n} - {ref}: " + "  ".join(f"{k} {100 * point[m]:+.1f} {[round(100 * x, 1) for x in diff[k]]}"
                                         for m, k in enumerate(keys)))
    if a.out:
        a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
