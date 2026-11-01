"""Compare search setups on the test questions: does the right official sheet come up?

    python scripts/eval_retrieval.py --index data/index --out results/retrieval.json

Metrics, computed on the questions that have a correct sheet ("gold"):
  hit@k   share of questions where a correct sheet is among the first k sheets returned
  mrr     mean reciprocal rank of the first correct sheet (1 = always first, 0.5 = always second...)
Questions are split into dev (every 3rd question, used to tune) and test (the rest, reported).
A 95% confidence interval for hit@5 is estimated by bootstrap (resampling the questions).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from fr_admin_assistant.chunking import load_chunks
from fr_admin_assistant.retrieval import BM25Searcher, DenseSearcher, rrf, to_sheets

KS = (1, 3, 5, 10)


def split_of(qid: str) -> str:
    return "dev" if int(qid[1:]) % 3 == 0 else "test"


def query_text(q: dict) -> str:
    return " ".join(q["turns"]) if "turns" in q else q["q"]


def metrics(ranked: list[list[str]], golds: list[list[str]]) -> dict:
    out = {f"hit@{k}": float(np.mean([any(g in r[:k] for g in gold) for r, gold in zip(ranked, golds)])) for k in KS}
    rr = []
    for r, gold in zip(ranked, golds):
        ranks = [r.index(g) + 1 for g in gold if g in r[:10]]
        rr.append(1 / min(ranks) if ranks else 0.0)
    out["mrr"] = float(np.mean(rr))
    return out


def bootstrap_ci(hits: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [hits[rng.integers(0, len(hits), len(hits))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, default=Path("data/index"))
    p.add_argument("--questions", type=Path, default=Path("data/eval/questions.yaml"))
    p.add_argument("--out", type=Path, default=Path("results/retrieval.json"))
    p.add_argument("--gpu-runs", type=Path, default=None, help="runs_gpu.json from search_gpu.py (scored only)")
    a = p.parse_args()

    all_questions = [q for q in yaml.safe_load(open(a.questions, encoding="utf-8")) if q["gold"]]
    all_ids = json.loads((a.index / "questions" / "ids.json").read_text())
    pos = {qid: i for i, qid in enumerate(all_ids)}
    questions = [q for q in all_questions if q["id"] in pos]  # questions embedded by embed.py
    qvecs = {m: np.load(a.index / "questions" / f"{m}.npy")[[pos[q["id"]] for q in questions]]
             for m in ("bge-m3", "e5-large")}

    runs: dict[str, list[list[str]]] = {}
    for strategy in ("section", "bare", "chapter", "fixed"):
        chunks = load_chunks(a.index / strategy / "chunks.jsonl")
        models = ["bge-m3", "e5-large"] if strategy == "section" else ["bge-m3"]
        dense = {}
        for m in models:
            searcher = DenseSearcher(np.load(a.index / strategy / f"{m}.npy"))
            dense[m] = [searcher.search_vector(v, 200) for v in qvecs[m]]
            runs[f"dense {m} / {strategy}"] = [[s for s, _ in to_sheets(h, chunks)] for h in dense[m]]
        if strategy == "section":
            bm25 = BM25Searcher(chunks)
            lex = [bm25.search(query_text(q), 200) for q in questions]
            runs["bm25 / section"] = [[s for s, _ in to_sheets(h, chunks)] for h in lex]
            for m in models:
                fused = [rrf(l, d) for l, d in zip(lex, dense[m])]
                runs[f"hybrid bm25+{m} / section"] = [[s for s, _ in to_sheets(h, chunks)] for h in fused]
            both = [rrf(d1, d2) for d1, d2 in zip(dense["bge-m3"], dense["e5-large"])]
            runs["dense bge-m3+e5 / section"] = [[s for s, _ in to_sheets(h, chunks)] for h in both]

    results = score(runs, questions)
    if a.gpu_runs:
        gpu = json.loads(a.gpu_runs.read_text(encoding="utf-8"))
        gq = [q for q in all_questions if all(q["id"] in r for r in gpu.values())]
        results_gpu = score({name: [r[q["id"]] for q in gq] for name, r in gpu.items()}, gq)
        results.update({f"[gpu] {k}": v for k, v in results_gpu.items()})
        runs.update({f"[gpu] {name}": [r[q["id"]] for q in gq] for name, r in gpu.items()})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(results)
    report["_per_question"] = {name: {q["id"]: ranked[i][:10] for i, q in enumerate(gq if name.startswith("[gpu]") else questions)}
                               for name, ranked in runs.items()} if a.gpu_runs else \
        {name: {q["id"]: ranked[i][:10] for i, q in enumerate(questions)} for name, ranked in runs.items()}
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    for group, prefix in (("embed.py runs", ""), ("search_gpu.py runs", "[gpu] ")):
        rows = [(n, r) for n, r in results.items() if (n.startswith("[gpu]") == bool(prefix))]
        if not rows:
            continue
        n_q = rows[0][1]["n"]["all"]
        print(f"\n{group} ({n_q} questions)")
        print(f"{'setup':44s} {'hit@1':>6s} {'hit@3':>6s} {'hit@5':>6s} {'hit@10':>6s} {'mrr':>6s}  test hit@5 [95% CI]")
        for name, res in sorted(rows, key=lambda x: -x[1]["all"]["mrr"]):
            m, t = res["all"], res["test"]
            ci = t["hit@5_ci95"]
            print(f"{name:44s} {m['hit@1']:6.3f} {m['hit@3']:6.3f} {m['hit@5']:6.3f} {m['hit@10']:6.3f} {m['mrr']:6.3f}  {t['hit@5']:.3f} [{ci[0]:.2f}-{ci[1]:.2f}]")


def score(runs: dict[str, list[list[str]]], questions: list[dict]) -> dict:
    golds = [q["gold"] for q in questions]
    results = {}
    splits = [split_of(q["id"]) for q in questions]
    cats = [q["category"] for q in questions]
    for name, ranked in runs.items():
        res = {"all": metrics(ranked, golds)}
        for sp in ("dev", "test"):
            idx = [i for i, s in enumerate(splits) if s == sp]
            res[sp] = metrics([ranked[i] for i in idx], [golds[i] for i in idx])
            hits = np.array([any(g in ranked[i][:5] for g in golds[i]) for i in idx], float)
            res[sp]["hit@5_ci95"] = bootstrap_ci(hits)
        by_cat = defaultdict(list)
        for i, c in enumerate(cats):
            by_cat[c].append(i)
        res["by_category_hit@5"] = {c: round(float(np.mean([any(g in ranked[i][:5] for g in golds[i]) for i in ix])), 3)
                                     for c, ix in sorted(by_cat.items())}
        res["n"] = {"all": len(questions), "dev": splits.count("dev"), "test": splits.count("test")}
        results[name] = res

    return results


if __name__ == "__main__":
    main()
