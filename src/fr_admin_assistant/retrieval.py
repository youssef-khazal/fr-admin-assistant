"""Search: find the passages (and so the official sheets) that answer a question.

Three searchers are compared:
  BM25     keyword search: scores passages by the question's words, rarer words count more
  Dense    meaning search: compares embedding vectors (the question and the passage are close in
           meaning even with different words, typos or another language)
  Hybrid   both lists merged by Reciprocal Rank Fusion: a passage ranked high by either one rises

Results are grouped by sheet, because the answer must cite an official page.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from fr_admin_assistant.chunking import Chunk

STOPWORDS = set("""
a au aux avec ce ces cet cette d dans de des du elle en et eux il ils je j l la le les leur leurs lui ma mais me
mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos
votre vous c s y est sont ai as a avons avez ont etre avoir fait faire comment quel quelle quels quelles quoi
est-ce peut peux dois doit the is are do i my to of in for and what how can
""".split())


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalize(text)) if t not in STOPWORDS and len(t) > 1]


@dataclass
class Hit:
    chunk: int     # index in the chunk list
    score: float


class BM25Searcher:
    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 100) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        return [Hit(int(i), float(scores[i])) for i in sorted(top, key=lambda i: -scores[i])]


class DenseSearcher:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors.astype(np.float32)

    def search_vector(self, qvec: np.ndarray, k: int = 100) -> list[Hit]:
        scores = self.vectors @ qvec.astype(np.float32)
        top = np.argpartition(-scores, k)[:k]
        return [Hit(int(i), float(scores[i])) for i in sorted(top, key=lambda i: -scores[i])]


def rrf(*rankings: list[Hit], k: int = 60, weights: list[float] | None = None) -> list[Hit]:
    """Reciprocal Rank Fusion: score = sum over lists of weight / (k + rank)."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        for rank, h in enumerate(ranking):
            fused[h.chunk] += w / (k + rank + 1)
    return [Hit(i, s) for i, s in sorted(fused.items(), key=lambda x: -x[1])]


def to_sheets(hits: list[Hit], chunks: list[Chunk]) -> list[tuple[str, float]]:
    """Rank sheets by their best passage."""
    best: dict[str, float] = {}
    for h in hits:
        sid = chunks[h.chunk].sheet
        if sid not in best:
            best[sid] = h.score
    return list(best.items())
