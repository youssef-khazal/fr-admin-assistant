"""Split the sheets into passages ("chunks") that the search engine indexes.

How a sheet is cut decides what the search engine can find, so several strategies are
compared on the test set:

  section   one chunk per deepest section (Situation > Chapitre > Cas), with the sheet title and
            the heading path written at the top; long sections are split into overlapping windows
  bare      the same chunks without the title and heading path (shows what the context adds)
  chapter   one chunk per top-level chapter (sections merged), with the heading path
  fixed     fixed windows of ~250 words over the whole sheet, ignoring its structure (naive baseline)

Every chunk keeps the id of its sheet, so a retrieved chunk always points back to an official page.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

MAX_WORDS = 350
OVERLAP = 50
STRATEGIES = ("section", "bare", "chapter", "fixed")


@dataclass
class Chunk:
    id: str          # "<sheet id>#<n>"
    sheet: str
    path: list[str]  # heading path inside the sheet (without the sheet title)
    text: str        # what is embedded and shown to the model


def load_sheets(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def windows(words: list[str], size: int = MAX_WORDS, overlap: int = OVERLAP) -> Iterator[list[str]]:
    if len(words) <= size:
        yield words
        return
    step = size - overlap
    for start in range(0, len(words), step):
        yield words[start:start + size]
        if start + size >= len(words):
            break


def header(sheet: dict, path: list[str]) -> str:
    lines = [sheet["title"]]
    if path:
        lines.append(" > ".join(path))
    return "\n".join(lines)


def _chunks_from_groups(sheet: dict, groups: list[tuple[list[str], str]], with_header: bool) -> list[Chunk]:
    out = []
    for path, text in groups:
        words = text.split()
        if not words:
            continue
        for part in windows(words):
            body = " ".join(part) if len(words) > MAX_WORDS else text
            full = f"{header(sheet, path)}\n{body}" if with_header else body
            out.append(Chunk(f"{sheet['id']}#{len(out)}", sheet["id"], path, full))
    return out


def chunk_sheet(sheet: dict, strategy: str = "section") -> list[Chunk]:
    blocks = [(b["path"][1:], b["text"]) for b in sheet["blocks"]]
    if not blocks:  # e.g. "Recherche guidée" pages: keep at least the title and description
        blocks = [([], sheet.get("description", ""))]
    # the first block of a sheet is its introduction: give it the description too
    if blocks and not blocks[0][0] and sheet.get("description"):
        blocks[0] = ([], f"{sheet['description']}\n{blocks[0][1]}".strip())

    if strategy in ("section", "bare"):
        return _chunks_from_groups(sheet, blocks, with_header=(strategy == "section"))
    if strategy == "chapter":
        merged: dict[tuple, list[str]] = {}
        for path, text in blocks:
            key = tuple(path[:2]) if path and path[0].startswith("Situation") else tuple(path[:1])
            merged.setdefault(key, []).append(text)
        return _chunks_from_groups(sheet, [(list(k), "\n".join(v)) for k, v in merged.items()], with_header=True)
    if strategy == "fixed":
        words = " ".join(t for _, t in blocks).split()
        out = []
        for part in windows(words, 250, 50):
            out.append(Chunk(f"{sheet['id']}#{len(out)}", sheet["id"], [], " ".join(part)))
        return out
    raise ValueError(f"unknown strategy {strategy!r}")


def build_chunks(sheets: list[dict], strategy: str) -> list[Chunk]:
    return [c for s in sheets for c in chunk_sheet(s, strategy)]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]
