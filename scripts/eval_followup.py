"""Test the follow-up detector: continuation of the previous message, or a new question?

    python scripts/eval_followup.py          (Ollama must be running)

Prints the accuracy, the errors, and writes results/followup.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from fr_admin_assistant.pipeline import is_followup


def main() -> None:
    cases = yaml.safe_load(Path("data/eval/followups.yaml").read_text(encoding="utf-8"))
    out, ok = [], 0
    for c in cases:
        got = is_followup([c["prev"], c["last"]])
        ok += got == c["suite"]
        out.append({**c, "got": got})
        if got != c["suite"]:
            print(f"ERROR expected {c['suite']}, got {got}: {c['prev']!r} -> {c['last']!r}")
    print(f"follow-up detection: {ok}/{len(cases)} correct")
    Path("results/followup.json").write_text(json.dumps({"accuracy": ok / len(cases), "cases": out}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")


if __name__ == "__main__":
    main()
