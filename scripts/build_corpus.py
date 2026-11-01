"""Build the clean corpus from the Service-Public export.

    python scripts/build_corpus.py --zip data/raw/vosdroits-latest.zip --out data/processed

Writes:
  sheets.jsonl     one practical sheet per line: metadata + blocks (text with its heading path)
  contacts.json    who-can-help entries (phone lines, national and local offices)
  manifest.json    source file, its date, counts (the licence asks to cite the file and its date)
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree

from fr_admin_assistant.parse import parse_folder

SOURCE_URL = "https://www.data.gouv.fr/fr/datasets/service-public-fr-guide-vos-droits-et-demarches-particuliers/"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    a = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(a.zip).extractall(tmp)
        folder = Path(tmp)
        sheets, contacts = parse_folder(folder)
        export_date = ""
        accueil = folder / "centresDeContact.xml"
        if accueil.exists():
            export_date = etree.parse(str(accueil)).getroot().findtext("{http://purl.org/dc/elements/1.1/}date") or ""

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "sheets.jsonl", "w", encoding="utf-8") as f:
        for s in sheets:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    (a.out / "contacts.json").write_text(
        json.dumps({k: vars(v) for k, v in sorted(contacts.items())}, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest = {
        "source": "Service-Public.gouv.fr / DILA, « Fiches pratiques et ressources pour les particuliers »",
        "licence": "Licence Ouverte v2.0",
        "dataset_url": SOURCE_URL,
        "file": a.zip.name,
        "export_date": export_date,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sheets": len(sheets),
        "blocks": sum(len(s.blocks) for s in sheets),
        "contacts": len(contacts),
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
