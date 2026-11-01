"""Parsing an official sheet and cutting it into passages."""
from pathlib import Path

from fr_admin_assistant.chunking import chunk_sheet, windows
from fr_admin_assistant.parse import parse_sheet

DATA = Path(__file__).parent / "data"


def sheet():
    return parse_sheet(DATA / "F9999.xml", {}, {})


def test_parse_keeps_situations_and_headings():
    s = sheet()
    assert s.id == "F9999" and s.title == "Préavis du locataire (fiche de test)"
    assert s.modified == "2026-04-14"
    paths = [b.path for b in s.blocks]
    assert ["Préavis du locataire (fiche de test)", "Situation : Logement vide", "Quel est le délai de préavis ?"] in paths
    assert ["Préavis du locataire (fiche de test)", "Situation : Logement meublé", "Quel est le délai de préavis ?"] in paths
    assert "R250" in s.contacts


def test_parse_renders_lists_notes_and_tables():
    text = "\n".join(b.text for b in sheet().blocks)
    assert "- 1 mois en zone tendue" in text
    assert "> À noter : Indiquez le motif dans la lettre." in text
    assert "| Bail | Délai |" in text and "| Meublé | 1 mois |" in text


def test_chunks_carry_the_heading_path():
    chunks = chunk_sheet(sheet().to_dict(), "section")
    meuble = [c for c in chunks if c.path and c.path[0] == "Situation : Logement meublé"]
    assert meuble and meuble[0].text.startswith("Préavis du locataire (fiche de test)\nSituation : Logement meublé > Quel est le délai de préavis ?")
    assert "1 mois" in meuble[0].text
    bare = chunk_sheet(sheet().to_dict(), "bare")
    assert not bare[1].text.startswith("Préavis du locataire")


def test_windows_overlap():
    words = [str(i) for i in range(800)]
    parts = list(windows(words, 350, 50))
    assert parts[0][-50:] == parts[1][:50]
    assert parts[-1][-1] == "799"
