"""Scoring the judge's output: a missing expected fact is not counted as invented content."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from score_answers import grade, is_missing_fact, summary  # noqa: E402

FACTS = ["Déclarer le nouveau domicile dans les 3 mois suivant le déménagement", "Nouvelle carte facultative : 50 €"]


def test_missing_fact_is_not_invention():
    assert is_missing_fact("déclarer le nouveau domicile dans les 3 mois", FACTS)
    assert not is_missing_fact("il faut payer 300 € de taxe", FACTS)


def test_grade_and_summary():
    g1 = grade({"behaviour_ok": True, "facts": [True, False], "unsupported": ["déclarer le nouveau domicile dans les 3 mois"], "wrong": []}, FACTS)
    g2 = grade({"behaviour_ok": False, "facts": [False], "unsupported": ["il faut payer 300 €"], "wrong": []}, FACTS)
    assert g1["invented_raw"] and not g1["invented"] and g2["invented"] and g2["expected"] == 2
    s = summary([g1, g2])
    assert s == {"n": 2, "behaviour": 0.5, "fact_recall": 0.25, "invented_raw": 1.0, "invented": 0.5}
