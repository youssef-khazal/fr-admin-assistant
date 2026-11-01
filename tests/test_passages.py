"""Choosing passages: the other situation of the same question is added only when the user did not say theirs."""
from fr_admin_assistant.chunking import Chunk
from fr_admin_assistant.pipeline import Assistant
from fr_admin_assistant.retrieval import Hit

Q = "Quel est le délai de préavis en cas de départ du locataire du logement {} ?"


def assistant():
    a = Assistant.__new__(Assistant)
    a.top_passages = 3
    a.chunks = [Chunk("F1#0", "F1", ["Situation : Logement vide", Q.format("vide")], "vide 3 mois"),
                Chunk("F1#1", "F1", ["Situation : Logement meublé", Q.format("meublé")], "meublé 1 mois"),
                Chunk("F2#0", "F2", [], "autre fiche")]
    a.sheets = {"F1": {"title": "Préavis", "url": "u1", "modified": "m", "situations": ["Logement vide", "Logement meublé"]},
                "F2": {"title": "Autre", "url": "u2", "modified": "m", "situations": []}}
    a._by_sheet = {"F1": [0, 1], "F2": [2]}
    return a


def test_sibling_added_when_situation_unknown():
    ps = assistant().passages([Hit(0, 1.0), Hit(2, 0.5)], query="je veux quitter mon appartement, quel préavis ?")
    assert [p.chunk.text for p in ps] == ["vide 3 mois", "autre fiche", "meublé 1 mois"]


def test_no_sibling_when_user_names_the_situation():
    ps = assistant().passages([Hit(0, 1.0), Hit(2, 0.5)], query="je quitte mon logement vide, quel préavis ?")
    assert [p.chunk.text for p in ps] == ["vide 3 mois", "autre fiche"]
