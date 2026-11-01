"""The web app with a fake assistant (no models): normal answer, live stream, feedback, guards in the response."""
import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import fr_admin_assistant.app as app_mod  # noqa: E402
from fr_admin_assistant.chunking import Chunk  # noqa: E402
from fr_admin_assistant.pipeline import Passage, Result  # noqa: E402

URL = "https://www.service-public.gouv.fr/particuliers/vosdroits/F1168"


class FakeAssistant:
    sheets = {"F1168": {"situations": ["Logement vide", "Logement meublé"]}}

    def is_followup(self, turns):
        return len(turns[-1].split()) < 5

    def retrieve(self, turns, followup=False):
        mk = lambda n, sit: Passage(n, Chunk(f"F1168#{n}", "F1168", [f"Situation : {sit}", "Quel est le délai de préavis ?"],
                                             f"Préavis\nSituation : {sit} > Quel est le délai de préavis ?\nDélai pour {sit}."),
                                    "Préavis et formalités", URL, "2026-04-14", 0.9)
        return [mk(1, "Logement vide"), mk(2, "Logement meublé")]

    def ask(self, turns, history, passages=None, followup=False, think=None, on_event=None):
        if on_event:
            on_event("thinking", "x" * 50)
            on_event("answer", "Meublé : 1 mois")
        r = Result("answer", "Meublé : 1 mois [2] ; vide : 3 mois [1].", [1, 2], passages, [], "",
                   [{"n": 1, "title": "Préavis et formalités", "url": URL, "modified": "2026-04-14"}], "fr")
        r.followup = followup
        return r


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "_assistant", FakeAssistant())
    monkeypatch.setattr(app_mod, "UI_LOG", str(tmp_path / "log.jsonl"))
    monkeypatch.setenv("UI_FEEDBACK", str(tmp_path / "fb.jsonl"))
    return TestClient(app_mod.app), tmp_path


def test_ask_returns_sources_with_excerpts(client):
    c, tmp = client
    r = c.post("/api/ask", json={"turns": ["je veux quitter mon appartement, quel préavis ?"]}).json()
    assert r["behaviour"] == "answer" and r["refs"] == {"1": 1, "2": 1}
    src = r["sources"][0]
    assert len(src["paths"]) == 2 and src["excerpts"][1]["text"] == "Délai pour Logement meublé."
    assert (tmp / "log.jsonl").exists()


def test_stream_events_end_with_done(client):
    c, _ = client
    lines = [json.loads(x) for x in c.post("/api/ask/stream", json={"turns": ["quel préavis pour quitter mon logement ?"]}).text.splitlines()]
    kinds = [e["type"] for e in lines]
    assert kinds[0] == "stage" and "found" in kinds and "thinking" in kinds and "answer" in kinds and kinds[-1] == "done"


def test_feedback_is_saved(client):
    c, tmp = client
    assert c.post("/api/feedback", json={"turns": ["q"], "answer": "a", "vote": "down", "reason": "wrong"}).json() == {"ok": True}
    assert json.loads((tmp / "fb.jsonl").read_text(encoding="utf-8"))["reason"] == "wrong"


def test_empty_question_is_rejected(client):
    c, _ = client
    assert c.post("/api/ask", json={"turns": ["   "]}).status_code == 400


def test_suggestions_offer_the_other_situation(client):
    c, _ = client
    r = c.post("/api/ask", json={"turns": ["je pars de mon studio meublé, quel préavis je dois donner ?"]}).json()
    assert "Et pour : Logement vide ?" in r["suggestions"]
    assert not any("meublé" in s for s in r["suggestions"])
    r = c.post("/api/ask", json={"turns": ["je veux quitter mon appartement, quel préavis ?"]}).json()
    assert "Et si : Logement vide ?" in r["suggestions"] and "Et si : Logement meublé ?" in r["suggestions"]
