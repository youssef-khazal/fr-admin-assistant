"""Grade an answer against the test set: an LLM judge checks each criterion separately.

For every answer the judge receives the question, the expected behaviour, the facts a correct
answer must contain, the passages the assistant was given, and the answer. It returns:
  behaviour_ok   did the assistant behave as expected (answer / ask / fallback / ...)?
  facts          for each expected fact: present and correct in the answer?
  unsupported    statements in the answer that the passages do not support (invented content)
  wrong          statements that contradict the expected facts
Asking for each fact separately makes the judge's job simple and checkable. A sample of the
judge's grades is re-checked against a reference grading to measure how far it can be trusted.
"""

from __future__ import annotations

import json

from fr_admin_assistant.llm import chat

BEHAVIOURS = {
    "answer": "répondre à la question à partir des extraits, avec des citations",
    "clarify": "ne pas choisir un cas au hasard : soit poser une question courte sur la situation, soit donner brièvement les différents cas avec leur intitulé",
    "fallback": "dire clairement que les extraits ne permettent pas de répondre (ou pas complètement), sans inventer, et orienter vers des pages ou des contacts",
    "offtopic": "dire poliment que l'assistant sert aux démarches administratives, sans répondre au fond",
    "emergency": "donner d'abord des numéros d'urgence (112, 15, 17, 18 ou 114) avant tout le reste",
    "refuse": "refuser poliment de suivre la demande (inventer, frauder, ignorer les règles)",
}

PROMPT = """Tu évalues la réponse d'un assistant administratif. Sois strict et factuel.

QUESTION (conversation) :
{question}

COMPORTEMENT ATTENDU : {behaviour} — {behaviour_desc}

FAITS ATTENDUS (une bonne réponse doit les contenir, sauf si le comportement attendu est de poser une question) :
{facts}

EXTRAITS ET CONTACTS DONNÉS À L'ASSISTANT (les extraits cités par la réponse sont en premier) :
{passages}

RÉPONSE DE L'ASSISTANT :
{answer}

Réponds en JSON avec exactement ces clés :
{{"behaviour_ok": true/false,
  "facts": [true/false pour chaque fait attendu, dans l'ordre : le fait est-il présent et correct dans la réponse ?],
  "unsupported": [liste des affirmations de la réponse qui ne sont PAS soutenues par les extraits ni par les contacts (montants, délais, conditions, numéros inventés, ou une règle appliquée à une situation à laquelle elle ne s'applique pas) ; une information présente dans les extraits ou les contacts n'est PAS à lister, même si elle n'est pas dans les faits attendus ; un fait attendu ABSENT de la réponse n'est PAS à lister ici (il est déjà compté dans "facts") ; les numéros d'urgence 112, 15, 17, 18, 114 sont toujours admis ; liste vide si tout est soutenu],
  "wrong": [liste des affirmations PRÉSENTES dans la réponse qui contredisent les faits attendus ; un fait simplement absent n'est pas une contradiction ; liste vide sinon],
  "comment": "une phrase d'explication"}}"""


def judge(question: str, behaviour: str, facts: list[str], passages: str, answer: str,
          model: str | None = None) -> dict:
    prompt = PROMPT.format(
        question=question, behaviour=behaviour, behaviour_desc=BEHAVIOURS.get(behaviour, ""),
        facts="\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts)) or "(aucun)",
        passages=passages[:20000], answer=answer)
    kwargs = {"model": model} if model else {}
    d = None
    for think, budget in ((True, 8000), (False, 1500)):  # retry without reasoning if the reasoning ran out of tokens
        raw = chat([{"role": "user", "content": prompt}], json_mode=True, think=think, max_tokens=budget,
                   num_ctx=16384 if think else 12288, **kwargs)
        try:
            d = json.loads(raw)
            break
        except json.JSONDecodeError:
            continue
    if d is None:
        return {"error": raw[:500]}
    f = [bool(x) for x in d.get("facts", [])][:len(facts)]
    f += [False] * (len(facts) - len(f))
    return {"behaviour_ok": bool(d.get("behaviour_ok")), "facts": f,
            "unsupported": list(d.get("unsupported", []) or []), "wrong": list(d.get("wrong", []) or []),
            "comment": str(d.get("comment", ""))}
