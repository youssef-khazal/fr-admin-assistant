"""Rewrite a user's question into a clean French search query before searching.

Users write with typos, in SMS style, in English or Arabic, or as a follow-up ("and if it's
refused?"). The official sheets are in standard French, so the LLM first turns the question (and the
conversation so far) into one self-contained French query. Whether this helps is measured.
"""

from __future__ import annotations

from fr_admin_assistant.llm import chat

PROMPT = """Tu prépares une recherche dans les fiches pratiques officielles de Service-Public.fr (droits et démarches des particuliers en France).

Réécris le dernier message de l'utilisateur en UNE requête de recherche en français standard :
- corrige les fautes et le style SMS, développe les abréviations quand tu es sûr (APL, RECE, OQTF...) ;
- traduis en français si le message est dans une autre langue ;
- si c'est une suite de conversation, intègre le contexte des messages précédents pour que la requête soit complète ;
- garde les détails utiles de la situation (nationalité, statut, type de logement, dates...) ;
- n'ajoute aucune information qui n'est pas dans la conversation.

Réponds uniquement par la requête, sans guillemets ni explication."""


def rewrite(turns: list[str], model: str | None = None) -> str:
    convo = "\n".join(f"Message {i + 1} : {t}" for i, t in enumerate(turns))
    kwargs = {"model": model} if model else {}
    out = chat([{"role": "system", "content": PROMPT}, {"role": "user", "content": convo}], max_tokens=120, **kwargs)
    return out.strip().strip('"').splitlines()[0] if out.strip() else turns[-1]
