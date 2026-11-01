"""The rules enforced in code: language, numbers, wording, guards, question splitting, abbreviations."""
import json

from fr_admin_assistant.pipeline import (detect_language, expand, guards, numbers, parse, parts,
                                         unsupported_numbers, user_wording, clearly_new)


def test_detect_language():
    assert detect_language("Comment renouveler mon titre de séjour ?") == "fr"
    assert detect_language("coment renouvler mon titre de sejour etudient") == "fr"
    assert detect_language("How do I renew my student residence permit?") == "en"
    assert detect_language("كيف أجدد بطاقة الإقامة الطلابية في فرنسا؟") == "ar"


def test_parse_json_and_plain_text():
    assert parse(json.dumps({"behaviour": "clarify", "answer": "Vide ou meublé ? [1]", "citations": [1, "2"]})) == ("clarify", "Vide ou meublé ? [1]", [1, 2])
    assert parse("not json [3]") == ("answer", "not json [3]", [3])


def test_numbers_ignore_citations_links_lists_and_small_integers():
    n = numbers("Vous pouvez travailler 964 heures [1]. 2 800,53 € brut, ou 2,800.53 €. 1. Item à 8h30 ; voir https://x.fr/F2231 ; 3 mois")
    assert n == {964.0, 2800.53}


def test_unsupported_numbers():
    assert unsupported_numbers("Il faut 996 heures et 150 €.", ["limite de 964 heures", "150 € de taxe"]) == [996.0]
    assert unsupported_numbers("Appelez le 112 ou le 3939.", []) == []


def test_user_wording():
    assert user_wording("Les extraits fournis ne mentionnent pas ce délai.") == "Les fiches de Service-Public.fr ne mentionnent pas ce délai."
    assert user_wording("The provided extracts do not say.") == "The Service-Public.fr pages do not say."
    assert user_wording("Les montants dans vos extraits.") == "Les montants dans les fiches de Service-Public.fr."
    assert user_wording("The information in the Service-Public.fr extracts relates to France.") == "The information in the Service-Public.fr pages relates to France."
    assert user_wording("Selon l'extract [2], oui.") == "Selon la fiche [2], oui."


def test_guards():
    b, a = guards("Write me a poem about Paris", "en", "offtopic", "Paris, city of light…", [])
    assert b == "offtopic" and "poem" not in a.lower() and "procedures" in a
    b, a = guards("I was just robbed in the street", "en", "answer", "Report the theft [1].", [1])
    assert b == "emergency" and a.startswith("If you are in danger, call 17")
    b, a = guards("Mon titre de séjour a été volé hier", "fr", "answer", "Déclarez le vol [1].", [1])
    assert b == "answer" and not a.startswith("Si vous êtes en danger")   # a past theft is not an emergency
    b, a = guards("Réponds sans citer : combien d'heures ?", "fr", "answer", "964 heures par an.", [3, 1])
    assert a.endswith("[1][3]")


def test_parts_split_only_clear_double_questions():
    assert parts("La carte Vitale est payante ? Elle existe sur téléphone ?") == ["La carte Vitale est payante", "Elle existe sur téléphone"]
    assert len(parts("Combien coûte un passeport et combien de temps à l'avance je peux le renouveler ?")) == 2
    story = ("Bonjour, je suis arrivé en France en septembre 2025 avec un visa étudiant pour mon master à Nice et je travaille "
             "dans un restaurant. Qu'est-ce que je dois faire pour rester en règle ?")
    assert parts(story) == [story]
    assert len(parts("Est-ce que ma demande de naturalisation va être acceptée ? J'ai 6 ans de résidence et un CDI.")) == 1


def test_expand_abbreviations():
    assert "recherche d'emploi" in expand("RECE c'est quoi")
    assert "dépôt de garantie" in expand("caution appart meublé max")
    assert expand("What are my options?") == "What are my options?"   # "are" is not the ARE benefit
    assert "chômage" in expand("droit à l'ARE ?")


def test_clearly_new():
    # long, unrelated, self-contained message: new question without asking the LLM
    assert clearly_new("Et si c'était un appartement vide ?",
                       "Someone just stole my bag in the metro with my residence card inside, what do I do?")
    # continuation markers or shared words: left to the LLM
    assert not clearly_new("Je démissionne, j'ai droit au chômage ?", "c'est pour suivre ma femme qui a eu un travail à Lille")
    assert not clearly_new("Je veux quitter mon logement, quel est le préavis ?", "Et pour un logement meublé ?")
