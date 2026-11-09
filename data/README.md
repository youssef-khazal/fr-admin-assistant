# Data

## Source
**"Fiches pratiques et ressources de Service-Public.gouv.fr Particuliers"** (the "vosdroits" XML export), published by the Direction de l'information légale et administrative (DILA).

- Dataset page: https://www.data.gouv.fr/datasets/fiches-pratiques-et-ressources-de-service-public-gouv-fr-particuliers
- File used: `vosdroits-latest.zip` (schema 3.5), export of 25 October 2026
- Licence: [Licence Ouverte / Open Licence 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/)
- Attribution: **Source : Service-Public.fr / DILA — mise à jour : 25/10/2026**

The text of the guides is not modified. It is only parsed and split into passages, and every answer links back to the original page on service-public.gouv.fr.

## Layout (not in git: rebuilt by the scripts)
| folder | content | made by |
|---|---|---|
| `raw/` | `vosdroits-latest.zip`, schema and DILA documentation, as downloaded | you (download) |
| `extracted/` | the unzipped XML files | `scripts/build_corpus.py` |
| `processed/` | clean sheets, passages with their heading path, contacts | `scripts/build_corpus.py` |
| `index/` | BM25 index and bge-m3 vectors | `scripts/embed.py` |

## In git
| folder | content |
|---|---|
| `eval/questions.yaml` | 163 test questions, with the expected behaviour, the official sheets that answer them and the facts the answer should give |
| `eval/paraphrases.yaml` | 24 of those questions rewritten in 5 styles (120 variants) |
| `eval/followups.yaml` | 28 cases for the follow-up detector |

## Refreshing the data
Download the latest `vosdroits-latest.zip` from the dataset page into `raw/`, then run `scripts/build_corpus.py` and `scripts/embed.py`. Update the date in the attribution above.
