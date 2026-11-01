"""Turn the Service-Public XML export ("vosdroits") into clean, structured sheets.

Each practical sheet (F*.xml) becomes one record with its metadata (title, URL, last important
change, theme path) and a list of *blocks*. A block is the text under the deepest heading, with
the full path of headings above it, for example:

    ["Étudiant étranger en France : visa de long séjour ou titre de séjour",
     "Situation : Après 1 an de séjour",
     "Quelle démarche doit effectuer un étudiant étranger quand le VLS-TS arrive en fin de validité ?",
     "Cas : Votre demande est refusée"]

Keeping that path matters: many sheets give different answers depending on the reader's
situation, and a passage is only correct for the situation it sits under.

Contacts ("Qui peut m'aider ?", "Où s'adresser ?") are collected into a separate table, used
when the assistant cannot answer from the sheets.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lxml import etree

DC = "{http://purl.org/dc/elements/1.1/}"
SP_URL = "https://www.service-public.gouv.fr/particuliers/vosdroits/"

# Elements that carry no reader-facing text (or duplicate a Titre)
SKIP = {"TitreRiche", "Condition", "PivotLocal", "RessourceWeb", "Source", "TitreAlternatif",
        "Credits", "Image", "ImageIllustration", "TexteDeRemplacement", "Video", "Infographie"}
NOTES = {"ASavoir": "À savoir", "ANoter": "À noter", "Attention": "Attention", "Rappel": "Rappel",
         "Exemple": "Exemple", "Complement": "Complément"}


def clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def inline(e: etree._Element) -> str:
    return clean("".join(e.itertext()))


@dataclass
class Block:
    path: list[str]
    text: str


@dataclass
class Sheet:
    id: str
    type: str
    title: str
    description: str
    url: str
    modified: str            # date of the last *important* change (shown to users)
    theme: list[str]         # breadcrumb, e.g. ["Étranger - Europe", "Titres, cartes de séjour ..."]
    blocks: list[Block] = field(default_factory=list)
    contacts: list[str] = field(default_factory=list)   # contact IDs, see Contact
    services: list[dict] = field(default_factory=list)  # online services and forms: title, url, type
    related: list[str] = field(default_factory=list)    # linked question-answer sheets

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Contact:
    id: str
    type: str                # "Centre de contact", "Local", "Local personnalisé sur SP", ...
    title: str
    text: str                # phone numbers, opening hours, scope, when Service-Public gives them
    sheets: list[str] = field(default_factory=list)


class _Renderer:
    """Walks a sheet's Introduction / Texte / ListeSituations and emits blocks."""

    def __init__(self, title: str, services: dict[str, dict], contacts: dict[str, Contact], sheet_id: str):
        self.blocks: list[Block] = []
        self.services, self.contacts, self.sheet_id = services, contacts, sheet_id
        self.path = [title]
        self.buf: list[str] = []

    # ---- output helpers
    def flush(self) -> None:
        text = "\n".join(self.buf).strip()
        if text:
            self.blocks.append(Block(list(self.path), text))
        self.buf = []

    def push(self, heading: str) -> None:
        self.flush()
        self.path.append(heading)

    def pop(self) -> None:
        self.flush()
        self.path.pop()

    # ---- elements
    def lines(self, e: etree._Element, depth: int = 0) -> list[str]:
        """Render an element that does not open a new section into text lines."""
        tag = e.tag
        if tag in SKIP or not isinstance(tag, str):
            return []
        if tag == "Paragraphe":
            t = inline(e)
            return [t] if t else []
        if tag == "Titre":
            return []  # titles are handled by the section that owns them
        if tag == "Liste":
            out = []
            for item in e.findall("Item"):
                sub = [ln for c in item for ln in self.lines(c, depth + 1)]
                own = clean((item.text or ""))
                first = own or (sub.pop(0) if sub else "")
                out.append("  " * depth + "- " + first)
                out += ["  " * (depth + 1) + ln for ln in sub]
            return out
        if tag in NOTES:
            label = inline(e.find("Titre")) if e.find("Titre") is not None else NOTES[tag]
            body = " ".join(ln for c in e for ln in self.lines(c, depth))
            return [f"> {label} : {body}"] if body else []
        if tag == "Tableau":
            return self.table(e)
        if tag in ("ServiceEnLigne", "ServiceEnLigneAnnexe", "NoticeLiee"):
            s = self.services.get(e.get("ID"), {})
            title = clean(e.findtext("Titre") or s.get("title", ""))
            url = s.get("url", "")
            return [f"[Service en ligne] {title}" + (f" : {url}" if url else "")] if title else []
        if tag == "OuSAdresser":
            self.add_contact(e)
            return [f"[Où s'adresser] {clean(e.findtext('Titre') or '')}"]
        if tag in ("Texte", "Introduction", "FragmentConditionne", "Description", "Contenu", "ContenuIllustre"):
            return [ln for c in e for ln in self.lines(c, depth)]
        # inline-ish leftovers (Expression, LienInterne at top level, etc.)
        t = inline(e)
        return [t] if t else []

    def table(self, e: etree._Element) -> list[str]:
        rows = []
        for r in e.findall("Rangée"):
            cells = [inline(c).replace("|", "/") for c in r.findall("Cellule")]
            if any(cells):
                rows.append(cells)
        if not rows:
            return []
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        out = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
        out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        caption = clean(e.findtext("TitreFlottant") or "")
        return ([caption] if caption else []) + out

    def add_contact(self, e: etree._Element) -> None:
        cid = e.get("ID")
        if not cid:
            return
        c = self.contacts.get(cid)
        if c is None:
            body = e.find("Texte")
            text = "\n".join(self.lines(body)) if body is not None else ""
            c = self.contacts[cid] = Contact(cid, e.get("type", ""), clean(e.findtext("Titre") or ""), text)
        if self.sheet_id not in c.sheets:
            c.sheets.append(self.sheet_id)

    def section(self, e: etree._Element, label: str = "") -> None:
        """Render an element that opens a new section (Situation, Chapitre, SousChapitre, Cas)."""
        t = e.find("Titre")
        title = inline(t) if t is not None else ""
        heading = (label + title) if title else ""
        if heading:
            self.push(heading)
        self.walk(e)
        if heading:
            self.pop()

    def walk(self, e: etree._Element) -> None:
        for c in e:
            tag = c.tag
            if tag == "Situation":
                self.section(c, "Situation : ")
            elif tag in ("Chapitre", "SousChapitre"):
                self.section(c)
            elif tag == "Cas":
                self.section(c, "Cas : ")
            elif tag in ("BlocCas", "ListeSituations", "Texte"):
                self.walk(c)
            else:
                self.buf += self.lines(c)


def _services(root: etree._Element, resources: dict[str, dict]) -> dict[str, dict]:
    out = dict(resources)
    for s in root.iter("ServiceEnLigne", "ServiceEnLigneAnnexe"):
        if s.get("URL"):
            out[s.get("ID")] = {"title": clean(s.findtext("Titre") or ""), "url": s.get("URL"), "type": s.get("type", "")}
    return out


def load_resources(folder: Path) -> dict[str, dict]:
    """Online services, forms and simulators (R*.xml): id -> title, url, type."""
    out = {}
    for f in folder.glob("R*.xml"):
        r = etree.parse(str(f)).getroot()
        url = r.get("URL") or r.findtext("URL") or r.get("spUrl", "")
        out[r.get("ID")] = {"title": clean(r.findtext(DC + "title") or ""), "url": url, "type": r.get("type", "")}
    return out


def parse_sheet(path: Path, resources: dict[str, dict], contacts: dict[str, Contact]) -> Sheet:
    root = etree.parse(str(path)).getroot()
    sid = root.get("ID")
    title = clean(root.findtext(DC + "title") or "")
    services = _services(root, resources)
    theme = [clean(n.text or "") for n in root.findall("FilDAriane/Niveau")
             if n.get("ID", "").startswith("N")]
    modified = (root.get("dateDerniereModificationImportante") or "")[:10]
    if not modified:
        modified = clean(root.findtext(DC + "date") or "").replace("modified", "").strip()[:10]

    r = _Renderer(title, services, contacts, sid)
    intro = root.find("Introduction")
    if intro is not None:
        r.buf += r.lines(intro)
    for tag in ("Texte", "ListeSituations"):
        e = root.find(tag)
        if e is not None:
            r.walk(e)   # Texte -> Chapitre..., ListeSituations -> Situation...
    r.flush()

    for q in root.findall("QuiPeutMAider"):
        r.add_contact(q)
    sheet = Sheet(
        id=sid, type=root.get("type", ""), title=title,
        description=clean(root.findtext(DC + "description") or ""),
        url=root.get("spUrl") or SP_URL + sid, modified=modified, theme=theme, blocks=r.blocks,
        contacts=[q.get("ID") for q in root.findall("QuiPeutMAider") if q.get("ID")],
        services=[{"title": s["title"], "url": s["url"], "type": s["type"]}
                  for s in (services.get(x.get("ID")) for x in root.findall("ServiceEnLigne")) if s],
        related=[q.get("ID") for q in root.findall("QuestionReponse") if q.get("ID")],
    )
    # offices named inside the text ("Où s'adresser") also count as contacts for this sheet
    for cid, c in contacts.items():
        if sid in c.sheets and cid not in sheet.contacts:
            sheet.contacts.append(cid)
    return sheet


def parse_folder(folder: Path) -> tuple[list[Sheet], dict[str, Contact]]:
    resources = load_resources(folder)
    contacts: dict[str, Contact] = {}
    sheets = [parse_sheet(f, resources, contacts) for f in sorted(folder.glob("F*.xml"))]
    return sheets, contacts
