"""
NER stage — Named Entity Recognition on transcribed page text.
==============================================================
Recall-first extraction for the Laubmann ornithologist field journals.

Two parallel axes are returned per entity:
  • type  — one of the nine TYPE classes (Tier, Pflanze, Lebensraum, Material,
            Klima, Person, Ort, Organisation, Datum)
  • scope — one of the five SCOPE classes (Singular, Kollektiv, Teil, Produkt,
            Referenz); default Singular.  Carries the specimen logic.

Plus optional entity-level attributes:
  • count           — stated cardinality ("2", "7", "Tausende")
  • scientific_name — Latin binomial when given in parentheses after a Tier

The model also returns a short verbatim ``quote`` per entity.  The renderer in
ner_stage.py matches entities back to the source text; start_char / end_char are
left at -1 here because LLMs are unreliable at exact character offsets.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .utils import clean_llm_json

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """One Named Entity detected on a page."""
    text: str
    entity_type: str
    scope: str = "Singular"
    count: Optional[str] = None
    scientific_name: Optional[str] = None
    start_char: int = -1
    end_char: int = -1
    context: Optional[str] = None
    region_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "text": self.text,
            "entity_type": self.entity_type,
            "scope": self.scope,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }
        if self.count:
            d["count"] = self.count
        if self.scientific_name:
            d["scientific_name"] = self.scientific_name
        if self.context:
            d["context"] = self.context
        if self.region_ref:
            d["region_ref"] = self.region_ref
        return d


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_GUIDELINE_HEADER = """\
Du bist Experte für Named Entity Recognition in historischen deutschen \
Feldtagebüchern eines Ornithologen (Laubmann, ~1930er Jahre). Der Text enthält \
Vogelbeobachtungen mit Datum und Ort, Sammler- und Autorennamen, sowie \
Beschreibungen von Präparaten (Federn, getrocknete Blätter) und Bildern.

ZIEL: HOHE VOLLSTÄNDIGKEIT (Recall). Erfasse JEDE Entität der unten genannten \
Typen. Bei den biologischen und geographischen Klassen gilt: lieber eine \
Entität zu viel als eine übersehen. Nur bei Person, Ort und Organisation ist \
echte Benanntheit Voraussetzung."""

_TYPE_RULES = """\
WICHTIGE REGELN:

1. MEHRWORT-SPANS: Annotiere immer den VOLLSTÄNDIGEN Namen, nicht nur einen Teil.
   "Große Sumpfschnepfe", "Starnberger See", "Dr. von Jäckel", "kleine Bekassine"
   sind je EINE Entität — nicht "Bekassine" allein.

2. ABGETRENNTE WÖRTER: Im Text wurden Trennstriche am Zeilenende bereits \
zusammengefügt. Gib zusammengesetzte Namen geschlossen zurück \
("Löffelreiher", "Purpurreiher", "Fischbeißer").

3. WISSENSCHAFTLICHE NAMEN: Steht hinter dem deutschen Tiernamen ein \
lateinisches Binomen in Klammern (z. B. "(Sterna tschegrava Lep.)"), \
annotiere das Latein NICHT als eigene Entität, sondern schreibe es in das \
Feld scientific_name der zugehörigen Tier-Entität.

4. ANZAHL: Ist eine Stückzahl genannt ("2 Raubseeschwalben", "7 Gänse", \
"Tausende von Schwalben"), trage die Zahl/Menge in das Feld count ein.

5. text MUSS WÖRTLICH aus dem Analyse-Text kopierbar sein (gleiche Schreibung).
   quote = ein kurzer wörtlicher Ausschnitt (3–8 Wörter), der die Entität enthält.

6. NICHT annotieren: Adjektive/Zahlwörter als solche ("dunkler", "drei"), \
Tiergeräusche, generische Rollen (Forscher, Jäger), bloße Jahreszeiten ohne Jahr."""

_SCOPE_RULES = """\
SCOPE (zweite Achse, Standard = Singular):
  - Singular : Einzeltier/-pflanze oder bloßer Plural (ein Habicht, die Schwalben, 7 Gänse)
  - Kollektiv: echte Gruppe als Einheit (Schwarm, Herde, Zug, Tausende von Schwalben)
  - Teil     : Körper-/Pflanzenteil, auch Präparat-Teil (Feder, Stoß, Flügel, Blatt)
  - Produkt  : abgelöstes Erzeugnis (Ei, Gelege, Same, Frucht)
  - Referenz : benanntes Einzelwesen / Rasse"""

NER_PROMPT_TEMPLATE = """\
{header}

ENTITÄTSTYPEN:
{entity_descriptions}

{type_rules}

{scope_rules}
{hints_block}
TEXT ZUR ANALYSE:
```
{text}
```

Antworte NUR mit einem JSON-Array (kein Markdown, kein Kommentar):
[
  {{
    "text": "exakter Text der Entität",
    "type": "einer der Typen oben",
    "scope": "einer der Scope-Werte",
    "count": "Stückzahl falls genannt, sonst weglassen",
    "scientific_name": "lat. Binomen falls vorhanden, sonst weglassen",
    "quote": "kurzer wörtlicher Ausschnitt mit der Entität"
  }}
]

Gib ein leeres Array [] zurück, wenn keine Entitäten gefunden werden."""

VERIFY_PROMPT_TEMPLATE = """\
Du hast diesen historischen ornithologischen Text bereits einmal annotiert.
Prüfe ihn auf ÜBERSEHENE Entitäten der Typen: {type_list}.
Achte besonders auf Vogelarten, Pflanzen, Personennamen (auch mit Dr./von), \
Datumsangaben (auch römische Monatszahlen) und benannte Orte/Gewässer.

Bereits gefunden (NICHT erneut nennen):
{found_list}

TEXT:
```
{text}
```

Gib NUR die ZUSÄTZLICH gefundenen Entitäten als JSON-Array im selben Format \
zurück (text, type, scope, count, scientific_name, quote). Leeres Array [] wenn \
nichts fehlt."""


# ---------------------------------------------------------------------------
# Response schema (google-genai structured output)
# ---------------------------------------------------------------------------

def _build_response_schema(entity_types: List[str], scope_types: List[str]):
    """Return a google-genai Schema constraining the model to valid labels.

    Returns None if the SDK Schema API is unavailable so the caller can fall
    back to free-form JSON + the robust parser below.
    """
    try:
        from google.genai import types
        s = types.Schema
        return s(
            type=types.Type.ARRAY,
            items=s(
                type=types.Type.OBJECT,
                properties={
                    "text": s(type=types.Type.STRING),
                    "type": s(type=types.Type.STRING, enum=list(entity_types)),
                    "scope": s(type=types.Type.STRING, enum=list(scope_types)),
                    "count": s(type=types.Type.STRING, nullable=True),
                    "scientific_name": s(type=types.Type.STRING, nullable=True),
                    "quote": s(type=types.Type.STRING, nullable=True),
                },
                required=["text", "type"],
            ),
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("response_schema unavailable, using free-form JSON: %s", exc)
        return None


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    """Robustly extract a JSON array from an LLM response."""
    if not text:
        return []
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("entities", "items", "results", "data"):
                if isinstance(data.get(key), list):
                    return data[key]
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    if start == -1:
        cleaned = clean_llm_json(text)
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                for key in ("entities", "items", "results", "data"):
                    if isinstance(obj.get(key), list):
                        return obj[key]
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
        return []

    depth, end = 0, -1
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return []
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError as exc:
        log.warning("NER JSON parse failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Single Gemini call
# ---------------------------------------------------------------------------

def _generate(client: Any, model_id: str, prompt: str, thinking_level: str,
              schema: Any, max_attempts: int, page_id: Optional[str]
              ) -> List[Dict[str, Any]]:
    """One prompt → parsed list of raw dicts.  Retries; falls back to no-schema."""
    from google.genai import types

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        use_schema = schema if attempt == 1 else None  # drop schema on retry
        try:
            cfg_kwargs: Dict[str, Any] = {
                "thinking_config": types.ThinkingConfig(thinking_level=thinking_level),
                "response_mime_type": "application/json",
            }
            if use_schema is not None:
                cfg_kwargs["response_schema"] = use_schema
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            return _parse_json_array(response.text or "")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log.warning("NER attempt %d/%d failed%s: %s", attempt, max_attempts,
                        f" (page={page_id})" if page_id else "", exc)
    if last_err is not None:
        log.error("NER giving up on %s after %d attempts: %s",
                  page_id or "<text>", max_attempts, last_err)
    return []


# ---------------------------------------------------------------------------
# Normalisation / validation of raw items
# ---------------------------------------------------------------------------

def _coerce_entity(item: Dict[str, Any], valid_types: set, valid_scopes: set
                   ) -> Optional[Entity]:
    if not isinstance(item, dict):
        return None
    etype = str(item.get("type") or item.get("entity_type") or "").strip()
    etext = str(item.get("text") or "").strip()
    if etype not in valid_types or not etext:
        return None
    scope = str(item.get("scope") or "").strip()
    if scope not in valid_scopes:
        scope = "Singular"
    count = item.get("count")
    count = str(count).strip() if count not in (None, "", "null") else None
    sci = item.get("scientific_name")
    sci = str(sci).strip() if sci not in (None, "", "null") else None
    ctx = item.get("quote") or item.get("context")
    ctx = str(ctx).strip() if ctx else None
    return Entity(text=etext, entity_type=etype, scope=scope,
                  count=count, scientific_name=sci, context=ctx)


# ---------------------------------------------------------------------------
# Core NER function
# ---------------------------------------------------------------------------

def perform_ner(
    client: Any,
    text: str,
    entity_types: Dict[str, str],
    model_id: str,
    scope_types: Optional[Dict[str, str]] = None,
    thinking_level: str = "medium",
    max_attempts: int = 2,
    verify_pass: bool = True,
    underline_hints: Optional[List[str]] = None,
    page_id: Optional[str] = None,
) -> List[Entity]:
    """Run NER on plain text and return a list of Entity objects."""
    if not text or not text.strip():
        return []

    scope_types = scope_types or {"Singular": ""}
    valid_types = set(entity_types.keys())
    valid_scopes = set(scope_types.keys())

    entity_descriptions = "\n".join(
        f"- {etype}: {desc}" for etype, desc in entity_types.items()
    )
    scope_rules = _SCOPE_RULES if len(scope_types) > 1 else ""

    hints_block = ""
    if underline_hints:
        uniq = list(dict.fromkeys(h for h in underline_hints if h.strip()))
        if uniq:
            joined = " · ".join(uniq[:40])
            hints_block = (
                "\nHINWEIS: Folgende Begriffe sind im Original unterstrichen und "
                "sind mit hoher Wahrscheinlichkeit Entitäten (Arten oder Orte) — "
                f"prüfe jeden:\n{joined}\n"
            )

    schema = _build_response_schema(list(valid_types), list(valid_scopes))

    prompt = NER_PROMPT_TEMPLATE.format(
        header=_GUIDELINE_HEADER,
        entity_descriptions=entity_descriptions,
        type_rules=_TYPE_RULES,
        scope_rules=scope_rules,
        hints_block=hints_block,
        text=text,
    )

    raw = _generate(client, model_id, prompt, thinking_level, schema,
                    max_attempts, page_id)

    # de-dup while preserving order; key on (text, type)
    seen: set = set()
    entities: List[Entity] = []
    for item in raw:
        ent = _coerce_entity(item, valid_types, valid_scopes)
        if ent is None:
            continue
        key = (ent.text, ent.entity_type)
        if key in seen:
            continue
        seen.add(key)
        entities.append(ent)

    # ── completeness pass ───────────────────────────────────────────────
    if verify_pass and entities:
        found_list = "\n".join(f"- {e.text} ({e.entity_type})" for e in entities)
        vprompt = VERIFY_PROMPT_TEMPLATE.format(
            type_list=", ".join(valid_types),
            found_list=found_list,
            text=text,
        )
        extra = _generate(client, model_id, vprompt, thinking_level, schema,
                          1, page_id)
        for item in extra:
            ent = _coerce_entity(item, valid_types, valid_scopes)
            if ent is None:
                continue
            key = (ent.text, ent.entity_type)
            if key in seen:
                continue
            seen.add(key)
            entities.append(ent)

    return entities
