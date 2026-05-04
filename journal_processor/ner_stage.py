"""
NER stage orchestration
=======================
Reads a PAGE-XML file produced by the layout + transcription stages,
extracts the plain text from each TextRegion, runs Gemini-based NER
on the combined page text, and writes a new PAGE-XML file that adds
a <NamedEntities> block under the <Page> element:

    <Page ...>
        ...
        <TextRegion id="r02">...</TextRegion>
        ...
        <NamedEntities>
            <NamedEntity regionRef="r02" type="Location" text="München"
                         context="Im englischen Garten ..."/>
            ...
        </NamedEntities>
    </Page>

Original PAGE-XML files in ``output/pagexml`` are NOT modified — annotated
copies are written to ``output/pagexml_ner``.  The Create_GUIs.py viewer
prefers ``pagexml_ner`` if it exists and falls back to ``pagexml`` otherwise.
"""

from __future__ import annotations

import json
import logging
import re
import xml.dom.minidom as minidom
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

from .ner import Entity, perform_ner

log = logging.getLogger(__name__)

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

# Strip transcription markup like <u>...</u> / <sup>...</sup> before sending to NER.
_MARKUP_RE = re.compile(r"</?(?:u|sup|sub|s|del|ins|mark|b|i|em|strong|small)\b[^>]*>",
                        flags=re.IGNORECASE)
# Soft-hyphen at end of line: "Zaunköni-\nge" → "Zaunkönige"
_LINE_HYPHEN_RE = re.compile(r"-\n")
_WS_RE = re.compile(r"[ \t]+\n")


def _strip_markup(text: str) -> str:
    """Remove HTML-like markup tags and join soft-hyphenated words."""
    if not text:
        return ""
    cleaned = _MARKUP_RE.sub("", text)
    cleaned = _LINE_HYPHEN_RE.sub("", cleaned)
    cleaned = _WS_RE.sub("\n", cleaned)
    return cleaned


def _local(tag: str) -> str:
    """Strip XML namespace from a tag."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def _read_pagexml(xml_path: Path) -> Tuple[ET.ElementTree, ET.Element, List[Dict[str, str]]]:
    """Parse a PAGE XML file and return (tree, page_element, regions).

    ``regions`` is a list of {id, type, text} for every TextRegion that
    has transcribed Unicode content.  Markup tags are preserved here —
    callers strip them with ``_strip_markup`` before sending to NER.
    """
    # Register the PAGE namespace so that ET writes default-namespaced output
    ET.register_namespace("", PAGE_NS)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    page = None
    for child in root.iter():
        if _local(child.tag) == "Page":
            page = child
            break
    if page is None:
        raise ValueError(f"No <Page> element found in {xml_path}")

    regions: List[Dict[str, str]] = []
    for child in page:
        if _local(child.tag) != "TextRegion":
            continue
        rid = child.get("id", "")
        custom = child.get("custom", "")
        # Try to parse "type:Foo" out of the custom attribute
        rtype = ""
        m = re.search(r"type:([A-Za-z]+)", custom)
        if m:
            rtype = m.group(1)

        # Find Unicode text under TextEquiv
        text = ""
        for sub in child.iter():
            if _local(sub.tag) == "Unicode" and sub.text:
                text = sub.text
                break

        if text:
            regions.append({"id": rid, "type": rtype, "text": text})

    return tree, page, regions


def _build_page_text(regions: List[Dict[str, str]],
                     skip_types: Optional[set] = None) -> Tuple[str, List[Tuple[str, str]]]:
    """Concatenate region texts (markup-stripped) into one page-level string.

    Returns ``(combined_text, [(region_id, clean_text), ...])`` so that
    callers can later attribute each entity hit back to a region.
    """
    skip_types = skip_types or {"PageNumberRegion", "ImageRegion", "ObjectRegion"}
    parts: List[Tuple[str, str]] = []
    pieces: List[str] = []
    for r in regions:
        if r["type"] in skip_types:
            continue
        clean = _strip_markup(r["text"])
        if not clean.strip():
            continue
        parts.append((r["id"], clean))
        pieces.append(clean)
    return "\n\n".join(pieces), parts


def _attribute_to_regions(entities: List[Entity],
                          region_texts: List[Tuple[str, str]]) -> None:
    """Best-effort: set ``Entity.region_ref`` to the first region containing the text."""
    for ent in entities:
        if not ent.text:
            continue
        for rid, rtext in region_texts:
            if ent.text in rtext:
                ent.region_ref = rid
                break


def _add_named_entities_element(page: ET.Element,
                                entities: List[Entity]) -> ET.Element:
    """Replace any existing <NamedEntities> block under ``page`` with a fresh one."""
    # Remove old block if present (idempotent re-runs)
    for existing in list(page):
        if _local(existing.tag) == "NamedEntities":
            page.remove(existing)

    ns = f"{{{PAGE_NS}}}"
    block = ET.SubElement(page, f"{ns}NamedEntities")
    for ent in entities:
        attrs = {
            "type": ent.entity_type,
            "text": ent.text,
        }
        if ent.region_ref:
            attrs["regionRef"] = ent.region_ref
        if ent.context:
            attrs["context"] = ent.context
        ET.SubElement(block, f"{ns}NamedEntity", attrs)
    return block


def _write_pretty(tree: ET.ElementTree, out_path: Path) -> None:
    """Pretty-print and write the XML, matching the original generator style."""
    xml_str = ET.tostring(tree.getroot(), encoding="unicode")
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ", encoding=None)
    # Drop the extra <?xml ...?> declaration minidom adds to match output_pagexml.py
    pretty = "\n".join(line for line in pretty.splitlines()[1:] if line.strip())
    out_path.write_text(pretty + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_pagexml(
    client: Any,
    xml_path: Path,
    out_path: Path,
    entity_types: Dict[str, str],
    model_id: str,
    thinking_level: str = "low",
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Run NER on one PAGE-XML file and write an annotated copy.

    Returns a small summary dict with keys: ``page``, ``status``, ``n_entities``.
    """
    page_id = xml_path.stem

    if skip_existing and out_path.exists():
        return {"page": page_id, "status": "skipped", "n_entities": -1}

    try:
        tree, page_el, regions = _read_pagexml(xml_path)
    except Exception as exc:
        log.error("Failed to parse %s: %s", xml_path, exc)
        return {"page": page_id, "status": "parse_error", "n_entities": 0}

    combined, region_texts = _build_page_text(regions)
    if not combined.strip():
        # No transcribed text — still write an empty annotated copy so the
        # GUI can find the file alongside the original pagexml.
        _add_named_entities_element(page_el, [])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pretty(tree, out_path)
        return {"page": page_id, "status": "empty", "n_entities": 0}

    entities = perform_ner(
        client=client,
        text=combined,
        entity_types=entity_types,
        model_id=model_id,
        thinking_level=thinking_level,
        page_id=page_id,
    )
    _attribute_to_regions(entities, region_texts)

    _add_named_entities_element(page_el, entities)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pretty(tree, out_path)

    return {"page": page_id, "status": "ok", "n_entities": len(entities)}


def parse_named_entities(xml_path: Path) -> List[Dict[str, Any]]:
    """Read back the <NamedEntity> entries from an annotated PAGE-XML file.

    Used by tools that consume NER output (e.g. evaluation scripts).
    """
    if not xml_path.exists():
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out: List[Dict[str, Any]] = []
    for el in root.iter():
        if _local(el.tag) != "NamedEntity":
            continue
        out.append({
            "text": el.get("text", ""),
            "entity_type": el.get("type", ""),
            "region_ref": el.get("regionRef") or None,
            "context": el.get("context") or None,
        })
    return out
