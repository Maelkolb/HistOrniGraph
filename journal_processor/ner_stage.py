"""
NER stage orchestration
=======================
Reads a PAGE-XML file, runs Gemini NER on its text-bearing regions, and writes
an annotated copy under ``pagexml_ner/`` with:

(1) inline Transkribus-style ``custom`` tags on each region:
        namedentity {offset:14; length:7; type:Tier; scope:Singular;}
    offset = char index into the region <Unicode>, length = char count.

(2) a denormalised ``<NamedEntities>`` index block under ``<Page>``, one
    ``<NamedEntity>`` per occurrence carrying regionRef, regionType, offset,
    length, type, scope and optional count / scientificName / context.

Key properties of this version
------------------------------
* Alignment hardening: detected entities are matched back to the source through
  a normalised search (case-, ß/ss-, ſ-, whitespace-insensitive) so model
  normalisation no longer silently drops real detections.  Entities that still
  cannot be located are kept in the index with ``unmatched="true"`` rather than
  vanishing.
* Overlap resolution: per region, occurrences are reduced to a non-overlapping
  longest-match set so the inline tags and the BIO export agree and stay legal.
* Specimen / picture descriptions: ImageRegion and ObjectRegion (GraphicRegion)
  descriptions are included.  When the original PAGE-XML lacks a <Unicode> for
  them (older output), the description is backfilled from ``regions/<page>.json``
  and injected into the annotated copy so offsets anchor to stored text.

Original ``pagexml/`` files are never modified.
"""

from __future__ import annotations

import json
import logging
import re
import xml.dom.minidom as minidom
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

from .ner import Entity, perform_ner

log = logging.getLogger(__name__)

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

_MARKUP_TAG_RE = re.compile(
    r"</?(?:u|sup|sub|s|del|ins|mark|b|i|em|strong|small)\b[^>]*>",
    flags=re.IGNORECASE,
)

INLINE_TAG_NAME = "namedentity"

# Region element tags we treat as carrying transcribable / describable text.
_TEXTLIKE_TAGS = frozenset({"TextRegion", "TableRegion", "ImageRegion", "GraphicRegion"})


# ---------------------------------------------------------------------------
# Markup stripping with offset map
# ---------------------------------------------------------------------------

def _strip_with_map(raw: str) -> Tuple[str, List[int]]:
    """Strip <u>/<sup>/… and join soft hyphens; return (clean, raw_idx_of_clean).

    Soft-hyphen joining is markup-aware: a line-break hyphen may be separated
    from its newline by markup tags (the journal splits species across two
    underline spans, e.g. ``<u>Löffel-</u>\\n<u>reiher</u>``).  Such a hyphen and
    the surrounding markup/newline are dropped so the word halves join.
    """
    clean_chars: List[str] = []
    raw_idx: List[int] = []
    i, n = 0, len(raw)
    while i < n:
        m = _MARKUP_TAG_RE.match(raw, i)
        if m:
            i = m.end()
            continue
        if raw[i] == "-":
            j = i + 1
            while True:
                mt = _MARKUP_TAG_RE.match(raw, j)
                if mt:
                    j = mt.end()
                    continue
                break
            if j < n and raw[j] == "\n":
                i = j + 1
                while True:
                    mt = _MARKUP_TAG_RE.match(raw, i)
                    if mt:
                        i = mt.end()
                        continue
                    break
                continue
        clean_chars.append(raw[i])
        raw_idx.append(i)
        i += 1
    return "".join(clean_chars), raw_idx


def _project_to_raw(span_in_clean: Tuple[int, int],
                    raw_idx_of_clean: List[int],
                    raw_len: int) -> Tuple[int, int]:
    s_clean, e_clean = span_in_clean
    n_clean = len(raw_idx_of_clean)
    if s_clean >= e_clean or n_clean == 0:
        return (raw_len, raw_len)
    raw_start = raw_idx_of_clean[s_clean]
    raw_last = raw_idx_of_clean[min(e_clean - 1, n_clean - 1)]
    return (raw_start, raw_last + 1)


# ---------------------------------------------------------------------------
# Normalised matching (alignment hardening)
# ---------------------------------------------------------------------------

def _normalize_with_map(s: str) -> Tuple[str, List[int]]:
    """Casefold + ſ→s + ß→ss + whitespace-collapse; map each norm char → src idx.

    casefold() already lowercases, folds 'ß'→'ss' and 'ſ'→'s', so a single
    character may expand to several; every produced char points back at the
    source index it came from.  Runs of whitespace collapse to one space.
    """
    out_chars: List[str] = []
    out_idx: List[int] = []
    prev_ws = False
    for i, ch in enumerate(s):
        if ch.isspace():
            if not prev_ws:
                out_chars.append(" ")
                out_idx.append(i)
                prev_ws = True
            continue
        prev_ws = False
        for cc in ch.casefold():
            out_chars.append(cc)
            out_idx.append(i)
    return "".join(out_chars), out_idx


def _norm_offsets_in_clean(clean: str, clean_norm: str, norm_to_clean: List[int],
                           needle: str) -> List[Tuple[int, int]]:
    """Find ``needle`` in ``clean`` via the normalised form; return clean spans."""
    nd_norm, _ = _normalize_with_map(needle)
    nd_norm = nd_norm.strip()
    if not nd_norm:
        return []
    spans: List[Tuple[int, int]] = []
    start = 0
    L = len(nd_norm)
    while True:
        k = clean_norm.find(nd_norm, start)
        if k == -1:
            break
        c_start = norm_to_clean[k]
        c_end = norm_to_clean[k + L - 1] + 1
        spans.append((c_start, c_end))
        start = k + 1
    return spans


_WORD_RE = re.compile(r"\w", flags=re.UNICODE)


def _find_all_offsets(haystack: str, needle: str,
                      prefer_whole_word: bool = True) -> List[int]:
    if not needle:
        return []
    positions: List[int] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    if not prefer_whole_word or not positions:
        return positions
    n, h_len = len(needle), len(haystack)
    whole = []
    for p in positions:
        left_ok = p == 0 or not _WORD_RE.match(haystack[p - 1])
        right_ok = p + n == h_len or not _WORD_RE.match(haystack[p + n])
        if left_ok and right_ok:
            whole.append(p)
    return whole or positions


# ---------------------------------------------------------------------------
# Custom-attribute syntax (Transkribus convention)
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"(?P<name>[A-Za-z_][\w\-]*)\s*\{(?P<body>[^}]*)\}",
                     flags=re.UNICODE)


def _parse_custom(attr: str) -> List[Tuple[str, Dict[str, str]]]:
    if not attr:
        return []
    out: List[Tuple[str, Dict[str, str]]] = []
    pos = 0
    for m in _TAG_RE.finditer(attr):
        gap = attr[pos:m.start()].strip()
        if gap:
            for token in gap.split():
                out.append((token, {}))
        body: Dict[str, str] = {}
        for piece in m.group("body").split(";"):
            piece = piece.strip()
            if not piece:
                continue
            if ":" in piece:
                k, _, v = piece.partition(":")
                body[k.strip()] = v.strip()
            else:
                body[piece] = ""
        out.append((m.group("name"), body))
        pos = m.end()
    tail = attr[pos:].strip()
    if tail:
        for token in tail.split():
            out.append((token, {}))
    return out


def _format_custom(parts: List[Tuple[str, Dict[str, str]]]) -> str:
    pieces: List[str] = []
    for name, body in parts:
        if not body:
            pieces.append(name)
            continue
        kv = " ".join(f"{k}:{v};" for k, v in body.items())
        pieces.append(f"{name} {{{kv}}}")
    return " ".join(pieces)


# ---------------------------------------------------------------------------
# PAGE-XML I/O
# ---------------------------------------------------------------------------

def _local(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _region_unicode(region_el: ET.Element) -> str:
    for sub in region_el.iter():
        if _local(sub.tag) == "Unicode" and sub.text:
            return sub.text
    return ""


def _ensure_unicode(region_el: ET.Element, text: str) -> bool:
    """Inject a <TextEquiv><Unicode>text</Unicode></TextEquiv> if absent."""
    ns = f"{{{PAGE_NS}}}"
    for te in region_el:
        if _local(te.tag) == "TextEquiv":
            for u in te:
                if _local(u.tag) == "Unicode":
                    if u.text:
                        return False
                    u.text = text
                    return True
            u = ET.SubElement(te, f"{ns}Unicode")
            u.text = text
            return True
    te = ET.SubElement(region_el, f"{ns}TextEquiv")
    ET.SubElement(te, f"{ns}Unicode").text = text
    return True


def _region_type(region_el: ET.Element) -> str:
    custom = region_el.get("custom", "")
    m = re.search(r"type:([A-Za-z]+)", custom)
    if m:
        return m.group(1)
    return _local(region_el.tag)


def _read_pagexml(xml_path: Path):
    """Return (tree, page_element, region_elements) for all text-like regions."""
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
    regions: List[Dict[str, Any]] = []
    for child in page:
        if _local(child.tag) not in _TEXTLIKE_TAGS:
            continue
        regions.append({
            "id": child.get("id", ""),
            "type": _region_type(child),
            "text": _region_unicode(child),
            "element": child,
        })
    return tree, page, regions


def _load_region_descriptions(xml_path: Path) -> Dict[str, str]:
    """Read image/object descriptions from the sibling regions/<page>.json."""
    page_id = xml_path.stem
    candidates = [
        xml_path.parent.parent / "regions" / f"{page_id}.json",
        xml_path.parent / "regions" / f"{page_id}.json",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        return {}
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not read descriptions from %s: %s", src, exc)
        return {}
    out: Dict[str, str] = {}
    for r in data if isinstance(data, list) else []:
        if r.get("type") not in ("ImageRegion", "ObjectRegion"):
            continue
        tr = r.get("transcription", {}) or {}
        desc = tr.get("description")
        if not desc:
            raw = tr.get("text", "") or ""
            # strip "DESCRIPTION:" / "TEXT:" / "*_TYPE:" scaffolding
            kept = [ln.split(":", 1)[1].strip() if ln.upper().startswith("DESCRIPTION:")
                    else ("" if re.match(r"(?i)^(text|drawing_type|object_type)\s*:", ln) else ln)
                    for ln in raw.splitlines()]
            desc = " ".join(p for p in kept if p).strip()
        rid = r.get("id", "")
        if rid and desc:
            out[rid] = desc
    return out


def _write_pretty(tree: ET.ElementTree, out_path: Path) -> None:
    xml_str = ET.tostring(tree.getroot(), encoding="unicode")
    pretty = minidom.parseString(xml_str).toprettyxml(indent="  ", encoding=None)
    pretty = "\n".join(pretty.splitlines()[1:])
    out_path.write_text(pretty, encoding="utf-8")


# ---------------------------------------------------------------------------
# Underline hints
# ---------------------------------------------------------------------------

def _extract_underline_hints(raw_text: str) -> List[str]:
    """Reconstruct underlined spans (<u>…</u>), rejoining hyphen-split words."""
    spans = re.findall(r"<u>(.*?)</u>", raw_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = []
    for sp in spans:
        sp = _MARKUP_TAG_RE.sub("", sp).replace("\n", " ").strip()
        cleaned.append(sp)
    merged: List[str] = []
    i = 0
    while i < len(cleaned):
        cur = cleaned[i]
        while cur.endswith("-") and i + 1 < len(cleaned):
            cur = cur[:-1] + cleaned[i + 1]
            i += 1
        merged.append(cur.strip())
        i += 1
    out: List[str] = []
    for h in merged:
        h = h.strip(" .,;:")
        if len(h) >= 2 and any(c.isalpha() for c in h):
            out.append(h)
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Entity → offset bookkeeping
# ---------------------------------------------------------------------------

def _occurrences_for_region(raw_text: str, clean_text: str,
                            raw_idx_of_clean: List[int],
                            entities: List[Entity]) -> Tuple[List[Dict[str, Any]], set]:
    """Locate every occurrence of every entity inside one region's raw text.

    Returns (occurrences, matched_entity_keys).  Matching tiers per entity:
      1. literal in clean text   2. literal in raw text   3. normalised in clean.
    Occurrences are overlap-resolved (longest wins) so the result is BIO-legal.
    """
    raw_len = len(raw_text)
    clean_norm, norm_to_clean = _normalize_with_map(clean_text)

    raw_occs: List[Dict[str, Any]] = []
    matched: set = set()

    for ent in entities:
        seen_raw_starts: set = set()
        hits_before = len(raw_occs)

        def _add(raw_start: int, raw_end: int) -> None:
            if raw_start in seen_raw_starts:
                return
            seen_raw_starts.add(raw_start)
            surface = " ".join(_strip_with_map(raw_text[raw_start:raw_end])[0].split())
            raw_occs.append({
                "offset": raw_start,
                "length": raw_end - raw_start,
                "type": ent.entity_type,
                "scope": ent.scope,
                "count": ent.count,
                "sci": ent.scientific_name,
                "context": ent.context,
                "text": surface,
            })

        # tier 1 — literal in clean
        for off_clean in _find_all_offsets(clean_text, ent.text):
            rs, re_ = _project_to_raw((off_clean, off_clean + len(ent.text)),
                                      raw_idx_of_clean, raw_len)
            _add(rs, re_)
        # tier 2 — literal in raw
        for off_raw in _find_all_offsets(raw_text, ent.text):
            _add(off_raw, off_raw + len(ent.text))
        # tier 3 — normalised in clean (orthography / whitespace tolerant)
        if len(raw_occs) == hits_before:
            for (cs, ce) in _norm_offsets_in_clean(clean_text, clean_norm,
                                                   norm_to_clean, ent.text):
                rs, re_ = _project_to_raw((cs, ce), raw_idx_of_clean, raw_len)
                _add(rs, re_)

        if len(raw_occs) > hits_before:
            matched.add((ent.text, ent.entity_type))

    resolved = _resolve_overlaps(raw_occs)
    resolved.sort(key=lambda r: (r["offset"], r["length"]))
    return resolved, matched


def _resolve_overlaps(occs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Greedy longest-match: drop occurrences overlapping a kept (longer) one."""
    ordered = sorted(occs, key=lambda r: (-(r["length"]), r["offset"]))
    kept: List[Dict[str, Any]] = []
    for occ in ordered:
        s, e = occ["offset"], occ["offset"] + occ["length"]
        if any(not (e <= k["offset"] or s >= k["offset"] + k["length"]) for k in kept):
            continue
        kept.append(occ)
    return kept


def _attach_inline_tags(region_el: ET.Element,
                        occurrences: List[Dict[str, Any]]) -> None:
    parts = _parse_custom(region_el.get("custom", ""))
    parts = [(n, b) for n, b in parts if n != INLINE_TAG_NAME]
    for occ in occurrences:
        body = {"offset": str(occ["offset"]),
                "length": str(occ["length"]),
                "type": occ["type"]}
        if occ.get("scope"):
            body["scope"] = occ["scope"]
        parts.append((INLINE_TAG_NAME, body))
    serialised = _format_custom(parts)
    if serialised:
        region_el.set("custom", serialised)
    elif "custom" in region_el.attrib:
        del region_el.attrib["custom"]


def _add_named_entities_index(page: ET.Element,
                              flat: List[Dict[str, Any]]) -> ET.Element:
    for existing in list(page):
        if _local(existing.tag) == "NamedEntities":
            page.remove(existing)
    ns = f"{{{PAGE_NS}}}"
    block = ET.SubElement(page, f"{ns}NamedEntities")
    for rec in flat:
        attrs: Dict[str, str] = {"type": rec["type"]}
        if rec.get("scope"):
            attrs["scope"] = rec["scope"]
        if rec.get("regionRef"):
            attrs["regionRef"] = rec["regionRef"]
        if rec.get("regionType"):
            attrs["regionType"] = rec["regionType"]
        if rec.get("offset") is not None and rec["offset"] >= 0:
            attrs["offset"] = str(rec["offset"])
            attrs["length"] = str(rec["length"])
        else:
            attrs["unmatched"] = "true"
        if rec.get("text"):
            attrs["text"] = rec["text"]
        if rec.get("count"):
            attrs["count"] = rec["count"]
        if rec.get("sci"):
            attrs["scientificName"] = rec["sci"]
        if rec.get("context"):
            attrs["context"] = rec["context"]
        ET.SubElement(block, f"{ns}NamedEntity", attrs)
    return block


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_pagexml(
    client: Any,
    xml_path: Path,
    out_path: Path,
    entity_types: Dict[str, str],
    model_id: str,
    scope_types: Optional[Dict[str, str]] = None,
    thinking_level: str = "medium",
    skip_existing: bool = True,
    verify_pass: bool = True,
    use_underline_hints: bool = True,
    include_object_regions: bool = True,
    include_image_regions: bool = True,
) -> Dict[str, Any]:
    """Run NER on one PAGE-XML file and write an annotated copy."""
    page_id = xml_path.stem
    if skip_existing and out_path.exists():
        return {"page": page_id, "status": "skipped", "n_entities": -1}

    try:
        tree, page_el, regions = _read_pagexml(xml_path)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to parse %s: %s", xml_path, exc)
        return {"page": page_id, "status": "parse_error", "n_entities": 0}

    skip_types = {"PageNumberRegion"}
    if not include_object_regions:
        skip_types.add("ObjectRegion")
    if not include_image_regions:
        skip_types.add("ImageRegion")

    # Backfill descriptions for image/object regions lacking <Unicode>.
    descriptions = _load_region_descriptions(xml_path)
    for r in regions:
        if r["type"] in ("ImageRegion", "ObjectRegion") and not r["text"]:
            desc = descriptions.get(r["id"])
            if desc and r["type"] not in skip_types:
                _ensure_unicode(r["element"], desc)
                r["text"] = desc

    region_views: List[Dict[str, Any]] = []
    prompt_pieces: List[str] = []
    hint_set: List[str] = []
    for r in regions:
        if r["type"] in skip_types or not r["text"]:
            continue
        clean, idx_map = _strip_with_map(r["text"])
        if not clean.strip():
            continue
        prompt_pieces.append(clean)
        if use_underline_hints:
            hint_set.extend(_extract_underline_hints(r["text"]))
        region_views.append({**r, "clean": clean, "idx_map": idx_map})

    combined = "\n\n".join(prompt_pieces)
    if not combined.strip():
        _add_named_entities_index(page_el, [])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_pretty(tree, out_path)
        return {"page": page_id, "status": "empty", "n_entities": 0}

    entities = perform_ner(
        client=client,
        text=combined,
        entity_types=entity_types,
        model_id=model_id,
        scope_types=scope_types,
        thinking_level=thinking_level,
        verify_pass=verify_pass,
        underline_hints=list(dict.fromkeys(hint_set)) if use_underline_hints else None,
        page_id=page_id,
    )

    flat_index: List[Dict[str, Any]] = []
    matched_all: set = set()
    for view in region_views:
        occs, matched = _occurrences_for_region(
            view["text"], view["clean"], view["idx_map"], entities)
        _attach_inline_tags(view["element"], occs)
        matched_all |= matched
        for occ in occs:
            flat_index.append({
                "regionRef": view["id"],
                "regionType": view["type"],
                "offset": occ["offset"],
                "length": occ["length"],
                "type": occ["type"],
                "scope": occ.get("scope"),
                "count": occ.get("count"),
                "sci": occ.get("sci"),
                "context": occ.get("context"),
                "text": occ.get("text"),
            })

    # entities the matcher could not locate anywhere — keep them visible
    unmatched = [e for e in entities if (e.text, e.entity_type) not in matched_all]
    for e in unmatched:
        flat_index.append({
            "regionRef": "", "regionType": "", "offset": -1, "length": -1,
            "type": e.entity_type, "scope": e.scope, "count": e.count,
            "sci": e.scientific_name, "context": e.context, "text": e.text,
        })
    if unmatched:
        log.info("%s: %d entit(y/ies) detected but not located in text: %s",
                 page_id, len(unmatched), ", ".join(e.text for e in unmatched[:8]))

    # wipe stale tags on skipped regions
    for r in regions:
        if r["type"] in skip_types:
            _attach_inline_tags(r["element"], [])

    _add_named_entities_index(page_el, flat_index)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_pretty(tree, out_path)

    n_located = sum(1 for r in flat_index if r["offset"] >= 0)
    return {"page": page_id, "status": "ok", "n_entities": n_located,
            "n_unmatched": len(unmatched)}


# ---------------------------------------------------------------------------
# Readers for downstream tools
# ---------------------------------------------------------------------------

def parse_named_entities(xml_path: Path) -> List[Dict[str, Any]]:
    if not xml_path.exists():
        return []
    root = ET.parse(xml_path).getroot()
    region_text: Dict[str, str] = {}
    for tr in root.iter():
        if _local(tr.tag) in _TEXTLIKE_TAGS:
            txt = _region_unicode(tr)
            if txt:
                region_text[tr.get("id", "")] = txt

    def _to_int(s):
        try:
            return int(s) if s not in (None, "") else None
        except (TypeError, ValueError):
            return None

    out: List[Dict[str, Any]] = []
    for el in root.iter():
        if _local(el.tag) != "NamedEntity":
            continue
        rid = el.get("regionRef") or ""
        offset = _to_int(el.get("offset"))
        length = _to_int(el.get("length"))
        text = el.get("text")
        if not text:
            if rid and offset is not None and length is not None:
                src = region_text.get(rid, "")
                raw_slice = src[offset:offset + length] if 0 <= offset < len(src) else ""
                text = " ".join(_strip_with_map(raw_slice)[0].split())
            else:
                text = ""
        out.append({
            "text": text,
            "entity_type": el.get("type", ""),
            "scope": el.get("scope") or None,
            "region_ref": rid or None,
            "region_type": el.get("regionType") or None,
            "offset": offset,
            "length": length,
            "count": el.get("count") or None,
            "scientific_name": el.get("scientificName") or None,
            "context": el.get("context") or None,
            "unmatched": el.get("unmatched") == "true",
        })
    return out


def parse_inline_custom_entities(xml_path: Path) -> List[Dict[str, Any]]:
    if not xml_path.exists():
        return []
    root = ET.parse(xml_path).getroot()
    out: List[Dict[str, Any]] = []
    for el in root.iter():
        if _local(el.tag) not in _TEXTLIKE_TAGS:
            continue
        rid = el.get("id", "")
        custom = el.get("custom", "")
        if not custom or INLINE_TAG_NAME not in custom:
            continue
        text = _region_unicode(el)
        for name, body in _parse_custom(custom):
            if name != INLINE_TAG_NAME:
                continue
            try:
                offset = int(body.get("offset", "-1"))
                length = int(body.get("length", "-1"))
            except ValueError:
                continue
            etext = text[offset:offset + length] if 0 <= offset < len(text) else ""
            out.append({
                "region_ref": rid,
                "offset": offset,
                "length": length,
                "entity_type": body.get("type", ""),
                "scope": body.get("scope") or None,
                "text": etext,
            })
    return out
