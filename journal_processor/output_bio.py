"""
BIO CSV export
==============
Turns annotated ``pagexml_ner/*.xml`` files into token-level BIO CSV, ready for
training a multitask token classifier (parallel Type + Scope heads, the same
shape as the Marlitt model).

Each row is one token of one region's cleaned text (markup stripped, soft
hyphens joined).  Regions are separated by a blank line (CoNLL style).

Columns
-------
doc_id, page_id, region_id, region_type, token_index, token, char_start,
char_end, bio_type, bio_scope

  bio_type  ∈ {O, B-Tier, I-Tier, B-Ort, …}
  bio_scope ∈ {O, B-Singular, I-Singular, …}   (same spans as bio_type)

Entity-level attributes (count, scientific_name) are intentionally NOT in the
token CSV — they live in the PAGE-XML ``<NamedEntities>`` index for the KG.
Detected-but-unlocated entities (offset = -1) are skipped here; see the index.
"""

from __future__ import annotations

import bisect
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from xml.etree import ElementTree as ET

from .ner_stage import (
    INLINE_TAG_NAME, _local, _parse_custom, _region_type, _region_unicode,
    _strip_with_map, _TEXTLIKE_TAGS,
)

CSV_HEADER = [
    "doc_id", "page_id", "region_id", "region_type", "token_index",
    "token", "char_start", "char_end", "bio_type", "bio_scope",
]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def _raw_span_to_clean(raw_idx_of_clean: List[int],
                       raw_off: int, raw_len: int) -> Tuple[int, int]:
    """Map a raw [off, off+len) span to a clean [start, end) span."""
    raw_end = raw_off + raw_len
    lo = bisect.bisect_left(raw_idx_of_clean, raw_off)
    hi = bisect.bisect_left(raw_idx_of_clean, raw_end)
    return lo, max(lo, hi)


def _region_spans(region_el: ET.Element, clean_len_idx: List[int]
                  ) -> List[Dict[str, Any]]:
    """Inline entity spans for one region, mapped to clean-text coordinates."""
    spans: List[Dict[str, Any]] = []
    for name, body in _parse_custom(region_el.get("custom", "")):
        if name != INLINE_TAG_NAME:
            continue
        try:
            off = int(body.get("offset", "-1"))
            length = int(body.get("length", "-1"))
        except ValueError:
            continue
        if off < 0 or length <= 0:
            continue
        cs, ce = _raw_span_to_clean(clean_len_idx, off, length)
        if ce > cs:
            spans.append({"start": cs, "end": ce,
                          "type": body.get("type", ""),
                          "scope": body.get("scope", "")})
    return spans


def pagexml_to_bio(xml_path: Path, doc_id: str) -> List[List[Any]]:
    """Return BIO rows for one page (region-separated by an empty list)."""
    if not xml_path.exists():
        return []
    root = ET.parse(xml_path).getroot()
    page_id = xml_path.stem
    rows: List[List[Any]] = []

    for region in root.iter():
        if _local(region.tag) not in _TEXTLIKE_TAGS:
            continue
        raw = _region_unicode(region)
        if not raw:
            continue
        clean, raw_idx_of_clean = _strip_with_map(raw)
        if not clean.strip():
            continue
        rid = region.get("id", "")
        rtype = _region_type(region)
        spans = _region_spans(region, raw_idx_of_clean)

        tokens = [(m.group(0), m.start(), m.end())
                  for m in _TOKEN_RE.finditer(clean)]
        bio_type = ["O"] * len(tokens)
        bio_scope = ["O"] * len(tokens)

        for sp in spans:
            first = True
            for ti, (_, ts, te) in enumerate(tokens):
                if ts < sp["end"] and te > sp["start"]:  # token overlaps span
                    pfx = "B-" if first else "I-"
                    bio_type[ti] = pfx + sp["type"]
                    bio_scope[ti] = (pfx + sp["scope"]) if sp["scope"] else "O"
                    first = False

        emitted = False
        for ti, (tok, ts, te) in enumerate(tokens):
            rows.append([doc_id, page_id, rid, rtype, ti, tok, ts, te,
                         bio_type[ti], bio_scope[ti]])
            emitted = True
        if emitted:
            rows.append([])  # region separator
    return rows


def _list_ner_pages(book_dir: Path, subdir: str = "pagexml_ner") -> List[Path]:
    src = book_dir / subdir
    if not src.is_dir():
        return []
    def _key(p: Path):
        return [int(c) if c.isdigit() else c.lower()
                for c in re.split(r"(\d+)", p.stem)]
    return sorted((p for p in src.iterdir() if p.suffix.lower() == ".xml"), key=_key)


def export_book_bio(book_dir: Path, out_csv: Path,
                    subdir: str = "pagexml_ner") -> int:
    """Write one CSV for every annotated page of a book. Returns token count."""
    doc_id = book_dir.name
    pages = _list_ner_pages(book_dir, subdir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_tokens = 0
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for page in pages:
            for row in pagexml_to_bio(page, doc_id):
                w.writerow(row)
                if row:
                    n_tokens += 1
    return n_tokens


def export_corpus_bio(book_dirs: List[Path], out_csv: Path,
                      subdir: str = "pagexml_ner") -> int:
    """Write a single merged CSV across several books. Returns token count."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_tokens = 0
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        for book_dir in book_dirs:
            for page in _list_ner_pages(book_dir, subdir):
                for row in pagexml_to_bio(page, book_dir.name):
                    w.writerow(row)
                    if row:
                        n_tokens += 1
    return n_tokens
