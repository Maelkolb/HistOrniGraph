#!/usr/bin/env python3
"""Build a metadata-rich text corpus from all processed Laubmann volumes.

Reads regions/*.json from every Laubmann_NN_gemini output directory and
reconstructs, per page, the transcribed regions in reading order together with
their metadata (volume, page id / image id, scan number, page side, region id,
region type, reading order, line count, detected page number, region-crop path).

It also detects the typical start of a Laubmann journal entry — a line-initial
date (with a year) whose location may be underlined or plain, e.g.

    7. April 1917. <u>München</u>.
    30. Juli 1960. München.
    26. I. 48 Karlsfeld

and segments each volume's reading-order stream into individual entries.

Usage:
    python build_corpus.py
    python build_corpus.py --output-base "D:/some/other/path"
    python build_corpus.py --per-volume --include-nontext
    python build_corpus.py --volumes 1 5 9

Output (written to corpus/ next to this script):
    corpus/corpus.md      — metadata-rich Markdown reconstruction (all volumes)
    corpus/corpus.json    — structured page-by-page corpus + per-region entries
    corpus/corpus.txt     — flat text with page/entry markers (grep-friendly)
    corpus/entries.jsonl  — one detected entry per line (full text)
    corpus/entries.csv    — entry index (cleaned text + preview)
    corpus/by_volume/Laubmann_NN.md   — only with --per-volume
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OUTPUT_BASE = Path(r"G:\My Drive\HistOrniGraph_output")
CORPUS_DIR  = Path(__file__).parent / "corpus"

BODY_TEXT_TYPES  = {"ParagraphRegion", "ListRegion", "FootnoteRegion", "TableRegion"}
ENTRY_SCAN_TYPES = {"ParagraphRegion", "ListRegion"}
NONTEXT_TYPES    = {"ImageRegion", "ObjectRegion", "MarginaliaRegion"}
PAGE_META_TYPES  = {"PageNumberRegion"}


# ── Entry-start detection ────────────────────────────────────────────────────
# A Laubmann entry header is a LINE-INITIAL date (with a year), e.g.
#
#     7. April 1917. <u>München</u>.        ← location underlined
#     30. Juli 1960. München.               ← location plain
#     26. I. 48 Karlsfeld                   ← roman month, 2-digit year, plain loc
#     10. August 1938.                      ← date only (no location on the line)
#
# The date is the reliable anchor (almost never underlined); the location may be
# underlined or plain.  A date inside running prose is NOT a header — those are
# rejected because a lowercase word (or "," / ";") follows the date.

_MONTH_NAME = (
    r"J[äa]n(?:ner|uar|\.)?|Feb(?:ruar|\.)?|M[äa]r(?:z|\.)?|Apr(?:il|\.)?|Mai|"
    r"Jun[i]?|Jul[i]?|Aug(?:ust|\.)?|Sept?(?:ember|\.)?|Okt(?:ober|\.)?|"
    r"Nov(?:ember|\.)?|Dez(?:ember|\.)?"
)
# Laubmann (or the transcriber) sometimes used English/Latin month spellings,
# e.g. "29. October 1934", "28. December 1935".
_MONTH_EN = (
    r"January|February|March|May|June|July|"
    r"September|October|November|December|"
    r"Oct\.?|Nov\.?|Dec\.?|Sept?\.?"
)
_MONTH_ROMAN = r"VIII|XII|VII|III|XI|IX|IV|VI|II|X|V|I"
_MONTH = rf"(?:{_MONTH_NAME}|{_MONTH_EN}|(?:{_MONTH_ROMAN})\.?)"
_DAY   = r"\d{1,2}\.?"
_YEAR  = r"(?:1[5-9]\d{2}|20\d{2}|\d{2})"
# Date with a required year (strict).  ``[\s.]*`` lets a stray period sit between
# month and year, e.g. "21. August. 1917".
_DATEY = rf"{_DAY}\s*{_MONTH}[\s.]*{_YEAR}"
# Date with an optional year (used only with --loose; lower precision).
_DATE  = rf"{_DAY}\s*{_MONTH}[\s.]*(?:{_YEAR})?"
# A header line may be prefixed with a list bullet ("- 22. April 1938 …") because
# the transcriber rendered some entry starts as ListRegion items.
_BULLET = r"(?:[-*\u2022]\s*)?"


def _header_re(year_required: bool) -> "re.Pattern":
    date = _DATEY if year_required else _DATE
    return re.compile(
        rf"^[ \t]*{_BULLET}(?:<u>[ \t]*)?(?P<date>(?i:{date}))[ \t]*(?:</u>)?[ \t]*"
        rf"[.:]?(?P<rest>[^\n]*)$",
        re.MULTILINE,
    )


HEADER_STRICT = _header_re(True)
HEADER_LOOSE  = _header_re(False)

_U_FIRST  = re.compile(r"^<u>\s*(.*?)\s*</u>")
_WEEKDAY  = re.compile(
    r"(?:Montag|Dienstag|Mittwoch|Donnerstag|Freitag|Samstag|Sonnabend|Sonntag)",
    re.IGNORECASE,
)
_WD_LEAD  = re.compile(rf"^{_WEEKDAY.pattern}\b\.?,?\s*", re.IGNORECASE)

_DATE_PARSE = re.compile(
    rf"^\s*(?P<d>\d{{1,2}})\.?\s*(?P<m>{_MONTH})[\s.]*(?P<y>{_YEAR})?", re.IGNORECASE
)

_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6,
          "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12}
_MNAME = {"jan": 1, "jän": 1, "feb": 2, "mar": 3, "mär": 3, "apr": 4,
          "mai": 5, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
          "okt": 10, "oct": 10, "nov": 11, "dez": 12, "dec": 12}

_MARKUP_RE   = re.compile(r"</?(?:u|sup|sub|b|i|em|strong)\s*>", re.IGNORECASE)
_DEHYPHEN_RE = re.compile(r"(\w)-[ \t]*\n+[ \t]*([a-zäöüß])")
_WS_RE       = re.compile(r"\s*\n+\s*")


def _month_to_num(raw: str) -> Optional[int]:
    s = raw.strip().rstrip(".").lower()
    if s in _ROMAN:
        return _ROMAN[s]
    if s.isdigit():
        n = int(s)
        return n if 1 <= n <= 12 else None
    return _MNAME.get(s[:3])


def normalize_date(date_raw: str) -> Tuple[Optional[str], Optional[int]]:
    """Best-effort ISO date.  2-digit years are read as 19xx — Laubmann's diary
    is entirely 20th-century, so "48" → 1948, "03" → 1903."""
    m = _DATE_PARSE.match(date_raw)
    if not m:
        return None, None
    day = int(m.group("d"))
    mon = _month_to_num(m.group("m") or "")
    yraw = m.group("y") or ""
    if len(yraw) == 4:
        year: Optional[int] = int(yraw)
    elif len(yraw) == 2:
        year = 1900 + int(yraw)
    else:
        year = None
    if year and mon and 1 <= day <= 31:
        return f"{year:04d}-{mon:02d}-{day:02d}", year
    return None, year


def _extract_location(rest: str) -> Optional[str]:
    """Pull the location out of the text after a date header.

    Returns "" for a date-only header, or None when the line is actually prose
    (a lowercase word / "," / ";" follows the date) and must be rejected.
    """
    r = rest.strip().lstrip(".:").strip()
    if not r:
        return ""
    if r[0].islower() or r[0] in ",;":
        return None
    r = _WD_LEAD.sub("", r).strip()          # Laubmann sometimes notes the weekday
    if not r or r[0].islower():
        return ""
    m = _U_FIRST.match(r)
    if m:
        loc = m.group(1)
    else:
        loc = re.split(r"\.(?:\s|$)", r, maxsplit=1)[0]
        loc = re.split(r"\s{2,}|\t", loc)[0]
    loc = loc.strip().strip('"').rstrip(".").strip()
    if loc and (not loc[0].isalpha() or _WEEKDAY.fullmatch(loc)):
        return ""
    return loc


def find_entry_starts(text: str, loose: bool = False) -> List[Dict[str, Any]]:
    rx = HEADER_LOOSE if loose else HEADER_STRICT
    out: List[Dict[str, Any]] = []
    for m in rx.finditer(text):
        loc = _extract_location(m.group("rest"))
        if loc is None:
            continue
        date = re.sub(r"\s+", " ", m.group("date").strip())
        rest_l = m.group("rest").strip().lstrip(".:").strip()
        variant = ("date-only" if not loc
                   else "underlined" if rest_l.startswith("<u>") else "plain")
        dn, yr = normalize_date(date)
        out.append({"offset": m.start(), "end": m.end(), "date": date,
                    "location": loc, "date_norm": dn, "year": yr,
                    "variant": variant})
    return out


def strip_markup(text: str) -> str:
    t = _MARKUP_RE.sub("", text)
    t = _DEHYPHEN_RE.sub(r"\1\2", t)          # join words split across line breaks
    t = _WS_RE.sub(" ", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()


# ── Page-id parsing / sorting ────────────────────────────────────────────────

def _parse_pid(stem: str) -> Tuple[int, str]:
    side = ""
    base = stem
    if len(stem) > 2 and stem[-2] == "_" and stem[-1] in "LRlr":
        side = stem[-1].upper()
        base = stem[:-2]
    nums = re.findall(r"\d+", base)
    scan = int(nums[-1]) if nums else 0
    return scan, side


def _page_sort_key(page_id: str) -> Tuple[int, int]:
    scan, side = _parse_pid(page_id)
    return (scan, {"": 0, "L": 0, "R": 1}.get(side, 0))


# ── Volume loading ───────────────────────────────────────────────────────────

def _region_text(r: Dict[str, Any]) -> str:
    tr = r.get("transcription") or {}
    if tr.get("status") != "success" or tr.get("skipped"):
        return ""
    rtype = r.get("type")
    if rtype in ("ImageRegion", "ObjectRegion"):
        desc = tr.get("description") or tr.get("text") or ""
        vis = tr.get("visible_text", "")
        if vis and vis.lower() != "none":
            desc = f"{desc}\nText: {vis}".strip()
        return desc.strip()
    return (tr.get("text") or "").strip()


def _page_number(regions: List[Dict[str, Any]]) -> str:
    for r in regions:
        if r.get("type") == "PageNumberRegion":
            pn = r.get("page_number") or (r.get("transcription") or {}).get("text", "")
            if pn:
                return str(pn).strip()
    return ""


def load_volume(vol_dir: Path, include_nontext: bool) -> Tuple[int, List[Dict[str, Any]]]:
    vol_num = int(vol_dir.name.split("_")[1])
    regions_dir = vol_dir / "regions"
    if not regions_dir.exists():
        return vol_num, []

    json_files = sorted(regions_dir.glob("*.json"), key=lambda p: _page_sort_key(p.stem))
    pages: List[Dict[str, Any]] = []

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [!] Cannot read {jf.name}: {exc}")
            continue
        if not isinstance(data, list):
            continue

        page_id = jf.stem
        scan, _ = _parse_pid(page_id)
        page_num = _page_number(data)

        regions: List[Dict[str, Any]] = []
        for r in sorted(data, key=lambda r: r.get("reading_order", 99)):
            rtype = r.get("type", "")
            if rtype in PAGE_META_TYPES:
                continue
            if r.get("insert_state") == "folded":
                continue
            is_body = rtype in BODY_TEXT_TYPES
            if not is_body and not (include_nontext and rtype in NONTEXT_TYPES):
                continue
            text = _region_text(r)
            if not text:
                continue
            rid = r.get("id", "")
            regions.append({
                "id": rid,
                "type": rtype,
                "reading_order": r.get("reading_order", len(regions) + 1),
                "page_side": r.get("page_side", ""),
                "line_count": r.get("line_count"),
                "crop": f"regions/{page_id}/{rid}_{rtype}.png",
                "text": text,
                "is_body": is_body,
                "scan_entries": rtype in ENTRY_SCAN_TYPES,
            })
        if regions:
            pages.append({
                "page_id": page_id,
                "image": f"{page_id}.png",
                "scan": scan,
                "page_number": page_num,
                "regions": regions,
            })
    return vol_num, pages


# ── Entry segmentation over a volume's reading-order stream ──────────────────

def build_stream(pages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    chunks: List[str] = []
    units: List[Dict[str, Any]] = []
    pos = 0
    for page in pages:
        for reg in page["regions"]:
            if not reg["scan_entries"]:
                continue
            text = reg["text"]
            units.append({
                "page_id": page["page_id"], "image": page["image"],
                "scan": page["scan"], "region_id": reg["id"],
                "region_type": reg["type"], "reading_order": reg["reading_order"],
                "start": pos, "end": pos + len(text),
            })
            chunks.append(text)
            pos += len(text) + 2
    return "\n\n".join(chunks), units


def _unit_for_offset(units: List[Dict[str, Any]], offset: int) -> Optional[Dict[str, Any]]:
    for u in units:
        if u["start"] <= offset < u["end"] + 2:
            return u
    return units[-1] if units else None


def segment_entries(vol_num: int, pages: List[Dict[str, Any]],
                    loose: bool = False) -> List[Dict[str, Any]]:
    vol_text, units = build_stream(pages)
    if not units:
        return []
    starts = find_entry_starts(vol_text, loose=loose)
    entries: List[Dict[str, Any]] = []
    for i, h in enumerate(starts):
        seg_start = h["offset"]
        seg_end = starts[i + 1]["offset"] if i + 1 < len(starts) else len(vol_text)
        raw = vol_text[seg_start:seg_end].strip()
        clean = strip_markup(raw)
        u = _unit_for_offset(units, seg_start) or {}
        entries.append({
            "entry_id": f"L{vol_num:02d}-e{i + 1:04d}",
            "volume": vol_num,
            "scan": u.get("scan"),
            "page_id": u.get("page_id", ""),
            "image": u.get("image", ""),
            "region_id": u.get("region_id", ""),
            "region_type": u.get("region_type", ""),
            "reading_order": u.get("reading_order"),
            "date_raw": h["date"],
            "date_norm": h.get("date_norm"),
            "year": h.get("year"),
            "location_raw": h["location"],
            "variant": h["variant"],
            "stream_start": seg_start,
            "stream_end": seg_end,
            "n_chars": len(clean),
            "n_words": len(clean.split()),
            "text_raw": raw,
            "text_clean": clean,
        })
    return entries


# ── Markdown rendering ───────────────────────────────────────────────────────

# Escape a leading "12." so Markdown viewers don't turn date lines into an
# ordered list (cosmetic; only affects corpus.md, not the data artifacts).
_MD_LIST_RE = re.compile(r"(?m)^([ \t]*)(\d{1,2})\.(?=\s)")


def _md_safe(text: str) -> str:
    return _MD_LIST_RE.sub(r"\1\2\\.", text)


def _entry_label(h: Dict[str, Any]) -> str:
    loc = f" · {h['location']}" if h["location"] else ""
    tag = f" `{h['date_norm']}`" if h.get("date_norm") else ""
    return f"{h['date']}{loc}{tag}"


def render_volume_md(vol_num: int, pages: List[Dict[str, Any]], loose: bool) -> str:
    out: List[str] = [f"# Laubmann · Vol. {vol_num:02d}", ""]
    last_entry: Optional[Dict[str, Any]] = None     # running entry across pages
    for page in pages:
        pid, scan, pnum = page["page_id"], page["scan"], page["page_number"]
        head = f"## Vol. {vol_num:02d} · scan {scan:04d}"
        if pnum:
            head += f" · p. {pnum}"
        out.append(
            f"<!-- page volume={vol_num} page_id={pid} image={page['image']} "
            f"scan={scan} page_number={pnum or ''} regions={len(page['regions'])} -->"
        )
        out.append(head)
        out.append(f"`{pid}`")
        out.append("")
        for reg in page["regions"]:
            lc = reg["line_count"]
            starts = find_entry_starts(reg["text"], loose=loose) if reg["scan_entries"] else []
            out.append(
                f"<!-- region id={reg['id']} type={reg['type']} "
                f"order={reg['reading_order']} side={reg['page_side'] or ''} "
                f"lines={lc if lc is not None else ''} entries={len(starts)} "
                f"crop={reg['crop']} -->"
            )
            if starts and starts[0]["offset"] == 0:
                out.append(f"**⮞ Entry — {_entry_label(starts[0])}**")
                if len(starts) > 1:
                    extra = "; ".join(_entry_label(h) for h in starts[1:])
                    out.append(f"*[+ further entry start(s): {extra}]*")
            elif starts:
                if last_entry:
                    out.append(f"*(…continued from {_entry_label(last_entry)})*")
                joined = "; ".join(_entry_label(h) for h in starts)
                out.append(f"*[entry start(s) mid-region: {joined}]*")
            elif reg["scan_entries"] and last_entry:
                out.append(f"*(…continued from {_entry_label(last_entry)})*")
            if starts:
                last_entry = starts[-1]
            out.append(_md_safe(reg["text"]))
            out.append("")
    return "\n".join(out)


def render_txt(vol_num: int, pages: List[Dict[str, Any]], loose: bool) -> List[str]:
    lines: List[str] = []
    for page in pages:
        lines.append(f"\n=== Vol.{vol_num:02d}  scan {page['scan']:04d}  {page['page_id']} ===")
        for reg in page["regions"]:
            if reg["scan_entries"]:
                for h in find_entry_starts(reg["text"], loose=loose):
                    lines.append(f"--- ENTRY  {h['date']}  |  {h['location']} ---")
            lines.append(reg["text"])
            lines.append("")
    return lines


# ── Driver ───────────────────────────────────────────────────────────────────

def build(output_base: Path, corpus_dir: Path, per_volume: bool,
          include_nontext: bool, loose: bool, only_volumes: Optional[List[int]]) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)

    vol_dirs = sorted(
        [d for d in output_base.iterdir()
         if d.is_dir() and d.name.startswith("Laubmann_") and d.name.endswith("_gemini")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if only_volumes:
        vol_dirs = [d for d in vol_dirs if int(d.name.split("_")[1]) in set(only_volumes)]
    print(f"Found {len(vol_dirs)} volume(s).")

    corpus_json: List[Dict[str, Any]] = []
    all_entries: List[Dict[str, Any]] = []
    md_parts: List[str] = []
    txt_lines: List[str] = []
    by_vol_dir = corpus_dir / "by_volume"
    if per_volume:
        by_vol_dir.mkdir(exist_ok=True)

    total_pages = total_regions = 0

    for vi, vol_dir in enumerate(vol_dirs, 1):
        vol_num, pages = load_volume(vol_dir, include_nontext)
        if not pages:
            print(f"  [{vi:02d}/{len(vol_dirs)}] Vol.{vol_num:02d}  [!] no usable regions, skipping.")
            continue

        entries = segment_entries(vol_num, pages, loose=loose)
        all_entries.extend(entries)

        for page in pages:
            page_regions = []
            for reg in page["regions"]:
                starts = find_entry_starts(reg["text"], loose=loose) if reg["scan_entries"] else []
                page_regions.append({
                    "id": reg["id"], "type": reg["type"],
                    "reading_order": reg["reading_order"],
                    "page_side": reg["page_side"], "line_count": reg["line_count"],
                    "crop": reg["crop"], "text": reg["text"],
                    "entry_starts": [
                        {"date": h["date"], "location": h["location"],
                         "date_norm": h.get("date_norm"), "offset": h["offset"]}
                        for h in starts
                    ],
                })
            corpus_json.append({
                "volume": vol_num, "page_id": page["page_id"], "image": page["image"],
                "scan": page["scan"], "page_number": page["page_number"],
                "regions": page_regions,
            })

        md = render_volume_md(vol_num, pages, loose)
        md_parts.append(md)
        txt_lines.extend(render_txt(vol_num, pages, loose))
        if per_volume:
            (by_vol_dir / f"Laubmann_{vol_num:02d}.md").write_text(md, encoding="utf-8")

        n_regions = sum(len(p["regions"]) for p in pages)
        total_pages += len(pages)
        total_regions += n_regions
        print(f"  [{vi:02d}/{len(vol_dirs)}] Vol.{vol_num:02d}  "
              f"{len(pages)} pages, {n_regions} regions, {len(entries)} entries")

    (corpus_dir / "corpus.json").write_text(
        json.dumps(corpus_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (corpus_dir / "corpus.md").write_text("\n\n".join(md_parts), encoding="utf-8")
    (corpus_dir / "corpus.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    with (corpus_dir / "entries.jsonl").open("w", encoding="utf-8") as fh:
        for e in all_entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    csv_cols = ["entry_id", "volume", "scan", "page_id", "image", "region_id",
                "region_type", "reading_order", "date_raw", "date_norm", "year",
                "location_raw", "variant", "n_chars", "n_words", "preview", "text_clean"]
    with (corpus_dir / "entries.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        for e in all_entries:
            row = dict(e)
            row["preview"] = (e["text_clean"][:120] + "…") if len(e["text_clean"]) > 120 else e["text_clean"]
            w.writerow(row)

    print(f"\nCorpus: {total_pages} pages, {total_regions} regions, {len(all_entries)} entries")
    for name in ("corpus.md", "corpus.json", "corpus.txt", "entries.jsonl", "entries.csv"):
        print(f"  → {corpus_dir / name}")
    if per_volume:
        print(f"  → {by_vol_dir}/Laubmann_NN.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-base", type=Path, default=OUTPUT_BASE,
                    help="Root dir containing Laubmann_NN_gemini folders")
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR,
                    help="Where to write the corpus (default: corpus/ next to this script)")
    ap.add_argument("--per-volume", action="store_true",
                    help="Also write one Markdown file per volume under corpus/by_volume/")
    ap.add_argument("--include-nontext", action="store_true",
                    help="Include Image/Object/Marginalia region descriptions in the body")
    ap.add_argument("--loose", action="store_true",
                    help="Also detect date headers that omit the year "
                         "(higher recall, lower precision)")
    ap.add_argument("--volumes", type=int, nargs="*", default=None,
                    help="Restrict to specific volume numbers, e.g. --volumes 1 5 9")
    args = ap.parse_args()
    build(args.output_base, args.corpus_dir, args.per_volume,
          args.include_nontext, args.loose, args.volumes)
