"""Compare old vs new NER output for one book (span-level, label-agnostic)."""
import csv, sys
from pathlib import Path
sys.path.insert(0, ".")
from journal_processor.ner_stage import parse_named_entities

IMG_TYPES = {"ImageRegion", "ObjectRegion", "GraphicRegion"}

def _located(ents):
    out = []
    for e in ents:
        if e.get("unmatched"): continue
        o, l = e.get("offset"), e.get("length")
        if o is None or l is None or l <= 0: continue
        out.append(e)
    return out

def _by_region(ents):
    d = {}
    for e in ents:
        d.setdefault(e.get("region_ref") or "", []).append(e)
    return d

def _overlap(a, b):
    return a["offset"] < b["offset"] + b["length"] and b["offset"] < a["offset"] + a["length"]

def compare_book(book, old_sub="pagexml_ner_old", new_sub="pagexml_ner", csv_out=None):
    book = Path(book)
    old_dir, new_dir = book / old_sub, book / new_sub
    new_pages = {p.name for p in new_dir.glob("*.xml")}
    old_pages = {p.name for p in old_dir.glob("*.xml")}
    pages = sorted(new_pages & old_pages)
    print(f"Pages: new={len(new_pages)}  old={len(old_pages)}  "
          f"compared (in both)={len(pages)}\n")
    if not pages:
        print("  No overlapping pages — nothing to compare.")
        return []
    tot_old = tot_new = old_only = new_only = matched_old = 0
    new_img = new_scope = new_count = new_sci = new_unloc = old_unloc = 0
    rows = []
    per_page = []
    for name in pages:
        new_e_all = parse_named_entities(new_dir / name)
        old_e_all = parse_named_entities(old_dir / name) if (old_dir / name).exists() else []
        new_unloc += sum(1 for e in new_e_all if e.get("unmatched"))
        old_unloc += sum(1 for e in old_e_all if e.get("unmatched"))
        new_e, old_e = _located(new_e_all), _located(old_e_all)
        tot_old += len(old_e); tot_new += len(new_e)
        for e in new_e:
            if (e.get("region_type") or "") in IMG_TYPES: new_img += 1
            if e.get("scope"): new_scope += 1
            if e.get("count"): new_count += 1
            if e.get("scientific_name"): new_sci += 1
        on, nn = _by_region(old_e), _by_region(new_e)
        page_old_only = page_new_only = page_match = 0
        for rid, olds in on.items():
            news = nn.get(rid, [])
            for a in olds:
                if any(_overlap(a, b) for b in news):
                    matched_old += 1; page_match += 1
                else:
                    old_only += 1; page_old_only += 1
                    rows.append([name, rid, e.get("region_type") or "", "OLD_ONLY",
                                 a.get("entity_type") or "", a.get("text") or ""])
        for rid, news in nn.items():
            olds = on.get(rid, [])
            for b in news:
                if not any(_overlap(b, a) for a in olds):
                    new_only += 1; page_new_only += 1
                    rows.append([name, rid, b.get("region_type") or "", "NEW_ONLY",
                                 b.get("entity_type") or "", b.get("text") or ""])
        per_page.append((name, len(old_e), len(new_e), page_match, page_old_only, page_new_only))

    print(f"{'':22}{'OLD':>8}{'NEW':>8}")
    print(f"{'located entities':22}{tot_old:>8}{tot_new:>8}   ({tot_new - tot_old:+d})")
    print(f"{'unlocated (surfaced)':22}{old_unloc:>8}{new_unloc:>8}")
    print()
    print("Span overlap (label-agnostic):")
    print(f"  old spans also found by new : {matched_old}/{tot_old}"
          + (f"  ({100*matched_old/tot_old:.0f}%)" if tot_old else ""))
    print(f"  OLD-only spans (check these): {old_only}")
    print(f"  NEW-only spans (recall gain): {new_only}")
    print()
    print("New-only capabilities (absent in old by design):")
    print(f"  entities in image/object regions: {new_img}")
    print(f"  entities carrying scope:          {new_scope}")
    print(f"  entities carrying count:          {new_count}")
    print(f"  entities carrying sci. name:      {new_sci}")
    if csv_out:
        with open(csv_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["page", "region", "region_type", "diff", "type", "text"])
            w.writerows(rows)
        print(f"\nDetailed per-span diff -> {csv_out}  ({len(rows)} rows)")
    return per_page

if __name__ == "__main__":
    compare_book(sys.argv[1], csv_out=(sys.argv[2] if len(sys.argv) > 2 else None))


# ── N-variant comparison (e.g. old vs new-medium vs new-low) ──────────────

def _variant_spans(book, sub, pages):
    d = Path(book) / sub
    by_pr, st = {}, dict(pages=0, located=0, unloc=0, img=0, scope=0, count=0, sci=0)
    for name in pages:
        f = d / name
        if not f.exists():
            continue
        st["pages"] += 1
        all_e = parse_named_entities(f)
        st["unloc"] += sum(1 for e in all_e if e.get("unmatched"))
        for e in _located(all_e):
            st["located"] += 1
            if (e.get("region_type") or "") in IMG_TYPES: st["img"] += 1
            if e.get("scope"): st["scope"] += 1
            if e.get("count"): st["count"] += 1
            if e.get("scientific_name"): st["sci"] += 1
            by_pr.setdefault((name, e.get("region_ref") or ""), []).append(e)
    return by_pr, st


def _coverage(a_by, b_by):
    tot = cov = 0
    for key, alist in a_by.items():
        blist = b_by.get(key, [])
        for a in alist:
            tot += 1
            if any(_overlap(a, b) for b in blist):
                cov += 1
    return cov, tot


def compare_variants(book, variants, ref=None, pages=None, csv_out=None):
    """variants: dict {label: subdir}. Compares all on pages present in every dir.

    Prints per-variant totals and a coverage matrix (row covered by column,
    label-agnostic span overlap). Diagonal is 100%. Reading example: the
    (new_low, new_medium) cell = % of low's spans that medium also found.
    """
    book = Path(book)
    labels = list(variants)
    ref = ref or labels[0]
    page_sets = {lab: {p.name for p in (book / sub).glob("*.xml")}
                 for lab, sub in variants.items()}
    common = sorted(pages) if pages is not None else sorted(set.intersection(*page_sets.values()))
    print("Variants & page coverage:")
    for lab, sub in variants.items():
        print(f"  {lab:22} {sub:24} pages={len(page_sets[lab])}")
    print(f"  -> compared on {len(common)} page(s) present in all\n")
    if not common:
        print("No common pages."); return {}

    spans, stats = {}, {}
    for lab, sub in variants.items():
        spans[lab], stats[lab] = _variant_spans(book, sub, common)

    print("Per-variant totals (common pages):")
    print(f"  {'variant':22}{'located':>8}{'unloc':>7}{'img':>5}{'scope':>7}{'count':>7}{'sci':>5}")
    for lab in labels:
        s = stats[lab]
        print(f"  {lab:22}{s['located']:>8}{s['unloc']:>7}{s['img']:>5}"
              f"{s['scope']:>7}{s['count']:>7}{s['sci']:>5}")
    print()

    print("Coverage matrix — % of ROW's spans also found by COLUMN:")
    print("  " + " " * 22 + "".join(f"{l[:11]:>12}" for l in labels))
    for a in labels:
        cells = ""
        for b in labels:
            cov, tot = _coverage(spans[a], spans[b])
            cells += f"{(100 * cov / tot if tot else 0):>11.0f}%"
        print(f"  {a:22}{cells}")
    print()

    rows = []
    for lab in labels:
        if lab == ref:
            continue
        for key, reflist in spans[ref].items():
            for r in reflist:
                if not any(_overlap(r, v) for v in spans[lab].get(key, [])):
                    rows.append([key[0], key[1], r.get("region_type") or "", lab,
                                 f"MISSED_vs_{ref}", r.get("entity_type") or "", r.get("text") or ""])
        for key, vlist in spans[lab].items():
            for v in vlist:
                if not any(_overlap(v, r) for r in spans[ref].get(key, [])):
                    rows.append([key[0], key[1], v.get("region_type") or "", lab,
                                 f"ADDED_vs_{ref}", v.get("entity_type") or "", v.get("text") or ""])
    if csv_out:
        import csv as _csv
        with open(csv_out, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["page", "region", "region_type", "variant", "diff", "type", "text"])
            w.writerows(rows)
        print(f"Per-span diffs vs '{ref}' -> {csv_out}  ({len(rows)} rows)")
    return stats


# ── token-level comparison: which tokens each variant tagged ──────────────
from xml.etree import ElementTree as _ET
from journal_processor.ner_stage import (
    _strip_with_map as _swm, _region_unicode as _ru, _local as _lc,
    _TEXTLIKE_TAGS as _TLT,
)
from journal_processor.output_bio import _TOKEN_RE as _TRE, _raw_span_to_clean as _r2c


def _page_token_labels(xml_path):
    """{(region_id, token_index): (token, type)} for one annotated page."""
    if not Path(xml_path).exists():
        return {}
    ents = [e for e in parse_named_entities(xml_path)
            if not e.get("unmatched") and e.get("offset") is not None
            and e.get("length") and e["length"] > 0]
    by_region = {}
    for e in ents:
        by_region.setdefault(e.get("region_ref") or "", []).append(e)
    root = _ET.parse(xml_path).getroot()
    out = {}
    for region in root.iter():
        if _lc(region.tag) not in _TLT:
            continue
        raw = _ru(region)
        if not raw:
            continue
        rid = region.get("id", "")
        clean, raw_idx = _swm(raw)
        toks = [(m.group(0), m.start(), m.end()) for m in _TRE.finditer(clean)]
        labels = ["O"] * len(toks)
        for e in by_region.get(rid, []):
            cs, ce = _r2c(raw_idx, e["offset"], e["length"])
            for ti, (_, ts, te) in enumerate(toks):
                if ts < ce and te > cs:
                    labels[ti] = e.get("entity_type") or "?"
        for ti, (tok, _, _) in enumerate(toks):
            out[(rid, ti)] = (tok, labels[ti])
    return out


def _abbr(t):
    return "·" if t in ("O", "", None) else t[:2]


def token_table(book, variants, pages=None, csv_out=None, show="disagree", max_print=45):
    """Per-token type label for every variant, side by side.

    show: 'disagree' (tokens where variants differ), 'tagged' (any tagged it),
    or 'all'. Writes a wide CSV: one column per variant.
    """
    book = Path(book)
    labels = list(variants)
    page_sets = {lab: {p.name for p in (book / sub).glob("*.xml")}
                 for lab, sub in variants.items()}
    common = sorted(pages) if pages is not None else sorted(set.intersection(*page_sets.values()))
    print(f"Token comparison on {len(common)} page(s); variants: {', '.join(labels)}\n")
    if not common:
        print("No common pages."); return []

    merged = {}
    for lab, sub in variants.items():
        for name in common:
            for (rid, ti), (tok, ty) in _page_token_labels(book / sub / name).items():
                rec = merged.setdefault((name, rid, ti), {"token": tok, "labs": {}})
                rec["labs"][lab] = ty

    nvar = len(labels)
    n_all = n_none = n_dis = 0
    rows = []
    for key in sorted(merged):
        rec = merged[key]
        types = [rec["labs"].get(lab, "O") for lab in labels]
        tagged = [t for t in types if t != "O"]
        if not tagged:
            n_none += 1
        elif len(tagged) == nvar and len(set(tagged)) == 1:
            n_all += 1
        else:
            n_dis += 1
        rows.append((key, rec["token"], types, len(tagged)))

    print(f"tokens compared:       {len(rows)}")
    print(f"  tagged by all (same): {n_all}")
    print(f"  tagged by none:       {n_none}")
    print(f"  disagreement:         {n_dis}\n")

    def _is_dis(r):
        tg = [t for t in r[2] if t != "O"]
        return tg and not (len(tg) == nvar and len(set(tg)) == 1)
    sel = (rows if show == "all"
           else [r for r in rows if r[3] >= 1] if show == "tagged"
           else [r for r in rows if _is_dis(r)])

    print(f"{'token':16}" + "".join(f"{l[:7]:>9}" for l in labels))
    for (key, tok, types, nt) in sel[:max_print]:
        print(f"{tok[:16]:16}" + "".join(f"{_abbr(t):>9}" for t in types))
    if len(sel) > max_print:
        print(f"… {len(sel) - max_print} more (see CSV)")

    if csv_out:
        import csv as _csv
        with open(csv_out, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["page", "region", "token_index", "token", "n_tagged"] + labels)
            for (key, tok, types, nt) in sel:
                w.writerow([key[0], key[1], key[2], tok, nt] + types)
        print(f"\n-> {csv_out}  ({len(sel)} rows)")
    return rows
