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
    pages = sorted(p.name for p in new_dir.glob("*.xml"))
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

    print(f"Pages compared: {len(pages)}\n")
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
