#!/usr/bin/env python3
"""Build a text corpus from all processed Laubmann volumes.

Reads regions/*.json from every Laubmann_NN_gemini output directory.
Excludes: ImageRegion, ObjectRegion, MarginaliaRegion, PageNumberRegion.

Usage:
    python build_corpus.py
    python build_corpus.py --output-base "D:/some/other/path"

Output (written to corpus/ next to this script):
    corpus/corpus.json   — structured page-by-page corpus
    corpus/corpus.txt    — flat text with page markers
"""

import argparse
import json
from pathlib import Path

OUTPUT_BASE = Path(r"G:\My Drive\HistOrniGraph_output")
CORPUS_DIR  = Path(__file__).parent / "corpus"

EXCLUDED = {"ImageRegion", "ObjectRegion", "MarginaliaRegion", "PageNumberRegion"}


def _page_sort_key(path: Path) -> tuple:
    """Sort region JSONs by scan number then L before R."""
    parts = path.stem.rsplit("_", 2)
    try:
        num = int(parts[-2])
    except (ValueError, IndexError):
        num = 0
    side = 0 if parts[-1].upper() == "L" else 1
    return (num, side)


def build(output_base: Path) -> None:
    CORPUS_DIR.mkdir(exist_ok=True)

    volumes = sorted(
        [
            d for d in output_base.iterdir()
            if d.is_dir()
            and d.name.startswith("Laubmann_")
            and d.name.endswith("_gemini")
        ],
        key=lambda p: int(p.name.split("_")[1]),
    )
    print(f"Found {len(volumes)} volume(s).")

    corpus = []
    total_regions = 0

    for vol_idx, vol_dir in enumerate(volumes, 1):
        vol_num = int(vol_dir.name.split("_")[1])
        regions_dir = vol_dir / "regions"
        if not regions_dir.exists():
            print(f"  [{vol_idx:02d}/{len(volumes)}] Vol.{vol_num:02d}  [!] No regions/ dir, skipping.")
            continue

        json_files = sorted(
            [f for f in regions_dir.glob("*.json")],
            key=_page_sort_key,
        )

        vol_pages = 0
        vol_regions = 0

        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  [!] Cannot read {jf.name}: {e}")
                continue

            regions = [
                r for r in data
                if r.get("type") not in EXCLUDED
                and r.get("transcription", {}).get("status") == "success"
                and not r.get("transcription", {}).get("skipped")
            ]
            regions.sort(key=lambda r: r.get("reading_order", 99))
            if not regions:
                continue

            corpus.append({
                "volume": vol_num,
                "page_id": jf.stem,
                "regions": [
                    {"type": r["type"], "text": r["transcription"]["text"]}
                    for r in regions
                ],
            })
            vol_pages   += 1
            vol_regions += len(regions)

        total_regions += vol_regions
        print(f"  [{vol_idx:02d}/{len(volumes)}] Vol.{vol_num:02d}  {vol_pages} pages, {vol_regions} regions")

    # Write corpus.json
    out_json = CORPUS_DIR / "corpus.json"
    out_json.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Write corpus.txt (flat text for human reading / grep)
    lines = []
    for page in corpus:
        lines.append(f"\n=== Vol.{page['volume']:02d}  {page['page_id']} ===")
        for reg in page["regions"]:
            lines.append(reg["text"])
            lines.append("")
    (CORPUS_DIR / "corpus.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nCorpus: {len(corpus)} pages, {total_regions} regions")
    print(f"  → {out_json}")
    print(f"  → {CORPUS_DIR / 'corpus.txt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--output-base", type=Path, default=OUTPUT_BASE,
        help="Root dir containing Laubmann_NN_gemini folders",
    )
    args = ap.parse_args()
    build(args.output_base)
