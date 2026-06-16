"""
============================================================
HistOrniGraph — BIO Export Runner (Step 4, Colab)
============================================================
Turns the annotated ``pagexml_ner/*.xml`` files into token-level BIO CSV
for training a multitask token classifier (parallel Type + Scope heads).

Run this AFTER Run_NER_Stage.py.

Usage in Colab
--------------
    import os
    os.environ['BOOK_ROOT_DIR'] = '/content/drive/.../Laubmann_01_gemini'
    %run Run_BIO_Export.py

or process every book under a root and also write a merged corpus CSV::

    import os
    os.environ['BOOKS_ROOT_DIR'] = '/content/drive/MyDrive/HistOrniGraph_output'
    %run Run_BIO_Export.py

Output
------
Per book:   <book>/bio/<book>_bio.csv
Merged:     <BOOKS_ROOT_DIR>/corpus_bio.csv   (only in multi-book mode)

Columns: doc_id, page_id, region_id, region_type, token_index, token,
char_start, char_end, bio_type, bio_scope.  Regions separated by a blank line.
============================================================
"""

import os
import sys
from pathlib import Path


def _resolve(name: str, default):
    val = os.environ.get(name)
    if val:
        return val
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is not None and name in ip.user_ns:
            v = ip.user_ns.get(name)
            if v not in (None, ""):
                return v
    except Exception:
        pass
    if name in globals():
        v = globals().get(name)
        if v not in (None, ""):
            return v
    return default


BOOK_ROOT_DIR  = _resolve("BOOK_ROOT_DIR_OVERRIDE",  "") or _resolve("BOOK_ROOT_DIR", "")
BOOKS_ROOT_DIR = _resolve("BOOKS_ROOT_DIR_OVERRIDE", "") or _resolve(
    "BOOKS_ROOT_DIR", "/content/drive/MyDrive/HistOrniGraph_output"
)
PAGEXML_NER_SUBDIR = "pagexml_ner"
BIO_SUBDIR         = "bio"


def _list_books(books_root: Path) -> list:
    if not books_root.is_dir():
        return []
    return [sub for sub in sorted(books_root.iterdir())
            if sub.is_dir() and (sub / PAGEXML_NER_SUBDIR).is_dir()]


def main() -> None:
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from journal_processor.output_bio import export_book_bio, export_corpus_bio

    print("🔬 HistOrniGraph — BIO Export Runner")

    if BOOK_ROOT_DIR:
        book = Path(BOOK_ROOT_DIR.rstrip("/"))
        if not (book / PAGEXML_NER_SUBDIR).is_dir():
            raise SystemExit(f"❌ No {PAGEXML_NER_SUBDIR}/ under {book} — run NER first.")
        books = [book]
        merged_root = None
        print(f"   Target: single book → {book}")
    else:
        root = Path(BOOKS_ROOT_DIR.rstrip("/"))
        books = _list_books(root)
        merged_root = root
        print(f"   Target: {len(books)} book(s) under {root}")
        if not books:
            print(f"   ⚠ No books with a {PAGEXML_NER_SUBDIR}/ folder found.")
            return

    grand = 0
    for book in books:
        out_csv = book / BIO_SUBDIR / f"{book.name}_bio.csv"
        n = export_book_bio(book, out_csv, subdir=PAGEXML_NER_SUBDIR)
        grand += n
        print(f"   ✓ {book.name:<32} {n:>7} tokens → {out_csv}")

    if merged_root is not None and len(books) > 1:
        merged = merged_root / "corpus_bio.csv"
        total = export_corpus_bio(books, merged, subdir=PAGEXML_NER_SUBDIR)
        print(f"\n   📦 Merged corpus: {total} tokens → {merged}")

    print(f"\n   ✅ Total tokens: {grand}")


if __name__ == "__main__":
    main()
