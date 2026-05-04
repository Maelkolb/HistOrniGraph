#!/usr/bin/env python3
"""Batch GUI generator + zipper for all Laubmann output folders.

Sequentially runs Create_GUIs.main() for every folder that exists,
then zips all generated HTML files into one archive.

Edit OUTPUT_BASE_DIR and FOLDERS below, then run:
    python batch_gui.py
"""

import sys
import zipfile
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

OUTPUT_BASE_DIR = Path(r"G:\My Drive\HistOrniGraph_output")

# Folder names to process, in order.
# Remove or comment out any that don't exist yet.
FOLDERS = [f"Laubmann_{i:02d}_gemini" for i in range(1, 36)]

# Where to save the zip of all GUIs (relative to this script, or absolute)
ZIP_OUTPUT = Path(r"G:\My Drive\HistOrniGraph_output\all_guis.zip")

# ─── Runner ───────────────────────────────────────────────────────────────────

def main() -> None:
    import Create_GUIs as cg

    generated: list[Path] = []
    skipped:   list[str]  = []
    failed:    list[str]  = []

    for folder_name in FOLDERS:
        folder_path = OUTPUT_BASE_DIR / folder_name
        if not folder_path.exists():
            print(f"  SKIP  {folder_name}  (folder not found)")
            skipped.append(folder_name)
            continue

        gui_name = f"{folder_name}_validation_gui.html"
        print(f"\n{'─'*60}")
        print(f"  Processing: {folder_name}")

        try:
            cg.BOOK_ROOT_DIR  = str(folder_path)
            cg.OUTPUT_FILENAME = gui_name
            cg.main()
            gui_path = folder_path / gui_name
            if gui_path.exists():
                generated.append(gui_path)
                print(f"  ✓  {gui_name}")
            else:
                print(f"  ✗  {gui_name} not found after generation")
                failed.append(folder_name)
        except Exception as exc:
            print(f"  ✗  {folder_name}: {exc}")
            failed.append(folder_name)

    # ── Zip all generated GUIs ────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    if generated:
        ZIP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_OUTPUT, "w", zipfile.ZIP_DEFLATED) as zf:
            for gui_path in generated:
                zf.write(gui_path, gui_path.name)
                print(f"  zipped  {gui_path.name}")
        print(f"\n✓  Zip saved → {ZIP_OUTPUT}  ({len(generated)} files)")
    else:
        print("No GUIs were generated — nothing to zip.")

    if skipped:
        print(f"\nSkipped ({len(skipped)}): {', '.join(skipped)}")
    if failed:
        print(f"Failed  ({len(failed)}): {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
