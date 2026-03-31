"""Generate Markdown reconstruction of each page.

MarginaliaRegion is intentionally excluded from Markdown output — marginalia
are peripheral annotations that clutter reading flow.  They remain present in
PageXML, ShareGPT, and the regions JSON.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from .config import MD_EXCLUDED_TYPES

log = logging.getLogger(__name__)


def generate_md(
    page_id: str,
    regions: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write a Markdown file that reconstructs the full page content.

    Regions are rendered in reading order with type-appropriate formatting.
    """
    lines: List[str] = []

    # Page-level header with detected page number (if any)
    page_num = _find_page_number(regions)
    header = f"# Page {page_num}" if page_num else f"# {page_id}"
    lines.append(header)
    lines.append("")

    for r in sorted(regions, key=lambda r: r["reading_order"]):
        rtype = r["type"]

        # Skip types excluded from MD output (e.g. MarginaliaRegion)
        if rtype in MD_EXCLUDED_TYPES:
            continue

        # Skip folded inserts — nothing readable to render
        if r.get("insert_state") == "folded":
            continue

        text = r.get("transcription", {}).get("text", "")
        if not text:
            continue

        if rtype == "PageNumberRegion":
            # Already used in header; skip body
            continue

        if rtype in ("ParagraphRegion", "ListRegion", "FootnoteRegion"):
            if rtype == "FootnoteRegion":
                lines.append(f"[^footnote]: {text}")
            else:
                lines.append(text)
            lines.append("")

        elif rtype == "TableRegion":
            lines.append(text)  # already in Markdown table format
            lines.append("")

        elif rtype in ("ImageRegion", "ObjectRegion"):
            desc = r.get("transcription", {}).get("description", text)
            dtype = r.get("transcription", {}).get("drawing_type",
                    r.get("transcription", {}).get("object_type", "unknown"))
            visible = r.get("transcription", {}).get("visible_text", "")
            lines.append(f"*[{rtype}: {dtype}]* {desc}")
            if visible and visible.lower() != "none":
                lines.append(f"Text: {visible}")
            lines.append("")

    md_path = output_dir / f"{page_id}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log.debug("Wrote %s", md_path.name)
    return md_path


def _find_page_number(regions: List[Dict]) -> str:
    """Extract page number from PageNumberRegion if present."""
    for r in regions:
        if r["type"] == "PageNumberRegion":
            pn = r.get("page_number") or r.get("transcription", {}).get("text", "")
            if pn:
                return str(pn)
    return ""
