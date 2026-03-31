"""Agentic scan analysis: decide how each scan should be processed.

A fast, cheap Gemini call inspects each raw scan and returns:
  - layout        : "double_page" or "single_page"
  - use_double_page: True  → send full scan to region detector (no split)
                    False → split down the centre first
  - has_inserts   : whether any loose insert sheets are visible
  - folded_inserts: count of inserts that are folded / unreadable

The result drives per-scan routing inside the Pipeline, so different images
in the same batch can be handled differently.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from .utils import MIME_BY_EXT, clean_llm_json

log = logging.getLogger(__name__)


# ── Prompt ───────────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """\
You are examining a scan from a bound field journal.

Analyse the image and answer the following questions as accurately as possible.

1. LAYOUT
   Is this a double-page spread (two pages visible side by side from an open
   bound book) or a single page?
   → "double_page" or "single_page"

2. USE_DOUBLE_PAGE
   Should this image be sent to the layout detector WITHOUT splitting it first?

   Answer true (no split) if ANY of the following apply:
   • The image is a single page.
   • Loose insert sheets are visible anywhere in the image — inserts that cross
     the centre line would be destroyed by a crop-split.
   • The central gutter / spine is unclear or missing.

   Answer false (split recommended) ONLY if ALL of these hold:
   • It is clearly a double-page spread.
   • No loose insert sheets are visible.
   • There is a well-defined central gutter separating the two pages.

   → true or false

3. HAS_INSERTS
   Are any loose insert sheets (separate pieces of paper tucked into the
   journal) visible in the image?
   → true or false

4. FOLDED_INSERTS
   How many of those inserts are clearly folded — showing only their back or
   blank/exterior side with no readable text content?
   → integer (0 if no inserts or all inserts are fully open and readable)

5. NOTES
   One short sentence about anything unusual (e.g. severely damaged page,
   extremely faint writing, insert covers most of the left page, etc.).
   Omit if nothing notable.
   → string or ""

OUTPUT — JSON only, no commentary, no markdown fences:
{{"layout": "…", "use_double_page": …, "has_inserts": …, "folded_inserts": …, "notes": "…"}}
"""


class ScanAnalyzer:
    """Run a fast Gemini call to decide how each scan should be processed."""

    def __init__(self, client: Any, cfg: Any) -> None:
        self.client = client
        self.cfg = cfg

    def analyse(self, image_path: Path) -> Dict[str, Any]:
        """Return an analysis dict for *image_path*.

        Keys:
            layout           – "double_page" | "single_page"
            use_double_page  – bool: send full scan to detector (no split)
            has_inserts      – bool
            folded_inserts   – int
            notes            – str

        Falls back to a conservative default (no split, double-page prompt)
        on any error so the scan is still processed rather than silently skipped.
        """
        from google.genai import types

        ext = image_path.suffix.lower()
        mime = MIME_BY_EXT.get(ext, "image/png")
        img_bytes = image_path.read_bytes()

        try:
            resp = self.client.models.generate_content(
                model=self.cfg.analyzer_model_id,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime),
                    ANALYSIS_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=256,
                    # "minimal" = lowest latency/cost thinking level for flash-lite
                    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                ),
            )
            raw = clean_llm_json(resp.text)
            data = json.loads(raw)

            result: Dict[str, Any] = {
                "layout": str(data.get("layout", "double_page")),
                "use_double_page": bool(data.get("use_double_page", True)),
                "has_inserts": bool(data.get("has_inserts", False)),
                "folded_inserts": int(data.get("folded_inserts", 0)),
                "notes": str(data.get("notes", "")),
            }
            log.info(
                "Analysis %s → layout=%s use_double_page=%s inserts=%s folded=%d%s",
                image_path.name,
                result["layout"],
                result["use_double_page"],
                result["has_inserts"],
                result["folded_inserts"],
                f" notes: {result['notes']}" if result["notes"] else "",
            )
            return result

        except Exception as exc:
            log.warning(
                "Scan analysis failed for %s (%s) — defaulting to no-split double-page mode",
                image_path.name, exc,
            )
            return {
                "layout": "double_page",
                "use_double_page": True,
                "has_inserts": False,
                "folded_inserts": 0,
                "notes": f"[analysis_error: {exc}]",
            }
