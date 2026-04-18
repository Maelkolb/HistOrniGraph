"""Region detection using Gemini 3 Flash Preview.

Two prompt variants:
  DETECTION_PROMPT        – single-page (after splitting, or a single-page scan)
  DETECTION_PROMPT_DOUBLE – full double-page scan

The double-page prompt enforces:
  • Each physical insert is ONE region (no sub-region splitting by content type)
  • Folded inserts get a dedicated region with insert_state="folded" so the
    transcriber can skip them gracefully
"""

import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .config import REGION_TYPES, MAX_REGIONS_PER_PAGE, PipelineConfig
from .utils import MIME_BY_EXT, clean_llm_json

log = logging.getLogger(__name__)


# ── Prompts ──────────────────────────────────────────────────────────────────

# Single-page prompt — used after the scan has been split into L/R halves,
# or when the analyser determines the scan is already a single page.
DETECTION_PROMPT = """\
You are a document-layout analyst for digitised pages of a German ornithologist's \
handwritten/printed field journal (19th–20th century).

TASK: Detect every distinct content region on this page and return precise \
bounding boxes.  Try to keep whole journal entries together — an entry \
typically begins with a date and city/location name.  Do NOT split a single \
entry across multiple regions unless it clearly changes type (e.g. a table \
inside an entry).

COORDINATE SYSTEM
Normalised 0–1000 scale.  (0, 0) = top-left, (1000, 1000) = bottom-right.
bbox format: {{"x": left, "y": top, "width": w, "height": h}}

REGION TYPES (use exactly these names):
ParagraphRegion · ListRegion · TableRegion · ObjectRegion · PageNumberRegion · \
MarginaliaRegion · FootnoteRegion · ImageRegion · InsertRegion

INSERT SHEETS
If any type of paper insert is visible in this image, treat its entire
visible insert as a single InsertRegion (do not subdivide it) and add:
  "page_side": "insert", "insert_id": 1, "insert_state": "visible"
If the insert is folded and unreadable, use type "InsertRegion" and add:
  "page_side": "insert", "insert_id": 1, "insert_state": "folded"

RULES
1. Maximum {max_regions} regions – merge small adjacent blocks of the same type.
2. Tight bounding boxes, no overlaps.
3. Reading order = natural top-to-bottom, left-to-right sequence.

EXTRA METADATA per region:
• PageNumberRegion → "page_number": <int or string>.
• ParagraphRegion / ListRegion / FootnoteRegion → "line_count": <int>.
• TableRegion → "rows": <int>, "cols": <int>.

OUTPUT (JSON only, no commentary):
{{"regions": [
  {{"id": "r1", "type": "…", "bbox": {{"x":…,"y":…,"width":…,"height":…}}, \
"reading_order": 1, …metadata…}},
  …
], "total_regions": N}}
"""

# Double-page prompt — used when the full unsplit scan is sent to the detector.
DETECTION_PROMPT_DOUBLE = """\
You are a document-layout analyst for digitised double-page scans of a German \
ornithologist's handwritten field journal (20th century).

CONTEXT
This image contains TWO journal pages side by side (left page and right page).
It may also contain loose INSERT sheets — separate pieces of paper tucked inside
the journal that overlap one or both main pages.

TASK: Detect every distinct content region across BOTH pages and any inserts,
then return precise bounding boxes with metadata.  Keep whole journal entries
together — an entry typically begins with a date and city/location name.  Do NOT
split a single entry across multiple regions unless it clearly changes type
(e.g. a table inside an entry).

COORDINATE SYSTEM
Normalised 0–1000 scale.  (0, 0) = top-left, (1000, 1000) = bottom-right.
bbox format: {{"x": left, "y": top, "width": w, "height": h}}

REGION TYPES (use exactly these names):
ParagraphRegion · ListRegion · TableRegion · ObjectRegion · PageNumberRegion · \
MarginaliaRegion · FootnoteRegion · ImageRegion · InsertRegion

PAGE ASSIGNMENT — required for every region:
• "page_side": "left" | "right" | "insert"
  – "left"   → belongs to the left-hand main page
  – "right"  → belongs to the right-hand main page
  – "insert" → belongs to a loose insert sheet

INSERT RULES — read carefully:
① Group all content from the SAME physical insert under one "insert_id" integer
  (insert_id: 1, 2, …).
② A FULLY VISIBLE insert MUST be captured as a SINGLE region with one bounding box
  that encloses the entire insert — do NOT subdivide it into separate paragraph /
  table / image sub-regions.  Use the region type InsertRegion and add "page_side": "insert", "insert_id": 1, "insert_state": "visible"
③ A FOLDED insert (showing only its back or blank/exterior side — no readable
  text) → create exactly ONE region with:
    type: "InsertRegion", page_side: "insert", insert_id: <n>,
    insert_state: "folded"
  Do NOT create sub-regions inside a folded insert.


INSERT STATE — required for every insert region:
• "insert_state": "visible"  → readable content
• "insert_state": "folded"   → back/exterior only, not readable

RULES
1. Maximum {max_regions} regions total across both pages and all inserts.
   Merge small adjacent same-type blocks rather than creating many tiny regions.
2. Tight bounding boxes, no overlaps.
3. Reading order = left-to-right, top-to-bottom
   (left page first, then right page, inserts interleaved by their position).

EXTRA METADATA per region:
• PageNumberRegion → "page_number": <int or string>.
• ParagraphRegion / ListRegion / FootnoteRegion → "line_count": <int>.
• TableRegion → "rows": <int>, "cols": <int>.

OUTPUT (JSON only, no commentary):
{{"regions": [
  {{"id": "r1", "type": "…", "page_side": "left", \
"bbox": {{"x":…,"y":…,"width":…,"height":…}}, "reading_order": 1, …metadata…}},
  …
], "total_regions": N}}
"""


class RegionDetector:
    """Detect and classify layout regions with Gemini 3 Flash."""

    def __init__(self, client: Any, cfg: PipelineConfig) -> None:
        self.client = client
        self.cfg = cfg

    # ── margin helper ───────────────────────────────────────────────────

    @staticmethod
    def _add_margin(
        bbox: Dict, img_w: int, img_h: int, margin_frac: float
    ) -> Dict[str, int]:
        mx = int(img_w * margin_frac)
        my = int(img_h * margin_frac)
        x = max(0, bbox["x"] - mx)
        y = max(0, bbox["y"] - my)
        w = min(bbox["width"] + 2 * mx, img_w - x)
        h = min(bbox["height"] + 2 * my, img_h - y)
        return {"x": x, "y": y, "width": w, "height": h}

    # ── main entry point ────────────────────────────────────────────────

    def detect(
        self,
        image_path: Path,
        use_double_page: Optional[bool] = None,
        max_regions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run region detection on a page image.

        Parameters
        ----------
        image_path:
            Path to the image to analyse.
        use_double_page:
            Override whether to use the double-page prompt for this specific
            call.  When None, falls back to cfg.double_page_mode (or False
            when auto_mode is True, since the pipeline passes the value
            explicitly for each scan).

        Returns a dict with keys:
            status, image_path, image_dimensions, regions, total_regions,
            reading_order, region_types_detected
        Each region carries: id, type, bbox (pixel coords), reading_order,
        and type-specific metadata.  Double-page / insert regions also carry
        page_side, insert_id, insert_state.
        """
        from google.genai import types

        # Resolve settings for this call
        if use_double_page is None:
            use_double_page = self.cfg.double_page_mode
        effective_max = max_regions if max_regions is not None else self.cfg.max_regions

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        ext = image_path.suffix.lower()
        mime = MIME_BY_EXT.get(ext, "image/png")
        img_bytes = image_path.read_bytes()

        if use_double_page:
            prompt = DETECTION_PROMPT_DOUBLE.format(max_regions=effective_max)
        else:
            prompt = DETECTION_PROMPT.format(max_regions=effective_max)

        raw = ""
        for attempt in range(self.cfg.detection_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.cfg.model_id,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type=mime),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=self.cfg.detection_temperature,
                        max_output_tokens=16384,
                        thinking_config=types.ThinkingConfig(
                            thinking_level=self.cfg.detection_thinking
                        ),
                    ),
                )
                raw = clean_llm_json(resp.text)
                data = json.loads(raw)
                break  # success
            except json.JSONDecodeError as exc:
                if attempt < self.cfg.detection_retries:
                    log.warning(
                        "JSON parse failed for %s (attempt %d/%d), retrying…",
                        image_path.name, attempt + 1, self.cfg.detection_retries + 1,
                    )
                    continue
                log.error("JSON parse error for %s: %s\nRaw: %s", image_path.name, exc, raw[:400])
                return self._error(image_path, f"JSON parse: {exc}", raw)
            except Exception as exc:
                log.error("Detection failed for %s: %s", image_path.name, exc)
                return self._error(image_path, str(exc), traceback.format_exc())

        regions = self._validate(data.get("regions", []), w, h, use_double_page, effective_max)
        return {
            "status": "success",
            "image_path": str(image_path),
            "image_dimensions": {"width": w, "height": h},
            "regions": regions,
            "total_regions": len(regions),
            "reading_order": [r["id"] for r in regions],
            "region_types_detected": sorted(set(r["type"] for r in regions)),
        }

    # ── validation / normalisation ──────────────────────────────────────

    _VALID_PAGE_SIDES = {"left", "right", "insert"}
    _VALID_INSERT_STATES = {"visible", "folded"}

    def _validate(
        self,
        raw_regions: List[Dict],
        img_w: int,
        img_h: int,
        double_page_context: bool,
        max_regions: int,
    ) -> List[Dict]:
        out: List[Dict] = []
        for i, r in enumerate(raw_regions[:max_regions]):
            rtype = self._normalise_type(r.get("type", "ParagraphRegion"))
            bbox = r.get("bbox", {})

            # normalised 0-1000 → pixel coordinates
            nx, ny = float(bbox.get("x", 0)), float(bbox.get("y", 0))
            nw, nh = float(bbox.get("width", 100)), float(bbox.get("height", 50))
            px = max(0, min(int(nx * img_w / 1000), img_w - 1))
            py = max(0, min(int(ny * img_h / 1000), img_h - 1))
            pw = max(10, min(int(nw * img_w / 1000), img_w - px))
            ph = max(10, min(int(nh * img_h / 1000), img_h - py))
            pixel_bbox = self._add_margin(
                {"x": px, "y": py, "width": pw, "height": ph},
                img_w, img_h, self.cfg.region_margin_frac,
            )

            entry: Dict[str, Any] = {
                "id": "",              # set after sorting
                "type": rtype,
                "bbox": pixel_bbox,
                "reading_order": int(r.get("reading_order", i + 1)),
            }

            # Page / insert assignment fields (present in both prompts for inserts)
            raw_side = str(r.get("page_side", "")).lower()
            if raw_side in self._VALID_PAGE_SIDES:
                entry["page_side"] = raw_side
            elif double_page_context:
                entry["page_side"] = "left"   # safe fallback in double-page mode

            if entry.get("page_side") == "insert":
                entry["insert_id"] = int(r.get("insert_id", 1))
                raw_state = str(r.get("insert_state", "visible")).lower()
                entry["insert_state"] = (
                    raw_state if raw_state in self._VALID_INSERT_STATES else "visible"
                )

            # type-specific metadata
            if rtype == "PageNumberRegion":
                entry["page_number"] = r.get("page_number", None)
            if rtype in ("ParagraphRegion", "ListRegion", "FootnoteRegion", "MarginaliaRegion"):
                entry["line_count"] = int(r.get("line_count", 0))
            if rtype == "TableRegion":
                entry["rows"] = int(r.get("rows", 0))
                entry["cols"] = int(r.get("cols", 0))

            out.append(entry)

        # deterministic ordering
        out.sort(key=lambda r: r["reading_order"])
        for idx, region in enumerate(out):
            region["id"] = f"r{idx + 1:02d}"
            region["reading_order"] = idx + 1

        return out

    @staticmethod
    def _normalise_type(raw: str) -> str:
        """Map a potentially misspelled type to the canonical name."""
        if raw in REGION_TYPES:
            return raw
        low = raw.lower()
        for valid in REGION_TYPES:
            if low in valid.lower() or valid.lower() in low:
                return valid
        return "ParagraphRegion"

    # ── error helper ────────────────────────────────────────────────────

    @staticmethod
    def _error(path: Path, msg: str, detail: str = "") -> Dict[str, Any]:
        return {
            "status": "error",
            "image_path": str(path),
            "error": msg,
            "detail": detail[:500],
            "regions": [],
            "total_regions": 0,
        }
