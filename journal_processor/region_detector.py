"""Region detection using Gemini 3 Flash Preview.

Detects layout regions in single journal pages and returns bounding boxes
with metadata (page numbers, line counts, table dimensions).
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


# ── Prompt ──────────────────────────────────────────────────────────────────

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
MarginaliaRegion · FootnoteRegion · ImageRegion

RULES  
1. Maximum {max_regions} regions per page – merge small adjacent blocks of the \
   same type rather than creating many tiny regions.  
2. Tight bounding boxes, no overlaps.  
3. Reading order = natural top-to-bottom, left-to-right sequence.

EXTRA METADATA per region (include in the JSON):  
• PageNumberRegion → add "page_number": <int or string of the printed number>.  
• ParagraphRegion / ListRegion / FootnoteRegion / MarginaliaRegion → add \
  "line_count": <estimated number of text lines>.  
• TableRegion → add "rows": <int>, "cols": <int>.

OUTPUT (JSON only, no commentary):
{{"regions": [
  {{"id": "r1", "type": "…", "bbox": {{"x":…,"y":…,"width":…,"height":…}}, \
"reading_order": 1, …metadata…}},
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

    def detect(self, image_path: Path) -> Dict[str, Any]:
        """Run region detection on a single page image.

        Returns a dict with keys:
            status, image_path, image_dimensions, regions, total_regions,
            reading_order, region_types_detected
        Each region carries: id, type, bbox (pixel coords), reading_order,
        and type-specific metadata (page_number / line_count / rows+cols).
        """
        from google.genai import types

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        ext = image_path.suffix.lower()
        mime = MIME_BY_EXT.get(ext, "image/png")
        img_bytes = image_path.read_bytes()

        prompt = DETECTION_PROMPT.format(max_regions=self.cfg.max_regions)

        raw = ""
        try:
            resp = self.client.models.generate_content(
                model=self.cfg.model_id,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=self.cfg.detection_temperature,
                    max_output_tokens=4096,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=self.cfg.detection_thinking
                    ),
                ),
            )
            raw = clean_llm_json(resp.text)
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.error("JSON parse error for %s: %s\nRaw: %s", image_path.name, exc, raw[:400])
            return self._error(image_path, f"JSON parse: {exc}", raw)
        except Exception as exc:
            log.error("Detection failed for %s: %s", image_path.name, exc)
            return self._error(image_path, str(exc), traceback.format_exc())

        regions = self._validate(data.get("regions", []), w, h)
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

    def _validate(
        self, raw_regions: List[Dict], img_w: int, img_h: int
    ) -> List[Dict]:
        out: List[Dict] = []
        for i, r in enumerate(raw_regions[: self.cfg.max_regions]):
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
