"""Scan analysis: rotation detection and split routing for each raw scan.

Rotation is detected from the image aspect ratio — no model needed:
  landscape (width > height) → already upright, no rotation
  portrait  (height > width) → scan is sideways, rotate 90° before splitting

The model's only job is to decide split vs. no-split and count loose inserts.
"""

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from .utils import clean_llm_json

log = logging.getLogger(__name__)

# Aspect ratio threshold: if height/width exceeds this, treat as rotated.
# 1.1 gives a small margin so near-square images don't false-trigger.
_PORTRAIT_THRESHOLD = 1.1


_PROMPT = """\
You are analysing a correctly-oriented scan from a bound German ornithologist's \
field journal. The image is already upright — text reads horizontally.

1. USE_DOUBLE_PAGE
   Does this scan show two journal pages side by side (a double-page spread)?
     false = two pages visible — split down the centre (default)
     true  = send the full image unsplit

   Set true ONLY when:
     • if splitting the image results in cut off pages because of scan orientation OR
     • A loose unattached sheet of paper or physical object (not glued down)
       crosses the centre line and would be cut by a vertical split.
   Glued items (photos, maps, postcards) do NOT prevent splitting.

2. NOTES  one short sentence about anything unusual, or ""

Output JSON only:
{"use_double_page": false, "notes": ""}
"""


class ScanAnalyzer:

    def __init__(self, client: Any, cfg: Any) -> None:
        self.client = client
        self.cfg = cfg

    def analyse(self, image_path: Path) -> Dict[str, Any]:
        img_bytes = image_path.read_bytes()

        try:
            # ── Rotation: aspect ratio, no model needed ───────────────────────
            img = Image.open(io.BytesIO(img_bytes))
            rotation = 90 if img.height > img.width * _PORTRAIT_THRESHOLD else 0
            if rotation:
                log.info(
                    "%s is portrait (%dx%d) — auto rotation 90°",
                    image_path.name, img.width, img.height,
                )
                img = img.rotate(-90, expand=True)

            # ── Split decision: send upright image to model ───────────────────
            buf = io.BytesIO()
            img.save(buf, format="PNG")

            from google.genai import types

            resp = self.client.models.generate_content(
                model=self.cfg.analyzer_model_id,
                contents=[
                    types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
                    _PROMPT,
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=4096,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    response_mime_type="application/json",
                ),
            )

            text = resp.text
            if text is None:
                raise ValueError("empty response")
            data = json.loads(clean_llm_json(text))

            result: Dict[str, Any] = {
                "rotation":        rotation,
                "use_double_page": bool(data.get("use_double_page", False)),
                "notes":           str(data.get("notes", "")),
            }

            log.info(
                "Analysis %s → rotation=%d use_double_page=%s%s",
                image_path.name,
                result["rotation"],
                result["use_double_page"],
                f"  notes: {result['notes']}" if result["notes"] else "",
            )
            return result

        except Exception as exc:
            log.warning(
                "Scan analysis failed for %s (%s) — defaulting to full-image",
                image_path.name, exc,
            )
            return {"rotation": 0, "use_double_page": True, "notes": f"[error: {exc}]"}
