"""Generate ShareGPT-format JSONL training data.

Each line is a conversation turn: system prompt → user image → assistant
transcription.  Only ParagraphRegion, ListRegion, TableRegion, and
FootnoteRegion are included.
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from .config import SHAREGPT_REGION_TYPES, PipelineConfig

log = logging.getLogger(__name__)


def _image_to_base64(img: Image.Image, max_side: int = 1024) -> str:
    """Resize (if needed) and encode a PIL image as base64 PNG."""
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_sharegpt_entries(
    page_id: str,
    page_image: Image.Image,
    regions: List[Dict[str, Any]],
    cfg: PipelineConfig,
) -> List[Dict[str, Any]]:
    """Return a list of ShareGPT conversation dicts for eligible regions."""
    entries: List[Dict[str, Any]] = []

    for r in regions:
        if r["type"] not in SHAREGPT_REGION_TYPES:
            continue

        text = r.get("transcription", {}).get("text", "")
        if not text:
            continue

        # Crop the region from the full page
        bbox = r["bbox"]
        crop = page_image.crop((
            bbox["x"], bbox["y"],
            bbox["x"] + bbox["width"],
            bbox["y"] + bbox["height"],
        ))
        img_b64 = _image_to_base64(crop)

        entry = {
            "id": f"{page_id}_{r['id']}",
            "conversations": [
                {
                    "from": "system",
                    "value": cfg.sharegpt_system_prompt,
                },
                {
                    "from": "human",
                    "value": f"<image>\nRegion type: {r['type']}",
                },
                {
                    "from": "gpt",
                    "value": text,
                },
            ],
            "image": img_b64,
            "metadata": {
                "page_id": page_id,
                "region_id": r["id"],
                "region_type": r["type"],
                "bbox": r["bbox"],
            },
        }
        entries.append(entry)

    return entries


def append_sharegpt(entries: List[Dict[str, Any]], output_path: Path) -> None:
    """Append entries to a JSONL file (one JSON object per line)."""
    with open(output_path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.debug("Appended %d ShareGPT entries to %s", len(entries), output_path.name)
