"""Generate ShareGPT-format JSONL training data.

Each line is a conversation turn: system prompt → user image → assistant
transcription.  Only ParagraphRegion, ListRegion, TableRegion, and
FootnoteRegion are included.

Region crops are saved as PNG files on disk; the JSONL references them
via relative path in the ``images`` list (compatible with most multimodal
training frameworks like LLaVA, ShareGPT4V, etc.).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from .config import SHAREGPT_REGION_TYPES, PipelineConfig

log = logging.getLogger(__name__)


def _save_resized_image(
    img: Image.Image, save_path: Path, max_side: int = 1024
) -> None:
    """Resize (if needed) and save a PIL image to disk."""
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img.save(save_path, format="PNG")


def build_sharegpt_entries(
    page_id: str,
    page_image: Image.Image,
    regions: List[Dict[str, Any]],
    cfg: PipelineConfig,
    image_save_dir: Path,
) -> List[Dict[str, Any]]:
    """Return a list of ShareGPT conversation dicts for eligible regions.

    Crops are saved into *image_save_dir* and referenced by path in the
    ``images`` field of each entry.
    """
    entries: List[Dict[str, Any]] = []
    image_save_dir.mkdir(parents=True, exist_ok=True)

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

        # Save the image crop to disk
        image_filename = f"{page_id}_{r['id']}.png"
        image_path = image_save_dir / image_filename
        _save_resized_image(crop, image_path)

        entry = {
            "id": f"{page_id}_{r['id']}",
            "messages": [
                {
                    "role": "system",
                    "content": cfg.sharegpt_system_prompt,
                },
                {
                    "role": "user",
                    "content": f"<image>\nRegion type: {r['type']}",
                },
                {
                    "role": "assistant",
                    "content": text,
                },
            ],
            "images": [
                str(image_path),
            ],
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
