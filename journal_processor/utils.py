"""Shared utility helpers."""

import json
import re
from pathlib import Path
from typing import Any, Dict


MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


def clean_llm_json(text: str) -> str:
    """Strip markdown fences and stray whitespace around JSON."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def safe_json_parse(text: str) -> Dict[str, Any]:
    """Attempt to parse LLM output as JSON, returning {} on failure."""
    try:
        return json.loads(clean_llm_json(text))
    except json.JSONDecodeError:
        return {}


def page_id(scan_stem: str, side: str) -> str:
    """Canonical page identifier: ``<scan_stem>_L`` or ``<scan_stem>_R``."""
    return f"{scan_stem}_{side}"


def natural_sort_key(p: Path):
    """Sort key that handles embedded numbers naturally."""
    return [
        int(c) if c.isdigit() else c.lower()
        for c in re.split(r"(\d+)", p.stem)
    ]
