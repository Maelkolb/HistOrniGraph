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
    """Extract JSON from LLM output that may contain reasoning/commentary.

    Handles:
    - Markdown ```json fences
    - Leading/trailing commentary or thinking text
    - JSON embedded in the middle of reasoning
    """
    text = text.strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # If it already starts with { and ends with }, return as-is
    if text.startswith("{") and text.endswith("}"):
        return text

    # Try to find a JSON object in the text (first { to last matching })
    start = text.find("{")
    if start != -1:
        # Find the matching closing brace
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    # Don't break — keep going to find the outermost match
        if end != -1:
            return text[start:end + 1]

    return text


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
