"""Split double-page scans into individual left / right pages."""

import logging
from pathlib import Path
from typing import List, Tuple

from PIL import Image

from .config import PipelineConfig
from .utils import natural_sort_key

log = logging.getLogger(__name__)


def split_double_page(
    image_path: Path,
    output_dir: Path,
    overlap_px: int = 10,
) -> Tuple[Path, Path]:
    """Crop a double-page scan down the centre.

    Returns paths ``(left_page, right_page)``.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    mid = w // 2

    left = img.crop((0, 0, mid + overlap_px, h))
    right = img.crop((mid - overlap_px, 0, w, h))

    stem = image_path.stem
    ext = ".png"

    left_path = output_dir / f"{stem}_L{ext}"
    right_path = output_dir / f"{stem}_R{ext}"

    left.save(left_path, "PNG")
    right.save(right_path, "PNG")

    log.info("Split %s → %s, %s", image_path.name, left_path.name, right_path.name)
    return left_path, right_path


def split_all(cfg: PipelineConfig) -> List[Path]:
    """Split every image in *cfg.input_dir* and return sorted page paths."""
    pages_dir = cfg.output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    scans = sorted(
        [
            p
            for p in cfg.input_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")
        ],
        key=natural_sort_key,
    )

    if not scans:
        log.warning("No images found in %s", cfg.input_dir)
        return []

    page_paths: List[Path] = []
    for scan in scans:
        left, right = split_double_page(scan, pages_dir, cfg.split_overlap_px)
        page_paths.extend([left, right])

    log.info("Split %d scans → %d pages", len(scans), len(page_paths))
    return sorted(page_paths, key=natural_sort_key)
