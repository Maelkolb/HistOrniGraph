"""Optional image pre-processing: deskew, contrast enhancement."""

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .config import PipelineConfig

log = logging.getLogger(__name__)


def _estimate_skew(gray: np.ndarray) -> float:
    """Estimate skew angle in degrees using horizontal projection profile."""
    best_angle = 0.0
    best_score = 0.0
    for angle_10x in range(-30, 31):  # -3.0 … +3.0 degrees
        angle = angle_10x / 10.0
        from scipy.ndimage import rotate as nd_rotate
        rotated = nd_rotate(gray, angle, reshape=False, order=0)
        profile = rotated.sum(axis=1).astype(float)
        score = float(np.var(profile))
        if score > best_score:
            best_score = score
            best_angle = angle
    return best_angle


def preprocess_page(page_path: Path, cfg: PipelineConfig) -> Path:
    """Apply optional pre-processing *in-place* and return the same path."""
    img = Image.open(page_path).convert("RGB")
    changed = False

    # --- Deskew ---
    if cfg.deskew:
        try:
            gray = np.array(ImageOps.grayscale(img))
            angle = _estimate_skew(gray)
            if abs(angle) > 0.2:
                img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
                log.debug("Deskewed %s by %.1f°", page_path.name, angle)
                changed = True
        except ImportError:
            log.warning("scipy not installed – skipping deskew")

    # --- Contrast enhancement (CLAHE-like via adaptive equalisation) ---
    if cfg.enhance_contrast:
        img = ImageOps.autocontrast(img, cutoff=1)
        changed = True

    if changed:
        img.save(page_path, "PNG")

    return page_path
