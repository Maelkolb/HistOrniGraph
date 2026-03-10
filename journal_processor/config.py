"""Configuration for the journal processing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Region taxonomy
# ---------------------------------------------------------------------------
REGION_TYPES: List[str] = [
    "ParagraphRegion",
    "ListRegion",
    "TableRegion",
    "ObjectRegion",
    "PageNumberRegion",
    "MarginaliaRegion",
    "FootnoteRegion",
    "ImageRegion",
]

# Region types that contain transcribable running text
TEXT_REGION_TYPES = {"ParagraphRegion", "ListRegion", "FootnoteRegion", "MarginaliaRegion"}

# Region types included in the ShareGPT training-data export
SHAREGPT_REGION_TYPES = {"ParagraphRegion", "ListRegion", "TableRegion", "FootnoteRegion"}

MAX_REGIONS_PER_PAGE = 5

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Pipeline defaults
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All tuneable knobs live here."""

    # I/O paths
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")

    # Model
    model_id: str = MODEL_ID

    # Splitting
    split_overlap_px: int = 10          # slight overlap when cropping centre

    # Pre-processing
    deskew: bool = False                # optional deskew step
    enhance_contrast: bool = False      # optional CLAHE contrast boost

    # Region detection
    max_regions: int = MAX_REGIONS_PER_PAGE
    region_margin_frac: float = 0.005   # margin added around detected boxes
    detection_temperature: float = 0.3
    detection_thinking: str = "medium"

    # Transcription
    transcription_temperature: float = 0.2
    transcription_thinking: str = "low"

    # Output formats (all enabled by default)
    output_md: bool = True
    output_pagexml: bool = True
    output_sharegpt: bool = True

    # ShareGPT
    sharegpt_system_prompt: str = "Transcribe the german text in this image region."

    # Concurrency
    workers: int = 4                    # parallel page processing threads

    def ensure_dirs(self) -> None:
        """Create output sub-directories."""
        for sub in ("pages", "regions", "md", "pagexml", "sharegpt"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
