"""Configuration for the journal processing pipeline."""

from dataclasses import dataclass
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

# Marginalia is intentionally excluded from Markdown output (kept in all others)
MD_EXCLUDED_TYPES = {"MarginaliaRegion"}

MAX_REGIONS_PER_PAGE = 5
MAX_REGIONS_DOUBLE_PAGE = 10

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL_ID = "gemini-3-flash-preview"
ANALYZER_MODEL_ID = "gemini-3.1-flash-lite-preview"   # cheap routing model

# ---------------------------------------------------------------------------
# Pipeline defaults
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All tuneable knobs live here.

    Processing mode
    ---------------
    auto_mode (default True)
        Before each scan a lightweight Gemini call analyses the image and
        decides whether to split it or send it unsplit to the detector.
        Different scans in the same batch may be handled differently.

    double_page_mode
        Only used when auto_mode=False.  When True, ALL scans are sent to the
        detector without splitting (full double-page prompt).  When False, ALL
        scans are split down the centre first.

    Insert handling
    ---------------
    The detector tags regions from loose insert sheets with page_side="insert"
    and insert_state="visible" | "folded".  Folded inserts are not transcribed.
    """

    # I/O paths
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")

    # Models
    model_id: str = MODEL_ID
    analyzer_model_id: str = ANALYZER_MODEL_ID  # fast/cheap model for scan routing

    # Processing mode
    auto_mode: bool = True              # agentic: model decides split vs full per scan
    double_page_mode: bool = False      # fallback when auto_mode=False

    # Splitting (only used when a scan is determined to need splitting)
    split_overlap_px: int = 10          # slight overlap when cropping centre

    # Pre-processing
    deskew: bool = False                # optional deskew step
    enhance_contrast: bool = False      # optional CLAHE contrast boost

    # Region detection
    max_regions: int = MAX_REGIONS_PER_PAGE
    region_margin_frac: float = 0.005   # margin added around detected boxes
    # Gemini 3 docs: keep temperature at 1.0 (default); values < 1.0 may cause
    # looping or degraded performance on complex tasks.
    detection_temperature: float = 1.0
    detection_thinking: str = "medium"
    detection_retries: int = 2          # retry on bad JSON from Gemini

    # Transcription (Gemini — always used for tables/images/objects)
    transcription_temperature: float = 1.0
    transcription_thinking: str = "low"

    # Transcription (GLM-OCR — used for text regions when enabled)
    use_glm_ocr: bool = False
    glm_ocr_base_model: str = "zai-org/GLM-OCR"
    glm_ocr_lora_path: str = ""              # path to LoRA adapter dir
    glm_ocr_max_new_tokens: int = 2048

    # Output formats (all enabled by default)
    output_md: bool = True
    output_pagexml: bool = True
    output_sharegpt: bool = True

    # ShareGPT
    sharegpt_system_prompt: str = "Transcribe the german text in this image region."

    # Concurrency
    workers: int = 4                    # parallel page processing threads

    def __post_init__(self) -> None:
        # In double-page / auto mode we typically need more regions.
        # Raise the default only when the user hasn't overridden it.
        if not self.auto_mode and self.double_page_mode:
            if self.max_regions == MAX_REGIONS_PER_PAGE:
                self.max_regions = MAX_REGIONS_DOUBLE_PAGE

    def ensure_dirs(self) -> None:
        """Create output sub-directories."""
        for sub in ("pages", "regions", "md", "pagexml", "sharegpt", "sharegpt/images"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
