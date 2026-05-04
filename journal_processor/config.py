"""Configuration for the journal processing pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Entity types  (NER stage — historical German ornithologist's journals)
# ---------------------------------------------------------------------------
ENTITY_TYPES: Dict[str, str] = {
    "Animal": (
        "Tier, Tiergruppe oder Tierart (z. B. Wolf, Forelle, Rinderherde)"
    ),
    "Artefact": (
        "Menschengemachtes, unbelebtes Artefakt (z. B. Brücke, Mühle, Eisenbahn)"
    ),
    "Environment": (
        "Biotop/Habitat, natürliche Umgebung, kein Eigenname einer Stadt/Ort "
        "(z. B. Wald, Uferzone, Auenlandschaft)"
    ),
    "Environmental Impact": (
        "Umweltauswirkung/Effekt (z. B. Überschwemmung, Erosion, Abholzung)"
    ),
    "Person": (
        "NUR einzelne, namentlich identifizierbare historische Persönlichkeiten "
        "mit Eigennamen (z. B. Kaiser Karl IV., Herzog Ernst, Fürst Reuß). "
        "KEINE Berufsgruppen, Bevölkerungsgruppen, Völker oder generische Bezeichnungen."
    ),
    "Location": (
        "NUR eindeutig identifizierbare, konkrete geographische Orte mit Eigennamen: "
        "Länder, Regionen, Städte, Dörfer (z. B. Weimar, Thüringen, Böhmen, Sachsen). "
        "KEINE abstrakten Gebietsbezeichnungen."
    ),
    "Organisation": (
        "Organisation/Verband/Institution (z. B. Universität Jena, Forstamt Saalfeld, "
        "Kloster Ettal)"
    ),
    "Natural Object": (
        "Natürlich vorkommendes Objekt ohne Veränderung durch menschliches Zutun "
        "(z. B. Donau, Fichtelgebirge, Lech, Brocken)"
    ),
    "Plant": "Pflanze/Pflanzenart (z. B. Eiche, Buche, Weizen)",
    "Resource": (
        "Natürlich vorkommende Ressource (z. B. Holz, Erz, Kohle, Quellwasser)"
    ),
    "Climate": (
        "Klima-/Wetter-/Temperatur-Phänomen (z. B. Frost, Dürre, Schneesturm, Regen)"
    ),
}

# Entity colours and labels for the HTML viewer / GUI
ENTITY_COLORS: Dict[str, str] = {
    "Animal":               "#c62828",
    "Artefact":             "#e65100",
    "Environment":          "#2e7d32",
    "Environmental Impact": "#bf360c",
    "Person":               "#6a1b9a",
    "Location":             "#1565c0",
    "Organisation":         "#37474f",
    "Natural Object":       "#5d4037",
    "Plant":                "#558b2f",
    "Resource":             "#f9a825",
    "Climate":              "#546e7a",
}
ENTITY_LABELS: Dict[str, str] = {
    "Animal":               "Tiere",
    "Artefact":             "Artefakte",
    "Environment":          "Umgebung",
    "Environmental Impact": "Umwelteinflüsse",
    "Person":               "Personen",
    "Location":             "Orte",
    "Organisation":         "Organisationen",
    "Natural Object":       "Naturobjekte",
    "Plant":                "Pflanzen",
    "Resource":             "Ressourcen",
    "Climate":              "Klima",
}

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
TEXT_REGION_TYPES = {"ParagraphRegion", "ListRegion", "FootnoteRegion"}

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
ANALYZER_MODEL_ID = "gemini-3-flash-preview"   # same as main model — lite was too weak for rotation detection

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
    detection_thinking: str = "high"
    detection_retries: int = 2          # retry on bad JSON from Gemini

    # Transcription (Gemini — always used for tables/images/objects)
    transcription_temperature: float = 0.0
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

    # NER (Stage 7 – run separately in Colab via Run_NER_Stage.py)
    ner_model_id: str = MODEL_ID
    ner_thinking_level: str = "low"
    ner_retries: int = 2

    # Pre-rotation (applied before splitting or full-page processing)
    force_rotation: int = 0             # 0, 90, 180, or 270 degrees clockwise

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
        for sub in ("pages", "regions", "md", "pagexml", "pagexml_ner",
                    "sharegpt", "sharegpt/images"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)
