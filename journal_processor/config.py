"""Configuration for the journal processing pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Entity types  (NER stage — historical German ornithologist's journals)
# ---------------------------------------------------------------------------
# Hybrid tagset for the Laubmann ornithologist journals.  Two parallel axes:
#   • TYPE  — the entity class (the BIO label backbone, KG node type)
#   • SCOPE — how the entity is referenced (EcoCor-style: individual, group,
#             body/plant part, product, named reference).  Carries the
#             specimen logic: a feather = Tier/Teil, a dried leaf = Pflanze/Teil,
#             an egg/clutch = Tier|Pflanze/Produkt.
# Entity-level attributes (count, scientific_name) live on the index/JSON, not
# in the token-level BIO export.
ENTITY_TYPES: Dict[str, str] = {
    "Tier": (
        "Jedes nicht-menschliche Tier — Vögel, Fische, Säuger, Insekten — ob "
        "Art/Taxon oder Trivialname (z. B. Schwalbe, Habicht, Raubseeschwalbe, "
        "Löffelreiher, Graugans, Bekassine, Wasserralle). Auch Tierteile (Feder, "
        "Flügel, Stoß) und Tierprodukte (Ei, Gelege) → über scope kennzeichnen. "
        "Ein wissenschaftliches Binomen wird NICHT als eigene Entität annotiert, "
        "sondern in das Feld scientific_name geschrieben."
    ),
    "Pflanze": (
        "Jede Pflanze, Baum, Strauch, Blume, Gras — als Taxon oder Trivialname "
        "(z. B. Eiche, Schilf, Fichte, Konifere). Auch Pflanzenteile (Blatt, Ast, "
        "Knospe) und Pflanzenprodukte (Same, Frucht) → über scope kennzeichnen."
    ),
    "Lebensraum": (
        "Generischer Lebensraum / Biotop / Habitat OHNE Eigennamen (z. B. Wald, "
        "Ufer, Sumpf, Fischteich, Wiese, Speicherweiher, Garten, Westufer, Feld). "
        "Trägt der Ort einen geokodierbaren Eigennamen, gehört er zu Ort."
    ),
    "Material": (
        "Unbelebtes Natur- oder Werkstoffmaterial, das weder Lebewesen noch "
        "eigenständiger Lebensraum ist (z. B. Stein, Sand, Wasser als Substanz, "
        "Erz, Holz als Stoff)."
    ),
    "Klima": (
        "Wetter-, Klima- und atmosphärisches Phänomen (z. B. Wettersturz, Regen, "
        "Frost, Schnee, Sturm, Sonne, Herbsttag, Nebel)."
    ),
    "Person": (
        "Namentlich genannter Mensch — Beobachter, Sammler, Korrespondenten, "
        "zitierte Autoren (z. B. Dr. Hirt, Dr. Dietz, Adolf Müller, Dr. von Jäckel, "
        "Hellmayr, Schillinger). Titel/Anrede gehören in den Span. KEINE "
        "generischen Rollen (Forscher, Jäger, Leute, beide Forscher)."
    ),
    "Ort": (
        "Benannter, geokodierbarer geographischer Ort: Städte, Dörfer, Länder, "
        "Regionen sowie benannte Gewässer und Berge (z. B. München, Wien, Venedig, "
        "Ismaning, Bayern, Starnberger See, Inn, Alpen, Karlstein, Rosenheim). "
        "Eigenname ⇒ Ort; generisches Habitat ⇒ Lebensraum."
    ),
    "Organisation": (
        "Benannte Institution, Verein, Behörde oder Sammlung (z. B. "
        "Tierschutzverein, Universität, Forstamt, zoologische Staatssammlung)."
    ),
    "Datum": (
        "Datums- und Zeitangabe — Tag, Monat, Jahr, römische Monatszahlen, "
        "Jahreszeit+Jahr, Uhrzeit (z. B. 28. September 1931, 10. VIII. 1849, "
        "Herbst 1821, 7. VIII. 31, morgens gegen 5h). Eine bloße Jahreszeit ohne "
        "Jahr wird NICHT annotiert."
    ),
}

# Scope axis (parallel BIO column).  Default = Singular when not stated.
SCOPE_TYPES: Dict[str, str] = {
    "Singular": (
        "Einzelexemplar oder bloße Plural-/Verallgemeinerungsnennung (ein Habicht, "
        "die Schwalben, 7 Gänse, Sonnenschein). Plural ⇒ Singular, nicht Kollektiv."
    ),
    "Kollektiv": (
        "Echte kollektive Einheit, als Gruppe aufgefasst (Schwarm, Rudel, Herde, "
        "Zug, Tausende von Schwalben, Gebüsch)."
    ),
    "Teil": (
        "Bestandteil einer übergeordneten Entität — Tierteil (Feder, Flügel, Stoß), "
        "Pflanzenteil (Blatt, Ast), getrocknetes Präparat als Teil."
    ),
    "Produkt": (
        "Abgelöstes Erzeugnis eines Organismus (Ei, Gelege, Same, Frucht)."
    ),
    "Referenz": (
        "Eigenname / direkte Benennung eines spezifischen Einzelwesens "
        "(benannter oder beringter Vogel, Rasse)."
    ),
}
DEFAULT_SCOPE = "Singular"

# Entity colours and labels for the HTML viewer / GUI
ENTITY_COLORS: Dict[str, str] = {
    "Tier":         "#c62828",
    "Pflanze":      "#2e7d32",
    "Lebensraum":   "#00838f",
    "Material":     "#6d4c41",
    "Klima":        "#546e7a",
    "Person":       "#6a1b9a",
    "Ort":          "#1565c0",
    "Organisation": "#37474f",
    "Datum":        "#ef6c00",
}
ENTITY_LABELS: Dict[str, str] = {
    "Tier":         "Tiere",
    "Pflanze":      "Pflanzen",
    "Lebensraum":   "Lebensräume",
    "Material":     "Materialien",
    "Klima":        "Klima",
    "Person":       "Personen",
    "Ort":          "Orte",
    "Organisation": "Organisationen",
    "Datum":        "Datum/Zeit",
}
SCOPE_LABELS: Dict[str, str] = {
    "Singular":  "Singular",
    "Kollektiv": "Kollektiv",
    "Teil":      "Teil",
    "Produkt":   "Produkt",
    "Referenz":  "Referenz",
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
    ner_thinking_level: str = "medium"   # raised from "low" — dense pages under-recall at low
    ner_retries: int = 2
    ner_verify_pass: bool = True         # cheap second "what did you miss" pass (recall)
    ner_use_underline_hints: bool = True # feed <u>…</u> spans to the model as candidates
    ner_include_object_regions: bool = True  # tag specimens in ObjectRegion descriptions
    ner_include_image_regions: bool = True   # tag content in ImageRegion descriptions

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
