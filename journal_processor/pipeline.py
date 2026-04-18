"""Main processing pipeline.

Processing modes
----------------

auto_mode=True (default)
    A ScanAnalyzer call inspects each raw scan and decides:
      • use_double_page=True  → send full scan to detector (no split)
      • use_double_page=False → split down the centre first
    Different scans in the same batch can be routed differently.

auto_mode=False, double_page_mode=False
    All scans are split down the centre (original workflow).

auto_mode=False, double_page_mode=True
    All scans are sent to the detector unsplit.

Flow per scan:
    [analyse] → [split or copy] → [preprocess] → [detect] → [transcribe] → [output]
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .config import PipelineConfig
from .scan_analyzer import ScanAnalyzer
from .splitter import split_double_page
from .preprocessor import preprocess_page
from .region_detector import RegionDetector
from .transcriber import Transcriber
from .output_md import generate_md
from .output_pagexml import generate_pagexml
from .output_sharegpt import append_sharegpt, build_sharegpt_entries
from .utils import natural_sort_key, MIME_BY_EXT

log = logging.getLogger(__name__)

# Supported input image extensions
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


class Pipeline:
    """End-to-end journal processing pipeline."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self._init_client()

    # ── Gemini client + components ──────────────────────────────────────

    def _init_client(self) -> None:
        from google import genai

        self.client = genai.Client(
            http_options={"api_version": "v1alpha"},
        )
        self.analyzer = ScanAnalyzer(self.client, self.cfg)
        self.detector = RegionDetector(self.client, self.cfg)

        gemini_transcriber = Transcriber(self.client, self.cfg)

        if self.cfg.use_glm_ocr:
            from .transcriber_glm_ocr import GlmOcrTranscriber
            if self.cfg.workers > 1:
                log.warning(
                    "GLM-OCR runs on GPU → forcing workers=1 (was %d).",
                    self.cfg.workers,
                )
                self.cfg.workers = 1
            self.transcriber = GlmOcrTranscriber(
                self.cfg, gemini_fallback=gemini_transcriber,
            )
        else:
            self.transcriber = gemini_transcriber

    # ── Full run ────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute the complete pipeline.  Returns a summary dict."""
        t0 = time.time()
        summary: Dict[str, Any] = {
            "mode": "auto" if self.cfg.auto_mode else (
                "double_page" if self.cfg.double_page_mode else "split"
            ),
            "pages_processed": 0,
            "errors": [],
        }

        # Stage 1 — collect raw scans
        scans = self._find_scans()
        if not scans:
            log.error("No images found in %s", self.cfg.input_dir)
            return summary
        log.info("Found %d scan(s) in %s", len(scans), self.cfg.input_dir)

        # Stage 2 — per-scan analysis (auto_mode) or static routing
        log.info("=== Stage 2: Routing scans ===")
        scan_tasks = self._route_scans(scans, summary)
        # scan_tasks: List[(page_path, use_double_page)]

        if not scan_tasks:
            return summary

        # Stage 3 — preprocess working copies
        log.info("=== Stage 3: Pre-processing %d page images ===", len(scan_tasks))
        for page_path, _ in scan_tasks:
            preprocess_page(page_path, self.cfg)

        # Stage 4-6 — detect → transcribe → output
        log.info("=== Stages 4-6: Detect → Transcribe → Output ===")
        sharegpt_path = self.cfg.output_dir / "sharegpt" / "training_data.jsonl"
        if sharegpt_path.exists():
            sharegpt_path.unlink()

        if self.cfg.workers > 1:
            self._run_parallel(scan_tasks, sharegpt_path, summary)
        else:
            self._run_sequential(scan_tasks, sharegpt_path, summary)

        elapsed = time.time() - t0
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["pages_processed"] = len(scan_tasks) - len(summary["errors"])

        summary_path = self.cfg.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        log.info(
            "Done. %d page images in %.0fs (%d errors)",
            summary["pages_processed"], elapsed, len(summary["errors"]),
        )
        return summary

    # ── Rotation helper ─────────────────────────────────────────────────

    @staticmethod
    def _apply_rotation(scan: Path, pages_dir: Path, rotation: int) -> Path:
        """Return a rotated copy of *scan* saved under the original stem in *pages_dir*.

        Using the original stem ensures that downstream split_double_page produces
        clean _L / _R names (e.g. uuid_0048_L.png) rather than uuid_0048_rot_L.png.
        Returns *scan* itself when no rotation is needed.
        """
        if not rotation:
            return scan
        img = Image.open(scan).convert("RGB")
        img = img.rotate(-rotation, expand=True)
        dest = pages_dir / (scan.stem + ".png")
        img.save(dest, "PNG")
        log.info("Rotated %s by %d° → %s", scan.name, rotation, dest.name)
        return dest

    # ── Scan discovery ───────────────────────────────────────────────────

    def _find_scans(self) -> List[Path]:
        scans = sorted(
            [p for p in self.cfg.input_dir.iterdir()
             if p.suffix.lower() in _IMAGE_EXTS],
            key=natural_sort_key,
        )
        return scans

    # ── Routing: analysis → page list ───────────────────────────────────

    def _route_scans(
        self, scans: List[Path], summary: Dict
    ) -> List[Tuple[Path, bool]]:
        """Analyse each scan and prepare working copies in pages/.

        Returns a list of (page_path, use_double_page) tuples.
        page_path is a PNG under output_dir/pages/ ready for detection.
        """
        pages_dir = self.cfg.output_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        analysis_log: Dict[str, Any] = {}
        tasks: List[Tuple[Path, bool]] = []

        for scan in scans:
            # Determine routing for this scan
            if self.cfg.auto_mode:
                log.info("Analysing %s …", scan.name)
                analysis = self.analyzer.analyse(scan)
                analysis_log[scan.name] = analysis
                use_double_page: bool = analysis["use_double_page"]
            else:
                use_double_page = self.cfg.double_page_mode
                analysis = None

            rotation = (
                analysis.get("rotation", 0)
                if analysis
                else self.cfg.force_rotation
            )

            if not use_double_page:
                # Split path: produce _L and _R working copies
                try:
                    src = self._apply_rotation(scan, pages_dir, rotation)
                    left, right = split_double_page(
                        src, pages_dir, self.cfg.split_overlap_px
                    )
                    if src != scan:
                        src.unlink(missing_ok=True)
                    tasks.append((left, False))
                    tasks.append((right, False))
                except Exception as exc:
                    log.error("Splitting failed for %s: %s", scan.name, exc)
                    summary["errors"].append({"page": scan.name, "error": f"split: {exc}"})
            else:
                # No-split path: convert/copy to pages/ dir
                dest = pages_dir / (scan.stem + ".png")
                try:
                    if not dest.exists() or dest.stat().st_mtime < scan.stat().st_mtime:
                        img = Image.open(scan).convert("RGB")
                        if rotation:
                            img = img.rotate(-rotation, expand=True)
                            log.info("Rotated %s by %d°", scan.name, rotation)
                        img.save(dest, "PNG")
                    tasks.append((dest, True))
                except Exception as exc:
                    log.error("Copying scan failed for %s: %s", scan.name, exc)
                    summary["errors"].append({"page": scan.name, "error": f"copy: {exc}"})

        # Save analysis decisions for inspection
        if analysis_log:
            analysis_path = self.cfg.output_dir / "scan_analysis.json"
            analysis_path.write_text(
                json.dumps(analysis_log, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info("Scan analysis saved → %s", analysis_path.name)

        log.info(
            "Routing complete: %d page images (%d split, %d unsplit)",
            len(tasks),
            sum(1 for _, dp in tasks if not dp),
            sum(1 for _, dp in tasks if dp),
        )
        summary["routing"] = {
            "total_page_images": len(tasks),
            "split": sum(1 for _, dp in tasks if not dp),
            "unsplit": sum(1 for _, dp in tasks if dp),
        }
        return tasks

    # ── Sequential / parallel dispatch ──────────────────────────────────

    def _run_sequential(
        self,
        tasks: List[Tuple[Path, bool]],
        sharegpt_path: Path,
        summary: Dict,
    ) -> None:
        for idx, (page_path, use_dp) in enumerate(tasks, 1):
            log.info("[%d/%d] %s", idx, len(tasks), page_path.name)
            try:
                self._process_page(page_path, use_dp, sharegpt_path)
            except Exception as exc:
                log.error("Failed %s: %s", page_path.name, exc)
                summary["errors"].append({"page": page_path.name, "error": str(exc)})

    def _run_parallel(
        self,
        tasks: List[Tuple[Path, bool]],
        sharegpt_path: Path,
        summary: Dict,
    ) -> None:
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as pool:
            futures = {
                pool.submit(self._process_page, pp, dp, sharegpt_path): pp
                for pp, dp in tasks
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                pp = futures[future]
                try:
                    future.result()
                    log.info("[%d/%d] ✓ %s", done, len(tasks), pp.name)
                except Exception as exc:
                    log.error("[%d/%d] ✗ %s: %s", done, len(tasks), pp.name, exc)
                    summary["errors"].append({"page": pp.name, "error": str(exc)})

    # ── Single-page processing ──────────────────────────────────────────

    def _process_page(
        self,
        page_path: Path,
        use_double_page: bool,
        sharegpt_path: Path,
    ) -> None:
        pid = page_path.stem
        page_img = Image.open(page_path).convert("RGB")

        # Determine effective max_regions for this call.
        # Double-page images get a higher budget when the user hasn't
        # overridden from the single-page default (5).
        from .config import MAX_REGIONS_PER_PAGE, MAX_REGIONS_DOUBLE_PAGE
        if use_double_page and self.cfg.max_regions == MAX_REGIONS_PER_PAGE:
            effective_max = MAX_REGIONS_DOUBLE_PAGE
        else:
            effective_max = self.cfg.max_regions

        det = self.detector.detect(
            page_path,
            use_double_page=use_double_page,
            max_regions=effective_max,
        )

        if det["status"] != "success":
            raise RuntimeError(f"Detection failed: {det.get('error', 'unknown')}")

        regions = det["regions"]
        dims = det["image_dimensions"]

        # Save region crops
        regions_dir = self.cfg.output_dir / "regions" / pid
        regions_dir.mkdir(parents=True, exist_ok=True)

        # Transcription (per region)
        for r in regions:
            bbox = r["bbox"]
            crop = page_img.crop((
                bbox["x"], bbox["y"],
                bbox["x"] + bbox["width"],
                bbox["y"] + bbox["height"],
            ))
            crop_path = regions_dir / f"{r['id']}_{r['type']}.png"
            crop.save(crop_path, "PNG")

            result = self.transcriber.transcribe_region(crop, r)
            r["transcription"] = result

        # Output: Markdown
        if self.cfg.output_md:
            generate_md(pid, regions, self.cfg.output_dir / "md")

        # Output: PageXML
        if self.cfg.output_pagexml:
            generate_pagexml(
                pid, regions, dims, page_path.name,
                self.cfg.output_dir / "pagexml",
            )

        # Output: ShareGPT
        if self.cfg.output_sharegpt:
            sharegpt_images_dir = self.cfg.output_dir / "sharegpt" / "images"
            entries = build_sharegpt_entries(
                pid, page_img, regions, self.cfg, sharegpt_images_dir
            )
            if entries:
                append_sharegpt(entries, sharegpt_path)

        # Debug JSON: detection + transcription for every region
        page_json = self.cfg.output_dir / "regions" / f"{pid}.json"
        serialisable = []
        for r in regions:
            sr = dict(r)
            if "transcription" in sr:
                sr["transcription"] = {
                    k: v for k, v in sr["transcription"].items()
                    if isinstance(v, (str, int, float, bool, type(None)))
                }
            serialisable.append(sr)
        page_json.write_text(
            json.dumps(serialisable, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
