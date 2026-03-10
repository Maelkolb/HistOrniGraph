"""Main processing pipeline.

Orchestrates:  split → preprocess → detect → transcribe → output
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from .config import PipelineConfig
from .splitter import split_all
from .preprocessor import preprocess_page
from .region_detector import RegionDetector
from .transcriber import Transcriber
from .output_md import generate_md
from .output_pagexml import generate_pagexml
from .output_sharegpt import append_sharegpt, build_sharegpt_entries

log = logging.getLogger(__name__)


class Pipeline:
    """End-to-end journal processing pipeline."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self._init_client()

    # ── Gemini client ───────────────────────────────────────────────────

    def _init_client(self) -> None:
        from google import genai

        self.client = genai.Client(
            http_options={"api_version": "v1alpha"},
        )
        self.detector = RegionDetector(self.client, self.cfg)
        self.transcriber = Transcriber(self.client, self.cfg)

    # ── Full run ────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Execute the complete pipeline.  Returns a summary dict."""
        t0 = time.time()
        summary: Dict[str, Any] = {"pages_processed": 0, "errors": []}

        # 1 — Split
        log.info("=== Stage 1: Splitting double pages ===")
        page_paths = split_all(self.cfg)
        if not page_paths:
            log.error("No pages produced after splitting.")
            return summary

        # 2 — Preprocess
        log.info("=== Stage 2: Pre-processing %d pages ===", len(page_paths))
        for pp in page_paths:
            preprocess_page(pp, self.cfg)

        # 3+4+5 — Detect + Transcribe + Output (per page)
        log.info("=== Stages 3-5: Detect → Transcribe → Output ===")
        sharegpt_path = self.cfg.output_dir / "sharegpt" / "training_data.jsonl"
        # Clear previous JSONL
        if sharegpt_path.exists():
            sharegpt_path.unlink()

        if self.cfg.workers > 1:
            self._run_parallel(page_paths, sharegpt_path, summary)
        else:
            self._run_sequential(page_paths, sharegpt_path, summary)

        elapsed = time.time() - t0
        summary["elapsed_seconds"] = round(elapsed, 1)
        summary["pages_processed"] = len(page_paths) - len(summary["errors"])

        # Write summary
        summary_path = self.cfg.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        log.info(
            "Done. %d pages in %.0fs (%d errors)",
            summary["pages_processed"], elapsed, len(summary["errors"]),
        )
        return summary

    # ── Sequential / parallel helpers ───────────────────────────────────

    def _run_sequential(
        self, page_paths: List[Path], sharegpt_path: Path, summary: Dict
    ) -> None:
        for idx, pp in enumerate(page_paths, 1):
            log.info("[%d/%d] %s", idx, len(page_paths), pp.name)
            try:
                self._process_page(pp, sharegpt_path)
            except Exception as exc:
                log.error("Failed %s: %s", pp.name, exc)
                summary["errors"].append({"page": pp.name, "error": str(exc)})

    def _run_parallel(
        self, page_paths: List[Path], sharegpt_path: Path, summary: Dict
    ) -> None:
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as pool:
            futures = {
                pool.submit(self._process_page, pp, sharegpt_path): pp
                for pp in page_paths
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                pp = futures[future]
                try:
                    future.result()
                    log.info("[%d/%d] ✓ %s", done, len(page_paths), pp.name)
                except Exception as exc:
                    log.error("[%d/%d] ✗ %s: %s", done, len(page_paths), pp.name, exc)
                    summary["errors"].append({"page": pp.name, "error": str(exc)})

    # ── Single-page processing ──────────────────────────────────────────

    def _process_page(self, page_path: Path, sharegpt_path: Path) -> None:
        pid = page_path.stem  # e.g. "scan_042_L"
        page_img = Image.open(page_path).convert("RGB")

        # --- Region detection ---
        det = self.detector.detect(page_path)
        if det["status"] != "success":
            raise RuntimeError(f"Detection failed: {det.get('error', 'unknown')}")

        regions = det["regions"]
        dims = det["image_dimensions"]

        # Save region crops
        regions_dir = self.cfg.output_dir / "regions" / pid
        regions_dir.mkdir(parents=True, exist_ok=True)

        # --- Transcription (per region) ---
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

        # --- Output: Markdown ---
        if self.cfg.output_md:
            generate_md(pid, regions, self.cfg.output_dir / "md")

        # --- Output: PageXML ---
        if self.cfg.output_pagexml:
            generate_pagexml(
                pid, regions, dims, page_path.name,
                self.cfg.output_dir / "pagexml",
            )

        # --- Output: ShareGPT ---
        if self.cfg.output_sharegpt:
            sharegpt_images_dir = self.cfg.output_dir / "sharegpt" / "images"
            entries = build_sharegpt_entries(
                pid, page_img, regions, self.cfg, sharegpt_images_dir
            )
            if entries:
                append_sharegpt(entries, sharegpt_path)

        # Save detection + transcription JSON for debugging
        page_json = self.cfg.output_dir / "regions" / f"{pid}.json"
        serialisable = []
        for r in regions:
            sr = {k: v for k, v in r.items()}
            # Ensure transcription is serialisable
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
