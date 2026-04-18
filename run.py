#!/usr/bin/env python3
"""Run the journal processing pipeline.

Default mode (auto_mode):
    python run.py --input /path/to/scans --output /path/to/output

Force split-only (old workflow):
    python run.py --input … --output … --no-auto --split

Force double-page (no split):
    python run.py --input … --output … --no-auto --double-page

Pre-rotate all images 90° clockwise then split:
    python run.py --input … --output … --no-auto --split --rotate 90

Requires:
    - GOOGLE_API_KEY environment variable set
    - pip install google-genai Pillow
"""

import argparse
import logging
import sys
from pathlib import Path

from journal_processor import Pipeline, PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process digitised journal scans.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Directory containing scan images.",
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Output directory (will be created).",
    )
    parser.add_argument(
        "--model", default="gemini-3-flash-preview",
        help="Gemini model ID (default: gemini-3-flash-preview).",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Parallel page-processing threads (default: 4).",
    )

    # Processing mode
    mode_group = parser.add_argument_group("processing mode (default: auto)")
    mode_group.add_argument(
        "--no-auto", action="store_true",
        help=(
            "Disable agentic scan analysis.  Must be combined with --split or "
            "--double-page to choose a fixed mode for all scans."
        ),
    )
    mode_group.add_argument(
        "--split", action="store_true",
        help=(
            "Force split mode: all scans are cut down the centre into left/right "
            "pages before detection.  Only used with --no-auto."
        ),
    )
    mode_group.add_argument(
        "--double-page", action="store_true",
        help=(
            "Force double-page mode: all scans are sent to the detector unsplit. "
            "Only used with --no-auto."
        ),
    )

    parser.add_argument(
        "--rotate", type=int, choices=[0, 90, 180, 270], default=0,
        help=(
            "Pre-rotate all input images by this many degrees clockwise before "
            "splitting or full-page processing (0, 90, 180, or 270).  "
            "Ignored in auto mode (the analyzer detects rotation automatically)."
        ),
    )
    parser.add_argument(
        "--max-regions", type=int, default=None,
        help=(
            "Maximum regions per image. "
            "Defaults: 5 for split images, 10 for double-page/unsplit images."
        ),
    )
    parser.add_argument(
        "--deskew", action="store_true",
        help="Enable deskew pre-processing (requires scipy).",
    )
    parser.add_argument(
        "--enhance-contrast", action="store_true",
        help="Enable auto-contrast enhancement.",
    )
    parser.add_argument(
        "--no-md", action="store_true", help="Skip Markdown output.",
    )
    parser.add_argument(
        "--no-pagexml", action="store_true", help="Skip PAGE XML output.",
    )
    parser.add_argument(
        "--no-sharegpt", action="store_true", help="Skip ShareGPT JSONL output.",
    )
    parser.add_argument(
        "--use-glm-ocr", action="store_true",
        help="Use fine-tuned GLM-OCR for text transcription (requires GPU).",
    )
    parser.add_argument(
        "--glm-ocr-base", default="zai-org/GLM-OCR",
        help="GLM-OCR base model ID (default: zai-org/GLM-OCR).",
    )
    parser.add_argument(
        "--glm-ocr-lora", default="",
        help="Path to LoRA adapter directory for GLM-OCR.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Debug logging.",
    )

    args = parser.parse_args()

    # Validate mode flags
    if args.split and args.double_page:
        parser.error("--split and --double-page are mutually exclusive.")
    if (args.split or args.double_page) and not args.no_auto:
        parser.error("--split and --double-page require --no-auto.")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    auto_mode = not args.no_auto
    double_page_mode = args.double_page  # only meaningful when auto_mode=False

    cfg_kwargs: dict = dict(
        input_dir=args.input,
        output_dir=args.output,
        model_id=args.model,
        workers=args.workers,
        auto_mode=auto_mode,
        double_page_mode=double_page_mode,
        deskew=args.deskew,
        enhance_contrast=args.enhance_contrast,
        output_md=not args.no_md,
        output_pagexml=not args.no_pagexml,
        output_sharegpt=not args.no_sharegpt,
        use_glm_ocr=args.use_glm_ocr,
        glm_ocr_base_model=args.glm_ocr_base,
        glm_ocr_lora_path=args.glm_ocr_lora,
        force_rotation=args.rotate,
    )
    if args.max_regions is not None:
        cfg_kwargs["max_regions"] = args.max_regions

    cfg = PipelineConfig(**cfg_kwargs)

    pipeline = Pipeline(cfg)
    summary = pipeline.run()

    if summary.get("errors"):
        print(f"\n⚠  Completed with {len(summary['errors'])} error(s).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
