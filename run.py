#!/usr/bin/env python3
"""Run the journal processing pipeline.

Usage:
    python run.py --input /path/to/scans --output /path/to/output

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
        description="Process digitised double-page journal scans."
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Directory containing double-page scan images.",
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
    parser.add_argument(
        "--max-regions", type=int, default=5,
        help="Maximum regions per page (default: 5).",
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
        "--verbose", "-v", action="store_true", help="Debug logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = PipelineConfig(
        input_dir=args.input,
        output_dir=args.output,
        model_id=args.model,
        workers=args.workers,
        max_regions=args.max_regions,
        deskew=args.deskew,
        enhance_contrast=args.enhance_contrast,
        output_md=not args.no_md,
        output_pagexml=not args.no_pagexml,
        output_sharegpt=not args.no_sharegpt,
    )

    pipeline = Pipeline(cfg)
    summary = pipeline.run()

    if summary.get("errors"):
        print(f"\n⚠  Completed with {len(summary['errors'])} error(s).", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
