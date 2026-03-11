"""Per-region transcription using a fine-tuned GLM-OCR model (local GPU).

Only *text* regions are handled locally:
    ParagraphRegion, ListRegion, FootnoteRegion, MarginaliaRegion

Everything else (TableRegion, ImageRegion, ObjectRegion, PageNumberRegion)
is forwarded unchanged to the Gemini-based ``Transcriber`` so that tables
get Markdown formatting and images/objects get descriptive output.
"""

import logging
import os
import tempfile
from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

from .config import PipelineConfig, TEXT_REGION_TYPES

log = logging.getLogger(__name__)


class GlmOcrTranscriber:
    """Drop-in replacement for :class:`Transcriber`.

    Loads the GLM-OCR base model + optional LoRA adapter once at init.
    Regions whose ``type`` is in :data:`TEXT_REGION_TYPES` are transcribed
    locally; all other region types are delegated to *gemini_fallback*.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        gemini_fallback: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.gemini_fallback = gemini_fallback

        log.info("Loading GLM-OCR base model: %s", cfg.glm_ocr_base_model)
        self.processor = AutoProcessor.from_pretrained(
            cfg.glm_ocr_base_model, trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.glm_ocr_base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if cfg.glm_ocr_lora_path:
            log.info("Loading LoRA adapter from: %s", cfg.glm_ocr_lora_path)
            self.model = PeftModel.from_pretrained(self.model, cfg.glm_ocr_lora_path)

        self.model.eval()
        log.info("GLM-OCR ready  (text regions → local, others → Gemini).")

    # ── public API (same signature as Transcriber.transcribe_region) ──

    def transcribe_region(
        self,
        region_image: Image.Image,
        region: Dict[str, Any],
    ) -> Dict[str, Any]:
        rtype = region["type"]

        # ---- Text regions → local GLM-OCR ----
        if rtype in TEXT_REGION_TYPES:
            return self._glm_ocr_call(region_image, region)

        # ---- Everything else → Gemini fallback ----
        if self.gemini_fallback is not None:
            return self.gemini_fallback.transcribe_region(region_image, region)

        # No fallback available (should not happen in normal usage)
        log.warning("No Gemini fallback for region type %s — skipping.", rtype)
        return {"status": "skipped", "text": "", "note": f"No handler for {rtype}"}

    # ── local GLM-OCR inference ──────────────────────────────────────

    def _glm_ocr_call(
        self,
        image: Image.Image,
        region: Dict[str, Any],
    ) -> Dict[str, Any]:
        rtype = region["type"]

        # Write image to a temp file — the HF processor resolves file paths
        # more reliably than raw PIL objects for this model.
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        try:
            image.save(tmp_path, format="PNG")
            os.close(fd)

            # Build messages matching the fine-tuning chat format exactly:
            #   system: "Transcribe the german text in this image region."
            #   user:   <image> + "Region type: ParagraphRegion"
            messages = [
                {
                    "role": "system",
                    "content": self.cfg.sharegpt_system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": tmp_path},
                        {"type": "text", "text": f"Region type: {rtype}"},
                    ],
                },
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            inputs.pop("token_type_ids", None)

            with torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.glm_ocr_max_new_tokens,
                )

            text = self.processor.decode(
                gen_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            return {"status": "success", "text": text}

        except Exception as exc:
            log.error("GLM-OCR failed on %s region: %s", rtype, exc)
            return {"status": "error", "error": str(exc), "text": ""}

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
