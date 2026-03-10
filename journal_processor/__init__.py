"""journal_processor — digitised journal page processing pipeline."""

from .config import PipelineConfig
from .pipeline import Pipeline

__all__ = ["Pipeline", "PipelineConfig"]
