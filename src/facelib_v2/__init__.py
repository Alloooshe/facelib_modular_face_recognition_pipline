"""facelib_v2 package."""

from .config import PipelineConfig
from .core.pipeline import FacePipeline
from .interfaces.protocols import Aligner, Detector, Embedder, Matcher

__all__ = [
    "Aligner",
    "Detector",
    "Embedder",
    "FacePipeline",
    "Matcher",
    "PipelineConfig",
]
