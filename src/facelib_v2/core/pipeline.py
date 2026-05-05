from __future__ import annotations

import numpy as np

from facelib_v2.config import PipelineConfig
from facelib_v2.interfaces.protocols import Aligner, Detector, Embedder, Matcher


class FacePipeline:
    """Composable v2 pipeline that orchestrates detector->aligner->embedder->matcher."""

    def __init__(
        self,
        detector: Detector,
        aligner: Aligner,
        embedder: Embedder,
        matcher: Matcher,
        config: PipelineConfig | None = None,
    ) -> None:
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.matcher = matcher
        self.config = config or PipelineConfig()

    def process_image(self, image: np.ndarray) -> dict[str, object]:
        detections = self.detector.detect(image)
        if not detections:
            return {"detections": [], "matches": [], "embeddings": np.empty((0, 0))}

        aligned_faces = self.aligner.align(image, detections)
        embeddings = self.embedder.embed(aligned_faces)
        matches = self.matcher.match(embeddings)
        return {"detections": detections, "matches": matches, "embeddings": embeddings}
