from __future__ import annotations

import numpy as np

from facelib_v2.config import PipelineConfig
from facelib_v2.interfaces.protocols import Aligner, Detector, Embedder, Matcher


class FacePipeline:
    """Composable v2 pipeline that orchestrates detector->aligner->embedder->matcher.

    Usage:
        Instantiate with concrete implementations of the four stage interfaces,
        then call `process_image()` for each frame or image.
    """

    def __init__(
        self,
        detector: Detector,
        aligner: Aligner,
        embedder: Embedder,
        matcher: Matcher,
        config: PipelineConfig | None = None,
    ) -> None:
        """Create a new pipeline instance.

        Args:
            detector: Stage that detects faces and returns detection records.
            aligner: Stage that converts detections to normalized face crops.
            embedder: Stage that converts aligned crops to embeddings.
            matcher: Stage that maps embeddings to identities.
            config: Optional runtime settings. Defaults to `PipelineConfig()`.
        """
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.matcher = matcher
        self.config = config or PipelineConfig()

    def process_image(self, image: np.ndarray) -> dict[str, object]:
        """Run one full recognition pass on a single image.

        Args:
            image: Input image as a NumPy array.

        Returns:
            Dictionary with:
            - `detections`: list of detector outputs
            - `matches`: list of matcher outputs
            - `embeddings`: embedding matrix for detected/aligned faces
        """
        detections = self.detector.detect(image)
        if not detections:
            return {"detections": [], "matches": [], "embeddings": np.empty((0, 0))}

        aligned_faces = self.aligner.align(image, detections)
        embeddings = self.embedder.embed(aligned_faces)
        matches = self.matcher.match(embeddings)
        return {"detections": detections, "matches": matches, "embeddings": embeddings}
