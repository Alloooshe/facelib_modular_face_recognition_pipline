from typing import Any, Protocol

import numpy as np


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Return detected faces with at least a bounding box and score."""


class Aligner(Protocol):
    def align(self, image: np.ndarray, detections: list[dict[str, Any]]) -> list[np.ndarray]:
        """Return aligned face crops for each detection."""


class Embedder(Protocol):
    def embed(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        """Return embeddings with shape [N, D]."""


class Matcher(Protocol):
    def match(self, embeddings: np.ndarray) -> list[dict[str, Any]]:
        """Return identity predictions per embedding."""
