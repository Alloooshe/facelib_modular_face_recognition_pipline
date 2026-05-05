from typing import Any, Protocol

import numpy as np


class Detector(Protocol):
    """Face detector interface.

    Implementations receive an image and return zero or more detection records.
    Each record should include at least:
    - `bbox`: bounding box in pixel coordinates
    - `score`: detector confidence in [0, 1]
    """

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Detect faces in a single image.

        Args:
            image: Input RGB or BGR image as a NumPy array.

        Returns:
            A list of detection dictionaries. Typical keys are `bbox`, `score`,
            and optionally landmarks or metadata.
        """


class Aligner(Protocol):
    """Face aligner interface.

    Implementations transform raw detections into normalized face crops that are
    suitable for embedding extraction.
    """

    def align(self, image: np.ndarray, detections: list[dict[str, Any]]) -> list[np.ndarray]:
        """Align detected faces into canonical crops.

        Args:
            image: Original image that was used for detection.
            detections: Detector outputs associated with the image.

        Returns:
            A list of aligned face images, one entry per accepted detection.
        """


class Embedder(Protocol):
    """Face embedder interface.

    Implementations convert aligned face crops into numerical embeddings.
    """

    def embed(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        """Compute embeddings for aligned faces.

        Args:
            aligned_faces: List of aligned face crops.

        Returns:
            A NumPy array shaped `[N, D]` where `N` is number of faces and `D`
            is embedding dimensionality.
        """


class Matcher(Protocol):
    """Face matcher interface.

    Implementations map embeddings to identities or nearest gallery entries.
    """

    def match(self, embeddings: np.ndarray) -> list[dict[str, Any]]:
        """Match embeddings against a recognition backend.

        Args:
            embeddings: Embedding matrix shaped `[N, D]`.

        Returns:
            A list of prediction dictionaries, typically including identity and
            confidence or distance fields.
        """
