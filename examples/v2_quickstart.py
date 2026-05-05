import numpy as np

from facelib_v2 import FacePipeline


class DummyDetector:
    def detect(self, image: np.ndarray) -> list[dict]:
        return [{"bbox": [10, 10, 100, 100], "score": 0.99}]


class DummyAligner:
    def align(self, image: np.ndarray, detections: list[dict]) -> list[np.ndarray]:
        return [np.zeros((112, 112, 3), dtype=np.uint8) for _ in detections]


class DummyEmbedder:
    def embed(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        return np.ones((len(aligned_faces), 512), dtype=np.float32)


class DummyMatcher:
    def match(self, embeddings: np.ndarray) -> list[dict]:
        return [{"identity": "person_a", "confidence": 0.88} for _ in range(len(embeddings))]


if __name__ == "__main__":
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    pipeline = FacePipeline(
        detector=DummyDetector(),
        aligner=DummyAligner(),
        embedder=DummyEmbedder(),
        matcher=DummyMatcher(),
    )
    result = pipeline.process_image(image)
    print(result["matches"])
