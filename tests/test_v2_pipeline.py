import numpy as np

from facelib_v2 import FacePipeline


class _Detector:
    def detect(self, image: np.ndarray) -> list[dict]:
        return [{"bbox": [0, 0, 5, 5], "score": 0.95}]


class _Aligner:
    def align(self, image: np.ndarray, detections: list[dict]) -> list[np.ndarray]:
        return [np.zeros((112, 112, 3), dtype=np.uint8)]


class _Embedder:
    def embed(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        return np.ones((1, 128), dtype=np.float32)


class _Matcher:
    def match(self, embeddings: np.ndarray) -> list[dict]:
        return [{"identity": "x", "confidence": 0.9}]


def test_pipeline_runs_end_to_end() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    pipeline = FacePipeline(_Detector(), _Aligner(), _Embedder(), _Matcher())
    result = pipeline.process_image(image)

    assert len(result["detections"]) == 1
    assert len(result["matches"]) == 1
