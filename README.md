# facelib_modular_face_recognition_pipline

[![GitHub stars](https://img.shields.io/github/stars/Alloooshe/facelib_modular_face_recognition_pipline?style=social)](https://github.com/Alloooshe/facelib_modular_face_recognition_pipline/stargazers)

Star count: **9** (checked on 2026-05-05 via GitHub API).

> Legacy notice (v1.0.0): the original v1 codebase is archived and unsupported.
> No planned bug fixes, security updates, or dependency upgrades for v1.

This repository now contains:
- `v1` legacy implementation (historical reference)
- `v2` modern Python package scaffold under `src/facelib_v2`

## v2 goals

- Modern architecture with pluggable stages (detector, aligner, embedder, matcher)
- Clean package layout suitable for PyPI
- Type hints and testability first

## Quick start (v2, local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest
python examples/v2_quickstart.py
```

## Usage (v2 API)

```python
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

image = np.zeros((480, 640, 3), dtype=np.uint8)
pipeline = FacePipeline(DummyDetector(), DummyAligner(), DummyEmbedder(), DummyMatcher())
result = pipeline.process_image(image)
print(result["matches"])
```

## PyPI packaging and release

`pyproject.toml` is already configured for package builds.

```bash
pip install -U build twine
python -m build
twine check dist/*
twine upload dist/*
```

If package name `facelib-v2` is already taken on PyPI, change `project.name` in `pyproject.toml` before upload.

## v1 historical notes

The legacy project solved face recognition as independent stages:
- detection
- alignment
- embedding
- matching

It also supported video/live stream processing with tracking and decision accumulation.

## Demo (legacy v1)

[face recognition video](https://www.youtube.com/watch?v=kSNk_1QLzbQ)

## Acknowledgment

1. [MTCNN](https://github.com/ipazc/mtcnn)
2. [facenet](https://github.com/davidsandberg/facenet)
3. [SORT](https://github.com/abewley/sort)
4. [arcface](https://github.com/deepinsight/insightface)

