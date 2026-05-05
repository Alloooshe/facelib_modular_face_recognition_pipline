# Usage guide

## Pipeline stages

`FacePipeline` requires four collaborators:

1. `Detector.detect(image)` -> list of detections
2. `Aligner.align(image, detections)` -> list of aligned crops
3. `Embedder.embed(aligned_faces)` -> embedding matrix `[N, D]`
4. `Matcher.match(embeddings)` -> identity predictions

## Typical flow

```python
result = pipeline.process_image(image)
detections = result["detections"]
matches = result["matches"]
embeddings = result["embeddings"]
```

## Configuration

Use `PipelineConfig` to control runtime behavior:

- `min_face_size`: minimum accepted face size in pixels
- `confidence_threshold`: drop detections below this score
- `max_faces_per_frame`: cap number of faces processed per frame

```python
from facelib_v2 import PipelineConfig, FacePipeline

config = PipelineConfig(
    min_face_size=30,
    confidence_threshold=0.9,
    max_faces_per_frame=10,
)

pipeline = FacePipeline(detector, aligner, embedder, matcher, config=config)
```
