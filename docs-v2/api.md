# API reference

## `facelib_v2.FacePipeline`

### `__init__(detector, aligner, embedder, matcher, config=None)`

Creates a new pipeline using pluggable stage implementations.

### `process_image(image)`

Runs detector -> aligner -> embedder -> matcher and returns:

- `detections`: list
- `matches`: list
- `embeddings`: NumPy array

## `facelib_v2.PipelineConfig`

Pydantic model for pipeline runtime configuration.

Fields:

- `min_face_size: int`
- `confidence_threshold: float`
- `max_faces_per_frame: int`

## Protocol interfaces

### `facelib_v2.interfaces.Detector`
- `detect(image) -> list[dict]`

### `facelib_v2.interfaces.Aligner`
- `align(image, detections) -> list[np.ndarray]`

### `facelib_v2.interfaces.Embedder`
- `embed(aligned_faces) -> np.ndarray`

### `facelib_v2.interfaces.Matcher`
- `match(embeddings) -> list[dict]`
