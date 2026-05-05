from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Runtime settings for the v2 recognition pipeline.

    Attributes:
        min_face_size: Minimum accepted face size in pixels.
        confidence_threshold: Minimum confidence for accepted detections.
        max_faces_per_frame: Upper bound on processed detections per frame.
    """

    min_face_size: int = Field(default=20, ge=1)
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_faces_per_frame: int = Field(default=20, ge=1)
