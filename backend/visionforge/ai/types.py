"""VisionForge AI Core Type Definitions and Metadata Specifications."""

from enum import StrEnum

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    """Computer vision task domain classification."""

    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    DEPTH_ESTIMATION = "depth_estimation"
    OCR = "ocr"
    VLM = "vlm"  # Vision-Language Models
    RETRIEVAL = "retrieval"
    VIDEO_UNDERSTANDING = "video_understanding"
    RECONSTRUCTION_3D = "reconstruction_3d"


class InputType(StrEnum):
    """Supported input modality types."""

    IMAGE = "image"
    VIDEO = "video"
    TEXT_PROMPT = "text_prompt"
    TENSOR = "tensor"
    MULTIMODAL = "multimodal"


class OutputType(StrEnum):
    """Supported output payload types."""

    BOUNDING_BOXES = "bounding_boxes"
    SEGMENTATION_MASKS = "segmentation_masks"
    DEPTH_MAP = "depth_map"
    TEXT = "text"
    EMBEDDINGS = "embeddings"
    VIDEO_ANNOTATIONS = "video_annotations"
    POINT_CLOUD = "point_cloud"


class ModelStatus(StrEnum):
    """Lifecycle execution state of a model."""

    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    UNLOADED = "unloaded"


class MemoryRequirements(BaseModel):
    """Memory and resource footprint specification for a model."""

    vram_mb: int = Field(default=0, description="Estimated VRAM memory requirement in megabytes")
    ram_mb: int = Field(default=512, description="Estimated system RAM requirement in megabytes")
    disk_space_mb: int = Field(
        default=0, description="Checkpoint disk storage requirement in megabytes"
    )


class ModelMetadata(BaseModel):
    """Declarative metadata model defining a vision model specification."""

    name: str = Field(description="Unique model identifier name (e.g. 'yolo-v8-detector')")
    version: str = Field(default="1.0.0", description="Semantic model version")
    author: str = Field(default="VisionForge", description="Model creator or maintainer entity")
    task: TaskType = Field(description="Primary computer vision task classification")
    license: str = Field(default="MIT", description="Model license classification")
    supported_input_types: list[InputType] = Field(
        default_factory=lambda: [InputType.IMAGE],
        description="Supported input payload modalities",
    )
    supported_output_types: list[OutputType] = Field(
        default_factory=list,
        description="Supported output payload modalities",
    )
    memory_requirements: MemoryRequirements = Field(
        default_factory=MemoryRequirements,
        description="Hardware memory footprint specs",
    )
    device_support: list[str] = Field(
        default_factory=lambda: ["cpu", "cuda", "mps"],
        description="Supported hardware acceleration backends",
    )
    description: str = Field(default="", description="Detailed model capabilities summary")
    status: ModelStatus = Field(
        default=ModelStatus.UNINITIALIZED,
        description="Current lifecycle operational state",
    )
