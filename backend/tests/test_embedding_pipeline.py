"""Unit test suite for Image Embedding Pipeline stages."""

import io

import pytest
from PIL import Image

from visionforge.engine.context import ExecutionContext
from visionforge.engine.exceptions import TaskValidationError
from visionforge.engine.pipelines.embedding_pipeline import (
    EmbeddingPipeline,
    ImagePreprocessingStage,
    ImageValidationStage,
)


@pytest.fixture
def sample_image_bytes():
    """Generate JPEG image bytes for validation testing."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_image_validation_stage_success(sample_image_bytes):
    """Test validation stage with valid JPEG image bytes."""
    stage = ImageValidationStage()
    ctx = ExecutionContext(task_id="test-val-1", task_type="retrieval")

    result_img = await stage.process(ctx, sample_image_bytes)
    assert isinstance(result_img, Image.Image)
    assert result_img.width == 100
    assert result_img.height == 100


@pytest.mark.asyncio
async def test_image_validation_stage_invalid():
    """Test validation stage error handling for corrupted inputs."""
    stage = ImageValidationStage()
    ctx = ExecutionContext(task_id="test-val-2", task_type="retrieval")

    with pytest.raises(TaskValidationError):
        await stage.process(ctx, b"invalid_not_an_image_data")

    with pytest.raises(TaskValidationError):
        await stage.process(ctx, None)


@pytest.mark.asyncio
async def test_image_preprocessing_stage():
    """Test preprocessing stage converting RGBA to RGB."""
    stage = ImagePreprocessingStage()
    ctx = ExecutionContext(task_id="test-prep-1", task_type="retrieval")

    rgba_img = Image.new("RGBA", (50, 50), color=(0, 255, 0, 128))
    rgb_img = await stage.process(ctx, rgba_img)
    assert rgb_img.mode == "RGB"


@pytest.mark.asyncio
async def test_full_embedding_pipeline(sample_image_bytes):
    """Test end-to-end embedding pipeline run."""
    pipeline = EmbeddingPipeline()
    ctx = ExecutionContext(
        task_id="test-pipe-1",
        task_type="retrieval",
        model_name="siglip-base-patch16-224",
    )

    from visionforge.engine.metrics import MetricsCollector

    metrics = MetricsCollector(device_used="cpu")
    result = await pipeline.run(ctx, sample_image_bytes, metrics)

    assert result.success is True
    assert result.data is not None
    assert result.data.dimension == 768
