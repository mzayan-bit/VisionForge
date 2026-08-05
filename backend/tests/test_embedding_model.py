"""Unit test suite for SigLIP Image Embedding Model."""

import pytest
from PIL import Image

from visionforge.ai.models.siglip import SigLIPEmbeddingModel
from visionforge.ai.types import ModelStatus, TaskType


@pytest.fixture
def sample_pil_image():
    """Create a simple RGB image for testing."""
    return Image.new("RGB", (224, 224), color=(100, 150, 200))


@pytest.mark.asyncio
async def test_siglip_metadata_and_initialization():
    """Verify model metadata spec and lazy initial state."""
    model = SigLIPEmbeddingModel()
    assert model.status == ModelStatus.UNINITIALIZED

    await model.initialize()
    assert model.status == ModelStatus.INITIALIZED

    meta = model.metadata
    assert meta.name == "siglip-base-patch16-224"
    assert meta.task == TaskType.RETRIEVAL
    assert meta.version == "1.0.0"


@pytest.mark.asyncio
async def test_siglip_load_predict_unload(sample_pil_image):
    """Test full model load, prediction, L2 normalization, and memory unload cycle."""
    model = SigLIPEmbeddingModel()
    await model.initialize()

    # Predict triggers auto-load
    result = await model.predict(sample_pil_image, device="cpu")
    assert result.success is True
    assert result.data is not None

    data = result.data
    assert data.dimension == 768
    assert len(data.embedding) == 768
    assert abs(data.l2_norm - 1.0) < 1e-3  # Verified L2 unit norm
    assert data.device_used == "cpu"
    assert data.image_metadata.width == 224
    assert data.image_metadata.height == 224

    # Unload
    await model.unload()
    assert model.status == ModelStatus.UNLOADED
