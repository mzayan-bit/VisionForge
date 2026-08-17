"""Unit test suite for VisionForge Model Management System."""

from pathlib import Path

import pytest

from visionforge.ai.types import InputType, OutputType, TaskType
from visionforge.models.fs import (
    bytes_to_mb,
    calculate_sha256,
    directory_file_count,
    directory_size_bytes,
    ensure_directory,
    safe_delete_directory,
    safe_delete_file,
    safe_model_name,
)
from visionforge.models.manager import (
    ModelAlreadyExistsError,
    ModelManager,
    ModelNotInstalledError,
)
from visionforge.models.metadata import InstalledModelMetadata, InstallStatus, ModelSource
from visionforge.models.storage import ModelStorage
from visionforge.models.validation import (
    ModelValidationError,
    validate_model_name,
    validate_version,
)


@pytest.fixture
def temp_storage(tmp_path: Path) -> ModelStorage:
    """Provide a ModelStorage instance backed by a temporary directory."""
    storage = ModelStorage(storage_root=str(tmp_path))
    storage.initialize()
    return storage


@pytest.fixture
def manager(temp_storage: ModelStorage) -> ModelManager:
    """Provide a ModelManager instance using the temporary storage."""
    return ModelManager(storage=temp_storage)


@pytest.fixture
def sample_metadata() -> InstalledModelMetadata:
    """Provide a sample valid model metadata record."""
    return InstalledModelMetadata(
        name="test-model",
        version="1.0.0",
        task=TaskType.DETECTION,
        supported_input_types=[InputType.IMAGE],
        supported_output_types=[OutputType.BOUNDING_BOXES],
        source=ModelSource(provider="local"),
    )


def test_fs_utilities(tmp_path: Path):
    """Test filesystem utility functions."""
    assert safe_model_name("huggingface/model:name") == "huggingface_model_name"

    test_dir = tmp_path / "test_dir"
    ensure_directory(test_dir)
    assert test_dir.exists()

    file1 = test_dir / "file1.txt"
    file1.write_text("hello")
    file2 = test_dir / "file2.txt"
    file2.write_text("world!")

    assert directory_file_count(test_dir) == 2
    assert directory_size_bytes(test_dir) == 11
    assert bytes_to_mb(1024 * 1024) == 1.0

    # Test SHA-256
    sha = calculate_sha256(file1)
    assert len(sha) == 64

    # Test deletion
    assert safe_delete_file(file1) == 5
    assert directory_file_count(test_dir) == 1
    assert safe_delete_directory(test_dir) == 6
    assert not test_dir.exists()


def test_validation_logic():
    """Test model name and version validation rules."""
    # Valid names
    validate_model_name("yolov8")
    validate_model_name("meta-llama-3")
    validate_model_name("resnet.50_v2")

    # Invalid names
    with pytest.raises(ModelValidationError):
        validate_model_name("")
    with pytest.raises(ModelValidationError):
        validate_model_name("MyModel")  # Uppercase
    with pytest.raises(ModelValidationError):
        validate_model_name("model/name")  # Slash

    # Valid versions
    validate_version("1.0.0")
    validate_version("2.1.0-beta.1")

    # Invalid versions
    with pytest.raises(ModelValidationError):
        validate_version("v1.0")
    with pytest.raises(ModelValidationError):
        validate_version("1.0")


def test_storage_initialization(temp_storage: ModelStorage):
    """Verify storage directories are created correctly."""
    assert temp_storage.models_dir.exists()
    assert temp_storage.downloads_dir.exists()
    assert temp_storage.metadata_dir.exists()
    assert temp_storage.temp_dir.exists()


def test_manager_prepare_and_finalize_install(
    manager: ModelManager, sample_metadata: InstalledModelMetadata
):
    """Test the full installation lifecycle state transitions."""
    # 1. Prepare
    meta = manager.prepare_install(sample_metadata)
    assert meta.status == InstallStatus.INSTALLING
    assert meta.install_path.endswith(f"models/{sample_metadata.name}/{sample_metadata.version}")
    assert manager.is_installed(sample_metadata.name)

    # 2. Simulate file download
    model_dir = Path(meta.install_path)
    (model_dir / "weights.bin").write_bytes(b"0" * 1024)

    # 3. Finalize
    final_meta = manager.finalize_install(sample_metadata.name)
    assert final_meta.status == InstallStatus.INSTALLED
    assert final_meta.disk_size_bytes == 1024
    assert final_meta.disk_size_mb == 0.0


def test_manager_duplicate_install(manager: ModelManager, sample_metadata: InstalledModelMetadata):
    """Verify duplicate installations are blocked."""
    manager.prepare_install(sample_metadata)
    with pytest.raises(ModelAlreadyExistsError):
        manager.prepare_install(sample_metadata)


def test_manager_remove_model(manager: ModelManager, sample_metadata: InstalledModelMetadata):
    """Verify model removal cleans up disk and metadata."""
    manager.prepare_install(sample_metadata)
    model_dir = Path(sample_metadata.install_path)
    (model_dir / "weights.bin").write_bytes(b"0" * 1024)
    manager.finalize_install(sample_metadata.name)

    assert manager.is_installed(sample_metadata.name)

    freed = manager.remove_model(sample_metadata.name)
    assert freed == 1024
    assert not manager.is_installed(sample_metadata.name)
    assert not model_dir.exists()

    with pytest.raises(ModelNotInstalledError):
        manager.get_model(sample_metadata.name)


def test_manager_validation_report(manager: ModelManager, sample_metadata: InstalledModelMetadata):
    """Test diagnostic validation report generation."""
    # Setup healthy model
    manager.prepare_install(sample_metadata)
    model_dir = manager.storage.get_model_dir(sample_metadata.name, sample_metadata.version)
    (model_dir / "config.json").write_text("{}")
    manager.finalize_install(sample_metadata.name)

    # Healthy check
    report = manager.validate_model(sample_metadata.name)
    assert report["valid"] is True
    assert not report["errors"]
    assert not report["warnings"]

    # Corrupt the directory (delete it)
    safe_delete_directory(model_dir)

    # Unhealthy check
    report = manager.validate_model(sample_metadata.name)
    assert report["valid"] is False
    assert len(report["errors"]) == 1
    assert "directory missing" in report["errors"][0]


def test_manager_list_and_stats(manager: ModelManager, sample_metadata: InstalledModelMetadata):
    """Test listing installed models and gathering stats."""
    manager.prepare_install(sample_metadata)

    installed = manager.list_installed()
    assert len(installed) == 1
    assert installed[0].name == sample_metadata.name

    status = manager.get_manager_status()
    assert status["status"] == "ready"
    assert status["installed_models"] == 1
    assert status["storage"]["models_count"] == 1
