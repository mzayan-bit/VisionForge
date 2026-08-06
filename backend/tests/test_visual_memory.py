"""Unit test suite for Visual Memory Index and Storage."""

import pytest

from visionforge.memory.index import (
    MemoryRecordNotFoundError,
    VisualMemoryIndex,
    VisualMemoryRecord,
)


@pytest.fixture
def sample_record():
    """Generate sample visual memory record."""
    vector = [0.1 * i for i in range(768)]
    return VisualMemoryRecord(
        id="mem_sample_1",
        embedding=vector,
        dimension=768,
        image_metadata={"width": 224, "height": 224, "format": "JPEG"},
        tags=["test", "sample"],
    )


def test_visual_memory_add_get_delete(tmp_path, sample_record):
    """Test adding, retrieving, listing, and deleting memory records."""
    mem = VisualMemoryIndex(storage_dir=str(tmp_path))
    assert mem.get_stats().total_records == 0

    # Add
    mem.add_record(sample_record)
    assert mem.get_stats().total_records == 1

    # Retrieve
    retrieved = mem.get_record("mem_sample_1")
    assert retrieved.id == "mem_sample_1"
    assert len(retrieved.embedding) == 768

    # Matrix lookup
    matrix, ids = mem.get_matrix_and_ids()
    assert matrix.shape == (1, 768)
    assert ids == ["mem_sample_1"]

    # Delete
    assert mem.delete_record("mem_sample_1") is True
    assert mem.get_stats().total_records == 0

    with pytest.raises(MemoryRecordNotFoundError):
        mem.get_record("mem_sample_1")


def test_visual_memory_persistence(tmp_path, sample_record):
    """Test disk serialization and automatic loading."""
    mem1 = VisualMemoryIndex(storage_dir=str(tmp_path))
    mem1.add_record(sample_record)
    mem1.save_to_disk()

    # Create new instance pointing to same storage
    mem2 = VisualMemoryIndex(storage_dir=str(tmp_path))
    assert mem2.get_stats().total_records == 1
    rec = mem2.get_record("mem_sample_1")
    assert rec.tags == ["test", "sample"]
