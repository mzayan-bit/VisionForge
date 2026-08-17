"""Unit test suite for Dataset Preparation Pipeline."""

import pytest
from fastapi.testclient import TestClient

from visionforge.datasets.leakage import detect_data_leakage
from visionforge.datasets.manifest import build_manifest, export_manifest_csv
from visionforge.datasets.schemas import SplitConfig
from visionforge.datasets.service import DatasetPreparationService, PreparationHistoryStore
from visionforge.datasets.splitting import InvalidSplitRatioError, partition_dataset
from visionforge.datasets.validation import validate_dataset
from visionforge.main import app
from visionforge.memory.index import VisualMemoryIndex, VisualMemoryRecord

client = TestClient(app)


def test_validate_dataset_valid():
    """Verify validation passes for valid dataset records."""
    rec = VisualMemoryRecord(
        id="sample_01",
        embedding=[0.1] * 768,
        image_metadata={"width": 640, "height": 480, "format": "JPEG"},
    )
    report = validate_dataset([rec])
    assert report.status == "PASSED"
    assert report.total_samples == 1
    assert report.valid_samples == 1


def test_validate_dataset_corrupted_and_missing():
    """Verify validation identifies invalid dimensions and missing embeddings."""
    rec_bad_dim = VisualMemoryRecord(
        id="bad_dim",
        embedding=[0.1] * 768,
        image_metadata={"width": 0, "height": 0},
    )
    rec_no_embed = VisualMemoryRecord(
        id="no_embed",
        embedding=[],
        image_metadata={"width": 100, "height": 100},
    )

    report = validate_dataset([rec_bad_dim, rec_no_embed])
    assert report.status in ("PASSED_WITH_WARNINGS", "FAILED")
    assert report.corrupted_samples_count == 1
    assert report.missing_embeddings_count == 1
    assert len(report.issues) == 2


def test_detect_data_leakage_exact_and_near():
    """Verify exact duplicate content hash and near-duplicate embedding detection."""
    rec1 = VisualMemoryRecord(
        id="dup_1",
        embedding=[1.0] + [0.0] * 767,
        image_metadata={"width": 100, "height": 100, "file_size_bytes": 5000},
    )
    rec2 = VisualMemoryRecord(
        id="dup_2",
        embedding=[1.0] + [0.0] * 767,
        image_metadata={"width": 100, "height": 100, "file_size_bytes": 5000},
    )

    findings, group_map = detect_data_leakage([rec1, rec2])
    assert len(findings) >= 1
    assert "dup_1" in group_map
    assert "dup_2" in group_map
    assert group_map["dup_1"] == group_map["dup_2"]


def test_partition_dataset_seed_reproducibility():
    """Verify 100% deterministic seed-based dataset partitioning."""
    records = [
        VisualMemoryRecord(
            id=f"rec_{i}",
            embedding=[float(i)] + [0.0] * 767,
            image_metadata={"width": 224, "height": 224},
        )
        for i in range(20)
    ]
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    # Run 1
    refs1, counts1 = partition_dataset(records, {}, config)
    splits1 = {r.id: r.split for r in refs1}

    # Run 2 with identical seed
    refs2, counts2 = partition_dataset(records, {}, config)
    splits2 = {r.id: r.split for r in refs2}

    assert counts1 == counts2
    assert splits1 == splits2


def test_partition_dataset_leakage_safety():
    """Verify all members of a leakage group land in the SAME split partition."""
    records = [
        VisualMemoryRecord(
            id=f"rec_{i}", embedding=[0.1] * 768, image_metadata={"width": 200, "height": 200}
        )
        for i in range(10)
    ]
    # Group rec_0, rec_1, rec_2 together
    leakage_map = {"rec_0": "leak_01", "rec_1": "leak_01", "rec_2": "leak_01"}
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)

    refs, _ = partition_dataset(records, leakage_map, config)
    splits = {r.id: r.split for r in refs}

    assert splits["rec_0"] == splits["rec_1"] == splits["rec_2"]


def test_partition_dataset_invalid_ratios():
    """Verify InvalidSplitRatioError when ratios sum != 1.0."""
    records = [
        VisualMemoryRecord(
            id="sample_01", embedding=[0.1] * 768, image_metadata={"width": 100, "height": 100}
        )
    ]
    config = SplitConfig(train_ratio=0.50, val_ratio=0.50, test_ratio=0.50)  # Sums to 1.5

    with pytest.raises(InvalidSplitRatioError):
        partition_dataset(records, {}, config)


def test_manifest_building_and_export():
    """Verify manifest construction and CSV export."""
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    manifest = build_manifest(
        preparation_id="prep_test123",
        dataset_id="safety_v1",
        dataset_version="v1.0",
        split_config=config,
        samples=[],
        counts={"train": 0, "validation": 0, "test": 0},
        leakage_findings=[],
    )

    assert manifest.preparation_id == "prep_test123"
    assert manifest.software_version == "VisionForge v0.1.0"

    csv_text = export_manifest_csv(manifest)
    assert "sample_id,split" in csv_text


def test_preparation_service_end_to_end(tmp_path):
    """Verify full preparation service orchestration and API endpoints."""
    mem = VisualMemoryIndex(storage_dir=str(tmp_path / "memory"))
    history = PreparationHistoryStore(storage_dir=str(tmp_path / "datasets"))
    svc = DatasetPreparationService(memory_index=mem, history_store=history)

    # Populate 10 synthetic records
    for i in range(10):
        rec = VisualMemoryRecord(
            id=f"sample_{i:02d}",
            embedding=[float(i)] + [0.0] * 767,
            image_metadata={"width": 640, "height": 480, "format": "JPEG"},
            tags=["person"],
        )
        mem.add_record(rec)

    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    run = svc.create_preparation_run("safety_dataset", "v1.0", config)

    assert run.status.value == "COMPLETED"
    assert run.validation_report.status == "PASSED"
    assert run.split_stats["train"].count > 0

    # Test REST API history endpoint
    res = client.get("/api/v1/datasets/prepare/history")
    assert res.status_code == 200
    assert res.json()["success"] is True
