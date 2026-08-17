"""Unit & Integration Test Suite for VisionForge Model Training Pipeline."""

import pytest
from fastapi.testclient import TestClient

from visionforge.datasets.schemas import DatasetPreparationManifest, SampleRef, SplitConfig
from visionforge.datasets.service import PreparationHistoryStore
from visionforge.main import app
from visionforge.models.manager import ModelManager
from visionforge.training.adapter import YOLODataStoreAdapter
from visionforge.training.schemas import TrainingConfig, TrainingStatus
from visionforge.training.service import (
    InvalidTrainingConfigError,
    TrainingHistoryStore,
    TrainingService,
)
from visionforge.training.trainer import UltralyticsTrainer

client = TestClient(app)


def test_validate_training_config_valid(tmp_path):
    """Verify validation passes for valid training configuration."""
    svc = TrainingService()
    cfg = TrainingConfig(
        dataset_id="safety_v1",
        preparation_id="prep_test123",
        epochs=10,
        batch_size=16,
        imgsz=640,
    )
    svc.validate_config(cfg)  # Should not raise exception


def test_validate_training_config_invalid():
    """Verify ValidationError or InvalidTrainingConfigError raised for invalid hyperparameters."""
    svc = TrainingService()

    # Invalid epochs
    with pytest.raises(Exception):
        svc.validate_config(TrainingConfig(dataset_id="d1", preparation_id="p1", epochs=0))

    # Missing preparation ID
    with pytest.raises(InvalidTrainingConfigError):
        svc.validate_config(TrainingConfig(dataset_id="d1", preparation_id="", epochs=10))


def test_yolo_adapter_manifest_processing(tmp_path):
    """Verify YOLODataStoreAdapter generates dataset.yaml without re-splitting."""
    adapter = YOLODataStoreAdapter(output_root=str(tmp_path / "data"))
    config = SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    samples = [
        SampleRef(id="sample_01", split="train", file_path="", tags=["person"]),
        SampleRef(id="sample_02", split="validation", file_path="", tags=["person"]),
        SampleRef(id="sample_03", split="test", file_path="", tags=["person"]),
    ]

    manifest = DatasetPreparationManifest(
        preparation_id="prep_adapter_test",
        dataset_id="safety_dataset",
        dataset_version="v1.0",
        random_seed=42,
        split_config=config,
        software_version="VisionForge v0.1.0",
        total_samples=3,
        train_count=1,
        val_count=1,
        test_count=1,
        exact_duplicates_found=0,
        near_duplicates_found=0,
        samples=samples,
    )

    yaml_path = adapter.prepare_yolo_dataset(manifest)
    assert yaml_path.is_file()
    assert "dataset.yaml" in yaml_path.name


def test_yolo_trainer_synthetic_execution(tmp_path):
    """Verify UltralyticsTrainer metric parsing and evaluation."""
    trainer = UltralyticsTrainer(output_root=tmp_path / "runs")
    cfg = TrainingConfig(
        dataset_id="d1",
        preparation_id="p1",
        epochs=3,
        batch_size=4,
        experiment_name="exp_test",
    )

    ds_yaml = tmp_path / "dataset.yaml"
    ds_yaml.write_text("path: .\ntrain: .\nval: .\nnames:\n  0: object", encoding="utf-8")

    metrics_hist, best_m, best_pt, last_pt = trainer.train_model(ds_yaml, cfg)
    assert len(metrics_hist) == 3
    assert best_m.map50 > 0.0

    eval_res = trainer.evaluate_model(best_pt, ds_yaml)
    assert eval_res.map50 > 0.0


def test_training_service_end_to_end(tmp_path):
    """Verify end-to-end TrainingService workflow."""
    from visionforge.models.storage import ModelStorage

    prep_history = PreparationHistoryStore(storage_dir=str(tmp_path / "datasets"))
    model_storage = ModelStorage(storage_root=str(tmp_path / "models"))
    model_mgr = ModelManager(storage=model_storage)
    model_mgr.initialize()
    train_history = TrainingHistoryStore(storage_dir=str(tmp_path / "training"))

    # Seed mock preparation manifest in history
    manifest = DatasetPreparationManifest(
        preparation_id="prep_e2e_123",
        dataset_id="safety_v2",
        dataset_version="v2.0",
        random_seed=42,
        split_config=SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15),
        software_version="VisionForge v0.1.0",
        total_samples=3,
        train_count=1,
        val_count=1,
        test_count=1,
        exact_duplicates_found=0,
        near_duplicates_found=0,
        samples=[
            SampleRef(id="s1", split="train", tags=["helmet"]),
            SampleRef(id="s2", split="validation", tags=["helmet"]),
            SampleRef(id="s3", split="test", tags=["helmet"]),
        ],
    )
    # Save manifest file
    prep_dir = tmp_path / "datasets" / "prepared" / "prep_e2e_123"
    prep_dir.mkdir(parents=True, exist_ok=True)
    import json

    (prep_dir / "manifest.json").write_text(json.dumps(manifest.model_dump()), encoding="utf-8")

    from visionforge.datasets.service import DatasetPreparationService

    ds_svc = DatasetPreparationService(history_store=prep_history)
    prep_history._manifests["prep_e2e_123"] = manifest

    svc = TrainingService(
        dataset_service=ds_svc, model_manager=model_mgr, history_store=train_history
    )
    cfg = TrainingConfig(
        dataset_id="safety_v2",
        preparation_id="prep_e2e_123",
        epochs=2,
        batch_size=2,
        experiment_name="exp_e2e",
    )

    run = svc.create_training_run(cfg)
    assert run.status == TrainingStatus.COMPLETED
    assert run.best_metrics is not None
    assert run.test_evaluation is not None

    # Test model registration
    reg_model = svc.register_model_artifact(run.run_id, version_tag="1.0.0")
    assert reg_model.version == "1.0.0"

    # Test inference smoke test
    smoke_res = svc.run_inference_smoke_test(run.run_id)
    assert smoke_res.run_id == run.run_id
    assert len(smoke_res.predictions) >= 1


def test_training_api_endpoints():
    """Verify Training Pipeline REST API endpoints."""
    res = client.get("/api/v1/training/runs")
    assert res.status_code == 200
    assert res.json()["success"] is True
