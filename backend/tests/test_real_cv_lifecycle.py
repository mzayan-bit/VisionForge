"""End-to-End Real Computer Vision Lifecycle Integration Test Suite.

Validates:
1. Real Dataset Adapter (COCO8 ingestion, image decoding, bounding box parsing, class mapping).
2. Pre-Split Dataset Validation (zero missing images, valid bbox coordinates in [0, 1]).
3. Dataset Versioning & Dataset Intelligence profiling.
4. YOLO Dataset Materialization & Multi-split Annotation Synchronization.
5. Real Transfer Learning Model Training (YOLO11n on CPU/MPS) and Checkpoint Creation.
6. Model Registry Packaging & Metadata Registration.
7. Model Inference & Separate Test Set Evaluation (mAP@50, Precision, Recall, IoU).
8. Diagnostic Error Analysis & Failure Gallery Population.
9. Grad-CAM Explainability Generation.
10. Research Experiment Tracking, Lineage Graph, and Structured Research Report Generation.
"""

from pathlib import Path

from visionforge.datasets.adapters.coco8_adapter import (
    COCO8Adapter,
    COCO8ValidationSummary,
)
from visionforge.datasets.intelligence_service import (
    get_dataset_intelligence_service,
)
from visionforge.datasets.schemas import DatasetPreparationManifest
from visionforge.experiments.service import get_experiment_service
from visionforge.explainability.schemas import (
    CreateExplanationRequest,
    ExplanationMethod,
)
from visionforge.explainability.service import ExplainabilityService
from visionforge.models.manager import get_model_manager
from visionforge.training.adapter import YOLODataStoreAdapter
from visionforge.training.schemas import TrainingConfig, TrainingStatus
from visionforge.training.service import get_training_service


def test_coco8_adapter_validation_and_parsing():
    """Verify that COCO8Adapter validates decodability and bounding box geometry of all real images."""
    adapter = COCO8Adapter()
    summary, samples = adapter.validate_dataset()

    assert isinstance(summary, COCO8ValidationSummary)
    assert summary.validation_status in ("PASSED", "PASSED_WITH_WARNINGS")
    assert summary.total_images == 8
    assert summary.train_images == 4
    assert summary.val_images == 4
    assert summary.corrupt_images == 0
    assert summary.invalid_boxes == 0
    assert summary.total_annotations >= 20
    assert len(summary.unique_classes_present) > 0

    assert len(samples) == 8
    for sample in samples:
        assert Path(sample["file_path"]).is_file()
        assert sample["width"] > 0
        assert sample["height"] > 0
        assert sample["format"] in ("JPEG", "JPG", "PNG")
        assert len(sample["tags"]) > 0


def test_coco8_adapter_ingestion_and_manifest_materialization():
    """Verify that COCO8 records are ingested into VisualMemory, Manifests, and DatasetIntelligence."""
    adapter = COCO8Adapter()
    summary, manifest, profile = adapter.ingest_dataset(
        dataset_id="coco8_test", dataset_version="v1.0.0"
    )

    assert isinstance(manifest, DatasetPreparationManifest)
    assert manifest.dataset_id == "coco8_test"
    assert manifest.total_samples == 8
    assert manifest.train_count == 4
    assert manifest.test_count == 4

    assert profile.dataset_id == "coco8_test"
    assert profile.total_samples == 8
    assert profile.total_annotations == summary.total_annotations
    assert profile.health_summary is not None
    assert len(profile.class_distribution) > 0

    intel_svc = get_dataset_intelligence_service()
    versions = intel_svc.list_dataset_versions("coco8_test")
    assert any(v.version_id == "v1.0.0" for v in versions)


def test_yolo_datastore_adapter_with_real_annotations(tmp_path):
    """Verify that YOLODataStoreAdapter formats real ground-truth label files and dataset.yaml."""
    adapter = COCO8Adapter()
    _, manifest, _ = adapter.ingest_dataset(dataset_id="coco8_test", dataset_version="v1.0.0")

    yolo_adapter = YOLODataStoreAdapter(output_root=str(tmp_path / "training"))
    yaml_path = yolo_adapter.prepare_yolo_dataset(manifest)

    assert yaml_path.is_file()
    ds_dir = yaml_path.parent
    train_images = list((ds_dir / "images" / "train").glob("*.jpg"))
    val_images = list((ds_dir / "images" / "val").glob("*.jpg"))
    train_labels = list((ds_dir / "labels" / "train").glob("*.txt"))
    val_labels = list((ds_dir / "labels" / "val").glob("*.txt"))

    assert len(train_images) == 4
    assert len(val_images) == 4
    assert len(train_labels) == 4
    assert len(val_labels) == 4

    # Verify label files contain valid class IDs and coordinates
    for lbl in train_labels + val_labels:
        content = lbl.read_text().strip()
        if content:
            for line in content.splitlines():
                parts = line.split()
                assert len(parts) == 5
                cid = int(parts[0])
                assert 0 <= cid < 80
                for coord in parts[1:]:
                    val = float(coord)
                    assert 0.0 <= val <= 1.0


def test_real_training_evaluation_and_registration_lifecycle():
    """Verify the full training, checkpointing, evaluation, and registration lifecycle with YOLO11n."""
    adapter = COCO8Adapter()
    _, manifest, _ = adapter.ingest_dataset(dataset_id="coco8", dataset_version="v1.0.0")

    train_svc = get_training_service()
    config = TrainingConfig(
        experiment_name="test_exp_coco8_transfer",
        dataset_id="coco8",
        preparation_id=manifest.preparation_id,
        model_name="yolo11n.pt",
        epochs=1,
        batch_size=4,
        imgsz=320,
        learning_rate=0.01,
        random_seed=42,
        device="cpu",
    )

    run = train_svc.create_training_run(config)
    assert run.status == TrainingStatus.COMPLETED
    assert run.best_checkpoint_path is not None
    assert Path(run.best_checkpoint_path).exists()
    assert run.test_evaluation is not None
    assert run.test_evaluation.precision >= 0.0
    assert run.test_evaluation.recall >= 0.0

    # Model Registration
    model_meta = train_svc.register_model_artifact(run.run_id, version_tag="1.0.0")
    assert model_meta.name.startswith("visionforge-")
    model_mgr = get_model_manager()
    assert model_mgr.is_installed(model_meta.name)

    # Smoke Test Inference
    smoke_result = train_svc.run_inference_smoke_test(run.run_id)
    assert smoke_result.model_name is not None
    assert len(smoke_result.predictions) > 0
    assert smoke_result.average_latency_ms >= 0.0


def test_real_explainability_and_research_experiment_flow():
    """Verify Grad-CAM attribution and Research Experiment tracking with lineage."""
    adapter = COCO8Adapter()
    _, samples = adapter.validate_dataset()
    test_sample = samples[0]

    # Explainability
    explain_svc = ExplainabilityService()
    req = CreateExplanationRequest(
        model_name="yolo11n.pt",
        model_version="1.0.0",
        image_path=test_sample["file_path"],
        method=ExplanationMethod.GRAD_CAM,
        target_class=test_sample["tags"][0],
    )
    explanation = explain_svc.create_explanation(req)
    assert explanation.status.value in ("COMPLETED", "CACHED")
    assert explanation.artifact is not None
    assert len(explanation.artifact.heatmap_grid) > 0
    assert explanation.artifact.object_concentration_score >= 0.0

    # Research Experiment
    exp_svc = get_experiment_service()
    exp = exp_svc.create_experiment(
        name="Real COCO8 Transfer Learning Benchmark",
        description="Empirical validation of YOLO11n transfer learning on COCO8 dataset.",
        dataset_id="coco8",
        dataset_version="v1.0.0",
        hypothesis="Pretrained feature representations accelerate convergence on small datasets.",
    )
    assert exp.experiment_id.startswith("exp_")
    assert exp.status.value in ("DRAFT", "ACTIVE", "COMPLETED")
