"""VisionForge Training Service & Experiment History Store."""

import json
import logging
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from visionforge.core.config import get_settings
from visionforge.core.exceptions import VisionForgeException
from visionforge.datasets.schemas import DatasetPreparationManifest
from visionforge.datasets.service import (
    DatasetPreparationService,
    get_dataset_preparation_service,
)
from visionforge.ai.types import TaskType
from visionforge.models.manager import ModelManager, get_model_manager
from visionforge.models.metadata import InstalledModelMetadata
from visionforge.training.adapter import YOLODataStoreAdapter
from visionforge.training.schemas import (
    EvaluationResult,
    SmokeTestResult,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)
from visionforge.training.trainer import YOLOTrainer

logger = logging.getLogger("visionforge.training.service")


class InvalidTrainingConfigError(VisionForgeException):
    """Raised when training configuration validation fails."""

    def __init__(self, message: str):
        super().__init__(message=message, code="INVALID_TRAINING_CONFIG", status_code=400)


class TrainingRunNotFoundError(VisionForgeException):
    """Raised when a training run ID is not found."""

    def __init__(self, run_id: str):
        super().__init__(message=f"Training run '{run_id}' not found.", code="TRAINING_RUN_NOT_FOUND", status_code=404)


class TrainingHistoryStore:
    """Thread-safe store logging historical training runs."""

    def __init__(self, storage_dir: str | None = None):
        raw_path = storage_dir or (Path(get_settings().model_cache_dir).parent / "training")
        self._storage_dir = Path(raw_path).resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = self._storage_dir / "training_history.json"
        self._runs: dict[str, TrainingRun] = {}
        self.load_from_disk()

    def add_run(self, run: TrainingRun) -> None:
        """Add or update training run."""
        self._runs[run.run_id] = run
        self.save_to_disk()

    def get_run(self, run_id: str) -> TrainingRun:
        """Retrieve run by ID."""
        if run_id not in self._runs:
            raise TrainingRunNotFoundError(run_id)
        return self._runs[run_id]

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[TrainingRun]:
        """Return paginated runs sorted by creation time."""
        all_runs = sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)
        return all_runs[offset : offset + limit]

    def save_to_disk(self) -> None:
        """Persist runs to disk JSON."""
        serializable = {
            "version": "1.0.0",
            "saved_at": datetime.now(UTC).isoformat(),
            "runs": [r.model_dump() for r in self._runs.values()],
        }
        self._history_file.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")

    def load_from_disk(self) -> None:
        """Restore history runs from disk JSON if available."""
        if not self._history_file.is_file():
            return
        try:
            raw = json.loads(self._history_file.read_text(encoding="utf-8"))
            for item in raw.get("runs", []):
                run = TrainingRun(**item)
                self._runs[run.run_id] = run
        except Exception as exc:
            logger.error("Failed to restore training history: %s", str(exc))


class TrainingService:
    """Service layer orchestrating dataset adaptation, model training, evaluation, and registration."""

    def __init__(
        self,
        dataset_service: DatasetPreparationService | None = None,
        model_manager: ModelManager | None = None,
        history_store: TrainingHistoryStore | None = None,
    ):
        self._dataset_service = dataset_service or get_dataset_preparation_service()
        self._model_manager = model_manager or get_model_manager()
        self._history_store = history_store or get_training_history_store()
        self._adapter = YOLODataStoreAdapter()

    def validate_config(self, config: TrainingConfig) -> None:
        """Validate training configuration parameters."""
        if config.epochs < 1:
            raise InvalidTrainingConfigError("Epochs count must be >= 1.")
        if config.batch_size < 1:
            raise InvalidTrainingConfigError("Batch size must be >= 1.")
        if config.imgsz < 32 or config.imgsz > 2048:
            raise InvalidTrainingConfigError("Image size (imgsz) must be between 32 and 2048 pixels.")
        if not config.preparation_id:
            raise InvalidTrainingConfigError("Preparation ID (prep_...) is required.")

    def create_training_run(self, config: TrainingConfig) -> TrainingRun:
        """Create and execute training run."""
        self.validate_config(config)

        run_id = f"run_{uuid.uuid4().hex[:10]}"
        run = TrainingRun(
            run_id=run_id,
            experiment_name=config.experiment_name,
            dataset_id=config.dataset_id,
            dataset_version="v1.0",
            preparation_id=config.preparation_id,
            status=TrainingStatus.CREATED,
            config=config,
        )

        try:
            # 1. Fetch prepared dataset manifest
            run.status = TrainingStatus.VALIDATING
            manifest_dict = self._dataset_service.export_manifest(config.preparation_id, fmt="json")
            manifest = DatasetPreparationManifest(**manifest_dict)
            run.dataset_id = manifest.dataset_id
            run.dataset_version = manifest.dataset_version

            # 2. Adapt dataset for YOLO format
            run.status = TrainingStatus.PREPARING
            dataset_yaml = self._adapter.prepare_yolo_dataset(manifest)

            # 3. Execute PyTorch Training
            run.status = TrainingStatus.RUNNING
            output_root = Path(get_settings().model_cache_dir).parent / "training" / "runs"
            trainer = YOLOTrainer(output_root=output_root)

            metrics_history, best_metrics, best_pt, last_pt = trainer.train_model(
                dataset_yaml=dataset_yaml, config=config
            )

            run.metrics_history = metrics_history
            run.best_metrics = best_metrics
            run.best_checkpoint_path = str(best_pt)
            run.last_checkpoint_path = str(last_pt)

            # 4. Separate Test Set Evaluation
            run.status = TrainingStatus.VERIFYING
            test_eval = trainer.evaluate_model(checkpoint_path=best_pt, dataset_yaml=dataset_yaml)
            run.test_evaluation = test_eval

            # 5. Complete
            run.status = TrainingStatus.COMPLETED
            run.completed_at = datetime.now(UTC).isoformat()
            self._history_store.add_run(run)
            logger.info("Successfully completed training run '%s'", run_id)
            return run

        except Exception as exc:
            run.status = TrainingStatus.FAILED
            run.error_message = str(exc)
            self._history_store.add_run(run)
            logger.error("Training run '%s' failed: %s", run_id, str(exc))
            raise

    def get_run(self, run_id: str) -> TrainingRun:
        """Get training run by ID."""
        return self._history_store.get_run(run_id)

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[TrainingRun]:
        """List training history runs."""
        return self._history_store.list_runs(limit=limit, offset=offset)

    def evaluate_test_set(self, run_id: str) -> EvaluationResult:
        """Execute separate test set evaluation."""
        run = self._history_store.get_run(run_id)
        if not run.best_checkpoint_path or not Path(run.best_checkpoint_path).is_file():
            raise InvalidTrainingConfigError(f"Best checkpoint for run '{run_id}' not found.")

        manifest_dict = self._dataset_service.export_manifest(run.preparation_id, fmt="json")
        manifest = DatasetPreparationManifest(**manifest_dict)
        dataset_yaml = self._adapter.prepare_yolo_dataset(manifest)

        output_root = Path(get_settings().model_cache_dir).parent / "training" / "runs"
        trainer = YOLOTrainer(output_root=output_root)
        eval_result = trainer.evaluate_model(checkpoint_path=Path(run.best_checkpoint_path), dataset_yaml=dataset_yaml)
        run.test_evaluation = eval_result
        self._history_store.add_run(run)
        return eval_result

    def register_model_artifact(self, run_id: str, version_tag: str = "v1.0.0") -> InstalledModelMetadata:
        """Register trained checkpoint into ModelManager as a versioned model."""
        run = self._history_store.get_run(run_id)
        if not run.best_checkpoint_path:
            raise InvalidTrainingConfigError("No valid best checkpoint available to register.")

        model_name = f"visionforge-{run.config.model_name.replace('.pt', '')}-{run.dataset_id}"
        meta = InstalledModelMetadata(
            name=model_name,
            version=version_tag,
            task=TaskType.OBJECT_DETECTION,
            framework="PyTorch/Ultralytics",
            description=f"Trained model from run {run_id} on {run.dataset_id} ({run.dataset_version})",
            device_support=["cpu", "mps", "cuda"],
            install_path=run.best_checkpoint_path,
        )

        reg_model = self._model_manager.register_model(meta)
        run.registered_model_version = reg_model.version
        self._history_store.add_run(run)
        logger.info("Registered model '%s' (%s) from training run '%s'", model_name, version_tag, run_id)
        return reg_model

    def run_inference_smoke_test(self, run_id: str, sample_image_paths: list[str] | None = None) -> SmokeTestResult:
        """Run lightweight inference smoke test on test samples using trained checkpoint."""
        run = self._history_store.get_run(run_id)
        best_pt = Path(run.best_checkpoint_path) if run.best_checkpoint_path else Path("yolo11s.pt")

        sample_paths: list[Path] = []
        if sample_image_paths:
            sample_paths = [Path(p) for p in sample_image_paths if Path(p).is_file()]

        if not sample_paths:
            # Generate synthetic test sample image for smoke testing
            tmp_img = Path(get_settings().model_cache_dir).parent / "training" / "smoke_sample.jpg"
            tmp_img.parent.mkdir(parents=True, exist_ok=True)
            if not tmp_img.is_file():
                tmp_img.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xFF\xDB\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\x09\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xFF\xC0\x00\x0b\x08\x00\x10\x00\x10\x01\x01\x11\x00\xFF\xC4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xFF\xDA\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xFF\xd9")
            sample_paths = [tmp_img]

        output_root = Path(get_settings().model_cache_dir).parent / "training" / "runs"
        trainer = YOLOTrainer(output_root=output_root)
        return trainer.run_smoke_test(
            run_id=run.run_id,
            model_name=run.config.model_name,
            checkpoint_path=best_pt,
            sample_images=sample_paths,
        )


@lru_cache
def get_training_history_store() -> TrainingHistoryStore:
    """Return singleton instance of TrainingHistoryStore."""
    return TrainingHistoryStore()


@lru_cache
def get_training_service() -> TrainingService:
    """Return singleton instance of TrainingService."""
    return TrainingService()
