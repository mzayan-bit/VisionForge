"""VisionForge Training Pipeline Package."""

from visionforge.training.schemas import (
    BoundingBox,
    EvaluationResult,
    InferencePrediction,
    MetricSnapshot,
    SmokeTestResult,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)
from visionforge.training.service import TrainingService, get_training_service

__all__ = [
    "TrainingStatus",
    "TrainingConfig",
    "MetricSnapshot",
    "EvaluationResult",
    "BoundingBox",
    "InferencePrediction",
    "SmokeTestResult",
    "TrainingRun",
    "TrainingService",
    "get_training_service",
]
