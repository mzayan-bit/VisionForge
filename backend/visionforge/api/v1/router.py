"""API v1 Router registry."""

from fastapi import APIRouter

from visionforge.api.v1.active_learning import router as active_learning_router
from visionforge.api.v1.benchmarks import router as benchmarks_router
from visionforge.api.v1.datasets import router as datasets_router
from visionforge.api.v1.datasets_intelligence import router as datasets_intelligence_router
from visionforge.api.v1.embeddings import router as embeddings_router
from visionforge.api.v1.evaluation import router as evaluation_router
from visionforge.api.v1.events import router as events_router
from visionforge.api.v1.experiments import router as experiments_router
from visionforge.api.v1.explainability import router as explainability_router
from visionforge.api.v1.explorer import router as explorer_router
from visionforge.api.v1.health import router as health_router
from visionforge.api.v1.inference import router as inference_router
from visionforge.api.v1.lifecycle import router as lifecycle_router
from visionforge.api.v1.memory import router as memory_router
from visionforge.api.v1.models import router as models_router
from visionforge.api.v1.multimodal import router as multimodal_router
from visionforge.api.v1.query import router as query_router
from visionforge.api.v1.search import router as search_router
from visionforge.api.v1.system import router as system_router
from visionforge.api.v1.training import router as training_router
from visionforge.api.v1.video import router as video_router

router = APIRouter(prefix="/v1")
router.include_router(health_router)
router.include_router(system_router)
router.include_router(models_router)
router.include_router(lifecycle_router)
router.include_router(explainability_router)
router.include_router(benchmarks_router)
router.include_router(embeddings_router)
router.include_router(memory_router)
router.include_router(search_router)
router.include_router(explorer_router)
router.include_router(datasets_router)
router.include_router(datasets_intelligence_router)
router.include_router(training_router)
router.include_router(evaluation_router)
router.include_router(inference_router)
router.include_router(experiments_router)
router.include_router(active_learning_router)
router.include_router(video_router)
router.include_router(events_router)
router.include_router(query_router)
router.include_router(multimodal_router)

__all__ = ["router"]
