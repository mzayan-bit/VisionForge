"""VisionForge Interactive Inference Studio API Routes."""

import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from visionforge.inference.schemas import (
    InferenceBenchmarkConfig,
    InferenceBenchmarkResult,
    InferenceConfig,
    InferenceModelDescriptor,
    InferenceResult,
    ModelComparisonRequest,
    ModelComparisonResult,
)
from visionforge.inference.service import (
    ImageValidationError,
    ModelNotFoundError,
    get_inference_service,
)

logger = logging.getLogger("visionforge.api.v1.inference")

router = APIRouter(prefix="/inference", tags=["Inference Studio"])


@router.get(
    "/models",
    response_model=list[InferenceModelDescriptor],
    summary="List available inference models",
)
def list_inference_models() -> list[InferenceModelDescriptor]:
    """Retrieve all models ready for inference (base models, installed models, and trained runs)."""
    service = get_inference_service()
    return service.list_available_models()


@router.post(
    "/upload",
    response_model=dict[str, str],
    summary="Upload image for inference studio",
)
async def upload_inference_image(
    file: Annotated[UploadFile, File(description="Target image file (JPEG, PNG, WEBP)")],
) -> dict[str, str]:
    """Upload an image file to the Vision Lab sandbox."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Filename missing in upload."
        )

    service = get_inference_service()
    try:
        content = await file.read()
        saved_path = service.process_image_upload(content, file.filename)
        return {"image_path": saved_path, "filename": file.filename}
    except ImageValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to process image upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload image file.",
        ) from exc


@router.post(
    "/run",
    response_model=InferenceResult,
    summary="Run real model inference on target image",
)
def run_model_inference(
    image_path: str = Form(description="Path to target image file"),
    model_id: str = Form(default="yolo11s.pt", description="Model ID or checkpoint path"),
    confidence_threshold: float = Form(default=0.25, ge=0.01, le=1.0),
    iou_threshold: float = Form(default=0.45, ge=0.01, le=1.0),
    imgsz: int = Form(default=640, ge=32, le=2048),
    device: str = Form(default="auto"),
    image_id: str | None = Form(default=None),
) -> InferenceResult:
    """Execute model forward pass over an image and return standardized detections."""
    service = get_inference_service()
    config = InferenceConfig(
        model_id=model_id,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        device=device,
    )

    try:
        return service.run_inference(image_path=image_path, config=config, image_id=image_id)
    except ImageValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference execution failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {exc}",
        ) from exc


@router.get(
    "/history",
    response_model=list[InferenceResult],
    summary="List inference history",
)
def get_inference_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[InferenceResult]:
    """Retrieve historical inference execution records."""
    service = get_inference_service()
    return service.list_history(limit=limit, offset=offset)


@router.get(
    "/history/{inference_id}",
    response_model=InferenceResult,
    summary="Get single inference record",
)
def get_inference_record(inference_id: str) -> InferenceResult:
    """Retrieve details for a specific inference transaction ID."""
    service = get_inference_service()
    rec = service.get_inference_record(inference_id)
    if not rec:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inference record '{inference_id}' not found.",
        )
    return rec


@router.post(
    "/compare",
    response_model=ModelComparisonResult,
    summary="Compare two models side-by-side",
)
def compare_models(payload: ModelComparisonRequest) -> ModelComparisonResult:
    """Run two distinct models over the same target image and compare metrics & predictions."""
    if not payload.image_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="image_path is required for comparison."
        )

    service = get_inference_service()
    try:
        return service.run_comparison(
            image_path=payload.image_path,
            model_a_id=payload.model_a_id,
            model_b_id=payload.model_b_id,
            config_a=payload.config_a,
            config_b=payload.config_b,
        )
    except Exception as exc:
        logger.exception("Model comparison failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model comparison failed: {exc}",
        ) from exc


@router.post(
    "/benchmark",
    response_model=InferenceBenchmarkResult,
    summary="Run latency benchmark",
)
def run_inference_benchmark(config: InferenceBenchmarkConfig) -> InferenceBenchmarkResult:
    """Run multi-pass latency benchmark for a specific model."""
    service = get_inference_service()
    try:
        return service.run_benchmark(config)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference benchmarking failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference benchmarking failed: {exc}",
        ) from exc


@router.post(
    "/unload/{model_id}",
    response_model=dict[str, bool],
    summary="Unload model from RAM/VRAM memory",
)
def unload_model(model_id: str) -> dict[str, bool]:
    """Unload cached PyTorch model weights from memory."""
    service = get_inference_service()
    unloaded = service._lifecycle.unload_model(model_id)
    return {"unloaded": unloaded}
