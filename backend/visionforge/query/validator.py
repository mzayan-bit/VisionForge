"""VisionForge Query Validator for Visual Query Layer."""

from visionforge.core.exceptions import VisionForgeException
from visionforge.events.service import get_temporal_event_service
from visionforge.query.schemas import VisualQuery
from visionforge.video.service import get_video_intelligence_service


class QueryValidationError(VisionForgeException):
    """Raised when a VisualQuery fails validation checks."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="INVALID_VISUAL_QUERY",
            status_code=400,
        )


class QueryValidator:
    """Validator enforcing safety, boundaries, and scope constraints on VisualQuery DSL objects."""

    def validate_query(self, query: VisualQuery) -> None:
        """Validate structured query DSL against available video assets, runs, and regions."""
        # 1. Validate target VideoInferenceRun exists
        video_svc = get_video_intelligence_service()
        run = video_svc.get_run(query.run_id)

        # 2. Validate Region ROI existence if region_name specified
        if query.region_name:
            event_svc = get_temporal_event_service()
            active_regs = event_svc.list_regions(run.video_id)
            if active_regs:
                matched = any(
                    r.name.lower() == query.region_name.lower() or query.region_name.lower() in r.name.lower()
                    for r in active_regs
                )
                if not matched:
                    known_names = ", ".join(f"'{r.name}'" for r in active_regs)
                    raise QueryValidationError(
                        f"Region '{query.region_name}' was not found for video '{run.video_id}'. Active regions: {known_names}"
                    )

        # 3. Validate Time Range boundaries
        if query.time_range:
            if len(query.time_range) != 2:
                raise QueryValidationError("time_range filter must specify exactly [start_sec, end_sec].")
            if query.time_range[0] < 0.0 or query.time_range[1] < 0.0:
                raise QueryValidationError("time_range timestamps must be non-negative.")
            if query.time_range[0] > query.time_range[1]:
                raise QueryValidationError("time_range start_sec cannot be greater than end_sec.")

        # 4. Validate Duration Threshold
        if query.min_duration_sec is not None and query.min_duration_sec < 0.0:
            raise QueryValidationError("min_duration_sec threshold must be non-negative.")

        # 5. Validate Confidence Threshold
        if query.min_confidence is not None and not (0.0 <= query.min_confidence <= 1.0):
            raise QueryValidationError("min_confidence threshold must be between 0.0 and 1.0.")
