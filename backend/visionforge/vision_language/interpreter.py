"""VisionForge Multimodal Query Interpreter & Intent Parser.

Translates natural language questions into structured VisionQuery categories and DSL filters,
handling multi-turn context carry-over and detecting underspecified ambiguity without guessing.
"""

import logging
import re
import uuid
from typing import Any

from visionforge.vision_language.schemas import (
    MultimodalQueryStatus,
    MultiTurnContext,
    VisionQuery,
    VisionQueryType,
)

logger = logging.getLogger("visionforge.vision_language.interpreter")

# Common computer vision class synonyms
CLASS_SYNONYMS: dict[str, str] = {
    "person": "person",
    "people": "person",
    "worker": "person",
    "workers": "person",
    "human": "person",
    "humans": "person",
    "pedestrian": "person",
    "car": "car",
    "cars": "car",
    "vehicle": "vehicle",
    "vehicles": "vehicle",
    "truck": "vehicle",
    "trucks": "vehicle",
    "helmet": "helmet",
    "helmets": "helmet",
    "hard hat": "helmet",
    "hard hats": "helmet",
    "vest": "vest",
    "vests": "vest",
    "safety vest": "vest",
}

# Regex patterns for query classification
FAILURE_PATTERNS = [
    r"\bfailures?\b",
    r"\bfail\b",
    r"\bfalse positive\b",
    r"\bfalse negative\b",
    r"\bmissed detection\b",
    r"\bwrong prediction\b",
    r"\berror analysis\b",
    r"\bwhy is (?:this|sample|image)\b.*(?:\bfailure\b|\bwrong\b)",
]

DATASET_PATTERNS = [
    r"\bdataset\b",
    r"\bhow many samples\b",
    r"\bunderrepresented\b",
    r"\bclass distribution\b",
    r"\bduplicate candidates?\b",
    r"\blow[- ]quality samples?\b",
    r"\bdataset health\b",
    r"\bdataset profile\b",
    r"\bdata quality\b",
]

MODEL_EVAL_PATTERNS = [
    r"\bcompare (?:model|v\d+)\b",
    r"\bwhich model performs best\b",
    r"\bbenchmark\b",
    r"\bmodel comparison\b",
    r"\bmAP\b",
    r"\bprecision[- ]recall\b",
    r"\bconfusion matrix\b",
    r"\bwhat changed between\b",
    r"\bmetrics\b",
]

SEARCH_PATTERNS = [
    r"\bsimilar (?:to|images?|samples?)\b",
    r"\bvisually similar\b",
    r"\bfind images\b",
    r"\bsearch near[- ]duplicate\b",
    r"\bembedding search\b",
    r"\bvisual search\b",
]

EVENT_PATTERNS = [
    r"\bentered\b",
    r"\bleft\b",
    r"\bexited\b",
    r"\bdwelled\b",
    r"\bstayed\b",
    r"\bstopped\b",
    r"\bzone\b",
    r"\bregion\b",
    r"\bproximity\b",
]

TRACK_PATTERNS = [
    r"\btrack\s*(?:#)?\d+\b",
    r"\btrajectory\b",
    r"\bhow long did track\b",
    r"\bimage[- ]space velocity\b",
    r"\bmovement\b",
]

IMAGE_OBJECT_PATTERNS = [
    r"\bwhat objects\b",
    r"\bvisible in this image\b",
    r"\bwere detected\b",
    r"\bconfidence\b",
    r"\bbounding box\b",
    r"\bhighest confidence\b",
    r"\blow confidence\b",
    r"\bwhy did the model make this prediction\b",
    r"\battribution\b",
]

UNSUPPORTED_PATTERNS = [
    r"\bselect\s+\*\s+from\b",
    r"\bdrop\s+table\b",
    r"\brm\s+-rf\b",
    r"\bexec\s*\(",
    r"\bfeeling\b|\bfeel\b|\bemotion\b|\bhappy\b|\bsad\b|\bangry\b",
    r"\bwhat (?:is|was) (?:the person|he|she) thinking\b",
    r"\bpredict the future\b",
    r"\bfacial recognition\b|\bwho is (?:this|that)\b",
]


class MultimodalQueryInterpreter:
    """Deterministic, grounded interpreter for Multimodal Vision-Language queries."""

    def parse_query(
        self,
        query_text: str,
        context: MultiTurnContext | None = None,
        available_models: list[str] | None = None,
        available_datasets: list[str] | None = None,
    ) -> VisionQuery:
        """Parse natural language question into structured VisionQuery with context resolution."""
        clean_text = query_text.strip()
        lower_text = clean_text.lower()
        qid = f"vq_{uuid.uuid4().hex[:10]}"

        # 1. Security & Unsupported Intent Check
        for pat in UNSUPPORTED_PATTERNS:
            if re.search(pat, lower_text):
                return VisionQuery(
                    query_id=qid,
                    user_query=clean_text,
                    query_type=VisionQueryType.IMAGE_QUERY,
                    answer=(
                        "VisionForge does not support arbitrary code execution, subjective emotion "
                        "inference, facial recognition, or open-ended speculation. Answers are strictly "
                        "derived from observable visual detections, tracks, events, dataset profiles, and evaluations."
                    ),
                    status=MultimodalQueryStatus.UNSUPPORTED,
                    grounding_verified=True,
                )

        # 2. Extract Extracted Entities & Filters
        target: dict[str, Any] = {}
        filters: dict[str, Any] = {}

        # Class filter
        for syn, canonical in CLASS_SYNONYMS.items():
            if re.search(rf"\b{syn}\b", lower_text):
                filters["class_name"] = canonical
                break

        # Confidence filter
        if re.search(r"\blow[- ]confidence\b|\blow confidence\b", lower_text):
            filters["max_confidence"] = 0.50
        elif re.search(r"\bhigh[- ]confidence\b|\bhighest confidence\b", lower_text):
            filters["min_confidence"] = 0.80

        conf_match = re.search(r"\bconfidence\s*(?:below|<|less than)\s*(0?\.\d+|\d+%)", lower_text)
        if conf_match:
            val_str = conf_match.group(1).replace("%", "")
            val = float(val_str) / 100.0 if float(val_str) > 1.0 else float(val_str)
            filters["max_confidence"] = val

        # Sample ID filter (e.g. "sample 1024" or "sample_1024")
        sample_match = re.search(r"\bsample\s*(?:#)?([a-zA-Z0-9_-]+)\b", clean_text, re.IGNORECASE)
        if sample_match:
            target["sample_id"] = sample_match.group(1)

        # Track ID filter
        track_match = re.search(r"\btrack\s*(?:#)?(\d+)\b", lower_text)
        if track_match:
            target["track_id"] = int(track_match.group(1))
            filters["track_id"] = int(track_match.group(1))

        # Region name filter (e.g. "Zone A", "Zone B", "Loading Dock")
        region_match = re.search(
            r"\b(zone\s+[a-z0-9]|loading dock|restricted corridor|staging area)\b", lower_text
        )
        if region_match:
            filters["region_name"] = region_match.group(1).title()

        # Dwell duration filter (e.g. "more than 10 seconds", "longer than 5s")
        dwell_match = re.search(
            r"\b(?:longer than|more than|stayed|dwelled for)\s*(\d+(?:\.\d+)?)\s*(?:s|seconds|sec)\b",
            lower_text,
        )
        if dwell_match:
            filters["min_duration_sec"] = float(dwell_match.group(1))

        # Time range filter (e.g. "between 10s and 30s")
        range_match = re.search(
            r"\bbetween\s*(\d+(?:\.\d+)?)\s*(?:s|sec)?\s*and\s*(\d+(?:\.\d+)?)\s*(?:s|sec)?\b",
            lower_text,
        )
        if range_match:
            filters["time_range"] = [float(range_match.group(1)), float(range_match.group(2))]

        # Model names filter (e.g. "v7", "v8", "yolo11s")
        model_matches = re.findall(r"\b(v\d+|yolo11[snmlx]|yolo\d+)\b", lower_text)
        if len(model_matches) >= 2:
            target["model_a"] = model_matches[0]
            target["model_b"] = model_matches[1]
        elif len(model_matches) == 1:
            target["model_id"] = model_matches[0]

        # 3. Context Inheritance for Multi-Turn Conversations
        if context:
            if context.selected_dataset and "dataset_id" not in target:
                target["dataset_id"] = context.selected_dataset
            if context.selected_model and "model_id" not in target:
                target["model_id"] = context.selected_model
            if context.selected_video and "video_id" not in target:
                target["video_id"] = context.selected_video
            if context.selected_image and "sample_id" not in target:
                target["sample_id"] = context.selected_image

            # Merge previous query filters if this turn is a follow-up refinement
            # (e.g. User previously asked "Show helmet failures", now says "Only low confidence ones")
            if context.previous_query:
                prev_filters = context.previous_query.get("filters", {})
                for k, v in prev_filters.items():
                    if k not in filters:
                        filters[k] = v
                if "query_type" in context.previous_query and not any(
                    re.search(p, lower_text)
                    for p in DATASET_PATTERNS
                    + MODEL_EVAL_PATTERNS
                    + SEARCH_PATTERNS
                    + EVENT_PATTERNS
                ):
                    # Inherit previous query category for follow-up refinement
                    prev_qt = context.previous_query.get("query_type")
                    if prev_qt:
                        try:
                            resolved_qt = VisionQueryType(prev_qt)
                            return self._build_query(qid, clean_text, resolved_qt, target, filters)
                        except ValueError:
                            pass

        # 4. Classify Query Type
        query_type = self._classify_query_type(lower_text, target, filters)

        # 5. Ambiguity Validation
        models = available_models or ["yolo11s.pt", "yolo11m.pt"]
        datasets = available_datasets or ["safety_v2", "warehouse_coco"]

        if query_type in (VisionQueryType.FAILURE_QUERY, VisionQueryType.EVALUATION_QUERY):
            if "model_id" not in target and "model_a" not in target and len(models) > 1:
                # If query asks for generic failures without specifying model and no active context model
                if not re.search(r"\b(sample|image)\b", lower_text) and not target.get("sample_id"):
                    return VisionQuery(
                        query_id=qid,
                        user_query=clean_text,
                        query_type=query_type,
                        target=target,
                        filters=filters,
                        status=MultimodalQueryStatus.AMBIGUOUS,
                        clarification_needed="Which model evaluation should I inspect for failures?",
                        clarification_options=models,
                        answer=f"Multiple model evaluations are available. Please specify a model: {', '.join(models)}.",
                        grounding_verified=True,
                    )

        if query_type == VisionQueryType.DATASET_QUERY:
            if (
                "dataset_id" not in target
                and len(datasets) > 1
                and not (context and context.selected_dataset)
            ):
                target["dataset_id"] = datasets[0]  # default to primary active dataset

        return self._build_query(qid, clean_text, query_type, target, filters)

    def _classify_query_type(
        self, lower_text: str, target: dict[str, Any], filters: dict[str, Any]
    ) -> VisionQueryType:
        """Deterministically map text patterns to one of 10 VisionQueryType categories."""
        if any(re.search(p, lower_text) for p in FAILURE_PATTERNS):
            return VisionQueryType.FAILURE_QUERY

        if any(re.search(p, lower_text) for p in SEARCH_PATTERNS):
            return VisionQueryType.SEARCH_QUERY

        if any(re.search(p, lower_text) for p in DATASET_PATTERNS):
            return VisionQueryType.DATASET_QUERY

        if any(re.search(p, lower_text) for p in MODEL_EVAL_PATTERNS) or "model_a" in target:
            return (
                VisionQueryType.MODEL_QUERY
                if re.search(r"\bcompare\b", lower_text)
                else VisionQueryType.EVALUATION_QUERY
            )

        if any(re.search(p, lower_text) for p in EVENT_PATTERNS) or "region_name" in filters:
            return VisionQueryType.EVENT_QUERY

        if any(re.search(p, lower_text) for p in TRACK_PATTERNS) or "track_id" in target:
            return VisionQueryType.TRACK_QUERY

        if re.search(r"\bvideo\b|\bwhen did\b|\bstream\b", lower_text):
            return VisionQueryType.VIDEO_QUERY

        if re.search(r"\bconfidence\b|\bbounding box\b|\blowest\b|\bhighest\b", lower_text):
            return VisionQueryType.OBJECT_QUERY

        return VisionQueryType.IMAGE_QUERY

    def _build_query(
        self,
        query_id: str,
        user_query: str,
        query_type: VisionQueryType,
        target: dict[str, Any],
        filters: dict[str, Any],
    ) -> VisionQuery:
        structured = {
            "query_type": query_type.value,
            "target": target,
            "filters": filters,
        }
        return VisionQuery(
            query_id=query_id,
            user_query=user_query,
            query_type=query_type,
            target=target,
            filters=filters,
            structured_query=structured,
            answer="",
            evidence=[],
            status=MultimodalQueryStatus.SUCCESS,
            grounding_verified=True,
        )
