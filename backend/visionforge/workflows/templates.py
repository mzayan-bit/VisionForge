"""VisionForge Pre-Configured Computer Vision Research Workflow Templates."""

import uuid

from visionforge.workflows.schemas import (
    DatasetConfig,
    ResearchDefinition,
    ResearchWorkflow,
    WorkflowEvent,
    WorkflowStage,
    WorkflowStatus,
    WorkflowTemplateType,
)


def create_template_workflow(
    template_type: WorkflowTemplateType,
    name: str | None = None,
    dataset_id: str = "safety_v2",
    dataset_version: str = "v2.0.0",
) -> ResearchWorkflow:
    """Instantiate a standardized, observable research workflow from a vetted template."""
    wf_id = f"wf_{uuid.uuid4().hex[:10]}"

    if template_type == WorkflowTemplateType.ACTIVE_LEARNING_STUDY:
        wf_name = name or "Active Learning Annotation Efficiency Study"
        definition = ResearchDefinition(
            research_question="Can active learning reduce manual bounding box annotation requirements by 50% while maintaining equivalent mAP@50?",
            hypothesis="Entropy-diversity sampling prioritizes high-loss boundary failure clusters, matching baseline accuracy with 2,500 samples instead of 5,000.",
            objective="Evaluate annotation efficiency curves and small-object safety recall gains.",
            success_metrics=["map50", "recall", "label_count"],
            constraints=["Fixed test split (safety_v2:test)", "Label budget <= 5,000 samples"],
        )
        desc = "Standard 8-stage active learning curation, retrain, and failure comparison cycle."

    elif template_type == WorkflowTemplateType.BASELINE_VS_VARIANT:
        wf_name = name or "Resolution & Augmentation Component Ablation"
        definition = ResearchDefinition(
            research_question="Does scaling spatial resolution to 1024x1024 improve small-object helmet recall sufficiently to justify the 2.2x compute cost?",
            hypothesis="1024px resolution improves small-object mAP by at least +0.04 over 640px baseline.",
            objective="Quantify accuracy gain vs. compute latency trade-off.",
            success_metrics=["map50", "training_time_sec", "gpu_hours"],
            constraints=[
                "Identical model architecture (yolo11s.pt)",
                "Identical evaluation protocol",
            ],
        )
        desc = "Ablation workflow measuring isolated component impact against control baseline."

    elif template_type == WorkflowTemplateType.MODEL_ARCHITECTURE_COMPARISON:
        wf_name = name or "CNN vs Vision Transformer Benchmark"
        definition = ResearchDefinition(
            research_question="How does RT-DETR (Vision Transformer) compare with YOLO11s (CNN) on multi-scale safety hazard detection under identical evaluation protocols?",
            hypothesis="RT-DETR achieves higher global mAP@50:95 on complex overlapping geometries at the cost of higher memory footprint.",
            objective="Side-by-side benchmark comparison under locked evaluation protocol.",
            success_metrics=["map50", "map50_95", "inference_latency_ms"],
            constraints=["Identical dataset version and test split"],
        )
        desc = "Cross-architecture evaluation comparing CNN and Transformer detectors."

    else:
        wf_name = name or "Custom Research Study"
        definition = ResearchDefinition(
            research_question="Custom research inquiry",
            hypothesis="Custom hypothesis",
            objective="Custom objective",
            success_metrics=["map50"],
            constraints=["Fixed test split"],
        )
        desc = "Custom user-defined research workflow."

    init_event = WorkflowEvent(
        event_id=f"evt_{uuid.uuid4().hex[:8]}",
        stage=WorkflowStage.RESEARCH_DEFINITION,
        event_type="WORKFLOW_INITIALIZED",
        message=f"Initialized '{wf_name}' from template {template_type.value}.",
    )

    return ResearchWorkflow(
        workflow_id=wf_id,
        name=wf_name,
        description=desc,
        template_type=template_type,
        status=WorkflowStatus.READY,
        current_stage=WorkflowStage.RESEARCH_DEFINITION,
        research_definition=definition,
        dataset_config=DatasetConfig(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            is_locked=True,
        ),
        timeline_events=[init_event],
        reproducibility_metadata={
            "template": template_type.value,
            "locked_protocol": True,
        },
    )
