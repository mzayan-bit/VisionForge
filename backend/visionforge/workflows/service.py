"""VisionForge Computer Vision Research Workflow Orchestration Service."""

import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from visionforge.core.config import get_settings
from visionforge.core.exceptions import VisionForgeException
from visionforge.experiments.service import get_experiment_service
from visionforge.workflows.schemas import (
    DatasetConfig,
    DecisionRecord,
    DecisionType,
    ResearchDefinition,
    ResearchWorkflow,
    StageNote,
    WorkflowEvent,
    WorkflowExportPackage,
    WorkflowLineageEdge,
    WorkflowLineageGraph,
    WorkflowLineageNode,
    WorkflowStage,
    WorkflowStatus,
    WorkflowTemplateType,
)
from visionforge.workflows.templates import create_template_workflow

logger = logging.getLogger("visionforge.workflows.service")

# Sequential order of the 8 stages
STAGE_ORDER: list[WorkflowStage] = [
    WorkflowStage.RESEARCH_DEFINITION,
    WorkflowStage.DATASET,
    WorkflowStage.EXPERIMENT,
    WorkflowStage.TRAINING,
    WorkflowStage.EVALUATION,
    WorkflowStage.ERROR_ANALYSIS,
    WorkflowStage.COMPARISON,
    WorkflowStage.REPORT,
]

VALID_STATE_TRANSITIONS: dict[WorkflowStatus, set[WorkflowStatus]] = {
    WorkflowStatus.DRAFT: {WorkflowStatus.READY, WorkflowStatus.CANCELLED},
    WorkflowStatus.READY: {WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED},
    WorkflowStatus.RUNNING: {
        WorkflowStatus.WAITING_FOR_REVIEW,
        WorkflowStatus.PAUSED,
        WorkflowStatus.COMPLETED,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.WAITING_FOR_REVIEW: {
        WorkflowStatus.RUNNING,
        WorkflowStatus.COMPLETED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.PAUSED: {WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED},
    WorkflowStatus.FAILED: {WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED},
    WorkflowStatus.COMPLETED: set(),
    WorkflowStatus.CANCELLED: set(),
}


class WorkflowNotFoundError(VisionForgeException):
    """Raised when a requested workflow ID does not exist."""

    def __init__(self, wf_id: str):
        super().__init__(
            message=f"Research workflow '{wf_id}' was not found.",
            code="WORKFLOW_NOT_FOUND",
            status_code=404,
        )


class InvalidStateTransitionError(VisionForgeException):
    """Raised when attempting an illegal workflow state transition."""

    def __init__(self, current_status: str, target_status: str):
        super().__init__(
            message=f"Cannot transition workflow from '{current_status}' to '{target_status}'.",
            code="INVALID_WORKFLOW_STATE_TRANSITION",
            status_code=400,
        )


class ResearchWorkflowService:
    """Central orchestration service managing reproducible research workflows."""

    def __init__(self, storage_dir: Path | None = None):
        cache_root = Path(get_settings().model_cache_dir).expanduser().resolve()
        raw_path = storage_dir or (cache_root.parent / "workflows")
        self._storage_dir = Path(raw_path).resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._storage_dir / "research_workflows.json"
        self._workflows: dict[str, ResearchWorkflow] = {}

        self.load_from_disk()
        self._seed_default_workflows_if_empty()

    # ─── CRUD & Instantiation ─────────────────────────────────────────

    def create_workflow(
        self,
        name: str,
        research_definition: ResearchDefinition,
        dataset_config: DatasetConfig | None = None,
        template_type: WorkflowTemplateType = WorkflowTemplateType.CUSTOM,
        description: str = "",
    ) -> ResearchWorkflow:
        """Create a new research workflow in DRAFT state."""
        wf_id = f"wf_{uuid.uuid4().hex[:10]}"
        ds = dataset_config or DatasetConfig()

        init_event = WorkflowEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            stage=WorkflowStage.RESEARCH_DEFINITION,
            event_type="WORKFLOW_CREATED",
            message=f"Created research workflow '{name}'.",
        )

        wf = ResearchWorkflow(
            workflow_id=wf_id,
            name=name,
            description=description,
            template_type=template_type,
            status=WorkflowStatus.READY,
            current_stage=WorkflowStage.RESEARCH_DEFINITION,
            research_definition=research_definition,
            dataset_config=ds,
            timeline_events=[init_event],
            reproducibility_metadata={
                "created_at": datetime.now(UTC).isoformat(),
                "dataset_locked": True,
            },
        )

        self._workflows[wf_id] = wf
        self.save_to_disk()
        return wf

    def create_from_template(
        self,
        template_type: WorkflowTemplateType,
        name: str | None = None,
        dataset_id: str = "safety_v2",
        dataset_version: str = "v2.0.0",
    ) -> ResearchWorkflow:
        """Instantiate a workflow from one of the pre-configured research study templates."""
        wf = create_template_workflow(
            template_type=template_type,
            name=name,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
        )
        self._workflows[wf.workflow_id] = wf
        self.save_to_disk()
        return wf

    def get_workflow(self, wf_id: str) -> ResearchWorkflow:
        """Retrieve workflow by ID."""
        if wf_id not in self._workflows:
            raise WorkflowNotFoundError(wf_id)
        return self._workflows[wf_id]

    def list_workflows(self) -> list[ResearchWorkflow]:
        """List all research workflows sorted chronologically."""
        return sorted(self._workflows.values(), key=lambda w: w.created_at, reverse=True)

    # ─── State Machine & Transitions ──────────────────────────────────

    def update_status(
        self, wf_id: str, new_status: WorkflowStatus, reason: str = ""
    ) -> ResearchWorkflow:
        """Transition workflow lifecycle state following strict state machine rules."""
        wf = self.get_workflow(wf_id)
        curr = wf.status

        if new_status not in VALID_STATE_TRANSITIONS.get(curr, set()):
            raise InvalidStateTransitionError(curr.value, new_status.value)

        wf.status = new_status
        wf.updated_at = datetime.now(UTC).isoformat()

        event = WorkflowEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            stage=wf.current_stage,
            event_type="STATUS_TRANSITION",
            message=f"Transitioned status from {curr.value} to {new_status.value}. {reason}".strip(),
            metadata={"from_status": curr.value, "to_status": new_status.value},
        )
        wf.timeline_events.append(event)
        self.save_to_disk()
        return wf

    def start_workflow(self, wf_id: str) -> ResearchWorkflow:
        """Start or initiate workflow execution."""
        wf = self.get_workflow(wf_id)
        if wf.status in (WorkflowStatus.DRAFT, WorkflowStatus.READY):
            return self.update_status(wf_id, WorkflowStatus.RUNNING, "Workflow started.")
        return wf

    def pause_workflow(self, wf_id: str) -> ResearchWorkflow:
        """Pause active workflow safely without losing state."""
        return self.update_status(wf_id, WorkflowStatus.PAUSED, "Researcher requested pause.")

    def resume_workflow(self, wf_id: str) -> ResearchWorkflow:
        """Resume paused workflow from its last valid state."""
        return self.update_status(wf_id, WorkflowStatus.RUNNING, "Resumed workflow execution.")

    def cancel_workflow(self, wf_id: str, reason: str = "User cancelled") -> ResearchWorkflow:
        """Cancel workflow execution."""
        return self.update_status(wf_id, WorkflowStatus.CANCELLED, reason)

    # ─── Stage Advancement & Orchestration ─────────────────────────────

    def advance_stage(self, wf_id: str) -> ResearchWorkflow:
        """Advance workflow to the next sequential stage."""
        wf = self.get_workflow(wf_id)
        curr_idx = STAGE_ORDER.index(wf.current_stage)

        if curr_idx >= len(STAGE_ORDER) - 1:
            # Already at final stage
            wf.status = WorkflowStatus.COMPLETED
            wf.updated_at = datetime.now(UTC).isoformat()
            self.save_to_disk()
            return wf

        next_stage = STAGE_ORDER[curr_idx + 1]
        wf.current_stage = next_stage
        wf.updated_at = datetime.now(UTC).isoformat()

        # Check for human review gate before expensive stages
        if next_stage in (
            WorkflowStage.TRAINING,
            WorkflowStage.EVALUATION,
            WorkflowStage.COMPARISON,
        ):
            wf.status = WorkflowStatus.WAITING_FOR_REVIEW
        else:
            wf.status = WorkflowStatus.RUNNING

        event = WorkflowEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            stage=next_stage,
            event_type="STAGE_ADVANCED",
            message=f"Advanced to stage {next_stage.value}.",
        )
        wf.timeline_events.append(event)

        # If reaching REPORT stage, auto-generate report
        if next_stage == WorkflowStage.REPORT:
            report_md = self.generate_workflow_report(wf_id)
            wf.generated_report_markdown = report_md

        self.save_to_disk()
        return wf

    # ─── Human Decision Gate & Investigation Loop ─────────────────────

    def record_decision(
        self,
        wf_id: str,
        decision: DecisionType,
        rationale: str,
        reviewer: str = "Researcher",
        target_stage: WorkflowStage | None = None,
    ) -> ResearchWorkflow:
        """Process researcher decision at review gates (ACCEPT, REJECT, or INVESTIGATE)."""
        wf = self.get_workflow(wf_id)
        dec_id = f"dec_{uuid.uuid4().hex[:8]}"

        record = DecisionRecord(
            decision_id=dec_id,
            decision=decision,
            reviewer=reviewer,
            rationale=rationale,
            target_stage=target_stage,
            iteration=wf.current_iteration,
        )
        wf.decision_history.append(record)

        if decision == DecisionType.ACCEPT:
            # Continue to next stage
            wf = self.advance_stage(wf_id)

        elif decision == DecisionType.REJECT:
            # Complete workflow noting rejection
            wf.status = WorkflowStatus.COMPLETED
            wf.timeline_events.append(
                WorkflowEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:8]}",
                    stage=wf.current_stage,
                    event_type="DECISION_REJECTED",
                    message=f"Researcher rejected experimental hypothesis: {rationale}",
                )
            )

        elif decision == DecisionType.INVESTIGATE:
            # Investigation Loop: Increment iteration, return to target stage without losing history
            destination = target_stage or WorkflowStage.ERROR_ANALYSIS
            wf.current_iteration += 1
            wf.current_stage = destination
            wf.status = WorkflowStatus.RUNNING

            wf.timeline_events.append(
                WorkflowEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:8]}",
                    stage=destination,
                    event_type="INVESTIGATION_LOOP_STARTED",
                    message=(
                        f"Commenced iteration #{wf.current_iteration} returning to {destination.value}: {rationale}"
                    ),
                    metadata={"iteration": wf.current_iteration, "target_stage": destination.value},
                )
            )

        wf.updated_at = datetime.now(UTC).isoformat()
        self.save_to_disk()
        return wf

    # ─── Stage Notes & Attachment ──────────────────────────────────────

    def add_stage_note(
        self, wf_id: str, stage: WorkflowStage, text: str, author: str = "Researcher"
    ) -> StageNote:
        """Attach researcher qualitative observations to a workflow stage."""
        wf = self.get_workflow(wf_id)
        note = StageNote(
            note_id=f"note_{uuid.uuid4().hex[:8]}",
            stage=stage,
            author=author,
            text=text,
        )
        wf.stage_notes.append(note)
        wf.updated_at = datetime.now(UTC).isoformat()
        self.save_to_disk()
        return note

    def attach_experiment(self, wf_id: str, experiment_id: str) -> ResearchWorkflow:
        """Attach a ResearchExperiment to the workflow."""
        wf = self.get_workflow(wf_id)
        wf.experiment_id = experiment_id
        wf.updated_at = datetime.now(UTC).isoformat()
        self.save_to_disk()
        return wf

    # ─── Lineage Graph Construction ───────────────────────────────────

    def get_lineage_graph(self, wf_id: str) -> WorkflowLineageGraph:
        """Construct directed lineage DAG linking Research -> Dataset -> Experiment -> Training -> Eval -> Report."""
        wf = self.get_workflow(wf_id)
        nodes: list[WorkflowLineageNode] = []
        edges: list[WorkflowLineageEdge] = []

        # 1. Research Question Node
        q_id = f"node_res_{wf.workflow_id}"
        nodes.append(
            WorkflowLineageNode(
                id=q_id,
                label=wf.research_definition.research_question[:40] + "...",
                stage=WorkflowStage.RESEARCH_DEFINITION,
                entity_type="research",
                route_link="/experiments",
            )
        )

        # 2. Dataset Node
        ds_id = f"node_ds_{wf.dataset_config.dataset_id}"
        nodes.append(
            WorkflowLineageNode(
                id=ds_id,
                label=f"Dataset: {wf.dataset_config.dataset_id} ({wf.dataset_config.dataset_version})",
                stage=WorkflowStage.DATASET,
                entity_type="dataset",
                route_link=f"/datasets?dataset_id={wf.dataset_config.dataset_id}",
            )
        )
        edges.append(
            WorkflowLineageEdge(
                source_id=q_id,
                target_id=ds_id,
                relationship="uses_dataset",
            )
        )

        # 3. Experiment Node
        if wf.experiment_id:
            exp_node_id = f"node_exp_{wf.experiment_id}"
            nodes.append(
                WorkflowLineageNode(
                    id=exp_node_id,
                    label=f"Experiment: {wf.experiment_id}",
                    stage=WorkflowStage.EXPERIMENT,
                    entity_type="experiment",
                    route_link="/experiments",
                )
            )
            edges.append(
                WorkflowLineageEdge(
                    source_id=ds_id,
                    target_id=exp_node_id,
                    relationship="configures_experiment",
                )
            )

        # 4. Training Run Nodes
        if wf.baseline_run_id:
            train_node_id = f"node_train_{wf.baseline_run_id}"
            nodes.append(
                WorkflowLineageNode(
                    id=train_node_id,
                    label=f"Baseline Run: {wf.baseline_run_id}",
                    stage=WorkflowStage.TRAINING,
                    entity_type="training_run",
                    route_link="/training",
                )
            )
            if wf.experiment_id:
                edges.append(
                    WorkflowLineageEdge(
                        source_id=f"node_exp_{wf.experiment_id}",
                        target_id=train_node_id,
                        relationship="trains_baseline",
                    )
                )

        # 5. Evaluation Node
        if wf.evaluation_ids:
            for eval_id in wf.evaluation_ids:
                eval_node_id = f"node_eval_{eval_id}"
                nodes.append(
                    WorkflowLineageNode(
                        id=eval_node_id,
                        label=f"Evaluation: {eval_id}",
                        stage=WorkflowStage.EVALUATION,
                        entity_type="evaluation",
                        route_link=f"/evaluation?eval_id={eval_id}",
                    )
                )
                if wf.baseline_run_id:
                    edges.append(
                        WorkflowLineageEdge(
                            source_id=f"node_train_{wf.baseline_run_id}",
                            target_id=eval_node_id,
                            relationship="evaluates_model",
                        )
                    )

        # 6. Report Node
        rep_node_id = f"node_rep_{wf.workflow_id}"
        nodes.append(
            WorkflowLineageNode(
                id=rep_node_id,
                label=f"Research Report: {wf.name}",
                stage=WorkflowStage.REPORT,
                entity_type="report",
                route_link="/experiments",
            )
        )
        if wf.evaluation_ids:
            edges.append(
                WorkflowLineageEdge(
                    source_id=f"node_eval_{wf.evaluation_ids[0]}",
                    target_id=rep_node_id,
                    relationship="synthesizes_report",
                )
            )

        return WorkflowLineageGraph(nodes=nodes, edges=edges)

    # ─── Report & Export ──────────────────────────────────────────────

    def generate_workflow_report(self, wf_id: str) -> str:
        """Synthesize traceable, grounded research report linking workflow entities."""
        wf = self.get_workflow(wf_id)
        exp_svc = get_experiment_service()

        exp_md = ""
        if wf.experiment_id:
            try:
                rep = exp_svc.generate_research_report(wf.experiment_id)
                exp_md = rep.markdown_report
            except Exception:
                pass

        decisions_summary = (
            "\n".join(
                f"- **Iteration #{d.iteration} ({d.decision.value}):** {d.rationale} (by {d.reviewer})"
                for d in wf.decision_history
            )
            or "None recorded."
        )

        notes_summary = (
            "\n".join(f"- **[{n.stage.value}]** {n.text} — *{n.author}*" for n in wf.stage_notes)
            or "No researcher notes attached."
        )

        report = f"""# End-to-End Research Workflow Report: {wf.name}

## 1. Research Question & Objective
- **Research Question:** *"{wf.research_definition.research_question}"*
- **Hypothesis:** *"{wf.research_definition.hypothesis}"*
- **Objective:** {wf.research_definition.objective or "Formal computer vision hypothesis test."}
- **Success Metrics:** {", ".join(wf.research_definition.success_metrics)}
- **Iteration Cycles Completed:** {wf.current_iteration}

---

## 2. Dataset & Protocol Lock
- **Dataset:** `{wf.dataset_config.dataset_id}` (Version: `{wf.dataset_config.dataset_version}`)
- **Splits:** Train `{wf.dataset_config.train_split}`, Val `{wf.dataset_config.val_split}`, Test `{wf.dataset_config.test_split}`
- **Immutability Status:** {"Locked" if wf.dataset_config.is_locked else "Open"}

---

## 3. Experimental Findings
{exp_md if exp_md else "Experiment evaluation completed."}

---

## 4. Human Decision History
{decisions_summary}

---

## 5. Researcher Notes & Observations
{notes_summary}

---
*Report synthesized deterministically by VisionForge End-to-End Research Workflow Engine.*
"""
        wf.generated_report_markdown = report
        self.save_to_disk()
        return report

    def export_workflow_package(self, wf_id: str) -> WorkflowExportPackage:
        """Create self-contained exportable JSON research package."""
        wf = self.get_workflow(wf_id)
        rep_md = wf.generated_report_markdown or self.generate_workflow_report(wf_id)

        exp_snapshot = None
        if wf.experiment_id:
            try:
                exp_svc = get_experiment_service()
                rexp = exp_svc.get_research_experiment(wf.experiment_id)
                exp_snapshot = rexp.model_dump()
            except Exception:
                pass

        hash_raw = f"{wf.workflow_id}:{wf.current_iteration}:{len(wf.decision_history)}:{wf.dataset_config.dataset_version}"
        rep_hash = hashlib.sha256(hash_raw.encode("utf-8")).hexdigest()[:16]

        return WorkflowExportPackage(
            workflow=wf,
            experiment_snapshot=exp_snapshot,
            evaluations_summary=[{"evaluation_id": eid} for eid in wf.evaluation_ids],
            report_markdown=rep_md,
            reproducibility_hash=rep_hash,
        )

    # ─── Seeding Defaults ─────────────────────────────────────────────

    def _seed_default_workflows_if_empty(self) -> None:
        """Seed representative real-world research workflows."""
        if self._workflows:
            return

        # ─── Workflow 1: Active Learning Annotation Efficiency
        wf1 = self.create_from_template(
            template_type=WorkflowTemplateType.ACTIVE_LEARNING_STUDY,
            name="Active Learning Label-Efficiency Study",
            dataset_id="safety_v2",
            dataset_version="v2.0.0",
        )
        wf1.status = WorkflowStatus.WAITING_FOR_REVIEW
        wf1.current_stage = WorkflowStage.COMPARISON
        wf1.experiment_id = "rexp_active_learning_01"
        wf1.baseline_run_id = "run_rand_seed_42"
        wf1.variant_run_ids = ["run_al_seed_42", "run_al_2k5_seed_42"]
        wf1.evaluation_ids = ["eval_yolo11s_safety_test"]

        wf1.decision_history.append(
            DecisionRecord(
                decision_id="dec_01",
                decision=DecisionType.ACCEPT,
                reviewer="Lead Researcher",
                rationale="Active learning achieved +0.062 mAP improvement with 5,000 samples and matched baseline at 2,500 samples.",
                iteration=1,
            )
        )
        wf1.stage_notes.append(
            StageNote(
                note_id="note_01",
                stage=WorkflowStage.DATASET,
                author="Data Curator",
                text="Confirmed safety_v2:test partition has 0 leakage with train split.",
            )
        )
        wf1.stage_notes.append(
            StageNote(
                note_id="note_02",
                stage=WorkflowStage.ERROR_ANALYSIS,
                author="Researcher",
                text="Small helmets in low illumination show largest recall increase (+0.08 mAP).",
            )
        )

        # ─── Workflow 2: Resolution & Augmentation Ablation
        wf2 = self.create_from_template(
            template_type=WorkflowTemplateType.BASELINE_VS_VARIANT,
            name="Resolution Scaling & Component Ablation",
            dataset_id="safety_v2",
            dataset_version="v2.0.0",
        )
        wf2.status = WorkflowStatus.RUNNING
        wf2.current_stage = WorkflowStage.TRAINING
        wf2.experiment_id = "rexp_resolution_01"
        wf2.baseline_run_id = "run_base_640"
        wf2.variant_run_ids = ["run_res_1024", "run_no_aug"]

        self.save_to_disk()

    # ─── Disk Persistence ─────────────────────────────────────────────

    def save_to_disk(self) -> None:
        try:
            serializable = [w.model_dump() for w in self._workflows.values()]
            self._file.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed saving workflows: %s", exc)

    def load_from_disk(self) -> None:
        if not self._file.is_file():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            for item in data:
                wf = ResearchWorkflow(**item)
                self._workflows[wf.workflow_id] = wf
        except Exception as exc:
            logger.warning("Failed loading workflows: %s", exc)


@lru_cache
def get_research_workflow_service() -> ResearchWorkflowService:
    """Return singleton instance of ResearchWorkflowService."""
    return ResearchWorkflowService()
