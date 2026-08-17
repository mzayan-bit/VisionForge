"""Deterministic Unit & Statistical Tests for Research Benchmark, Ablation, and Experiment Lab."""

import pytest
from fastapi.testclient import TestClient

from visionforge.experiments.schemas import (
    EvaluationProtocol,
    ExperimentRunRecord,
)
from visionforge.experiments.service import ExperimentService
from visionforge.main import app

client = TestClient(app)


def test_research_experiment_creation_and_protocol_lock(tmp_path):
    """Verify research experiment creation with locked evaluation protocol."""
    service = ExperimentService(storage_dir=tmp_path)

    protocol = EvaluationProtocol(
        dataset_split="test",
        primary_metric="map50",
        iou_threshold=0.50,
        confidence_threshold=0.25,
        is_locked=True,
    )

    rexp = service.create_research_experiment(
        name="Augmentation Effectiveness Study",
        hypothesis="Mosaic and Mixup augmentation will increase small-object helmet recall by at least +0.04 mAP.",
        dataset_id="safety_v2",
        dataset_version="v2.0.0",
        baseline_name="Baseline (No Aug)",
        baseline_config={"augmentation": "none", "image_size": 640, "learning_rate": 0.001},
        protocol=protocol,
        description="Testing data augmentation hypotheses on safety PPE benchmark.",
    )

    assert rexp.experiment_id.startswith("rexp_")
    assert rexp.evaluation_protocol.is_locked is True
    assert rexp.evaluation_protocol.primary_metric == "map50"
    assert len(rexp.variants) == 1
    assert rexp.variants[0].is_baseline is True
    assert rexp.variants[0].config_changes["augmentation"] == "none"


def test_variant_addition_and_configuration_diff(tmp_path):
    """Verify adding an experimental variant and generating parameter-level diff."""
    service = ExperimentService(storage_dir=tmp_path)

    rexp = service.create_research_experiment(
        name="Resolution Scaling Benchmark",
        hypothesis="Increasing resolution from 640 to 1024 improves small object mAP.",
        baseline_config={"image_size": 640, "augmentation": "standard", "lr": 0.001},
    )

    # Add Variant
    var = service.add_variant(
        exp_id=rexp.experiment_id,
        name="Resolution 1024px",
        config_changes={"image_size": 1024},
        description="Upscaling input resolution to 1024x1024",
    )

    assert var.name == "Resolution 1024px"
    assert var.is_baseline is False

    # Compute Variable Diff
    diff = service.compute_configuration_diff(rexp.experiment_id, var.variant_id)
    assert len(diff) >= 3

    size_diff = next((d for d in diff if d.parameter == "image_size"), None)
    assert size_diff is not None
    assert size_diff.baseline_value == 640
    assert size_diff.variant_value == 1024
    assert size_diff.has_changed is True
    assert size_diff.component_type == "resolution"

    aug_diff = next((d for d in diff if d.parameter == "augmentation"), None)
    assert aug_diff is not None
    assert aug_diff.has_changed is False


def test_multi_seed_runs_aggregation_and_single_run_warning(tmp_path):
    """Verify mean, std dev, min, max calculation across multiple seeds, and single-run warning."""
    service = ExperimentService(storage_dir=tmp_path)

    rexp = service.create_research_experiment(
        name="Multi-Seed Reliability Test",
        hypothesis="Active learning results are stable across random seed initializations.",
    )

    var = service.add_variant(
        exp_id=rexp.experiment_id,
        name="Active Learning Branch",
        config_changes={"selection_strategy": "entropy_diversity"},
    )

    # 1. First run -> MUST HAVE SINGLE RUN WARNING
    service.record_run(
        exp_id=rexp.experiment_id,
        variant_id=var.variant_id,
        run_record=ExperimentRunRecord(
            run_id="run_seed_1",
            seed=42,
            model_id="yolo11s.pt",
            metrics={"map50": 0.800, "precision": 0.820, "recall": 0.780},
        ),
    )

    rexp_updated = service.get_research_experiment(rexp.experiment_id)
    var_1 = rexp_updated.variants[1]
    stat_single = var_1.aggregated_metrics["map50"]
    assert stat_single.count == 1
    assert stat_single.mean == 0.800
    assert stat_single.std_dev == 0.0
    assert stat_single.is_single_run is True
    assert "Single run" in stat_single.warning

    # 2. Second and Third runs (Run 1: 0.80, Run 2: 0.82, Run 3: 0.81)
    service.record_run(
        exp_id=rexp.experiment_id,
        variant_id=var.variant_id,
        run_record=ExperimentRunRecord(
            run_id="run_seed_2",
            seed=43,
            model_id="yolo11s.pt",
            metrics={"map50": 0.820, "precision": 0.840, "recall": 0.800},
        ),
    )
    service.record_run(
        exp_id=rexp.experiment_id,
        variant_id=var.variant_id,
        run_record=ExperimentRunRecord(
            run_id="run_seed_3",
            seed=44,
            model_id="yolo11s.pt",
            metrics={"map50": 0.810, "precision": 0.830, "recall": 0.790},
        ),
    )

    rexp_multi = service.get_research_experiment(rexp.experiment_id)
    var_multi = rexp_multi.variants[1]
    stat_multi = var_multi.aggregated_metrics["map50"]

    # Mean: (0.80 + 0.82 + 0.81) / 3 = 0.810
    assert stat_multi.count == 3
    assert pytest.approx(stat_multi.mean, 0.001) == 0.810
    assert pytest.approx(stat_multi.min, 0.001) == 0.800
    assert pytest.approx(stat_multi.max, 0.001) == 0.820
    assert stat_multi.std_dev > 0.0
    assert stat_multi.is_single_run is False
    assert stat_multi.confidence_interval_95 is not None
    assert (
        stat_multi.confidence_interval_95[0]
        < stat_multi.mean
        < stat_multi.confidence_interval_95[1]
    )


def test_ablation_study_matrix_generation(tmp_path):
    """Verify AblationStudy matrix computation with component presence and deltas."""
    service = ExperimentService(storage_dir=tmp_path)

    rexp = service.create_research_experiment(
        name="Component Ablation Suite",
        hypothesis="Ablating augmentation or active learning reduces mAP.",
    )
    base_id = rexp.variants[0].variant_id
    service.record_run(
        rexp.experiment_id,
        base_id,
        ExperimentRunRecord(
            run_id="base_run",
            seed=42,
            model_id="yolo11s.pt",
            metrics={"map50": 0.800},
        ),
    )

    # Add Ablation 1: No Augmentation
    v_noaug = service.add_variant(
        rexp.experiment_id,
        name="Ablation: No Augmentation",
        config_changes={"augmentation": "none"},
    )
    service.record_run(
        rexp.experiment_id,
        v_noaug.variant_id,
        ExperimentRunRecord(
            run_id="noaug_run",
            seed=42,
            model_id="yolo11s.pt",
            metrics={"map50": 0.760},
        ),
    )

    # Add Ablation 2: High Resolution
    v_res = service.add_variant(
        rexp.experiment_id,
        name="Component: Resolution 1024px",
        config_changes={"image_size": 1024},
    )
    service.record_run(
        rexp.experiment_id,
        v_res.variant_id,
        ExperimentRunRecord(
            run_id="res_run",
            seed=42,
            model_id="yolo11s.pt",
            metrics={"map50": 0.845},
        ),
    )

    # Compute Ablation Matrix
    ablation = service.compute_ablation_matrix(rexp.experiment_id)
    assert len(ablation.matrix) == 2
    assert ablation.measured_effects["Ablation: No Augmentation"] == -0.04
    assert ablation.measured_effects["Component: Resolution 1024px"] == +0.045


def test_research_report_synthesis_and_grounding(tmp_path):
    """Verify research report generation contains only grounded facts and valid deltas."""
    service = ExperimentService(storage_dir=tmp_path)

    rexp = service.create_research_experiment(
        name="Active Learning Efficiency Report",
        hypothesis="Active learning achieves +0.06 mAP gain over random sampling.",
    )
    # Record Baseline
    service.record_run(
        rexp.experiment_id,
        rexp.variants[0].variant_id,
        ExperimentRunRecord(
            run_id="rand_run",
            seed=42,
            model_id="yolo11s.pt",
            metrics={"map50": 0.710},
            per_class_metrics={"helmet": 0.750, "vest": 0.720},
            error_counts={"false_negatives": 50, "false_positives": 30},
        ),
    )

    # Record Variant
    var_al = service.add_variant(
        rexp.experiment_id,
        name="Active Learning 5k",
        config_changes={"strategy": "active_learning"},
    )
    service.record_run(
        rexp.experiment_id,
        var_al.variant_id,
        ExperimentRunRecord(
            run_id="al_run",
            seed=42,
            model_id="yolo11s.pt",
            metrics={"map50": 0.770},
            per_class_metrics={"helmet": 0.830, "vest": 0.770},
            error_counts={"false_negatives": 38, "false_positives": 24},
        ),
    )

    report = service.generate_research_report(rexp.experiment_id)
    assert report.experiment_id == rexp.experiment_id
    assert report.performance_deltas["Active Learning 5k"] == +0.060
    assert report.per_class_deltas["helmet"] == +0.080
    assert report.per_class_deltas["vest"] == +0.050
    assert report.error_deltas["false_negatives"] == -24.0
    assert report.error_deltas["false_positives"] == -20.0
    assert "Active Learning 5k" in report.markdown_report
    assert "+0.060" in report.markdown_report


def test_research_experiment_rest_api_endpoints():
    """Verify REST API routes for Research Experiments and Ablations."""
    # 1. Create Research Experiment
    res_create = client.post(
        "/api/v1/experiments/research",
        json={
            "name": "API Research Test",
            "hypothesis": "Test hypothesis via REST API.",
            "dataset_id": "safety_v2",
            "dataset_version": "v2.0.0",
            "baseline_name": "Base Control",
            "baseline_config": {"image_size": 640},
        },
    )
    assert res_create.status_code == 201
    rexp_data = res_create.json()
    exp_id = rexp_data["experiment_id"]
    rexp_data["variants"][0]["variant_id"]

    # 2. Add Variant
    res_var = client.post(
        f"/api/v1/experiments/research/{exp_id}/variants",
        json={
            "name": "Variant HighRes",
            "config_changes": {"image_size": 1024},
            "description": "High resolution branch",
        },
    )
    assert res_var.status_code == 201
    var_id = res_var.json()["variant_id"]

    # 3. Record Run
    res_run = client.post(
        f"/api/v1/experiments/research/{exp_id}/variants/{var_id}/runs",
        json={
            "run_id": "run_api_01",
            "seed": 42,
            "model_id": "yolo11s.pt",
            "metrics": {"map50": 0.835, "precision": 0.860, "recall": 0.810},
            "per_class_metrics": {"helmet": 0.870},
            "error_counts": {"false_positives": 15},
        },
    )
    assert res_run.status_code == 200
    assert res_run.json()["aggregated_metrics"]["map50"]["mean"] == 0.835

    # 4. Get Configuration Diff
    res_diff = client.get(f"/api/v1/experiments/research/{exp_id}/variants/{var_id}/diff")
    assert res_diff.status_code == 200
    diffs = res_diff.json()
    assert any(d["parameter"] == "image_size" and d["has_changed"] is True for d in diffs)

    # 5. Get Ablation Matrix
    res_abl = client.get(f"/api/v1/experiments/research/{exp_id}/ablation")
    assert res_abl.status_code == 200
    assert "matrix" in res_abl.json()

    # 6. Get Research Report
    res_rep = client.get(f"/api/v1/experiments/research/{exp_id}/research-report")
    assert res_rep.status_code == 200
    assert "markdown_report" in res_rep.json()
