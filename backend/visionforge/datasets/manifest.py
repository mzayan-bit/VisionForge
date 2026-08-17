"""Dataset Manifest Generation & Materialization System."""

import csv
import json
import logging
from io import StringIO
from pathlib import Path

from visionforge.core.config import get_settings
from visionforge.datasets.schemas import (
    DatasetPreparationManifest,
    LeakageFinding,
    SampleRef,
    SplitConfig,
)

logger = logging.getLogger("visionforge.datasets.manifest")


def build_manifest(
    preparation_id: str,
    dataset_id: str,
    dataset_version: str,
    split_config: SplitConfig,
    samples: list[SampleRef],
    counts: dict[str, int],
    leakage_findings: list[LeakageFinding],
) -> DatasetPreparationManifest:
    """Build structured machine-readable dataset preparation manifest."""
    exact_count = sum(1 for f in leakage_findings if f.leakage_type == "EXACT_DUPLICATE")
    near_count = sum(1 for f in leakage_findings if f.leakage_type == "POSSIBLE_NEAR_DUPLICATE")

    return DatasetPreparationManifest(
        preparation_id=preparation_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        random_seed=split_config.random_seed,
        split_config=split_config,
        software_version="VisionForge v0.1.0",
        total_samples=len(samples),
        train_count=counts.get("train", 0),
        val_count=counts.get("validation", 0),
        test_count=counts.get("test", 0),
        exact_duplicates_found=exact_count,
        near_duplicates_found=near_count,
        samples=samples,
    )


def export_manifest_json(manifest: DatasetPreparationManifest, output_path: Path) -> None:
    """Persist dataset manifest to disk JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.model_dump(), indent=2, default=str), encoding="utf-8"
    )
    logger.info("Exported prepared dataset manifest JSON to '%s'", output_path)


def export_manifest_csv(manifest: DatasetPreparationManifest) -> str:
    """Generate CSV string of sample references and split assignments."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "sample_id",
            "split",
            "content_hash",
            "width",
            "height",
            "format",
            "tags",
            "leakage_group_id",
        ]
    )

    for s in manifest.samples:
        tags_str = ";".join(s.tags) if s.tags else ""
        meta = s.image_metadata or {}
        writer.writerow(
            [
                s.id,
                s.split,
                s.content_hash,
                meta.get("width", ""),
                meta.get("height", ""),
                meta.get("format", ""),
                tags_str,
                s.leakage_group_id or "",
            ]
        )

    return output.getvalue()


def materialize_prepared_dataset(
    manifest: DatasetPreparationManifest, storage_dir: str | None = None
) -> Path:
    """Materialize prepared dataset manifest directory.

    Creates ~/.cache/visionforge/datasets/prepared/{prep_id}/ containing manifest.json & manifest.csv.
    """
    raw_path = storage_dir or (
        Path(get_settings().model_cache_dir).parent / "datasets" / "prepared"
    )
    prep_dir = Path(raw_path) / manifest.preparation_id
    prep_dir.mkdir(parents=True, exist_ok=True)

    json_file = prep_dir / "manifest.json"
    csv_file = prep_dir / "manifest.csv"

    export_manifest_json(manifest, json_file)
    csv_file.write_text(export_manifest_csv(manifest), encoding="utf-8")

    logger.info("Materialized prepared dataset manifest into '%s'", prep_dir)
    return json_file
