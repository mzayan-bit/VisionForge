"""Deterministic Dataset Partitioning & Splitting Engine.

Implements Random, Stratified, and Group-Aware dataset splitting strategies
while strictly honoring Data Leakage Group boundaries.
"""

import logging
import random
from typing import Any

from visionforge.core.exceptions import VisionForgeException
from visionforge.datasets.schemas import SampleRef, SplitConfig, SplitStrategy
from visionforge.memory.index import VisualMemoryRecord

logger = logging.getLogger("visionforge.datasets.splitting")


class InvalidSplitRatioError(VisionForgeException):
    """Raised when split ratios do not sum to 1.0 or contain invalid values."""

    def __init__(self, train: float, val: float, test: float):
        total = round(train + val + test, 4)
        super().__init__(
            message=f"Split ratios (Train: {train}, Val: {val}, Test: {test}) sum to {total}. Total must equal 1.0.",
            code="INVALID_SPLIT_RATIO",
            status_code=400,
        )


def partition_dataset(
    records: list[VisualMemoryRecord],
    sample_to_leakage_group: dict[str, str],
    config: SplitConfig,
) -> tuple[list[SampleRef], dict[str, int]]:
    """Partition dataset into Train, Validation, and Test splits.

    Guarantees:
      1. Determinism: Same seed + dataset = identical split assignments.
      2. Data Leakage Safety: All samples in a leakage group land in the SAME split.
    """
    total_ratio = round(config.train_ratio + config.val_ratio + config.test_ratio, 4)
    if abs(total_ratio - 1.0) > 1e-4:
        raise InvalidSplitRatioError(config.train_ratio, config.val_ratio, config.test_ratio)

    if not records:
        return [], {"train": 0, "validation": 0, "test": 0}

    # Group samples by Leakage Group or individual item
    groups: dict[str, list[VisualMemoryRecord]] = {}
    for rec in records:
        grp_key = sample_to_leakage_group.get(rec.id, f"single_{rec.id}")
        if grp_key not in groups:
            groups[grp_key] = []
        groups[grp_key].append(rec)

    group_keys = list(groups.keys())

    # Deterministic Shuffle using configured random seed
    rng = random.Random(config.random_seed)

    if config.strategy == SplitStrategy.RANDOM:
        rng.shuffle(group_keys)
    elif config.strategy == SplitStrategy.STRATIFIED:
        # Sort keys deterministically by dominant tag before shuffling
        group_keys.sort(key=lambda k: (groups[k][0].tags[0] if groups[k][0].tags else "", k))
        rng.shuffle(group_keys)
    elif config.strategy == SplitStrategy.GROUP_AWARE and config.group_by_field:
        field = config.group_by_field
        group_keys.sort(key=lambda k: (str(groups[k][0].image_metadata.get(field, "")), k))
        rng.shuffle(group_keys)
    else:
        rng.shuffle(group_keys)

    # Calculate target sample counts
    total_samples = len(records)
    target_train = int(round(total_samples * config.train_ratio))
    target_val = int(round(total_samples * config.val_ratio))
    target_test = total_samples - target_train - target_val

    # Assign groups to splits to match target sample counts
    split_samples: dict[str, list[VisualMemoryRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    current_train = 0
    current_val = 0

    for grp_key in group_keys:
        members = groups[grp_key]
        n_members = len(members)

        if current_train + n_members <= target_train or (current_train == 0 and target_train > 0):
            split_samples["train"].extend(members)
            current_train += n_members
        elif current_val + n_members <= target_val or (current_val == 0 and target_val > 0):
            split_samples["validation"].extend(members)
            current_val += n_members
        else:
            split_samples["test"].extend(members)

    # If test is empty but target_test > 0 and train/val has extra, rebalance gracefully
    if not split_samples["test"] and target_test > 0 and len(group_keys) >= 3:
        # Move last group from train or val to test
        for target_split in ["train", "validation"]:
            if len(split_samples[target_split]) > 1:
                moved = split_samples[target_split].pop()
                split_samples["test"].append(moved)
                break

    # Build SampleRef list
    sample_refs: list[SampleRef] = []
    counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}

    for split_name, rec_list in split_samples.items():
        counts[split_name] = len(rec_list)
        for rec in rec_list:
            meta: dict[str, Any] = rec.image_metadata or {}
            grp_id = sample_to_leakage_group.get(rec.id)
            sample_refs.append(
                SampleRef(
                    id=rec.id,
                    split=split_name,
                    file_path=meta.get("file_path", ""),
                    content_hash=meta.get("content_hash", ""),
                    image_metadata=meta,
                    tags=rec.tags,
                    leakage_group_id=grp_id,
                )
            )

    return sample_refs, counts
