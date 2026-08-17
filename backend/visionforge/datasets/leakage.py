"""Data Leakage Prevention & Duplicate Identification Engine.

Prevents identical (exact duplicate) or highly similar (near-duplicate) samples
from being split across Train, Validation, and Test partitions.
"""

import hashlib
import logging
from typing import Any

import numpy as np

from visionforge.datasets.schemas import LeakageFinding
from visionforge.memory.index import VisualMemoryRecord
from visionforge.search.similarity import compute_matrix_cosine_similarity

logger = logging.getLogger("visionforge.datasets.leakage")

NEAR_DUPLICATE_SIMILARITY_THRESHOLD = 0.92


def detect_data_leakage(
    records: list[VisualMemoryRecord],
    near_duplicate_threshold: float = NEAR_DUPLICATE_SIMILARITY_THRESHOLD,
) -> tuple[list[LeakageFinding], dict[str, str]]:
    """Analyze dataset for exact duplicates and near-duplicate embedding clusters.

    Returns tuple of (list of LeakageFinding objects, mapping of sample_id -> group_id).
    """
    if not records:
        return [], {}

    sample_to_group: dict[str, str] = {}
    findings: list[LeakageFinding] = []
    group_counter = 1

    # 1. Exact Duplicate Detection via Hashing (content_hash or metadata signature)
    hash_to_samples: dict[str, list[str]] = {}
    for rec in records:
        meta: dict[str, Any] = rec.image_metadata or {}
        raw_sig = (
            meta.get("content_hash")
            or f"{meta.get('width')}_{meta.get('height')}_{meta.get('file_size_bytes')}_{rec.id}"
        )
        sig_hash = hashlib.sha256(raw_sig.encode("utf-8")).hexdigest()

        if sig_hash not in hash_to_samples:
            hash_to_samples[sig_hash] = []
        hash_to_samples[sig_hash].append(rec.id)

    for sig_hash, sample_ids in hash_to_samples.items():
        if len(sample_ids) > 1:
            grp_id = f"leak_exact_{group_counter:03d}"
            group_counter += 1
            findings.append(
                LeakageFinding(
                    group_id=grp_id,
                    leakage_type="EXACT_DUPLICATE",
                    sample_ids=sample_ids,
                    similarity_score=1.0,
                )
            )
            for sid in sample_ids:
                sample_to_group[sid] = grp_id

    # 2. Near-Duplicate Detection via Vector Embedding Cosine Similarity Matrix
    valid_records = [r for r in records if r.embedding and len(r.embedding) == 768]
    if len(valid_records) > 1:
        matrix = np.array([r.embedding for r in valid_records], dtype=np.float32)
        v_ids = [r.id for r in valid_records]
        n_samples = len(v_ids)

        visited = set()
        for i in range(n_samples):
            if v_ids[i] in visited:
                continue

            query_vec = matrix[i]
            scores = compute_matrix_cosine_similarity(matrix, query_vec)

            # Find near-duplicate matches above threshold
            matches = []
            for j in range(n_samples):
                if j == i:
                    continue
                if float(scores[j]) >= near_duplicate_threshold:
                    matches.append((v_ids[j], float(scores[j])))

            if matches:
                # Found near-duplicate group
                cluster_members = [v_ids[i]] + [m[0] for m in matches]
                # Filter out samples already assigned to exact duplicate groups
                unassigned = [m for m in cluster_members if m not in sample_to_group]

                if len(unassigned) > 1:
                    grp_id = f"leak_near_{group_counter:03d}"
                    group_counter += 1
                    max_sim = max(m[1] for m in matches)
                    findings.append(
                        LeakageFinding(
                            group_id=grp_id,
                            leakage_type="POSSIBLE_NEAR_DUPLICATE",
                            sample_ids=unassigned,
                            similarity_score=round(max_sim, 4),
                        )
                    )
                    for sid in unassigned:
                        sample_to_group[sid] = grp_id
                        visited.add(sid)

    return findings, sample_to_group
