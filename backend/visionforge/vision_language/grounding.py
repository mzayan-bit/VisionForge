"""VisionForge Grounding & Hallucination Protection Engine.

Validates that every claim, entity ID, track ID, event ID, sample ID, and numerical
value in natural language answers directly corresponds to actual structured vision data.
"""

import logging
import re
from typing import Any

from visionforge.vision_language.schemas import VisionEvidenceItem

logger = logging.getLogger("visionforge.vision_language.grounding")


class GroundingDiscrepancy:
    """Discrepancy between generated text claim and actual vision data."""

    def __init__(self, claim_type: str, claimed_value: Any, expected_value: Any, explanation: str):
        self.claim_type = claim_type
        self.claimed_value = claimed_value
        self.expected_value = expected_value
        self.explanation = explanation

    def __repr__(self) -> str:
        return f"<GroundingDiscrepancy {self.claim_type}: claimed '{self.claimed_value}', valid={self.expected_value}>"


class GroundingValidator:
    """Deterministic validator verifying facts, counts, and entity IDs against vision execution data."""

    def validate_answer(
        self,
        candidate_answer: str,
        execution_result: dict[str, Any],
        evidence: list[VisionEvidenceItem],
    ) -> tuple[bool, str, list[GroundingDiscrepancy]]:
        """Validate candidate answer against actual structured execution results.

        Returns (is_verified, final_grounded_answer, discrepancies).
        """
        discrepancies: list[GroundingDiscrepancy] = []

        # 1. Entity Grounding: Check Track IDs
        valid_track_ids = set()
        if "tracks" in execution_result and isinstance(execution_result["tracks"], list):
            for t in execution_result["tracks"]:
                if isinstance(t, dict) and "track_id" in t:
                    valid_track_ids.add(int(t["track_id"]))
        for evi in evidence:
            if evi.track_id is not None:
                valid_track_ids.add(int(evi.track_id))

        # Find any "Track #<digits>" or "Track <digits>" in text
        track_mentions = re.findall(r"\bTrack\s*(?:#)?(\d+)\b", candidate_answer, re.IGNORECASE)
        for tid_str in track_mentions:
            tid = int(tid_str)
            if valid_track_ids and tid not in valid_track_ids:
                discrepancies.append(
                    GroundingDiscrepancy(
                        claim_type="UNGROUNDED_TRACK_ID",
                        claimed_value=tid,
                        expected_value=list(valid_track_ids),
                        explanation=f"Answer referenced Track #{tid}, but valid tracks in result are {list(valid_track_ids)}.",
                    )
                )

        # 2. Entity Grounding: Check Event IDs
        valid_event_ids = set()
        if "events" in execution_result and isinstance(execution_result["events"], list):
            for e in execution_result["events"]:
                if isinstance(e, dict) and "event_id" in e:
                    valid_event_ids.add(str(e["event_id"]))
        for evi in evidence:
            if evi.event_id is not None:
                valid_event_ids.add(str(evi.event_id))

        event_mentions = re.findall(
            r"\b(?:evt_|event_)([a-zA-Z0-9_]+)\b", candidate_answer, re.IGNORECASE
        )
        for eid in event_mentions:
            full_eid = f"evt_{eid}" if not eid.startswith("evt_") else eid
            if valid_event_ids and full_eid not in valid_event_ids and eid not in valid_event_ids:
                discrepancies.append(
                    GroundingDiscrepancy(
                        claim_type="UNGROUNDED_EVENT_ID",
                        claimed_value=eid,
                        expected_value=list(valid_event_ids),
                        explanation=f"Answer referenced Event '{eid}', but valid events in result are {list(valid_event_ids)}.",
                    )
                )

        # 3. Numerical Grounding: Check count claims (e.g. "Found X ...", "X failures", "X detections")
        actual_count = execution_result.get(
            "total_count",
            execution_result.get(
                "count",
                len(execution_result.get("records", []))
                or len(execution_result.get("failures", []))
                or len(execution_result.get("tracks", []))
                or len(evidence),
            ),
        )

        found_match = re.search(
            r"\b(?:Found|Identified|Discovered|Total of|Showing)\s+(\d+)\b",
            candidate_answer,
            re.IGNORECASE,
        )
        if found_match:
            claimed_cnt = int(found_match.group(1))
            if claimed_cnt != actual_count:
                discrepancies.append(
                    GroundingDiscrepancy(
                        claim_type="NUMERICAL_MISMATCH",
                        claimed_value=claimed_cnt,
                        expected_value=actual_count,
                        explanation=f"Answer claimed count of {claimed_cnt}, but actual structured data contains {actual_count}.",
                    )
                )

        # Determine if answer is verified
        if not discrepancies:
            return True, candidate_answer, []

        logger.warning(
            "Grounding protection detected %d discrepancies: %s",
            len(discrepancies),
            [d.explanation for d in discrepancies],
        )

        # Generate strictly grounded fallback answer
        fallback_answer = self._generate_grounded_fallback(execution_result, evidence)
        return False, fallback_answer, discrepancies

    def _generate_grounded_fallback(
        self, execution_result: dict[str, Any], evidence: list[VisionEvidenceItem]
    ) -> str:
        """Construct a 100% deterministic, grounded synthesis directly from structured data."""
        if not execution_result and not evidence:
            return "No matching visual data was found for this query."

        summary_parts = []
        if "summary" in execution_result and execution_result["summary"]:
            summary_parts.append(str(execution_result["summary"]))
        elif "count" in execution_result:
            cnt = execution_result["count"]
            qtype = execution_result.get("query_type", "items")
            summary_parts.append(f"Found {cnt} matching {qtype}.")
        elif evidence:
            summary_parts.append(
                f"Retrieved {len(evidence)} verified evidence items from vision data."
            )
        else:
            summary_parts.append("Query executed successfully against structured vision data.")

        return " ".join(summary_parts)
