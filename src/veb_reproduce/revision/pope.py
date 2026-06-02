"""POPE yes/no VEB revision."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from veb_reproduce.utils.answers import normalize_yes_no


@dataclass(frozen=True)
class PopeRevisionResult:
    original_answer: str
    revised_answer: str
    action: dict[str, Any]
    visual_evidence: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def revise_pope_answer(
    image_path: str | Path,
    target_object: str | None,
    base_answer: str,
    detector: Any,
    config: Mapping[str, Any],
) -> PopeRevisionResult:
    normalized_answer = normalize_yes_no(base_answer) or str(base_answer).strip().lower()
    if normalized_answer != "yes" or not target_object:
        return PopeRevisionResult(
            original_answer=base_answer,
            revised_answer=normalized_answer,
            action={"action": "keep", "rule": "pope_keep", "reason": "not_positive_or_missing_target"},
            visual_evidence=None,
        )

    tau_pope = float(config.get("veb", {}).get("tau_pope", 0.25))
    evidence = detector.verify(image_path, target_object)
    evidence_dict = evidence.to_dict() if hasattr(evidence, "to_dict") else dict(evidence)
    score = float(evidence_dict.get("score", 0.0))
    if score < tau_pope:
        return PopeRevisionResult(
            original_answer=base_answer,
            revised_answer="no",
            action={
                "action": "yes_to_no",
                "rule": "pope_yes_to_no",
                "reason": "low_visual_evidence",
                "object": target_object,
                "score": score,
            },
            visual_evidence=evidence_dict,
        )
    return PopeRevisionResult(
        original_answer=base_answer,
        revised_answer="yes",
        action={
            "action": "keep",
            "rule": "pope_keep_yes",
            "reason": "sufficient_visual_evidence",
            "object": target_object,
            "score": score,
        },
        visual_evidence=evidence_dict,
    )
