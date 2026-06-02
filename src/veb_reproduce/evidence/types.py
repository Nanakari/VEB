"""Visual evidence data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DetectionBox:
    box: list[float]
    score: float
    phrase: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualEvidence:
    normalized: str
    query: str
    score: float
    has_visual_evidence: bool
    detector: str
    boxes: list[DetectionBox] = field(default_factory=list)
    latency_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized": self.normalized,
            "query": self.query,
            "score": self.score,
            "has_visual_evidence": self.has_visual_evidence,
            "detector": self.detector,
            "boxes": [box.to_dict() for box in self.boxes],
            "latency_sec": self.latency_sec,
            "metadata": dict(self.metadata),
        }
