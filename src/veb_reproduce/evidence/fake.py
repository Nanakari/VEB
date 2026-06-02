"""Deterministic fake visual evidence detector for tests and smoke runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from veb_reproduce.evidence.types import DetectionBox, VisualEvidence


class FakeEvidenceDetector:
    detector_name = "fake"

    def __init__(self, config: Mapping[str, Any]) -> None:
        evidence_config = config.get("evidence", {})
        fake_config = evidence_config.get("fake", {})
        self.scores = {str(key).lower(): float(value) for key, value in fake_config.get("scores", {}).items()}
        self.default_score = float(fake_config.get("default_score", 0.0))
        self.evidence_threshold = float(evidence_config.get("evidence_threshold", 0.25))

    def verify(self, image_path: str | Path, object_name: str) -> VisualEvidence:
        start = time.perf_counter()
        score = self.scores.get(object_name.lower(), self.default_score)
        boxes = [DetectionBox(box=[0.0, 0.0, 1.0, 1.0], score=score, phrase=object_name)] if score > 0 else []
        return VisualEvidence(
            normalized=object_name,
            query=f"{object_name} .",
            score=score,
            has_visual_evidence=score >= self.evidence_threshold,
            detector=self.detector_name,
            boxes=boxes,
            latency_sec=time.perf_counter() - start,
            metadata={"image_path": str(image_path), "evidence_threshold": self.evidence_threshold},
        )
