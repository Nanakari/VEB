"""Visual evidence detector builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from veb_reproduce.evidence.cache import CachedEvidenceDetector
from veb_reproduce.evidence.fake import FakeEvidenceDetector
from veb_reproduce.evidence.grounding_dino import GroundingDinoDetector


def build_detector(config: Mapping[str, Any], project_root: str | Path) -> Any:
    detector_name = str(config.get("evidence", {}).get("detector", "groundingdino")).lower()
    if detector_name == "groundingdino":
        detector = GroundingDinoDetector(config, project_root)
    elif detector_name == "fake":
        detector = FakeEvidenceDetector(config)
    else:
        raise ValueError(f"Unsupported evidence.detector: {detector_name}")
    if bool(config.get("evidence", {}).get("cache", True)):
        return CachedEvidenceDetector(detector)
    return detector
