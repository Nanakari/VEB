"""Caching wrapper for visual evidence calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CachedEvidenceDetector:
    def __init__(self, detector: Any) -> None:
        self.detector = detector
        self.cache: dict[tuple[str, str], Any] = {}
        self.calls = 0

    def verify(self, image_path: str | Path, object_name: str) -> Any:
        key = (str(Path(image_path)), object_name.lower().strip())
        if key not in self.cache:
            self.cache[key] = self.detector.verify(image_path, object_name)
            self.calls += 1
        return self.cache[key]
