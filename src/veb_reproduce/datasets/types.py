"""Dataset record types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PopeSample:
    sample_id: str
    image_id: str
    image_path: Path
    question: str
    label: str | None
    setting: str
    target_object: str | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaptionSample:
    sample_id: str
    image_id: str
    image_path: Path
    gt_objects: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
