"""Object extraction data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExtractedObject:
    text: str
    normalized: str
    span: tuple[int, int]
    object_index: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["span"] = list(self.span)
        return value


def object_from_mapping(record: dict[str, Any]) -> ExtractedObject:
    span = record.get("span") or record.get("char_span")
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise ValueError(f"Object record missing valid span: {record}")
    return ExtractedObject(
        text=str(record["text"]),
        normalized=str(record["normalized"]),
        span=(int(span[0]), int(span[1])),
        object_index=int(record.get("object_index", 0)),
        source=str(record.get("source", "mapping")),
        metadata=dict(record.get("metadata", {})),
    )
