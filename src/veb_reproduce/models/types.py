"""Generation result types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TokenScore:
    token_id: int
    token_text: str
    token_char_span: tuple[int, int]
    logprob: float | None
    top1_prob: float | None
    top2_prob: float | None
    entropy: float | None
    margin: float | None
    position: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["token_char_span"] = list(self.token_char_span)
        return value


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_sec: float
    token_scores: list[TokenScore] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "latency_sec": self.latency_sec,
            "token_scores": [item.to_dict() for item in self.token_scores]
            if self.token_scores is not None
            else None,
            "metadata": dict(self.metadata),
        }
