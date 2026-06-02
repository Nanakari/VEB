"""Deterministic fake generator for tests and smoke runs."""

from __future__ import annotations

import time
from typing import Any, Mapping

from veb_reproduce.models.types import GenerationResult, TokenScore


class FakeGenerator:
    def __init__(self, config: Mapping[str, Any]) -> None:
        fake = config.get("generation", {}).get("fake", {})
        self.outputs = dict(fake.get("outputs", {}))
        self.default_caption = str(fake.get("default_caption", "A person is sitting with a laptop and a phone."))
        self.default_answer = str(fake.get("default_answer", "yes"))

    def generate(
        self,
        image_path: str,
        prompt: str,
        *,
        sample_id: str | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        start = time.perf_counter()
        text = str(self.outputs.get(sample_id or "", self._default_for_prompt(prompt)))
        return GenerationResult(
            text=text,
            latency_sec=time.perf_counter() - start,
            token_scores=_fake_token_scores(text),
            metadata={"backend": "fake", "max_new_tokens": max_new_tokens},
        )

    def _default_for_prompt(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "yes or no" in lowered or "answer" in lowered and "question" in lowered:
            return self.default_answer
        return self.default_caption


def _fake_token_scores(text: str) -> list[TokenScore]:
    tokens: list[TokenScore] = []
    position = 1
    cursor = 0
    for raw in text.split():
        start = text.find(raw, cursor)
        end = start + len(raw)
        cursor = end
        normalized = raw.strip(".,!?;:")
        entropy = 0.9 if normalized.lower() in {"laptop", "phone", "bicycle"} else 0.2
        margin = 0.1 if normalized.lower() in {"laptop", "phone", "bicycle"} else 0.8
        tokens.append(
            TokenScore(
                token_id=position,
                token_text=raw,
                token_char_span=(start, end),
                logprob=-0.1,
                top1_prob=0.8,
                top2_prob=0.2,
                entropy=entropy,
                margin=margin,
                position=position,
            )
        )
        position += 1
    return tokens
