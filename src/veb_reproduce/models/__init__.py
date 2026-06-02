"""Model builders."""

from __future__ import annotations

from typing import Any, Mapping

from veb_reproduce.models.fake import FakeGenerator
from veb_reproduce.models.llava_hf import LlavaHfGenerator


def build_generator(config: Mapping[str, Any]) -> Any:
    backend = str(config.get("generation", {}).get("backend", "llava_hf")).lower()
    if backend in {"llava", "llava_hf"}:
        return LlavaHfGenerator(config)
    if backend == "fake":
        return FakeGenerator(config)
    raise ValueError(f"Unsupported generation.backend: {backend}")
