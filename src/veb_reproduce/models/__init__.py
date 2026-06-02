"""Model builders."""

from __future__ import annotations

from typing import Any, Mapping

from veb_reproduce.models.fake import FakeGenerator
from veb_reproduce.models.llava_original import OriginalLlavaGenerator
from veb_reproduce.models.official_decoders import OfficialDecoderGenerator


def build_generator(config: Mapping[str, Any]) -> Any:
    backend = str(config.get("generation", {}).get("backend", "llava_original")).lower()
    if backend in {"llava", "llava_original", "original_llava"}:
        return OriginalLlavaGenerator(config)
    if backend == "halc":
        return OfficialDecoderGenerator(config, "halc")
    if backend == "opera":
        return OfficialDecoderGenerator(config, "opera")
    if backend == "llava_hf":
        raise ValueError(
            "generation.backend=llava_hf has been removed. Use llava_original with the "
            "original merged LLaVA checkpoint."
        )
    if backend == "fake":
        return FakeGenerator(config)
    raise ValueError(f"Unsupported generation.backend: {backend}")
