from __future__ import annotations

import pytest

from veb_reproduce.models import build_generator
from veb_reproduce.utils.config import (
    load_config,
    load_method_config,
    load_run_config,
    validate_method_config,
)


def test_default_config_uses_original_llava_backend() -> None:
    config = load_config("configs/default.yaml")
    assert config["generation"]["backend"] == "llava_original"
    assert config["generation"]["model_name_or_path"] == "models/llava-v1.5-7b"


def test_hf_llava_backend_is_removed() -> None:
    with pytest.raises(ValueError, match="llava_hf has been removed"):
        build_generator({"generation": {"backend": "llava_hf"}})


def test_official_decoder_rejects_non_fp16_runtime() -> None:
    config = {"runtime": {"dtype": "float32"}, "generation": {"backend": "opera"}}
    with pytest.raises(ValueError, match="same precision"):
        build_generator(config)


def test_method_configs_do_not_override_shared_experiment_settings() -> None:
    for method in ["base", "veb", "halc", "opera"]:
        validate_method_config(method, load_method_config(method))


def test_method_config_rejects_shared_generation_override() -> None:
    with pytest.raises(ValueError, match="shared generation settings"):
        validate_method_config(
            "bad",
            {"generation": {"model_name_or_path": "models/other-llava"}},
        )


def test_method_config_rejects_shared_section_override() -> None:
    with pytest.raises(ValueError, match="shared experiment sections"):
        validate_method_config("bad", {"runtime": {"seed": 7}})


def test_smoke_config_preserves_fake_backend_for_base_and_veb() -> None:
    assert load_run_config("configs/smoke.yaml", "base")["generation"]["backend"] == "fake"
    assert load_run_config("configs/smoke.yaml", "veb")["generation"]["backend"] == "fake"
