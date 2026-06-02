"""Configuration loading and path resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]

METHOD_CONFIG_TOP_LEVEL_KEYS = {"name", "description", "status", "todo", "generation"}
METHOD_GENERATION_KEYS = {"backend", "return_token_scores", "halc", "opera"}
SHARED_GENERATION_KEYS = {
    "model_family",
    "model_name_or_path",
    "model_base",
    "model_name",
    "conv_mode",
    "temperature",
    "top_p",
    "do_sample",
    "num_beams",
    "pope_max_new_tokens",
    "caption_max_new_tokens",
    "fake",
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else PROJECT_ROOT / "configs" / "default.yaml"
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return loaded


def load_run_config(
    config_path: str | Path,
    method: str,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    base_config = load_config(config_path)
    method_config = load_method_config(method, project_root)
    if base_config.get("generation", {}).get("backend") == "fake" and method in {"base", "veb"}:
        method_config = {
            key: value for key, value in method_config.items() if key != "generation"
        }
    validate_method_config(method, method_config)
    return deep_merge(base_config, method_config)


def load_method_config(method: str, project_root: str | Path | None = None) -> dict[str, Any]:
    method_path = Path(project_root or PROJECT_ROOT) / "configs" / "methods" / f"{method}.yaml"
    if not method_path.exists():
        raise FileNotFoundError(f"Method config not found: {method_path}")
    with method_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Method config root must be a mapping: {method_path}")
    return loaded


def validate_method_config(method: str, method_config: Mapping[str, Any]) -> None:
    extra_top_level = set(method_config) - METHOD_CONFIG_TOP_LEVEL_KEYS
    if extra_top_level:
        raise ValueError(
            f"Method '{method}' overrides shared experiment sections: "
            f"{sorted(extra_top_level)}. Put shared runtime/model/data/prompt settings "
            "in the base config so all methods use the same environment."
        )
    generation = method_config.get("generation", {})
    if not isinstance(generation, Mapping):
        return
    protected = sorted(set(generation) & SHARED_GENERATION_KEYS)
    if protected:
        raise ValueError(
            f"Method '{method}' overrides shared generation settings: {protected}. "
            "Only backend, return_token_scores, and method-specific decoder blocks are allowed."
        )
    unknown = sorted(set(generation) - METHOD_GENERATION_KEYS)
    if unknown:
        raise ValueError(
            f"Method '{method}' contains unsupported generation override keys: {unknown}."
        )


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_path(value: Any, project_root: str | Path | None = None) -> Path | None:
    if value in {None, ""}:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path(project_root or PROJECT_ROOT) / path


def output_root(config: Mapping[str, Any], project_root: str | Path | None = None) -> Path:
    root = config.get("outputs", {}).get("root", "outputs/main")
    resolved = resolve_path(root, project_root)
    assert resolved is not None
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
