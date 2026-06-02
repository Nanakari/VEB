"""Configuration loading and path resolution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else PROJECT_ROOT / "configs" / "default.yaml"
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return loaded


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
