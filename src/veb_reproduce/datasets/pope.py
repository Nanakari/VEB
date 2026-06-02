"""POPE dataset reader and target-object extraction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from veb_reproduce.datasets.types import PopeSample
from veb_reproduce.utils.config import resolve_path
from veb_reproduce.utils.io import read_jsonl


def load_pope_samples(
    dataset_config: Mapping[str, Any],
    project_root: str | Path,
    *,
    settings: list[str] | None = None,
    limit_per_setting: int | None = None,
) -> list[PopeSample]:
    paths = dataset_config.get("paths", {})
    image_root = resolve_path(paths.get("image_root"), project_root)
    annotation_files = paths.get("annotation_files", {})
    if image_root is None:
        raise ValueError("POPE config is missing paths.image_root")
    if not isinstance(annotation_files, Mapping):
        raise ValueError("POPE config must define paths.annotation_files")

    requested_settings = settings or list(annotation_files.keys())
    samples: list[PopeSample] = []
    for setting in requested_settings:
        annotation_path = resolve_path(annotation_files.get(setting), project_root)
        if annotation_path is None:
            raise ValueError(f"POPE annotation file is not configured for `{setting}`")
        if not annotation_path.exists():
            raise FileNotFoundError(f"POPE annotation file not found: {annotation_path}")

        count = 0
        for index, record in enumerate(read_jsonl(annotation_path)):
            image_ref = _first_present(record, ["image", "image_path", "file_name", "filename"])
            image_id = str(record.get("image_id") or _image_id_from_ref(image_ref))
            question = _first_present(record, ["question", "text", "query"])
            if question is None:
                raise ValueError(f"Missing POPE question in {annotation_path}:{index + 1}")
            label = normalize_label(_first_present(record, ["answer", "label", "gt_answer", "truth"]))
            target_object = extract_target_object(str(question), record)
            local_id = record.get("sample_id") or record.get("question_id") or record.get("id") or index
            samples.append(
                PopeSample(
                    sample_id=f"{setting}:{local_id}",
                    image_id=image_id,
                    image_path=_resolve_image_path(image_root, image_ref, image_id),
                    question=str(question),
                    label=label,
                    setting=setting,
                    target_object=target_object,
                    raw=dict(record),
                )
            )
            count += 1
            if limit_per_setting is not None and count >= limit_per_setting:
                break
    return samples


def extract_target_object(question: str, record: Mapping[str, Any] | None = None) -> str | None:
    if record is not None:
        structured = _first_present(
            record,
            ["target_object", "object", "obj", "category", "category_name", "class", "class_name"],
        )
        if structured is not None:
            value = str(structured).strip()
            return value or None

    normalized = " ".join(question.strip().rstrip("?.!").split())
    patterns = [
        r"(?i)^is there (?:a |an |the |any )?(?P<object>.+?) (?:in|on|at) (?:the |this )?image$",
        r"(?i)^are there (?:any |some )?(?P<object>.+?) (?:in|on|at) (?:the |this )?image$",
        r"(?i)^does (?:the |this )?image contain (?:a |an |the |any )?(?P<object>.+?)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized)
        if match:
            return match.group("object").strip() or None
    return None


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"yes", "y", "true", "1"}:
        return "yes"
    if lowered in {"no", "n", "false", "0"}:
        return "no"
    return lowered


def _first_present(record: Mapping[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return default


def _image_id_from_ref(image_ref: Any) -> str:
    if image_ref is None:
        return ""
    stem = Path(str(image_ref)).stem
    match = re.search(r"(\d+)$", stem)
    return match.group(1) if match else stem


def _resolve_image_path(image_root: Path, image_ref: Any, image_id: str) -> Path:
    if image_ref:
        image_path = Path(str(image_ref))
        if image_path.is_absolute():
            return image_path
        return image_root / image_path
    if image_id:
        if str(image_id).isdigit():
            return image_root / f"COCO_val2014_{int(image_id):012d}.jpg"
        return image_root / str(image_id)
    raise ValueError("POPE record must provide an image reference or image_id")
