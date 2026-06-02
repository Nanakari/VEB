"""COCO caption dataset reader for CHAIR-style evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from veb_reproduce.datasets.types import CaptionSample
from veb_reproduce.utils.config import resolve_path
from veb_reproduce.utils.io import load_json, read_jsonl


def load_coco_caption_samples(
    dataset_config: Mapping[str, Any],
    project_root: str | Path,
    *,
    limit: int | None = None,
) -> list[CaptionSample]:
    paths = dataset_config.get("paths", {})
    samples_jsonl = resolve_path(paths.get("samples_jsonl"), project_root)
    if samples_jsonl is not None:
        return _load_samples_jsonl(samples_jsonl, project_root, limit=limit)

    image_root = resolve_path(paths.get("image_root"), project_root)
    instances_file = resolve_path(paths.get("instances_file"), project_root)
    if image_root is None or instances_file is None:
        raise ValueError("COCO config must define paths.image_root and paths.instances_file")
    if not instances_file.exists():
        raise FileNotFoundError(f"COCO instances file not found: {instances_file}")

    instances = load_json(instances_file)
    categories = {int(item["id"]): str(item["name"]) for item in instances.get("categories", [])}
    gt_by_image: dict[str, set[str]] = {}
    for ann in instances.get("annotations", []):
        image_id = str(ann.get("image_id"))
        category = categories.get(int(ann.get("category_id")))
        if category:
            gt_by_image.setdefault(image_id, set()).add(category)

    samples: list[CaptionSample] = []
    for image in instances.get("images", []):
        image_id = str(image.get("id"))
        file_name = image.get("file_name") or f"COCO_val2014_{int(image_id):012d}.jpg"
        samples.append(
            CaptionSample(
                sample_id=f"coco:{image_id}",
                image_id=image_id,
                image_path=image_root / str(file_name),
                gt_objects=sorted(gt_by_image.get(image_id, set())),
                raw=dict(image),
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def _load_samples_jsonl(
    samples_jsonl: Path, project_root: str | Path, *, limit: int | None
) -> list[CaptionSample]:
    if not samples_jsonl.exists():
        raise FileNotFoundError(f"Caption samples file not found: {samples_jsonl}")
    samples: list[CaptionSample] = []
    for record in read_jsonl(samples_jsonl):
        image_path = Path(str(record["image_path"]))
        if not image_path.is_absolute():
            image_path = Path(project_root) / image_path
        samples.append(
            CaptionSample(
                sample_id=str(record.get("sample_id") or record.get("image_id")),
                image_id=str(record.get("image_id") or record.get("sample_id")),
                image_path=image_path,
                gt_objects=[str(item) for item in record.get("gt_objects", [])],
                raw=dict(record),
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples
