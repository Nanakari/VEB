"""Reusable pipeline steps for scripts and smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from veb_reproduce.extraction.extractors import build_extractor
from veb_reproduce.models import build_generator
from veb_reproduce.revision.caption import revise_caption
from veb_reproduce.revision.pope import revise_pope_answer
from veb_reproduce.veb.demand import compute_demands
from veb_reproduce.veb.gap import compute_gap_states
from veb_reproduce.veb.token_alignment import align_objects_to_tokens


def generate_for_sample(
    generator: Any,
    image_path: str | Path,
    prompt: str,
    *,
    sample_id: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    result = generator.generate(
        image_path,
        prompt,
        sample_id=sample_id,
        max_new_tokens=max_new_tokens,
    )
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


def run_caption_veb_record(
    record: Mapping[str, Any],
    detector: Any,
    config: Mapping[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    caption = str(record.get("caption") or record.get("text") or "")
    token_scores = list(record.get("token_scores") or [])
    extractor = build_extractor(config, project_root)
    objects = [item.to_dict() for item in extractor.extract(caption)]
    token_alignment = align_objects_to_tokens(objects, token_scores)
    demands = [item.to_dict() for item in compute_demands(caption, objects, token_scores, config)]
    evidences = []
    calls_before = int(getattr(detector, "calls", 0))
    for obj in objects:
        evidence = detector.verify(record["image_path"], obj["normalized"])
        evidences.append(evidence.to_dict() if hasattr(evidence, "to_dict") else dict(evidence))
    gap_states = [item.to_dict() for item in compute_gap_states(demands, evidences, config)]
    revision_config = config.get("veb", {}).get("revision", {})
    revision = revise_caption(
        caption,
        objects,
        gap_states,
        allowed_rules=list(revision_config.get("allowed_rules", [])) or None,
    )
    actions = [action.to_dict() for action in revision.actions]
    delete_count = sum(1 for action in actions if action.get("action") == "delete")
    calls_after = int(getattr(detector, "calls", calls_before + len(evidences)))
    external_queries = calls_after - calls_before
    return {
        "sample_id": record.get("sample_id"),
        "image_id": record.get("image_id"),
        "image_path": record.get("image_path"),
        "dataset": record.get("dataset", "coco_chair"),
        "method": "veb",
        "caption": caption,
        "revised_caption": revision.revised_caption,
        "gt_objects": list(record.get("gt_objects", [])),
        "objects": objects,
        "token_alignment": token_alignment,
        "demands": demands,
        "visual_evidence": evidences,
        "gap_states": gap_states,
        "revision_actions": actions,
        "removal_rate": delete_count / len(objects) if objects else 0.0,
        "external_queries": external_queries,
        "latency_sec": float(record.get("latency_sec", 0.0))
        + sum(float(item.get("latency_sec", 0.0)) for item in evidences),
    }


def run_pope_veb_record(
    record: Mapping[str, Any],
    detector: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    result = revise_pope_answer(
        record["image_path"],
        record.get("target_object"),
        str(record.get("answer") or record.get("text") or ""),
        detector,
        config,
    )
    result_dict = result.to_dict()
    return {
        **dict(record),
        "method": "veb",
        "original_answer": result.original_answer,
        "revised_answer": result.revised_answer,
        "answer": result.revised_answer,
        "action": result_dict["action"],
        "visual_evidence": result.visual_evidence,
        "external_queries": 1 if result.visual_evidence is not None else 0,
    }
