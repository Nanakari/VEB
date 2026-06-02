"""Simplified CHAIR-style object hallucination metrics."""

from __future__ import annotations

from typing import Any, Mapping


def compute_chair_metrics(records: list[Mapping[str, Any]]) -> dict[str, float]:
    sentence_total = 0
    hallucinated_sentence_total = 0
    object_total = 0
    hallucinated_object_total = 0
    caption_lengths: list[int] = []
    object_counts: list[int] = []
    removal_rates: list[float] = []

    for record in records:
        gt_objects = {str(item).lower() for item in record.get("gt_objects", [])}
        objects = record.get("objects", [])
        if objects is None:
            objects = []
        normalized_objects = [str(item.get("normalized", "")).lower() for item in objects]
        hallucinated = [
            obj for obj in normalized_objects if obj and gt_objects and obj not in gt_objects
        ]
        sentence_total += 1
        object_total += len(normalized_objects)
        hallucinated_object_total += len(hallucinated)
        hallucinated_sentence_total += int(len(hallucinated) > 0)
        caption = str(record.get("caption") or record.get("revised_caption") or "")
        caption_lengths.append(len(caption.split()))
        object_counts.append(len(normalized_objects))
        removal_rates.append(float(record.get("removal_rate", 0.0)))

    return {
        "chairs": hallucinated_sentence_total / sentence_total if sentence_total else 0.0,
        "chairi": hallucinated_object_total / object_total if object_total else 0.0,
        "caption_length": _mean(caption_lengths),
        "object_count": _mean(object_counts),
        "removal_rate": _mean(removal_rates),
        "num_samples": float(sentence_total),
    }


def _mean(values: list[int] | list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
