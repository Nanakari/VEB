"""POPE yes/no metrics."""

from __future__ import annotations

from typing import Any, Mapping

from veb_reproduce.utils.answers import normalize_yes_no


def compute_pope_metrics(records: list[Mapping[str, Any]]) -> dict[str, float]:
    total = 0
    correct = 0
    tp = fp = fn = 0
    yes_count = 0
    for record in records:
        label = normalize_yes_no(record.get("label"))
        pred = normalize_yes_no(record.get("answer") or record.get("revised_answer") or record.get("text"))
        if label not in {"yes", "no"} or pred not in {"yes", "no"}:
            continue
        total += 1
        yes_count += int(pred == "yes")
        correct += int(pred == label)
        if pred == "yes" and label == "yes":
            tp += 1
        elif pred == "yes" and label == "no":
            fp += 1
        elif pred == "no" and label == "yes":
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": correct / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_count / total if total else 0.0,
        "num_samples": float(total),
    }
