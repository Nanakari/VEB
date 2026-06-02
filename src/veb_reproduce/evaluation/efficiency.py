"""Runtime and call-count summaries."""

from __future__ import annotations

from typing import Any, Mapping


def summarize_efficiency(records: list[Mapping[str, Any]], *, base_latency: float | None = None) -> dict[str, float]:
    total = len(records)
    latency = sum(float(record.get("latency_sec", 0.0)) for record in records)
    calls = sum(float(record.get("external_queries", 0.0)) for record in records)
    mean_latency = latency / total if total else 0.0
    latency_x = mean_latency / base_latency if base_latency and base_latency > 0 else 1.0
    return {
        "latency_sec_mean": mean_latency,
        "calls_per_image": calls / total if total else 0.0,
        "latency_x": latency_x,
    }
