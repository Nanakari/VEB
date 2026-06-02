"""Cumulative visual evidence gap modeling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GapState:
    object_index: int
    normalized: str
    demand: float
    support: float
    gap_increment: float
    cumulative_gap: float
    state: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_gap_states(
    demands: list[Mapping[str, Any] | Any],
    evidences: list[Mapping[str, Any] | Any],
    config: Mapping[str, Any],
) -> list[GapState]:
    veb_config = config.get("veb", {})
    lambda_gap = float(veb_config.get("lambda_gap", 0.8))
    tau1 = float(veb_config.get("tau1", 0.4))
    tau2 = float(veb_config.get("tau2", 0.8))
    tau_s = float(veb_config.get("tau_s", 0.25))

    states: list[GapState] = []
    gap = 0.0
    for index, demand_record in enumerate(demands):
        evidence = evidences[index] if index < len(evidences) else {}
        demand = _value(demand_record, "demand", 0.0)
        support = _value(evidence, "score", 0.0)
        gap_increment = max(0.0, demand - support)
        gap = lambda_gap * gap + gap_increment
        if gap < tau1:
            state = "normal"
        elif gap < tau2:
            state = "audit"
        elif support < tau_s:
            state = "remove_candidate"
        else:
            state = "audit"
        states.append(
            GapState(
                object_index=int(_value(demand_record, "object_index", index + 1)),
                normalized=str(_value(demand_record, "normalized", "")),
                demand=demand,
                support=support,
                gap_increment=gap_increment,
                cumulative_gap=gap,
                state=state,
            )
        )
    return states


def _value(record: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)
