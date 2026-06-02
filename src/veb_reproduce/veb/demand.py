"""Token-level evidence demand aggregation for object-level VEB decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from veb_reproduce.extraction.types import ExtractedObject
from veb_reproduce.veb.token_alignment import align_object_to_tokens


@dataclass(frozen=True)
class DemandRecord:
    object_index: int
    normalized: str
    h_tok: float
    m_tok: float
    l_tok: float
    demand: float
    token_indices: list[int]
    alignment_failed: bool
    missing_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_demands(
    caption: str,
    objects: list[ExtractedObject | dict[str, Any]],
    token_scores: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> list[DemandRecord]:
    demand_config = config.get("veb", {}).get("demand", {})
    aggregation = str(demand_config.get("aggregation", "mean")).lower()
    if aggregation != "mean":
        raise ValueError("The main VEB implementation fixes demand aggregation to mean")
    fallback_entropy = float(demand_config.get("fallback_entropy", 0.5))
    fallback_margin = float(demand_config.get("fallback_margin", 0.5))

    total_tokens = max(len(token_scores), 1)
    caption_length = max(len(caption), 1)
    records: list[DemandRecord] = []
    for obj in objects:
        tokens = align_object_to_tokens(obj, token_scores)
        object_index = int(_object_value(obj, "object_index", len(records) + 1))
        normalized = str(_object_value(obj, "normalized", ""))
        missing_terms: list[str] = []
        if tokens:
            entropies = [_float_or_default(token.get("entropy"), fallback_entropy) for token in tokens]
            margins = [
                _float_or_default(
                    token.get("margin"),
                    _float_or_default(token.get("top1_prob"), 0.5)
                    - _float_or_default(token.get("top2_prob"), 0.0),
                )
                for token in tokens
            ]
            h_tok = _mean(entropies)
            m_tok = _mean(margins)
            head_position = int(tokens[0].get("position", 1))
            l_tok = _clamp(head_position / total_tokens)
            token_indices = [int(token.get("position", index + 1)) for index, token in enumerate(tokens)]
            alignment_failed = False
        else:
            h_tok = fallback_entropy
            m_tok = fallback_margin
            span = _object_span(obj)
            l_tok = _clamp(span[0] / caption_length)
            token_indices = []
            alignment_failed = True
            missing_terms.append("token_alignment")

        h_tok = _clamp(h_tok)
        m_tok = _clamp(m_tok)
        l_tok = _clamp(l_tok)
        demand = _clamp(0.5 * h_tok + 0.3 * (1.0 - m_tok) + 0.2 * l_tok)
        records.append(
            DemandRecord(
                object_index=object_index,
                normalized=normalized,
                h_tok=h_tok,
                m_tok=m_tok,
                l_tok=l_tok,
                demand=demand,
                token_indices=token_indices,
                alignment_failed=alignment_failed,
                missing_terms=missing_terms,
            )
        )
    return records


def _object_value(obj: ExtractedObject | dict[str, Any], key: str, default: Any) -> Any:
    if isinstance(obj, ExtractedObject):
        return getattr(obj, key, default)
    return obj.get(key, default)


def _object_span(obj: ExtractedObject | dict[str, Any]) -> tuple[int, int]:
    raw = _object_value(obj, "span", (0, 0))
    return int(raw[0]), int(raw[1])


def _float_or_default(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
