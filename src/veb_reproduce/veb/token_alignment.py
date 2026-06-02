"""Token-to-caption and object-to-token alignment utilities."""

from __future__ import annotations

from typing import Any, Callable

from veb_reproduce.extraction.types import ExtractedObject


def spans_from_incremental_decodes(prefix_decodes: list[str], final_text: str) -> list[tuple[int, int]]:
    """Map token prefix decodes to character spans in the stripped final caption."""

    if len(prefix_decodes) < 2:
        return []
    raw_final = prefix_decodes[-1]
    left_trim = len(raw_final) - len(raw_final.lstrip())
    clean_final = raw_final.strip()
    if final_text != clean_final:
        clean_final = final_text

    spans: list[tuple[int, int]] = []
    for index in range(len(prefix_decodes) - 1):
        start = len(prefix_decodes[index]) - left_trim
        end = len(prefix_decodes[index + 1]) - left_trim
        start = max(0, min(start, len(clean_final)))
        end = max(start, min(end, len(clean_final)))
        spans.append((start, end))
    return spans


def token_spans_from_ids(
    token_ids: list[int],
    decode: Callable[[list[int]], str],
    *,
    final_text: str | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    prefixes = [decode(token_ids[:index]) for index in range(0, len(token_ids) + 1)]
    text = final_text if final_text is not None else prefixes[-1].strip()
    return text, spans_from_incremental_decodes(prefixes, text)


def align_object_to_tokens(
    obj: ExtractedObject | dict[str, Any],
    token_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    span = _span_from_object(obj)
    aligned = [
        token
        for token in token_scores
        if _overlaps(span, _token_span(token)) and _token_has_text_overlap(token)
    ]
    return aligned


def align_objects_to_tokens(
    objects: list[ExtractedObject | dict[str, Any]],
    token_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    alignments: list[dict[str, Any]] = []
    for obj in objects:
        tokens = align_object_to_tokens(obj, token_scores)
        alignments.append(
            {
                "object_index": int(_object_value(obj, "object_index", 0)),
                "normalized": str(_object_value(obj, "normalized", "")),
                "span": list(_span_from_object(obj)),
                "token_indices": [int(token.get("position", index + 1)) for index, token in enumerate(tokens)],
                "tokens": tokens,
                "alignment_failed": len(tokens) == 0,
            }
        )
    return alignments


def _span_from_object(obj: ExtractedObject | dict[str, Any]) -> tuple[int, int]:
    raw = _object_value(obj, "span", None) or _object_value(obj, "char_span", None)
    if raw is None:
        raise ValueError(f"Object has no span: {obj}")
    return int(raw[0]), int(raw[1])


def _token_span(token: dict[str, Any]) -> tuple[int, int]:
    raw = token.get("token_char_span") or token.get("char_span")
    if raw is None:
        return (0, 0)
    return int(raw[0]), int(raw[1])


def _token_has_text_overlap(token: dict[str, Any]) -> bool:
    start, end = _token_span(token)
    return end > start


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and left[1] > right[0]


def _object_value(obj: ExtractedObject | dict[str, Any], key: str, default: Any) -> Any:
    if isinstance(obj, ExtractedObject):
        return getattr(obj, key, default)
    return obj.get(key, default)
