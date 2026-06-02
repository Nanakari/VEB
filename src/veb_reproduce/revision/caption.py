"""Object-phrase-level caption revision for VEB."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

_ARTICLE_BEFORE_RE = re.compile(r"(?:^|(?<=\s))(?:a|an|the)\s+$", re.IGNORECASE)
_PREP_BEFORE_RE = re.compile(
    r"\s+(?:with|including|containing|near|beside|next\s+to)\s+$", re.IGNORECASE
)
_COORD_BEFORE_RE = re.compile(r"(?:,\s*(?:and|or)\s+|\s+(?:and|or)\s+)$", re.IGNORECASE)
_COORD_AFTER_RE = re.compile(r"^(?:\s*,\s*|\s+(?:and|or)\s+)", re.IGNORECASE)
_TRAILING_LOCAL_PP_RE = re.compile(
    r"^\s+(?:on|in|at|near|beside|next\s+to)\s+(?:a|an|the)?\s*[A-Za-z][A-Za-z0-9 -]*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RevisionAction:
    object: str
    action: str
    rule: str
    reason: str
    span: tuple[int, int] | None
    replacement: str | None
    score: float | None
    cumulative_gap: float | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.span is not None:
            value["span"] = list(self.span)
        return value


@dataclass(frozen=True)
class CaptionRevisionResult:
    original_caption: str
    revised_caption: str
    actions: list[RevisionAction]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_caption": self.original_caption,
            "revised_caption": self.revised_caption,
            "actions": [action.to_dict() for action in self.actions],
        }


def revise_caption(
    caption: str,
    objects: list[Mapping[str, Any]],
    gap_states: list[Mapping[str, Any]],
    *,
    allowed_rules: list[str] | None = None,
) -> CaptionRevisionResult:
    """Delete only object-level local structures selected by VEB gap states."""

    allowed = set(allowed_rules or ["preposition_phrase", "coordination", "object_np_with_local_pp"])
    states_by_index = {int(state.get("object_index", 0)): state for state in gap_states}
    candidates = [
        (obj, states_by_index.get(int(obj.get("object_index", 0)), {}))
        for obj in objects
        if states_by_index.get(int(obj.get("object_index", 0)), {}).get("state") == "remove_candidate"
    ]
    candidates.sort(key=lambda pair: _object_span(pair[0])[0], reverse=True)

    revised = caption
    actions: list[RevisionAction] = []
    occupied: list[tuple[int, int]] = []
    for obj, state in candidates:
        span = _object_span(obj)
        object_name = str(obj.get("normalized") or obj.get("text") or "")
        if not _valid_span(caption, span):
            actions.append(_action(obj, state, "skip", "invalid_span", "missing_or_invalid_span", None, None))
            continue
        if _overlaps_any(span, occupied):
            actions.append(_action(obj, state, "skip", "overlap", "overlapping_prior_edit", span, None))
            continue
        edit = _choose_edit(caption, span, allowed)
        if edit is None:
            actions.append(
                _action(obj, state, "skip", "unsupported_structure", "no_object_phrase_rule_matched", span, None)
            )
            continue
        start, end, rule = edit
        revised = revised[:start] + revised[end:]
        occupied.append((start, end))
        actions.append(_action(obj, state, "delete", rule, "high_gap_low_visual_support", span, ""))

    actions.sort(key=lambda action: action.span[0] if action.span else 10**9)
    return CaptionRevisionResult(
        original_caption=caption,
        revised_caption=_cleanup_spacing(revised),
        actions=actions,
    )


def _choose_edit(
    caption: str, span: tuple[int, int], allowed: set[str]
) -> tuple[int, int, str] | None:
    article_start = _expand_article_left(caption, span[0])
    end = _extend_trailing_local_pp(caption, span[1])
    if "preposition_phrase" in allowed:
        before = caption[:article_start]
        match = _PREP_BEFORE_RE.search(before)
        if match is not None:
            after_coord = _COORD_AFTER_RE.match(caption[span[1] :])
            if after_coord is not None:
                return (article_start, span[1] + after_coord.end(), "coordination_in_preposition")
            return (match.start(), end, "preposition_phrase")

    if "coordination" in allowed:
        before = caption[:article_start]
        before_match = _COORD_BEFORE_RE.search(before)
        if before_match is not None:
            return (before_match.start(), end, "coordination")
        after_match = _COORD_AFTER_RE.match(caption[span[1] :])
        if after_match is not None:
            return (article_start, span[1] + after_match.end(), "coordination")

    if "object_np_with_local_pp" in allowed and end > span[1]:
        if _has_safe_left_boundary(caption, article_start):
            return (article_start, end, "object_np_with_local_pp")
    return None


def _extend_trailing_local_pp(caption: str, end: int) -> int:
    after = caption[end:]
    match = _TRAILING_LOCAL_PP_RE.match(after)
    if match is None:
        return end
    raw_end = end + match.end()
    stop_candidates = [
        index
        for index in [
            caption.find(",", end),
            caption.find(".", end),
            caption.find(";", end),
            caption.find(" and ", end),
            caption.find(" or ", end),
            caption.find(" with ", end),
        ]
        if index != -1 and index <= raw_end
    ]
    return min(stop_candidates) if stop_candidates else raw_end


def _expand_article_left(text: str, start: int) -> int:
    match = _ARTICLE_BEFORE_RE.search(text[:start])
    return match.start() if match is not None else start


def _has_safe_left_boundary(text: str, start: int) -> bool:
    prefix = text[:start]
    stripped = prefix.rstrip().lower()
    return stripped.endswith(("with", "and", "or", ",")) or bool(_COORD_BEFORE_RE.search(prefix))


def _object_span(obj: Mapping[str, Any]) -> tuple[int, int]:
    span = obj.get("span") or obj.get("char_span")
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return (0, 0)
    return int(span[0]), int(span[1])


def _valid_span(text: str, span: tuple[int, int]) -> bool:
    return 0 <= span[0] < span[1] <= len(text)


def _overlaps_any(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in spans)


def _action(
    obj: Mapping[str, Any],
    state: Mapping[str, Any],
    action: str,
    rule: str,
    reason: str,
    span: tuple[int, int] | None,
    replacement: str | None,
) -> RevisionAction:
    return RevisionAction(
        object=str(obj.get("normalized") or obj.get("text") or ""),
        action=action,
        rule=rule,
        reason=reason,
        span=span,
        replacement=replacement,
        score=float(state["support"]) if "support" in state else None,
        cumulative_gap=float(state["cumulative_gap"]) if "cumulative_gap" in state else None,
    )


def _cleanup_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+(?:with|including|containing)\s*([,.!?;:])", r"\1", text, flags=re.I)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*([.!?])", r"\1", text)
    return text.strip()
