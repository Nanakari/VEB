"""Object vocabulary and lightweight tokenization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from veb_reproduce.utils.config import resolve_path

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
_NON_OBJECT_WORDS = {
    "image",
    "photo",
    "picture",
    "scene",
    "view",
    "background",
    "foreground",
    "day",
    "night",
    "outdoor",
    "indoor",
}


class ObjectVocabulary:
    """Maps object aliases to canonical names."""

    def __init__(self, aliases: Mapping[str, str]) -> None:
        self.aliases = {
            normalize_for_matching(alias): canonical
            for alias, canonical in aliases.items()
            if normalize_for_matching(alias)
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any], project_root: str | Path) -> "ObjectVocabulary":
        extraction = config.get("object_extraction", {})
        path = resolve_path(extraction.get("vocabulary_path"), project_root)
        if path is None or not path.exists():
            return cls(_default_aliases())
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        aliases: dict[str, str] = {}
        for item in loaded.get("objects", []):
            canonical = str(item["name"]).strip().lower()
            aliases[canonical] = canonical
            for alias in item.get("aliases", []):
                aliases[str(alias).strip().lower()] = canonical
        return cls(aliases)

    def normalize(self, text: str) -> str | None:
        normalized = normalize_for_matching(text)
        if not normalized or normalized in _NON_OBJECT_WORDS:
            return None
        if normalized in self.aliases:
            return self.aliases[normalized]
        singular = singularize(normalized)
        if singular in self.aliases:
            return self.aliases[singular]
        return None

    def alias_items_by_length(self) -> list[tuple[str, str]]:
        return sorted(self.aliases.items(), key=lambda item: len(item[0].split()), reverse=True)


def tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def normalize_for_matching(text: str) -> str:
    words = [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
    return " ".join(singularize(word) for word in words)


def singularize(value: str) -> str:
    words = value.split()
    return " ".join(_singularize_word(word) for word in words)


def _singularize_word(word: str) -> str:
    if word in {"bus", "scissors", "skis", "sheep"}:
        return word
    if word.endswith("ies") and len(word) > 3:
        return f"{word[:-3]}y"
    if word.endswith("es") and len(word) > 3 and not word.endswith(("ses", "xes")):
        return word[:-2]
    if word.endswith("s") and len(word) > 3:
        return word[:-1]
    return word


def _default_aliases() -> dict[str, str]:
    names = [
        "person",
        "bicycle",
        "car",
        "dog",
        "cat",
        "chair",
        "dining table",
        "laptop",
        "cell phone",
        "cup",
        "bottle",
    ]
    aliases = {name: name for name in names}
    aliases.update({"man": "person", "woman": "person", "people": "person", "phone": "cell phone"})
    return aliases
