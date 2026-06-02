"""Answer normalization for yes/no tasks."""

from __future__ import annotations

import re
from typing import Any


def normalize_yes_no(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    text = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", text)
    if not text:
        return None
    first = text.split()[0]
    if first in {"yes", "y", "true", "1"}:
        return "yes"
    if first in {"no", "n", "false", "0"}:
        return "no"
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return None
