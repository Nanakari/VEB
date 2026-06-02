from __future__ import annotations

from veb_reproduce.extraction.types import ExtractedObject
from veb_reproduce.veb.token_alignment import (
    align_object_to_tokens,
    spans_from_incremental_decodes,
)


def test_incremental_decode_spans_map_to_final_caption() -> None:
    prefixes = ["", "A", "A man", "A man sits"]
    spans = spans_from_incremental_decodes(prefixes, "A man sits")
    assert spans == [(0, 1), (1, 5), (5, 10)]


def test_object_span_aligns_to_overlapping_tokens() -> None:
    obj = ExtractedObject("laptop", "laptop", (18, 24), 1, "test")
    tokens = [
        {"token_char_span": [0, 1], "position": 1, "token_text": "A"},
        {"token_char_span": [18, 21], "position": 2, "token_text": "lap"},
        {"token_char_span": [21, 24], "position": 3, "token_text": "top"},
    ]
    aligned = align_object_to_tokens(obj, tokens)
    assert [token["position"] for token in aligned] == [2, 3]
