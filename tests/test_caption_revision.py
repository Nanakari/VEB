from __future__ import annotations

from veb_reproduce.revision.caption import revise_caption


def _obj(text: str, caption: str, index: int = 1) -> dict:
    start = caption.index(text)
    return {
        "text": text,
        "normalized": text,
        "span": [start, start + len(text)],
        "object_index": index,
    }


def test_revision_deletes_object_phrase_inside_preposition_not_single_token() -> None:
    caption = "A man is sitting at a table with a laptop and a cup."
    objects = [_obj("laptop", caption)]
    gaps = [
        {
            "object_index": 1,
            "support": 0.0,
            "cumulative_gap": 0.9,
            "state": "remove_candidate",
        }
    ]
    result = revise_caption(caption, objects, gaps)
    assert result.revised_caption == "A man is sitting at a table with a cup."
    assert result.actions[0].action == "delete"
    assert result.actions[0].rule == "coordination_in_preposition"


def test_revision_deletes_next_to_phrase() -> None:
    caption = "A man is standing next to a bicycle."
    objects = [_obj("bicycle", caption)]
    gaps = [{"object_index": 1, "support": 0.0, "cumulative_gap": 0.9, "state": "remove_candidate"}]
    result = revise_caption(caption, objects, gaps)
    assert result.revised_caption == "A man is standing."


def test_revision_skips_unsupported_main_subject_np() -> None:
    caption = "A laptop is open on the desk."
    objects = [_obj("laptop", caption)]
    gaps = [{"object_index": 1, "support": 0.0, "cumulative_gap": 0.9, "state": "remove_candidate"}]
    result = revise_caption(caption, objects, gaps)
    assert result.revised_caption == caption
    assert result.actions[0].action == "skip"
