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


def test_revision_skips_glass_door_compound_fragment() -> None:
    caption = "The shower stall is a walk-in shower with a glass door."
    objects = [_obj("glass", caption)]
    gaps = [{"object_index": 1, "support": 0.0, "cumulative_gap": 0.9, "state": "remove_candidate"}]
    result = revise_caption(caption, objects, gaps)
    assert result.revised_caption == caption
    assert result.actions[0].action == "skip"
    assert result.actions[0].rule == "compound_fragment"


def test_revision_skips_truck_bed_compound_fragment() -> None:
    caption = "Some people are sitting on the truck bed and others are standing."
    objects = [_obj("bed", caption)]
    gaps = [{"object_index": 1, "support": 0.0, "cumulative_gap": 0.9, "state": "remove_candidate"}]
    result = revise_caption(caption, objects, gaps)
    assert result.revised_caption == caption
    assert result.actions[0].action == "skip"
    assert result.actions[0].rule == "compound_fragment"


def test_revision_skips_word_internal_match() -> None:
    caption = "The bookshelf is filled with books."
    start = caption.index("book")
    objects = [{"text": "book", "normalized": "book", "span": [start, start + 4], "object_index": 1}]
    gaps = [{"object_index": 1, "support": 0.0, "cumulative_gap": 0.9, "state": "remove_candidate"}]
    result = revise_caption(caption, objects, gaps)
    assert result.revised_caption == caption
    assert result.actions[0].action == "skip"
    assert result.actions[0].rule == "compound_fragment"


def test_revision_skips_of_glass_anchor() -> None:
    caption = "The shower stall is made of glass and has a silver shower head."
    objects = [_obj("glass", caption)]
    gaps = [{"object_index": 1, "support": 0.0, "cumulative_gap": 0.9, "state": "remove_candidate"}]
    result = revise_caption(caption, objects, gaps)
    assert result.revised_caption == caption
    assert result.actions[0].action == "skip"
    assert result.actions[0].rule == "compound_fragment"


def test_revision_skips_of_table_anchor_before_coordination() -> None:
    caption = "One chair is visible on the left side of the table and another is nearby."
    objects = [_obj("table", caption)]
    gaps = [{"object_index": 1, "support": 0.0, "cumulative_gap": 0.9, "state": "remove_candidate"}]
    result = revise_caption(caption, objects, gaps)
    assert result.revised_caption == caption
    assert result.actions[0].action == "skip"
    assert result.actions[0].rule == "compound_fragment"
