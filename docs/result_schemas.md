# VEB Result Schemas

All experiment outputs are JSON Lines unless noted otherwise.

## Caption VEB Revision

Required fields:

- `sample_id`, `image_id`, `image_path`
- `caption`: Base caption
- `revised_caption`: VEB output
- `objects`: extracted object phrases with normalized names and character spans
- `token_alignment`: object-to-token overlap records
- `demands`: object-level `D(e_i)` values aggregated from token statistics
- `visual_evidence`: GroundingDINO or fake detector scores
- `gap_states`: cumulative evidence gap states
- `revision_actions`: object-phrase-level edit decisions

## POPE VEB Prediction

Required fields:

- `sample_id`, `image_id`, `question`, `target_object`
- `original_answer`, `revised_answer`, `answer`
- `action`: `keep` or `yes_to_no`
- `visual_evidence`: present only when Base answered yes and a detector call was needed

## Metrics

POPE reports `accuracy`, `precision`, `recall`, `f1`, and `yes_ratio`.
COCO Caption reports `chairs`, `chairi`, `caption_length`, `object_count`, and
`removal_rate`.
