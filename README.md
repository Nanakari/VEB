# VEB

VEB is a reproducible experiment scaffold for Visual Evidence Balance: a
training-free, post-generation calibration method for reducing object hallucinations
in LVLM object-existence answers and image captions.

The implementation follows the main experiment design:

- Base: LLaVA-1.5-7B with greedy decoding.
- Ours/VEB: object-level calibration with token-level evidence demand estimation.
- Visual support: GroundingDINO maximum detection confidence.
- Tasks: POPE yes/no QA and MSCOCO Caption/CHAIR.

## Quick Start

Smoke tests do not load LLaVA or GroundingDINO:

```bash
python -m pip install -e .[dev]
pytest
python scripts/smoke_test.py
```

Real runs require local model and dataset assets configured in `configs/default.yaml`:

```bash
python scripts/run_pope.py --config configs/default.yaml --method base
python scripts/run_pope.py --config configs/default.yaml --method veb
python scripts/run_caption.py --config configs/default.yaml --method base
python scripts/run_caption.py --config configs/default.yaml --method veb
python scripts/evaluate.py --config configs/default.yaml --dataset pope
python scripts/evaluate.py --config configs/default.yaml --dataset coco_chair
```

## Method Contract

Caption calibration uses object phrases as the decision unit. Token-level statistics
are only used to estimate object evidence demand:

```text
D(e_i) = 0.5H_tok(e_i) + 0.3(1 - M_tok(e_i)) + 0.2L_tok(e_i)
G_i = lambda_gap * G_{i-1} + max(0, D(e_i) - S(e_i, I))
```

Deletion is phrase-level and conservative. The implementation never deletes a single
token as a correction unit.
