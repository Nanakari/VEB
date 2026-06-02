# VEB

VEB is a reproducible experiment scaffold for Visual Evidence Balance: a
training-free, post-generation calibration method for reducing object hallucinations
in LVLM object-existence answers and image captions.

The implementation follows the main experiment design:

- Base: original LLaVA-1.5-7B with greedy decoding.
- Ours/VEB: object-level calibration with token-level evidence demand estimation.
- Baselines: official HALC and OPERA decoder implementations.
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
git clone https://github.com/BillChan226/HALC.git external/HALC
git clone https://github.com/shikiw/OPERA.git external/OPERA

python scripts/run_pope.py --config configs/default.yaml --method base
python scripts/run_pope.py --config configs/default.yaml --method veb
python scripts/run_pope.py --config configs/default.yaml --method halc
python scripts/run_pope.py --config configs/default.yaml --method opera
python scripts/run_caption.py --config configs/default.yaml --method base
python scripts/run_caption.py --config configs/default.yaml --method veb
python scripts/run_caption.py --config configs/default.yaml --method halc
python scripts/run_caption.py --config configs/default.yaml --method opera
python scripts/evaluate.py --config configs/default.yaml --dataset pope
python scripts/evaluate.py --config configs/default.yaml --dataset coco_chair
```

The LLaVA checkpoint must be the original merged checkpoint, for example
`models/llava-v1.5-7b` or `liuhaotian/llava-v1.5-7b`. The previous
`llava-hf/llava-1.5-7b-hf` conversion is intentionally disabled so all methods
share the same original LLaVA backbone.

To run the main sequence with both official baselines:

```bash
python scripts/run_all.py --config configs/default.yaml --baselines halc opera
```

## Fair Comparison Contract

All methods share the same base config for runtime, model checkpoint, prompts,
datasets, output token budgets, sampling flags, seed, and evidence settings.
Method configs are validated at load time and may only set:

- `generation.backend`
- `generation.return_token_scores`
- method-private decoder blocks such as `generation.halc` and `generation.opera`

This prevents a baseline from silently changing the backbone, prompt, split,
`max_new_tokens`, seed, or sampling environment. HALC/OPERA-specific beam and
penalty parameters stay inside their own decoder blocks because they are part of
the algorithms being compared.

## Method Contract

Caption calibration uses object phrases as the decision unit. Token-level statistics
are only used to estimate object evidence demand:

```text
D(e_i) = 0.5H_tok(e_i) + 0.3(1 - M_tok(e_i)) + 0.2L_tok(e_i)
G_i = lambda_gap * G_{i-1} + max(0, D(e_i) - S(e_i, I))
```

Deletion is phrase-level and conservative. The implementation never deletes a single
token as a correction unit.
