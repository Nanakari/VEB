"""Evaluate POPE or COCO-CHAIR outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veb_reproduce.evaluation.chair import compute_chair_metrics
from veb_reproduce.evaluation.pope import compute_pope_metrics
from veb_reproduce.extraction.extractors import build_extractor
from veb_reproduce.utils.config import load_config, output_root
from veb_reproduce.utils.io import dump_json, read_jsonl


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    out_root = output_root(config, ROOT)
    if args.dataset == "pope":
        base = list(read_jsonl(out_root / "pope_base_predictions.jsonl"))
        veb = list(read_jsonl(out_root / "pope_veb_predictions.jsonl"))
        report = {
            "dataset": "pope",
            "base": compute_pope_metrics(base),
            "veb": compute_pope_metrics(veb),
        }
        dump_json(out_root / "metrics_pope.json", report)
    else:
        extractor = build_extractor(config, ROOT)
        base = []
        for record in read_jsonl(out_root / "coco_base_captions.jsonl"):
            caption = str(record.get("caption") or record.get("text") or "")
            base.append({**record, "objects": [obj.to_dict() for obj in extractor.extract(caption)]})
        veb = []
        for record in read_jsonl(out_root / "coco_veb_revisions.jsonl"):
            revised_caption = str(record.get("revised_caption") or "")
            veb.append(
                {
                    **record,
                    "caption": revised_caption,
                    "objects": [obj.to_dict() for obj in extractor.extract(revised_caption)],
                }
            )
        report = {
            "dataset": "coco_chair",
            "base": compute_chair_metrics(base),
            "veb": compute_chair_metrics(veb),
        }
        dump_json(out_root / "metrics_coco_chair.json", report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["pope", "coco_chair"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
