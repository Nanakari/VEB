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
        report = {"dataset": "pope"}
        for method in ["base", "veb", "halc", "opera"]:
            path = out_root / f"pope_{method}_predictions.jsonl"
            if path.exists():
                report[method] = compute_pope_metrics(list(read_jsonl(path)))
        dump_json(out_root / "metrics_pope.json", report)
    else:
        extractor = build_extractor(config, ROOT)
        report = {"dataset": "coco_chair"}
        for method in ["base", "halc", "opera"]:
            path = out_root / f"coco_{method}_captions.jsonl"
            if path.exists():
                records = []
                for record in read_jsonl(path):
                    caption = str(record.get("caption") or record.get("text") or "")
                    records.append(
                        {**record, "objects": [obj.to_dict() for obj in extractor.extract(caption)]}
                    )
                report[method] = compute_chair_metrics(records)
        veb_path = out_root / "coco_veb_revisions.jsonl"
        if veb_path.exists():
            records = []
            for record in read_jsonl(veb_path):
                revised_caption = str(record.get("revised_caption") or "")
                records.append(
                    {
                        **record,
                        "caption": revised_caption,
                        "objects": [obj.to_dict() for obj in extractor.extract(revised_caption)],
                    }
                )
            report["veb"] = compute_chair_metrics(records)
        dump_json(out_root / "metrics_coco_chair.json", report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["pope", "coco_chair"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
