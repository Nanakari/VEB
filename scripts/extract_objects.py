"""Extract objects from base captions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veb_reproduce.extraction.extractors import build_extractor
from veb_reproduce.utils.config import load_config, output_root
from veb_reproduce.utils.io import read_jsonl, write_jsonl


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    out_root = output_root(config, ROOT)
    extractor = build_extractor(config, ROOT)
    records = []
    for record in read_jsonl(out_root / "coco_base_captions.jsonl"):
        caption = str(record.get("caption") or record.get("text") or "")
        records.append(
            {
                "sample_id": record.get("sample_id"),
                "image_id": record.get("image_id"),
                "caption": caption,
                "objects": [obj.to_dict() for obj in extractor.extract(caption)],
            }
        )
    write_jsonl(out_root / "coco_base_objects.jsonl", records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    main()
