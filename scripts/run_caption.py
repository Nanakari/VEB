"""Run MSCOCO Caption Base or VEB inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veb_reproduce.datasets.coco import load_coco_caption_samples
from veb_reproduce.evidence import build_detector
from veb_reproduce.models import build_generator
from veb_reproduce.pipeline import generate_for_sample, run_caption_veb_record
from veb_reproduce.utils.config import load_config, output_root
from veb_reproduce.utils.io import read_jsonl, write_jsonl


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    out_root = output_root(config, ROOT)
    if args.method == "base":
        records = _run_base(config, limit=args.limit)
        write_jsonl(out_root / "coco_base_captions.jsonl", records)
    else:
        base_path = out_root / "coco_base_captions.jsonl"
        if not base_path.exists():
            raise FileNotFoundError(f"Run Caption base first: {base_path}")
        detector = build_detector(config, ROOT)
        revision_records = []
        object_records = []
        demand_records = []
        evidence_records = []
        gap_records = []
        for index, record in enumerate(read_jsonl(base_path)):
            if args.limit is not None and index >= args.limit:
                break
            revised = run_caption_veb_record(record, detector, config, ROOT)
            revision_records.append(revised)
            object_records.append(
                {
                    "sample_id": revised["sample_id"],
                    "image_id": revised["image_id"],
                    "caption": revised["caption"],
                    "objects": revised["objects"],
                }
            )
            demand_records.append(
                {
                    "sample_id": revised["sample_id"],
                    "image_id": revised["image_id"],
                    "demands": revised["demands"],
                    "token_alignment": revised["token_alignment"],
                }
            )
            evidence_records.append(
                {
                    "sample_id": revised["sample_id"],
                    "image_id": revised["image_id"],
                    "visual_evidence": revised["visual_evidence"],
                    "external_queries": revised["external_queries"],
                }
            )
            gap_records.append(
                {
                    "sample_id": revised["sample_id"],
                    "image_id": revised["image_id"],
                    "gap_states": revised["gap_states"],
                }
            )
        write_jsonl(out_root / "coco_base_objects.jsonl", object_records)
        write_jsonl(out_root / "coco_veb_demands.jsonl", demand_records)
        write_jsonl(out_root / "coco_veb_evidence.jsonl", evidence_records)
        write_jsonl(out_root / "coco_veb_gap_states.jsonl", gap_records)
        write_jsonl(out_root / "coco_veb_revisions.jsonl", revision_records)


def _run_base(config: dict, *, limit: int | None) -> list[dict]:
    generator = build_generator(config)
    samples = load_coco_caption_samples(config["datasets"]["coco_chair"], ROOT, limit=limit)
    prompt = config["prompts"]["caption"]
    max_new_tokens = int(config["generation"].get("caption_max_new_tokens", 64))
    records: list[dict] = []
    for sample in samples:
        result = generate_for_sample(
            generator,
            sample.image_path,
            prompt,
            sample_id=sample.sample_id,
            max_new_tokens=max_new_tokens,
        )
        records.append(
            {
                "sample_id": sample.sample_id,
                "image_id": sample.image_id,
                "image_path": str(sample.image_path),
                "dataset": "coco_chair",
                "method": "base",
                "prompt": prompt,
                "caption": result["text"],
                "text": result["text"],
                "token_scores": result.get("token_scores"),
                "latency_sec": result["latency_sec"],
                "gt_objects": sample.gt_objects,
            }
        )
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", choices=["base", "veb"], required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
