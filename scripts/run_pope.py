"""Run POPE Base or VEB inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veb_reproduce.datasets.pope import load_pope_samples
from veb_reproduce.evidence import build_detector
from veb_reproduce.models import build_generator
from veb_reproduce.pipeline import generate_for_sample, run_pope_veb_record
from veb_reproduce.utils.answers import normalize_yes_no
from veb_reproduce.utils.config import load_run_config, output_root
from veb_reproduce.utils.io import read_jsonl, write_jsonl


def main() -> None:
    args = _parse_args()
    config = load_run_config(args.config, args.method, ROOT)
    out_root = output_root(config, ROOT)
    if args.method in {"base", "halc", "opera"}:
        records = _run_generation(config, method=args.method, limit=args.limit)
        write_jsonl(out_root / f"pope_{args.method}_predictions.jsonl", records)
    elif args.method == "veb":
        base_path = out_root / "pope_base_predictions.jsonl"
        if not base_path.exists():
            raise FileNotFoundError(f"Run POPE base first: {base_path}")
        detector = build_detector(config, ROOT)
        records = []
        for index, record in enumerate(read_jsonl(base_path)):
            if args.limit is not None and index >= args.limit:
                break
            records.append(run_pope_veb_record(record, detector, config))
        write_jsonl(out_root / "pope_veb_predictions.jsonl", records)
    else:
        raise ValueError(f"Unsupported method: {args.method}")


def _run_generation(config: dict, *, method: str, limit: int | None) -> list[dict]:
    generator = build_generator(config)
    dataset_config = config["datasets"]["pope"]
    samples = load_pope_samples(dataset_config, ROOT, limit_per_setting=limit)
    prompt_template = config["prompts"]["pope"]
    max_new_tokens = int(config["generation"].get("pope_max_new_tokens", 10))
    records: list[dict] = []
    for sample in samples:
        prompt = prompt_template.format(question=sample.question)
        result = generate_for_sample(
            generator,
            sample.image_path,
            prompt,
            sample_id=sample.sample_id,
            max_new_tokens=max_new_tokens,
        )
        answer = normalize_yes_no(result["text"]) or str(result["text"]).strip().lower()
        records.append(
            {
                "sample_id": sample.sample_id,
                "image_id": sample.image_id,
                "image_path": str(sample.image_path),
                "dataset": "pope",
                "method": method,
                "setting": sample.setting,
                "question": sample.question,
                "target_object": sample.target_object,
                "label": sample.label,
                "prompt": prompt,
                "text": result["text"],
                "answer": answer,
                "latency_sec": result["latency_sec"],
                "token_scores": result.get("token_scores"),
            }
        )
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--method", choices=["base", "veb", "halc", "opera"], required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
