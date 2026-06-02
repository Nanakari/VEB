"""Export compact Markdown result tables."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veb_reproduce.utils.config import load_config, output_root
from veb_reproduce.utils.io import load_json


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    out_root = output_root(config, ROOT)
    sections: list[str] = []
    pope_path = out_root / "metrics_pope.json"
    if pope_path.exists():
        report = load_json(pope_path)
        sections.append(_table("POPE", report, ["accuracy", "precision", "recall", "f1", "yes_ratio"]))
    chair_path = out_root / "metrics_coco_chair.json"
    if chair_path.exists():
        report = load_json(chair_path)
        sections.append(_table("COCO CHAIR", report, ["chairs", "chairi", "caption_length", "object_count", "removal_rate"]))
    (out_root / "tables_main.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


def _table(title: str, report: dict, keys: list[str]) -> str:
    header = "| Method | " + " | ".join(keys) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(keys)) + " |"
    rows = [f"## {title}", "", header, sep]
    for method in ["base", "veb"]:
        metrics = report.get(method, {})
        values = [f"{float(metrics.get(key, 0.0)):.4f}" for key in keys]
        rows.append("| " + method + " | " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
