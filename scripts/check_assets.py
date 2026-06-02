"""Check configured model and dataset asset paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veb_reproduce.utils.config import load_config, resolve_path


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    paths = [
        config.get("generation", {}).get("model_name_or_path"),
        config.get("evidence", {}).get("groundingdino", {}).get("config_path"),
        config.get("evidence", {}).get("groundingdino", {}).get("checkpoint_path"),
        config.get("datasets", {}).get("coco_chair", {}).get("paths", {}).get("image_root"),
        config.get("datasets", {}).get("coco_chair", {}).get("paths", {}).get("instances_file"),
    ]
    missing = []
    for value in paths:
        path = resolve_path(value, ROOT)
        if path is not None and not path.exists():
            missing.append(str(path))
    if missing:
        print("Missing assets:")
        for path in missing:
            print(f"- {path}")
        raise SystemExit(1 if args.strict else 0)
    print("Configured assets found.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
