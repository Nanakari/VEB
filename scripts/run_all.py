"""Run the main VEB experiment sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = _parse_args()
    commands = [
        [sys.executable, "scripts/run_pope.py", "--config", args.config, "--method", "base"],
        [sys.executable, "scripts/run_pope.py", "--config", args.config, "--method", "veb"],
        [sys.executable, "scripts/run_caption.py", "--config", args.config, "--method", "base"],
        [sys.executable, "scripts/run_caption.py", "--config", args.config, "--method", "veb"],
        [sys.executable, "scripts/evaluate.py", "--config", args.config, "--dataset", "pope"],
        [sys.executable, "scripts/evaluate.py", "--config", args.config, "--dataset", "coco_chair"],
        [sys.executable, "scripts/export_results.py", "--config", args.config],
    ]
    if args.limit is not None:
        commands[0].extend(["--limit", str(args.limit)])
        commands[2].extend(["--limit", str(args.limit)])
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
