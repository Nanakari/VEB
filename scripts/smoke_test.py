"""Run the fake-generator/fake-detector smoke pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = "configs/smoke.yaml"
    commands = [
        [sys.executable, "scripts/run_pope.py", "--config", config, "--method", "base"],
        [sys.executable, "scripts/run_pope.py", "--config", config, "--method", "veb"],
        [sys.executable, "scripts/run_caption.py", "--config", config, "--method", "base"],
        [sys.executable, "scripts/run_caption.py", "--config", config, "--method", "veb"],
        [sys.executable, "scripts/evaluate.py", "--config", config, "--dataset", "pope"],
        [sys.executable, "scripts/evaluate.py", "--config", config, "--dataset", "coco_chair"],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    print("Smoke pipeline completed: outputs/smoke")


if __name__ == "__main__":
    main()
