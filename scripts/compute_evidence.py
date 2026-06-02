"""Compute VEB evidence by running the caption VEB stage."""

from __future__ import annotations

import subprocess
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = _parse_args()
    subprocess.run(
        [sys.executable, "scripts/run_caption.py", "--config", args.config, "--method", "veb"],
        cwd=ROOT,
        check=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    main()
