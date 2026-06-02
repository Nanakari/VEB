from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from veb_reproduce.utils.io import read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def test_fake_smoke_pipeline_runs() -> None:
    subprocess.run([sys.executable, "scripts/smoke_test.py"], cwd=ROOT, check=True)
    pope = list(read_jsonl(ROOT / "outputs" / "smoke" / "pope_veb_predictions.jsonl"))
    caption = list(read_jsonl(ROOT / "outputs" / "smoke" / "coco_veb_revisions.jsonl"))
    assert pope[0]["answer"] == "no"
    assert caption[0]["revised_caption"] == "A man is sitting at a table with a cup."
