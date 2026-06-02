from __future__ import annotations

from veb_reproduce.evaluation.chair import compute_chair_metrics
from veb_reproduce.evidence.cache import CachedEvidenceDetector
from veb_reproduce.evidence.fake import FakeEvidenceDetector


def test_chair_metrics() -> None:
    records = [
        {
            "caption": "A person with a laptop.",
            "gt_objects": ["person"],
            "objects": [
                {"normalized": "person"},
                {"normalized": "laptop"},
            ],
        },
        {
            "caption": "A cup.",
            "gt_objects": ["cup"],
            "objects": [{"normalized": "cup"}],
        },
    ]
    metrics = compute_chair_metrics(records)
    assert metrics["chairs"] == 0.5
    assert round(metrics["chairi"], 3) == 0.333


def test_evidence_cache_avoids_duplicate_calls() -> None:
    detector = CachedEvidenceDetector(
        FakeEvidenceDetector({"evidence": {"fake": {"scores": {"laptop": 0.0}}}})
    )
    detector.verify("image.jpg", "laptop")
    detector.verify("image.jpg", "laptop")
    assert detector.calls == 1
