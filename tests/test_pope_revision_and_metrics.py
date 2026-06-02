from __future__ import annotations

from veb_reproduce.evidence.fake import FakeEvidenceDetector
from veb_reproduce.evaluation.pope import compute_pope_metrics
from veb_reproduce.revision.pope import revise_pope_answer


def test_pope_only_changes_low_evidence_yes_to_no() -> None:
    config = {"evidence": {"evidence_threshold": 0.25, "fake": {"scores": {"dog": 0.0}}}, "veb": {"tau_pope": 0.25}}
    detector = FakeEvidenceDetector(config)
    result = revise_pope_answer("missing.jpg", "dog", "yes", detector, config)
    assert result.revised_answer == "no"
    assert result.action["action"] == "yes_to_no"

    no_result = revise_pope_answer("missing.jpg", "dog", "no", detector, config)
    assert no_result.revised_answer == "no"
    assert no_result.visual_evidence is None


def test_pope_metrics() -> None:
    records = [
        {"label": "yes", "answer": "yes"},
        {"label": "no", "answer": "yes"},
        {"label": "yes", "answer": "no"},
        {"label": "no", "answer": "no"},
    ]
    metrics = compute_pope_metrics(records)
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["yes_ratio"] == 0.5
