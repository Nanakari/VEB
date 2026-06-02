from __future__ import annotations

from veb_reproduce.extraction.types import ExtractedObject
from veb_reproduce.veb.demand import compute_demands
from veb_reproduce.veb.gap import compute_gap_states


CONFIG = {
    "veb": {
        "lambda_gap": 0.8,
        "tau1": 0.4,
        "tau2": 0.8,
        "tau_s": 0.25,
        "demand": {"aggregation": "mean", "fallback_entropy": 0.5, "fallback_margin": 0.5},
    }
}


def test_demand_uses_mean_token_aggregation() -> None:
    caption = "A laptop is on the table."
    obj = ExtractedObject("laptop", "laptop", (2, 8), 1, "test")
    tokens = [
        {"token_char_span": [2, 5], "position": 1, "entropy": 0.8, "margin": 0.2},
        {"token_char_span": [5, 8], "position": 2, "entropy": 0.6, "margin": 0.4},
    ]
    demand = compute_demands(caption, [obj], tokens, CONFIG)[0]
    assert demand.h_tok == 0.7
    assert demand.m_tok == 0.30000000000000004
    assert demand.l_tok == 0.5
    assert round(demand.demand, 3) == 0.66
    assert demand.token_indices == [1, 2]
    assert demand.alignment_failed is False


def test_demand_falls_back_when_token_alignment_fails() -> None:
    caption = "A laptop is on the table."
    obj = ExtractedObject("laptop", "laptop", (2, 8), 1, "test")
    demand = compute_demands(caption, [obj], [], CONFIG)[0]
    assert demand.h_tok == 0.5
    assert demand.m_tok == 0.5
    assert demand.alignment_failed is True
    assert demand.missing_terms == ["token_alignment"]


def test_gap_states_follow_thresholds() -> None:
    demands = [
        {"object_index": 1, "normalized": "person", "demand": 0.3},
        {"object_index": 2, "normalized": "laptop", "demand": 0.9},
        {"object_index": 3, "normalized": "phone", "demand": 0.9},
    ]
    evidences = [{"score": 0.3}, {"score": 0.0}, {"score": 0.0}]
    states = compute_gap_states(demands, evidences, CONFIG)
    assert states[0].state == "normal"
    assert states[1].state == "remove_candidate"
    assert states[2].state == "remove_candidate"
    assert round(states[2].cumulative_gap, 2) == 1.62
