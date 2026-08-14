"""Discovery-engine Pareto and surprise tests."""

from qneuro.discovery import detect_surprises, pareto_frontier


def test_pareto_frontier_removes_strictly_dominated_candidate() -> None:
    records = [
        {"candidate_id": "a", "accuracy": 0.8, "seconds": 2.0},
        {"candidate_id": "b", "accuracy": 0.7, "seconds": 3.0},
        {"candidate_id": "c", "accuracy": 0.9, "seconds": 4.0},
    ]
    frontier = pareto_frontier(records, maximize=("accuracy",), minimize=("seconds",))
    assert [value["candidate_id"] for value in frontier] == ["a", "c"]


def test_surprise_detector_flags_hidden_order_failure() -> None:
    record = {
        "candidate_id": "attractor",
        "in_domain_top1": 0.8,
        "shifted_top1": 0.4,
        "ambiguity_nll": 1.5,
        "counterfactual_pair_accuracy": 0.0,
    }
    flags = detect_surprises([record])
    assert {value["type"] for value in flags} == {"order_blind_success"}
