import math

import numpy as np

from neuroworld import NeuroWorld
from qneuro.evaluation import (
    ActiveStep,
    aggregate_active_trajectories,
    canonicalize_case,
    estimate_positive_likelihoods,
    global_information_order,
    partial_case,
)


def test_partial_and_canonical_cases_preserve_only_requested_evidence() -> None:
    full = NeuroWorld(observation_probability=1.0).generate(1, seed=41)[0]
    canonical = canonicalize_case(full)
    assert np.array_equal(canonical.tokens % NeuroWorld.num_findings, np.arange(40))
    partial = partial_case(full, {7: int(full.evidence[7]), 3: int(full.evidence[3])})
    assert np.flatnonzero(partial.evidence).tolist() == [3, 7]
    assert (partial.tokens % NeuroWorld.num_findings).tolist() == [3, 7]


def test_information_statistics_and_active_aggregation() -> None:
    cases = NeuroWorld().generate(300, seed=42)
    likelihoods = estimate_positive_likelihoods(cases)
    order = global_information_order(cases)
    assert likelihoods.shape == (20, 40)
    assert bool(((likelihoods > 0.0) & (likelihoods < 1.0)).all())
    assert sorted(order) == list(range(40))

    trajectories = [
        (
            1,
            [
                ActiveStep(0, 0, 0.6, 0.4, 1.0),
                ActiveStep(1, 1, 0.9, 0.9, 0.3),
            ],
        ),
        (
            0,
            [
                ActiveStep(0, 0, 0.8, 0.8, 0.5),
                ActiveStep(2, 0, 0.95, 0.95, 0.2),
            ],
        ),
    ]
    summary = aggregate_active_trajectories(trajectories, confidence_threshold=0.85)
    assert summary["curve"][0]["accuracy"] == 0.5
    assert summary["final_accuracy"] == 1.0
    assert summary["resolution_rate"] == 1.0
    assert math.isclose(summary["mean_queries_to_resolution_penalized"], 2.0)
