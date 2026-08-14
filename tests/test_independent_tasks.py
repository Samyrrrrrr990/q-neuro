from __future__ import annotations

import numpy as np
import pytest

from independent_tasks import INDEPENDENT_TASK_FAMILIES, build_independent_task
from neuroworld import NeuroWorld


@pytest.mark.parametrize("family", INDEPENDENT_TASK_FAMILIES)
def test_independent_generators_are_deterministic_and_valid(family: str) -> None:
    task = build_independent_task(family)
    first = task.generate(80, 7103, split="test", shift_strength=0.5)
    second = task.generate(80, 7103, split="test", shift_strength=0.5)
    assert first.metadata == second.metadata
    assert first.metadata["synthetic_nonclinical"] is True
    assert first.metadata["generator_independent_of_neuroworld_rules"] is True
    assert len(first.cases) == 80
    for left, right in zip(first.cases, second.cases, strict=True):
        assert 0 <= left.label < NeuroWorld.num_diagnoses
        assert len(left.tokens) >= 4
        assert np.array_equal(left.tokens, right.tokens)
        assert np.array_equal(left.evidence, right.evidence)


@pytest.mark.parametrize("family", INDEPENDENT_TASK_FAMILIES)
def test_counterfactual_semantics_match_declared_order_causality(family: str) -> None:
    task = build_independent_task(family)
    for pair in task.counterfactual_pairs(20, seed=7207):
        assert sorted(pair.first.tokens.tolist()) == sorted(pair.second.tokens.tolist())
        assert np.array_equal(pair.first.evidence, pair.second.evidence)
        if pair.causal_factor == "causal_order":
            assert pair.first.label != pair.second.label
        else:
            assert pair.first.label == pair.second.label


def test_analytic_controls_span_commuting_and_noncommuting_endpoints() -> None:
    commuting = build_independent_task("analytic_commutative").generate(400, 7309)
    noncommuting = build_independent_task("analytic_noncommutative").generate(400, 7309)
    assert commuting.metadata["analytic_normalized_commutator"] == 0.0
    assert noncommuting.metadata["analytic_normalized_commutator"] > 0.4
    assert commuting.metadata["empirical_observed_order_target_mutual_information"] < 0.05
    assert noncommuting.metadata["empirical_observed_order_target_mutual_information"] > 0.5


def test_observed_order_information_tracks_controlled_dependence() -> None:
    values = []
    for level in (0.0, 0.5, 1.0):
        dataset = build_independent_task(
            "analytic_noncommutative", order_dependence=level
        ).generate(5000, 7351)
        values.append(dataset.metadata["empirical_observed_order_target_mutual_information"])
    assert values[0] < 0.01
    assert values[0] < values[1] < values[2]
    assert values[2] > 0.68


def test_shift_changes_surface_distribution_without_changing_class_space() -> None:
    task = build_independent_task("hidden_causal_machine")
    source = task.generate(200, 7411, split="train", shift_strength=0.0)
    shifted = task.generate(200, 7411, split="test", shift_strength=1.0)
    assert source.metadata["num_task_classes"] == shifted.metadata["num_task_classes"]
    assert np.mean([len(case.tokens) for case in shifted.cases]) < np.mean(
        [len(case.tokens) for case in source.cases]
    )


def test_invalid_commutative_override_fails() -> None:
    with pytest.raises(ValueError, match="commutative"):
        build_independent_task("bayesian_urn", order_dependence=0.2)
