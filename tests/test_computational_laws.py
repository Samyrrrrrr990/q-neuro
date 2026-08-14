from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from research.computational_laws import (
    analytic_operator_pair,
    counterfactual_order_divergence,
    discrete_mutual_information,
    evaluate_frozen_law,
    fit_candidate_laws,
    freeze_best_candidate,
    frozen_law_from_dict,
    normalized_commutator,
    order_sensitivity_index,
    state_conditioned_commutator,
    trajectory_geometry,
)


def test_commutator_measures_known_commuting_and_noncommuting_pairs() -> None:
    diagonal_a = np.diag([1.0, 2.0])
    diagonal_b = np.diag([3.0, 4.0])
    assert normalized_commutator(diagonal_a, diagonal_b) == 0.0
    rotation, scale = analytic_operator_pair(1.0)
    assert normalized_commutator(rotation, scale) > 0.4
    states = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert state_conditioned_commutator(rotation, scale, states) > 0.0


def test_analytic_pair_varies_order_dependence_monotonically() -> None:
    values = [
        normalized_commutator(*analytic_operator_pair(strength))
        for strength in np.linspace(0.0, 1.0, 11)
    ]
    assert math.isclose(values[0], 0.0, abs_tol=1e-12)
    assert all(left <= right + 1e-12 for left, right in pairwise(values))


def test_order_probability_measures_have_known_limits() -> None:
    first = np.asarray([[1.0, 0.0], [0.5, 0.5]])
    same = first.copy()
    reverse = np.asarray([[0.0, 1.0], [0.5, 0.5]])
    assert order_sensitivity_index(first, same) == 0.0
    assert math.isclose(order_sensitivity_index(first, reverse), 0.5)
    assert counterfactual_order_divergence(first, reverse) > 0.0


def test_discrete_mutual_information_detects_order_target_dependence() -> None:
    order = [0, 0, 1, 1]
    assert math.isclose(discrete_mutual_information(order, order), math.log(2.0))
    assert math.isclose(discrete_mutual_information(order, [0, 1, 0, 1]), 0.0, abs_tol=1e-12)


def test_trajectory_geometry_distinguishes_straight_and_turning_paths() -> None:
    straight = np.asarray([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    turning = np.asarray([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
    assert math.isclose(trajectory_geometry(straight)["trajectory_curvature"], 0.0)
    assert trajectory_geometry(turning)["trajectory_curvature"] > 1.5
    assert math.isclose(trajectory_geometry(straight)["displacement_to_length"], 1.0)


def test_discovery_fit_and_untouched_evaluation_recover_interaction_law() -> None:
    order, shift = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    order_values = order.ravel()
    shift_values = shift.ravel()
    advantage = 0.01 + 0.18 * order_values * shift_values
    candidates = fit_candidate_laws(order_values, shift_values, advantage)
    frozen = freeze_best_candidate(candidates)
    assert frozen.family == "interaction"
    confirmation_order = np.asarray([0.15, 0.35, 0.65, 0.85])
    confirmation_shift = np.asarray([0.8, 0.6, 0.4, 0.2])
    confirmation_advantage = 0.01 + 0.18 * confirmation_order * confirmation_shift
    metrics = evaluate_frozen_law(
        frozen, confirmation_order, confirmation_shift, confirmation_advantage
    )
    assert metrics["r2"] > 0.999
    assert metrics["mean_absolute_error"] < 1e-10
    assert metrics["effect_sign_accuracy"] == 1.0


def test_frozen_law_round_trip_and_validation() -> None:
    value = {
        "family": "linear",
        "coefficients": [0.1, -0.2, 0.3],
        "hyperparameter": None,
        "discovery_r2": 0.6,
        "discovery_mae": 0.01,
        "discovery_n": 12,
    }
    parsed = frozen_law_from_dict(value)
    assert parsed.family == value["family"]
    assert parsed.coefficients == tuple(value["coefficients"])
    assert parsed.to_dict()["discovery_n"] == value["discovery_n"]
