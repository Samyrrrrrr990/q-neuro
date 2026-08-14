from __future__ import annotations

import math

import pytest

from research.statistics import (
    HierarchicalObservation,
    benjamini_hochberg_adjust,
    hierarchical_bootstrap,
    holm_adjust,
    paired_sign_flip_pvalue,
    paired_summary,
    robustness_slope,
    select_world_count,
    trapezoidal_robustness_auc,
)


def test_robustness_curve_metrics_have_known_linear_answer() -> None:
    severities = [0.0, 0.5, 1.0]
    values = [1.0, 0.8, 0.6]
    assert math.isclose(trapezoidal_robustness_auc(severities, values), 0.8)
    assert math.isclose(robustness_slope(severities, values), -0.4)


def test_multiplicity_adjustments_preserve_original_order() -> None:
    p_values = [0.01, 0.04, 0.03]
    assert holm_adjust(p_values) == pytest.approx([0.03, 0.06, 0.06])
    assert benjamini_hochberg_adjust(p_values) == pytest.approx([0.03, 0.04, 0.04])


def test_paired_summary_and_exact_sign_flip_use_top_level_differences() -> None:
    differences = [0.1, 0.2, 0.3]
    summary = paired_summary(differences)
    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(0.2)
    assert summary["probability_of_superiority"] == 1.0
    assert paired_sign_flip_pvalue(differences) == 0.25


def test_hierarchical_bootstrap_equal_weights_families() -> None:
    observations = [
        HierarchicalObservation(family, f"world-{world}", seed, value)
        for family, value in (("a", 0.1), ("b", 0.3))
        for world in range(2)
        for seed in (1, 2)
    ]
    result = hierarchical_bootstrap(observations, resamples=2_000, seed=17)
    assert result["estimate"] == pytest.approx(0.2)
    assert result["ci_low"] == pytest.approx(0.1)
    assert result["ci_high"] == pytest.approx(0.3)
    assert result["worlds"] == 4


def test_power_selection_uses_smallest_candidate_reaching_target() -> None:
    result = select_world_count(
        standard_deviation=0.02,
        minimum_effect=0.02,
        candidates=[32, 40, 48],
        target_power=0.80,
        simulations=5_000,
        seed=19,
    )
    assert result["selected_worlds"] == 32
    assert result["target_reached"] is True


def test_statistics_reject_nonfinite_or_misaligned_inputs() -> None:
    with pytest.raises(ValueError):
        trapezoidal_robustness_auc([0.0, 1.0], [0.5])
    with pytest.raises(ValueError):
        holm_adjust([0.1, float("nan")])
