from __future__ import annotations

from research.analyses.analyze_falsification_phase import _confirmation_world_seed_effects


def test_confirmation_effects_average_severity_before_top_level_summary() -> None:
    effects = [
        {
            "family": "a",
            "world_seed": 1,
            "training_seed": 2,
            "difference": value,
        }
        for value in (-0.01, -0.02, -0.03)
    ]
    observations, worlds = _confirmation_world_seed_effects(effects)
    assert len(observations) == 1
    assert observations[0].value == -0.02
    assert worlds == [-0.02]
