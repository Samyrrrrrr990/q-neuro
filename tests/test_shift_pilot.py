from __future__ import annotations

import yaml

from experiments.run_shift_pilot import _curves_and_effects, smoke_config, variant_specs


def _config() -> dict:
    with open("experiments/configs/shift_pilot.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_pilot_covers_every_family_and_training_profile() -> None:
    variants = variant_specs(_config())
    assert len(variants) == 23
    assert {family for family, _, _ in variants} == set(_config()["shift_families"])
    assert {profile for _, _, profile in variants} == {"base", "spurious", "class_expansion"}


def test_smoke_profile_is_explicitly_distinct() -> None:
    smoke = smoke_config(_config())
    assert smoke["profile"] == "smoke"
    assert "NOT THE FROZEN PILOT" in smoke["description"]
    assert smoke["training"]["seeds"] == [1103]
    assert len(smoke["dataset"]["world_seeds"]) == 2


def test_curve_builder_uses_best_real_per_cell() -> None:
    severities = [0.0, 0.5, 1.0]
    records = []
    values = {
        "complex_operator": [0.8, 0.7, 0.6],
        "exact_real_block_operator": [0.8, 0.6, 0.4],
        "gru": [0.8, 0.65, 0.5],
    }
    for model, outcomes in values.items():
        for severity, top1 in zip(severities, outcomes, strict=True):
            records.append(
                {
                    "train_size": 250,
                    "training_seed": 1,
                    "variant": "noise:default",
                    "world_seed": 11,
                    "model": model,
                    "severity": severity,
                    "metrics": {"top1": top1, "nll": 1.0, "ece": 0.1},
                }
            )
    curves, effects = _curves_and_effects(records, list(values), severities)
    assert len(curves) == 3
    assert effects[0]["best_real_model"] == "gru"
    assert effects[0]["difference"] > 0.0
