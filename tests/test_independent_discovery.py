from __future__ import annotations

from pathlib import Path

import yaml

from experiments.run_independent_discovery import _effects_and_laws, smoke_config


def _config() -> dict:
    return yaml.safe_load(Path("experiments/configs/independent_discovery.yaml").read_text())


def test_discovery_split_contains_only_preregistered_discovery_families() -> None:
    assert set(_config()["families"]) == {
        "hidden_causal_machine",
        "sequential_detective",
        "analytic_noncommutative",
        "analytic_commutative",
    }
    assert len(_config()["dataset"]["world_seeds"]) == 24
    assert len(_config()["training"]["seeds"]) == 5
    assert _config()["protocol_deviations"]


def test_smoke_profile_is_not_discovery_evidence() -> None:
    smoke = smoke_config(_config())
    assert smoke["profile"] == "smoke"
    assert "NOT DISCOVERY EVIDENCE" in smoke["description"]
    assert len(smoke["families"]) == 1


def test_effect_builder_uses_cellwise_best_real_and_fits_laws() -> None:
    records = []
    models = ["complex_operator", "exact_real_block_operator", "state_space"]
    for family, order in (("low", 0.0), ("high", 0.7)):
        for severity in (0.0, 0.5, 1.0):
            for model, offset in (
                ("complex_operator", 0.02 * order * severity),
                ("exact_real_block_operator", 0.0),
                ("state_space", -0.01),
            ):
                records.append(
                    {
                        "family": family,
                        "train_size": 250,
                        "training_seed": 1,
                        "world_seed": 11,
                        "severity": severity,
                        "model": model,
                        "order_dependence": order,
                        "order_information": order,
                        "metrics": {"top1": 0.5 + offset},
                    }
                )
    # Duplicate across a second world to provide standard deviations and enough law cells.
    records.extend([{**record, "world_seed": 13} for record in records])
    # Add a third order level to meet the eight-cell law minimum.
    template = [record for record in records if record["family"] == "high"]
    records.extend(
        [
            {
                **record,
                "family": "middle",
                "order_dependence": 0.35,
                "order_information": 0.35,
            }
            for record in template
        ]
    )
    effects, candidates, cells = _effects_and_laws(records, models)
    assert effects
    assert set(candidates) == {
        "linear",
        "logarithmic",
        "saturating",
        "threshold",
        "interaction",
        "quadratic",
    }
    assert len(cells) == 9
