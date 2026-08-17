from __future__ import annotations

from pathlib import Path

import yaml

from experiments.run_independent_confirmation import smoke_config


def _config() -> dict:
    return yaml.safe_load(Path("experiments/configs/independent_confirmation.yaml").read_text())


def test_confirmation_families_and_seeds_are_disjoint_from_discovery() -> None:
    confirmation = _config()
    discovery = yaml.safe_load(Path("experiments/configs/independent_discovery.yaml").read_text())
    grand = yaml.safe_load(Path("experiments/configs/grand_falsification.yaml").read_text())
    assert set(confirmation["families"]).isdisjoint(discovery["families"])
    assert set(confirmation["training"]["seeds"]).isdisjoint(discovery["training"]["seeds"])
    assert set(confirmation["training"]["seeds"]).isdisjoint(
        grand["replication"]["confirmation"]["training_seeds"]
    )
    assert len(confirmation["dataset"]["world_seeds"]) == 32
    assert confirmation["split"] == "confirmation"


def test_confirmation_smoke_is_execution_only() -> None:
    smoke = smoke_config(_config())
    assert smoke["profile"] == "smoke_confirmation"
    assert "NOT CONFIRMATION EVIDENCE" in smoke["description"]
    assert len(smoke["families"]) == 2
