from __future__ import annotations

from pathlib import Path

import yaml

from experiments.run_qn_grand_001 import build_preflight


def test_current_grand_preflight_blocks_without_opening_sealed_data() -> None:
    config = yaml.safe_load(Path("experiments/configs/qn_grand_001.yaml").read_text())
    result = build_preflight(config)
    assert result["status"] == "blocked"
    assert result["sealed_benchmark_opened"] is False
    assert result["all_required_gates_pass"] is False
    assert {
        "complete_preregistered_real_envelope",
        "equal_or_real_favoring_search_budgets",
        "compute_matching_records",
        "full_shiftgauntlet_outcome_grid",
        "full_discovery_protocol",
        "raw_predictions_preserved",
    }.issubset(result["blocking_failures"])
