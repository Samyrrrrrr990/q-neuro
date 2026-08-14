from __future__ import annotations

import numpy as np
import pytest

from neuroworld import NeuroWorld
from neuroworld.shifts import ShiftGauntlet, ShiftSpec


@pytest.fixture(scope="module")
def cases():
    world = NeuroWorld(demographic_signal_strength=0.0, shared_nuisance_stages=True)
    return world.generate(120, seed=7123)


def test_families_match_preregistration() -> None:
    assert set(ShiftGauntlet.families) == {
        "nuisance_variable",
        "prevalence",
        "conditional_feature",
        "spurious_correlation_inversion",
        "missingness",
        "observation_noise",
        "evidence_order",
        "distractor_evidence",
        "contradictory_evidence",
        "delayed_decisive_evidence",
        "unseen_factor_combinations",
        "unseen_world_mechanisms",
        "evidence_deletion",
        "evidence_duplication",
        "temporal_dependency_change",
        "class_expansion",
        "irreducible_ambiguity",
    }


@pytest.mark.parametrize("family", ShiftGauntlet.families)
def test_zero_severity_is_identity(cases, family: str) -> None:
    result = ShiftGauntlet().apply(cases, ShiftSpec(family, 0.0, 9))
    assert result.audit["changed_aligned_cases"] == 0
    assert len(result.cases) == len(cases)
    for source, shifted in zip(cases, result.cases, strict=True):
        assert source is not shifted
        assert np.array_equal(source.evidence, shifted.evidence)
        assert np.array_equal(source.tokens, shifted.tokens)


@pytest.mark.parametrize("family", ShiftGauntlet.families)
def test_shift_is_deterministic_and_does_not_mutate_source(cases, family: str) -> None:
    mode = "mnar_like" if family == "missingness" else None
    spec = ShiftSpec(family, 0.8, 19, mode=mode, split="train")
    before = [(case.evidence.copy(), case.tokens.copy()) for case in cases]
    first = ShiftGauntlet().apply(cases, spec)
    second = ShiftGauntlet().apply(cases, spec)
    assert first.audit == second.audit
    assert len(first.cases) == len(second.cases)
    for left, right in zip(first.cases, second.cases, strict=True):
        assert left.label == right.label
        assert np.array_equal(left.evidence, right.evidence)
        assert np.array_equal(left.tokens, right.tokens)
    for case, (evidence, tokens) in zip(cases, before, strict=True):
        assert np.array_equal(case.evidence, evidence)
        assert np.array_equal(case.tokens, tokens)


@pytest.mark.parametrize("mode", ["mcar", "mar", "mnar_like"])
def test_missingness_modes_reduce_observations(cases, mode: str) -> None:
    result = ShiftGauntlet().apply(cases, ShiftSpec("missingness", 1.0, 31, mode=mode))
    assert result.audit["mean_output_length"] < result.audit["mean_input_length"]
    assert all(len(case.tokens) >= 1 for case in result.cases)


@pytest.mark.parametrize(
    "mode", ["canonical", "random", "partially_corrupted", "adversarial", "reversed"]
)
def test_evidence_order_modes_are_supported(cases, mode: str) -> None:
    result = ShiftGauntlet().apply(cases, ShiftSpec("evidence_order", 1.0, 41, mode=mode))
    assert len(result.cases) == len(cases)
    for source, shifted in zip(cases, result.cases, strict=True):
        assert sorted(source.tokens.tolist()) == sorted(shifted.tokens.tolist())


def test_ambiguity_creates_conflicting_observable_collisions(cases) -> None:
    result = ShiftGauntlet().apply(cases, ShiftSpec("irreducible_ambiguity", 1.0, 53))
    assert result.audit["conflicting_observable_collisions"] > 0


def test_class_expansion_withholds_classes_only_from_training(cases) -> None:
    train = ShiftGauntlet().apply(cases, ShiftSpec("class_expansion", 1.0, 61, split="train"))
    test = ShiftGauntlet().apply(cases, ShiftSpec("class_expansion", 1.0, 61, split="test"))
    assert max(case.label for case in train.cases) < NeuroWorld.num_diagnoses - 4
    assert len(test.cases) == len(cases)


def test_invalid_specs_fail_loudly(cases) -> None:
    with pytest.raises(ValueError, match="severity"):
        ShiftSpec("noise", 1.1, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown shift"):
        ShiftGauntlet().apply(cases, ShiftSpec("noise", 0.5, 1))  # type: ignore[arg-type]
