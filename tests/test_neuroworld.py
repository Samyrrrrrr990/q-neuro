import numpy as np

from neuroworld import NeuroWorld


def test_missing_is_distinct_from_observed_absence() -> None:
    world = NeuroWorld(observation_probability=0.65)
    cases = world.generate(200, seed=7)
    values = np.concatenate([case.evidence for case in cases])
    assert {-1, 0, 1}.issubset(set(values.tolist()))
    vector = world.vector_features(cases[0])
    value, mask = vector[:40], vector[40:80]
    assert np.all(mask[value == 0] == 0)
    assert np.all(mask[value == -1] == 1)


def test_counterfactual_pair_changes_only_marker_order_and_target() -> None:
    world = NeuroWorld()
    pair = world.counterfactual_pairs(1, seed=17)[0]
    assert np.array_equal(pair.first.evidence, pair.second.evidence)
    assert pair.first.age_scaled == pair.second.age_scaled
    assert pair.first.sex_binary == pair.second.sex_binary
    assert pair.second.label == pair.first.label + 1
    assert sorted(pair.first.tokens.tolist()) == sorted(pair.second.tokens.tolist())
    changed_positions = np.flatnonzero(pair.first.tokens != pair.second.tokens)
    assert changed_positions.size == 2
    assert pair.causal_factor == "evidence_order"


def test_generation_is_deterministic() -> None:
    world = NeuroWorld()
    first = world.generate(10, seed=101)
    second = world.generate(10, seed=101)
    for case_a, case_b in zip(first, second, strict=True):
        assert case_a.label == case_b.label
        assert np.array_equal(case_a.evidence, case_b.evidence)
        assert np.array_equal(case_a.tokens, case_b.tokens)
