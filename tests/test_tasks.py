import numpy as np

from neuroworld import (
    HIDDEN_SYNDROME_SIGNATURE,
    NeuroWorld,
    ambiguous_order_pairs,
    composition_reference_cases,
    composition_split,
    hidden_syndrome_cases,
    label_filtered_cases,
)
from neuroworld.tasks import contains_composition


def test_composition_split_enforces_conjunction_holdout() -> None:
    train, validation, test = composition_split(NeuroWorld(), 100, 40, 60, seed=12)
    assert all(case.label >= 8 for case in (*train, *validation, *test))
    assert not any(contains_composition(case) for case in (*train, *validation))
    assert all(contains_composition(case) for case in test)
    reference = composition_reference_cases(NeuroWorld(), 75, seed=16)
    assert len(reference) == 75
    assert all(case.label >= 8 and not contains_composition(case) for case in reference)


def test_ambiguous_pairs_are_observationally_identical() -> None:
    pairs = ambiguous_order_pairs(NeuroWorld(), 20, seed=13)
    for pair in pairs:
        assert pair.first.label != pair.second.label
        assert np.array_equal(pair.first.evidence, pair.second.evidence)
        assert np.array_equal(pair.first.tokens, pair.second.tokens)
        assert not pair.first.order_evidence_complete


def test_label_filter_and_hidden_syndrome() -> None:
    world = NeuroWorld()
    cases = label_filtered_cases(world, 100, seed=14, included_labels={0, 2, 4})
    assert {case.label for case in cases} <= {0, 2, 4}
    hidden = hidden_syndrome_cases(500, seed=15)
    assert all(case.label == -1 for case in hidden)
    signature_rate = np.mean(
        [case.evidence[list(HIDDEN_SYNDROME_SIGNATURE)] == 1 for case in hidden], axis=0
    )
    assert np.all(signature_rate > 0.55)
