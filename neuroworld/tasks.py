"""Controlled NeuroWorld task constructions beyond ordinary held-out classification."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np

from neuroworld.simulator import Case, CounterfactualPair, NeuroWorld

DEFAULT_COMPOSITION_PAIRS: tuple[tuple[int, int], ...] = (
    (8, 18),
    (10, 18),
    (12, 18),
    (14, 20),
)
HIDDEN_SYNDROME_SIGNATURE: tuple[int, ...] = (9, 17, 25, 31, 39)


def _filtered_cases(
    world: NeuroWorld,
    n_cases: int,
    seed: int,
    predicate: Callable[[Case], bool],
) -> list[Case]:
    rng = np.random.default_rng(seed)
    selected: list[Case] = []
    attempts = 0
    while len(selected) < n_cases:
        batch_seed = int(rng.integers(0, 2**31 - 1))
        batch = world.generate(max(512, n_cases - len(selected)), seed=batch_seed)
        selected.extend(case for case in batch if predicate(case))
        attempts += len(batch)
        if attempts > max(100_000, 200 * n_cases):
            raise RuntimeError("task rejection sampler could not find enough valid cases")
    return [replace(case, case_id=index) for index, case in enumerate(selected[:n_cases])]


def contains_composition(
    case: Case, heldout_pairs: Sequence[tuple[int, int]] = DEFAULT_COMPOSITION_PAIRS
) -> bool:
    return any(
        case.evidence[first] == 1 and case.evidence[second] == 1 for first, second in heldout_pairs
    )


def composition_split(
    world: NeuroWorld,
    n_train: int,
    n_validation: int,
    n_test: int,
    seed: int,
    heldout_pairs: Sequence[tuple[int, int]] = DEFAULT_COMPOSITION_PAIRS,
) -> tuple[list[Case], list[Case], list[Case]]:
    """Exclude selected positive finding conjunctions from train/validation and require them in test."""

    eligible = lambda case: case.label >= 8
    train = _filtered_cases(
        world,
        n_train,
        seed,
        lambda case: eligible(case) and not contains_composition(case, heldout_pairs),
    )
    validation = _filtered_cases(
        world,
        n_validation,
        seed + 1,
        lambda case: eligible(case) and not contains_composition(case, heldout_pairs),
    )
    test = _filtered_cases(
        world,
        n_test,
        seed + 2,
        lambda case: eligible(case) and contains_composition(case, heldout_pairs),
    )
    return train, validation, test


def composition_reference_cases(
    world: NeuroWorld,
    n_cases: int,
    seed: int,
    heldout_pairs: Sequence[tuple[int, int]] = DEFAULT_COMPOSITION_PAIRS,
) -> list[Case]:
    """Generate the exact inverse-filter reference population for the composition test."""

    return _filtered_cases(
        world,
        n_cases,
        seed,
        lambda case: case.label >= 8 and not contains_composition(case, heldout_pairs),
    )


def ambiguous_order_pairs(world: NeuroWorld, n_pairs: int, seed: int) -> list[CounterfactualPair]:
    """Remove both decisive chronology markers from otherwise paired order-twin cases."""

    pairs = world.counterfactual_pairs(n_pairs, seed)
    ambiguous: list[CounterfactualPair] = []
    for pair in pairs:
        pair_index = pair.first.label // 2
        markers = {2 * pair_index, 2 * pair_index + 1}

        def without_markers(case: Case, marker_set: frozenset[int] = frozenset(markers)) -> Case:
            evidence = case.evidence.copy()
            evidence[list(marker_set)] = 0
            tokens = np.asarray(
                [
                    token
                    for token in case.tokens
                    if int(token % NeuroWorld.num_findings) not in marker_set
                ],
                dtype=np.int64,
            )
            return replace(
                case,
                evidence=evidence,
                tokens=tokens,
                order_evidence_complete=False,
            )

        first = without_markers(pair.first)
        second = without_markers(pair.second)
        ambiguous.append(CounterfactualPair(first, second, "both_order_markers_missing"))
    return ambiguous


def label_filtered_cases(
    world: NeuroWorld,
    n_cases: int,
    seed: int,
    included_labels: set[int],
) -> list[Case]:
    return _filtered_cases(world, n_cases, seed, lambda case: case.label in included_labels)


def hidden_syndrome_cases(
    n_cases: int, seed: int, observation_probability: float = 0.72
) -> list[Case]:
    """Generate an unlabeled syndrome with a consistent cross-factor signature."""

    if n_cases <= 0:
        raise ValueError("n_cases must be positive")
    rng = np.random.default_rng(seed)
    probabilities = np.full(NeuroWorld.num_findings, 0.12, dtype=np.float32)
    probabilities[list(HIDDEN_SYNDROME_SIGNATURE)] = 0.92
    probabilities[[8, 16, 24, 30, 38]] = 0.02
    base_stages = rng.uniform(0.15, 0.85, size=NeuroWorld.num_findings)
    base_stages[list(HIDDEN_SYNDROME_SIGNATURE)] = np.asarray([0.10, 0.28, 0.46, 0.68, 0.88])
    cases: list[Case] = []
    for case_id in range(n_cases):
        present = rng.random(NeuroWorld.num_findings) < probabilities
        observed = rng.random(NeuroWorld.num_findings) < observation_probability
        evidence = np.zeros(NeuroWorld.num_findings, dtype=np.int8)
        evidence[observed & present] = 1
        evidence[observed & ~present] = -1
        finding_ids = np.flatnonzero(observed)
        times = base_stages[finding_ids] + rng.normal(0.0, 0.03, size=finding_ids.size)
        ordered = finding_ids[np.argsort(times, kind="stable")]
        tokens = ordered + (evidence[ordered] < 0) * NeuroWorld.num_findings
        cases.append(
            Case(
                case_id=case_id,
                label=-1,
                evidence=evidence,
                tokens=tokens.astype(np.int64),
                age_scaled=float(np.clip(rng.normal(0.48, 0.08), 0.0, 1.0)),
                sex_binary=float(rng.random() < 0.5),
                is_order_dependent=False,
                order_evidence_complete=True,
            )
        )
    return cases
