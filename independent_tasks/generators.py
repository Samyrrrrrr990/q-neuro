"""Nonmedical generators with controlled relational order dependence.

These tasks intentionally reuse Q-Neuro's 80-token transport and 20-way output envelope so every
architecture can be evaluated without architecture-specific adapters. Their causal rules and
random streams are independent of NeuroWorld.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuroworld.simulator import Case, CounterfactualPair, NeuroWorld
from research.computational_laws import (
    analytic_operator_pair,
    discrete_mutual_information,
    normalized_commutator,
)

INDEPENDENT_TASK_FAMILIES: tuple[str, ...] = (
    "hidden_causal_machine",
    "sequential_detective",
    "machine_fault_diagnosis",
    "network_intrusion_reasoning",
    "hidden_rule_relational",
    "bayesian_urn",
    "analytic_noncommutative",
    "analytic_commutative",
)
GENERATOR_VERSION = "1.1.0"


@dataclass(frozen=True)
class TaskDataset:
    family: str
    cases: tuple[Case, ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class _FamilyDefinition:
    token_offset: int
    pair_count: int
    causal_order: bool
    order_dependence: float
    sequence_length: int
    narrative: str


_DEFINITIONS: dict[str, _FamilyDefinition] = {
    "hidden_causal_machine": _FamilyDefinition(
        0,
        4,
        True,
        0.80,
        18,
        "infer a hidden machine from the composition order of controlled actions",
    ),
    "sequential_detective": _FamilyDefinition(
        8,
        4,
        True,
        0.65,
        20,
        "resolve a suspect from the order of corroborating and contradicting clues",
    ),
    "machine_fault_diagnosis": _FamilyDefinition(
        16,
        4,
        True,
        0.30,
        24,
        "identify a latent equipment fault from ordered sensor events",
    ),
    "network_intrusion_reasoning": _FamilyDefinition(
        24,
        4,
        True,
        0.75,
        22,
        "classify an event trace from reconnaissance/execution order",
    ),
    "hidden_rule_relational": _FamilyDefinition(
        32,
        4,
        True,
        1.00,
        16,
        "infer a hidden relational rule from symbol order",
    ),
    "bayesian_urn": _FamilyDefinition(
        4,
        4,
        False,
        0.00,
        28,
        "infer an urn from exchangeable draw counts",
    ),
    "analytic_noncommutative": _FamilyDefinition(
        12,
        4,
        True,
        1.00,
        12,
        "classify a controlled noncommutative two-operator composition",
    ),
    "analytic_commutative": _FamilyDefinition(
        20,
        4,
        False,
        0.00,
        12,
        "classify a commuting composition where order is a negative control",
    ),
}


def _evidence_from_tokens(tokens: np.ndarray) -> np.ndarray:
    evidence = np.zeros(NeuroWorld.num_findings, dtype=np.int8)
    for token in tokens:
        finding = int(token % NeuroWorld.num_findings)
        evidence[finding] = -1 if token >= NeuroWorld.num_findings else 1
    return evidence


class IndependentSequentialTask:
    """One independent family with exact counterfactual-order semantics."""

    def __init__(
        self,
        family: str,
        *,
        order_dependence: float | None = None,
        sequence_length: int | None = None,
        world_seed: int = 0,
    ):
        if family not in _DEFINITIONS:
            raise ValueError(f"unknown independent task family: {family}")
        definition = _DEFINITIONS[family]
        controlled_order = (
            definition.order_dependence if order_dependence is None else float(order_dependence)
        )
        if not 0.0 <= controlled_order <= 1.0:
            raise ValueError("order_dependence must be in [0, 1]")
        if not definition.causal_order and controlled_order != 0.0:
            raise ValueError("commutative controls require order_dependence=0")
        length = definition.sequence_length if sequence_length is None else int(sequence_length)
        if length < 4:
            raise ValueError("sequence_length must be at least four")
        self.family = family
        self.definition = definition
        self.order_dependence = controlled_order
        self.sequence_length = length
        self.world_seed = int(world_seed)
        family_code = sum(
            (index + 1) * ord(character) for index, character in enumerate(self.family)
        )
        world_rng = np.random.default_rng(self.world_seed + family_code)
        self._support_weights = world_rng.lognormal(0.0, 0.35, NeuroWorld.num_findings)
        self._negative_probability_offset = float(world_rng.uniform(-0.05, 0.05))
        self._nuisance_polarity = bool(world_rng.integers(0, 2))

    @property
    def num_task_classes(self) -> int:
        return (
            2 * self.definition.pair_count
            if self.definition.causal_order
            else self.definition.pair_count
        )

    def _marker_tokens(self, pair: int, direction: int) -> tuple[int, int]:
        first = (self.definition.token_offset + 2 * pair) % NeuroWorld.num_findings
        second = (first + 1) % NeuroWorld.num_findings
        return (first, second) if direction == 0 else (second, first)

    def _sample_case(
        self,
        case_id: int,
        pair: int,
        direction: int,
        rng: np.random.Generator,
        *,
        split: str,
        shift_strength: float,
        force_causal_order: bool = False,
    ) -> Case:
        marker_first, marker_second = self._marker_tokens(pair, direction)
        causal_direction = direction
        use_causal_order = force_causal_order or rng.random() < self.order_dependence
        if self.definition.causal_order and not use_causal_order:
            # At lower controlled dependence, an explicit relational tag carries the same target
            # information without relying on marker order.
            marker_first, marker_second = self._marker_tokens(pair, int(rng.integers(0, 2)))
        label = 2 * pair + causal_direction if self.definition.causal_order else pair
        relational_tag = (
            self.definition.token_offset + 16 + 2 * pair + causal_direction
        ) % NeuroWorld.num_findings
        tokens: list[int] = [marker_first, marker_second]
        if self.definition.causal_order and not use_causal_order:
            tokens.append(relational_tag)

        support_pool = np.asarray(
            [
                finding
                for finding in range(NeuroWorld.num_findings)
                if finding not in {marker_first, marker_second, relational_tag}
            ]
        )
        target_length = max(4, round(self.sequence_length * (1.0 - 0.25 * shift_strength)))
        remaining = max(0, target_length - len(tokens))
        replace = remaining > len(support_pool)
        support_probabilities = self._support_weights[support_pool]
        support_probabilities = support_probabilities / support_probabilities.sum()
        support = rng.choice(
            support_pool, size=remaining, replace=replace, p=support_probabilities
        ).astype(int)
        negative_probability = float(
            np.clip(0.20 + self._negative_probability_offset + 0.35 * shift_strength, 0.0, 1.0)
        )
        support_tokens = [
            int(finding + (rng.random() < negative_probability) * NeuroWorld.num_findings)
            for finding in support
        ]
        insertion = int(rng.integers(0, len(support_tokens) + 1))
        sequence = support_tokens[:insertion] + tokens + support_tokens[insertion:]
        if split == "test" and shift_strength > 0.0:
            # Reverse a nuisance-only prefix; the declared causal marker relation is retained.
            prefix = round(shift_strength * insertion)
            sequence[:prefix] = reversed(sequence[:prefix])
        token_array = np.asarray(sequence, dtype=np.int64)
        nuisance_high = ((pair % 2 == 0) == (split == "train")) == self._nuisance_polarity
        nuisance_target = 0.8 if nuisance_high else 0.2
        age_scaled = float(
            np.clip(
                (1.0 - shift_strength) * rng.uniform(0.35, 0.65) + shift_strength * nuisance_target,
                0.0,
                1.0,
            )
        )
        return Case(
            case_id=case_id,
            label=label,
            evidence=_evidence_from_tokens(token_array),
            tokens=token_array,
            age_scaled=age_scaled,
            sex_binary=float(rng.integers(0, 2)),
            is_order_dependent=self.definition.causal_order,
            order_evidence_complete=True,
        )

    def generate(
        self,
        n_cases: int,
        seed: int,
        *,
        split: str = "train",
        shift_strength: float = 0.0,
    ) -> TaskDataset:
        if n_cases <= 0:
            raise ValueError("n_cases must be positive")
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if not 0.0 <= shift_strength <= 1.0:
            raise ValueError("shift_strength must be in [0, 1]")
        rng = np.random.default_rng(seed)
        label_rng = np.random.default_rng(seed + 1_000_003)
        cases: list[Case] = []
        latent_direction_bits: list[int] = []
        observed_order_bits: list[int] = []
        labels: list[int] = []
        for case_id in range(n_cases):
            pair = int(label_rng.integers(0, self.definition.pair_count))
            direction = int(label_rng.integers(0, 2))
            case = self._sample_case(
                case_id,
                pair,
                direction,
                rng,
                split=split,
                shift_strength=shift_strength,
            )
            cases.append(case)
            marker_a, marker_b = self._marker_tokens(pair, 0)
            position_a = int(np.flatnonzero(case.tokens == marker_a)[0])
            position_b = int(np.flatnonzero(case.tokens == marker_b)[0])
            latent_direction_bits.append(direction)
            observed_order_bits.append(int(position_b < position_a))
            labels.append(case.label)
        first, second = analytic_operator_pair(self.order_dependence)
        metadata: dict[str, object] = {
            "family": self.family,
            "generator_version": GENERATOR_VERSION,
            "narrative": self.definition.narrative,
            "synthetic_nonclinical": True,
            "generator_independent_of_neuroworld_rules": True,
            "split": split,
            "seed": int(seed),
            "world_seed": self.world_seed,
            "cases": n_cases,
            "sequence_length_target": self.sequence_length,
            "shift_strength": float(shift_strength),
            "causal_order": self.definition.causal_order,
            "controlled_order_dependence": self.order_dependence,
            "analytic_normalized_commutator": normalized_commutator(first, second),
            "empirical_observed_order_target_mutual_information": discrete_mutual_information(
                observed_order_bits, labels
            ),
            "latent_direction_target_mutual_information": discrete_mutual_information(
                latent_direction_bits, labels
            ),
            "num_task_classes": self.num_task_classes,
        }
        return TaskDataset(self.family, tuple(cases), metadata)

    def counterfactual_pairs(self, n_pairs: int, seed: int) -> tuple[CounterfactualPair, ...]:
        if n_pairs <= 0:
            raise ValueError("n_pairs must be positive")
        rng = np.random.default_rng(seed)
        pairs: list[CounterfactualPair] = []
        for pair_id in range(n_pairs):
            pair = int(rng.integers(0, self.definition.pair_count))
            first = self._sample_case(
                2 * pair_id,
                pair,
                0,
                rng,
                split="test",
                shift_strength=0.0,
                force_causal_order=True,
            )
            second_tokens = first.tokens.copy()
            marker_a, marker_b = self._marker_tokens(pair, 0)
            position_a = int(np.flatnonzero(second_tokens == marker_a)[0])
            position_b = int(np.flatnonzero(second_tokens == marker_b)[0])
            second_tokens[position_a], second_tokens[position_b] = (
                second_tokens[position_b],
                second_tokens[position_a],
            )
            second_label = 2 * pair + 1 if self.definition.causal_order else pair
            second = Case(
                case_id=2 * pair_id + 1,
                label=second_label,
                evidence=first.evidence.copy(),
                tokens=second_tokens,
                age_scaled=first.age_scaled,
                sex_binary=first.sex_binary,
                is_order_dependent=self.definition.causal_order,
                order_evidence_complete=True,
            )
            pairs.append(
                CounterfactualPair(
                    first,
                    second,
                    "causal_order" if self.definition.causal_order else "noncausal_order_control",
                )
            )
        return tuple(pairs)


def build_independent_task(
    family: str,
    *,
    order_dependence: float | None = None,
    sequence_length: int | None = None,
    world_seed: int = 0,
) -> IndependentSequentialTask:
    return IndependentSequentialTask(
        family,
        order_dependence=order_dependence,
        sequence_length=sequence_length,
        world_seed=world_seed,
    )
