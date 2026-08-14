"""A small causal world for controlled ordered-evidence experiments.

The simulator is intentionally not a clinical simulator. Disease labels are synthetic archetypes,
and probabilities encode experimental structure rather than epidemiology or medical advice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Case:
    """One partially observed synthetic case."""

    case_id: int
    label: int
    evidence: np.ndarray
    tokens: np.ndarray
    age_scaled: float
    sex_binary: float
    is_order_dependent: bool
    order_evidence_complete: bool


@dataclass(frozen=True)
class CounterfactualPair:
    """Cases differing only in an explicitly identified causal factor."""

    first: Case
    second: Case
    causal_factor: str


class NeuroWorld:
    """Generate reproducible cases from disease -> factors -> findings -> observations.

    The first eight disease archetypes form four order-twin pairs. Within a pair, the two labels
    have identical marginal evidence and differ only in the chronology of two marker findings.
    This makes the benefit and cost of ordered computation directly measurable.
    """

    num_diagnoses = 20
    num_findings = 40
    num_tokens = 80
    pad_token = 80

    disease_names = tuple(
        [f"order_twin_{pair}_{direction}" for pair in range(4) for direction in ("ab", "ba")]
        + [f"factorial_archetype_{idx:02d}" for idx in range(12)]
    )
    finding_names = tuple(
        [f"order_marker_{idx}" for idx in range(8)]
        + [f"mechanism_signal_{idx}" for idx in range(10)]
        + [f"localization_signal_{idx}" for idx in range(8)]
        + [f"temporal_signal_{idx}" for idx in range(6)]
        + [f"context_signal_{idx}" for idx in range(8)]
    )

    def __init__(
        self,
        world_seed: int = 314159,
        observation_probability: float = 0.72,
        probability_mixing: float = 0.0,
        temporal_jitter: float = 0.025,
        order_marker_visibility: float = 1.0,
    ):
        if not 0.0 < observation_probability <= 1.0:
            raise ValueError("observation_probability must be in (0, 1]")
        if not 0.0 <= probability_mixing < 1.0:
            raise ValueError("probability_mixing must be in [0, 1)")
        if temporal_jitter < 0.0:
            raise ValueError("temporal_jitter must be non-negative")
        if not 0.0 <= order_marker_visibility <= 1.0:
            raise ValueError("order_marker_visibility must be in [0, 1]")
        self.world_seed = int(world_seed)
        self.observation_probability = float(observation_probability)
        self.probability_mixing = float(probability_mixing)
        self.temporal_jitter = float(temporal_jitter)
        self.order_marker_visibility = float(order_marker_visibility)
        self._probabilities, self._stages, self._age_means, self._sex_probs = (
            self._build_causal_templates()
        )
        self._probabilities = (
            (1.0 - self.probability_mixing) * self._probabilities + self.probability_mixing * 0.5
        ).astype(np.float32)

    def _build_causal_templates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.world_seed)
        probabilities = np.full((self.num_diagnoses, self.num_findings), 0.10, dtype=np.float32)
        stages = rng.uniform(0.15, 0.85, size=(self.num_diagnoses, self.num_findings)).astype(
            np.float32
        )
        age_means = np.zeros(self.num_diagnoses, dtype=np.float32)
        sex_probs = np.zeros(self.num_diagnoses, dtype=np.float32)

        # The first eight diagnoses are four marginally identical order-twin pairs.
        for pair in range(4):
            disease_a = 2 * pair
            disease_b = disease_a + 1
            marker_a = 2 * pair
            marker_b = marker_a + 1
            base = np.full(self.num_findings, 0.08, dtype=np.float32)
            base[marker_a] = 0.98
            base[marker_b] = 0.98
            support_pool = np.arange(8, self.num_findings)
            support = rng.choice(support_pool, size=8, replace=False)
            base[support[:4]] = 0.82
            base[support[4:]] = 0.56
            probabilities[disease_a] = base
            probabilities[disease_b] = base
            shared_stages = rng.uniform(0.18, 0.82, size=self.num_findings).astype(np.float32)
            stages[disease_a] = shared_stages
            stages[disease_b] = shared_stages
            stages[disease_a, marker_a], stages[disease_a, marker_b] = 0.05, 0.95
            stages[disease_b, marker_a], stages[disease_b, marker_b] = 0.95, 0.05
            age_means[disease_a] = age_means[disease_b] = 0.25 + 0.16 * pair
            sex_probs[disease_a] = sex_probs[disease_b] = 0.35 + 0.10 * (pair % 3)

        # Remaining diagnoses are built from mechanism, localization, temporal, and context causes.
        for local_idx, disease in enumerate(range(8, self.num_diagnoses)):
            mechanism = local_idx % 5
            localization = (local_idx // 3) % 4
            temporal = (2 * local_idx + 1) % 3
            context = (3 * local_idx + 2) % 4
            hallmark = [
                8 + 2 * mechanism,
                8 + 2 * mechanism + 1,
                18 + 2 * localization,
                18 + 2 * localization + 1,
                26 + 2 * temporal,
                26 + 2 * temporal + 1,
                32 + 2 * context,
                32 + 2 * context + 1,
            ]
            probabilities[disease, hallmark[:4]] = 0.86
            probabilities[disease, hallmark[4:]] = 0.68
            # Weak cross-factor signals prevent the task from becoming a lookup table.
            secondary = rng.choice(np.setdiff1d(np.arange(8, 40), hallmark), size=5, replace=False)
            probabilities[disease, secondary] = 0.38
            stages[disease, hallmark[0:2]] = 0.20
            stages[disease, hallmark[2:4]] = 0.42
            stages[disease, hallmark[4:6]] = 0.64
            stages[disease, hallmark[6:8]] = 0.80
            age_means[disease] = 0.18 + 0.065 * local_idx
            sex_probs[disease] = 0.30 + 0.10 * (local_idx % 5)

        return probabilities, stages, age_means, sex_probs

    def generate(self, n_cases: int, seed: int) -> list[Case]:
        """Generate independent cases with uniform disease prevalence."""

        if n_cases <= 0:
            raise ValueError("n_cases must be positive")
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, self.num_diagnoses, size=n_cases)
        return [self._sample_case(int(label), idx, rng) for idx, label in enumerate(labels)]

    def _sample_case(
        self,
        label: int,
        case_id: int,
        rng: np.random.Generator,
        force_order_visible: bool = False,
    ) -> Case:
        present = rng.random(self.num_findings) < self._probabilities[label]
        observed = rng.random(self.num_findings) < self.observation_probability

        is_order = label < 8
        if is_order:
            marker_a = 2 * (label // 2)
            marker_b = marker_a + 1
            markers = [marker_a, marker_b]
            present[markers] = True
            if force_order_visible or self.order_marker_visibility == 1.0:
                observed[markers] = True
            else:
                observed[markers] = rng.random(2) < self.order_marker_visibility

        order_evidence_complete = not is_order or bool(observed[[marker_a, marker_b]].all())

        evidence = np.zeros(self.num_findings, dtype=np.int8)
        evidence[observed & present] = 1
        evidence[observed & ~present] = -1

        finding_ids = np.flatnonzero(observed)
        jitter = rng.normal(0.0, self.temporal_jitter, size=finding_ids.size)
        times = self._stages[label, finding_ids] + jitter
        ordered_findings = finding_ids[np.argsort(times, kind="stable")]
        token_ids = ordered_findings + (evidence[ordered_findings] < 0) * self.num_findings

        age_scaled = float(np.clip(rng.normal(self._age_means[label], 0.10), 0.0, 1.0))
        sex_binary = float(rng.random() < self._sex_probs[label])
        return Case(
            case_id=int(case_id),
            label=label,
            evidence=evidence,
            tokens=token_ids.astype(np.int64),
            age_scaled=age_scaled,
            sex_binary=sex_binary,
            is_order_dependent=is_order,
            order_evidence_complete=order_evidence_complete,
        )

    def counterfactual_pairs(self, n_pairs: int, seed: int) -> list[CounterfactualPair]:
        """Generate order-twin pairs with identical values and reversed marker chronology."""

        if n_pairs <= 0:
            raise ValueError("n_pairs must be positive")
        rng = np.random.default_rng(seed)
        pairs: list[CounterfactualPair] = []
        for pair_id in range(n_pairs):
            pair_index = int(rng.integers(0, 4))
            label_ab = 2 * pair_index
            label_ba = label_ab + 1
            first = self._sample_case(label_ab, 2 * pair_id, rng, force_order_visible=True)
            marker_a = 2 * pair_index
            marker_b = marker_a + 1

            # Reuse every observed value and demographic variable; only exchange marker positions.
            second_tokens = first.tokens.copy()
            token_a = marker_a
            token_b = marker_b
            pos_a = int(np.flatnonzero(second_tokens == token_a)[0])
            pos_b = int(np.flatnonzero(second_tokens == token_b)[0])
            second_tokens[pos_a], second_tokens[pos_b] = second_tokens[pos_b], second_tokens[pos_a]
            second = Case(
                case_id=2 * pair_id + 1,
                label=label_ba,
                evidence=first.evidence.copy(),
                tokens=second_tokens,
                age_scaled=first.age_scaled,
                sex_binary=first.sex_binary,
                is_order_dependent=True,
                order_evidence_complete=True,
            )
            pairs.append(CounterfactualPair(first, second, "evidence_order"))
        return pairs

    @staticmethod
    def vector_features(case: Case) -> np.ndarray:
        """Return values, explicit observation mask, and demographics for non-sequential models."""

        values = case.evidence.astype(np.float32)
        observed = (case.evidence != 0).astype(np.float32)
        return np.concatenate(
            [values, observed, np.array([case.age_scaled, case.sex_binary], dtype=np.float32)]
        )
