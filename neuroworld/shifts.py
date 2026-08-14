"""Deterministic distribution-shift interventions for the ShiftGauntlet benchmark.

The interventions in this module modify synthetic observations, never clinical data.  Each
intervention is deliberately isolated so that a robustness result can be attributed to a named
shift rather than to an opaque mixture of simulator changes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np

from neuroworld.simulator import Case, NeuroWorld

ShiftFamily = Literal[
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
]


@dataclass(frozen=True)
class ShiftSpec:
    """One frozen ShiftGauntlet intervention."""

    family: ShiftFamily
    severity: float
    seed: int
    mode: str | None = None
    split: Literal["train", "validation", "test"] = "test"

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")


@dataclass(frozen=True)
class ShiftedCases:
    """Shifted cases and diagnostics required to audit the intervention."""

    cases: tuple[Case, ...]
    audit: dict[str, Any]


def _copy_case(case: Case, **changes: Any) -> Case:
    copied = replace(
        case,
        evidence=case.evidence.copy(),
        tokens=case.tokens.copy(),
    )
    return replace(copied, **changes)


def _rebuild_evidence(tokens: np.ndarray) -> np.ndarray:
    evidence = np.zeros(NeuroWorld.num_findings, dtype=np.int8)
    for token in tokens:
        finding = int(token % NeuroWorld.num_findings)
        evidence[finding] = -1 if token >= NeuroWorld.num_findings else 1
    return evidence


def _token_with_sign(finding: int, negative: bool) -> int:
    return int(finding + negative * NeuroWorld.num_findings)


def _ensure_one_token(original: np.ndarray, kept: np.ndarray) -> np.ndarray:
    if kept.size or original.size == 0:
        return kept
    return original[:1].copy()


def _changed(first: Case, second: Case) -> bool:
    return bool(
        first.label != second.label
        or first.age_scaled != second.age_scaled
        or first.sex_binary != second.sex_binary
        or not np.array_equal(first.evidence, second.evidence)
        or not np.array_equal(first.tokens, second.tokens)
    )


def _class_frequencies(cases: Sequence[Case]) -> dict[str, float]:
    counts = Counter(case.label for case in cases)
    total = max(1, len(cases))
    return {str(label): counts.get(label, 0) / total for label in range(NeuroWorld.num_diagnoses)}


def _observable_collision_count(cases: Sequence[Case]) -> int:
    observed: dict[tuple[bytes, bytes, float, float], int] = {}
    conflicts = 0
    for case in cases:
        key = (case.evidence.tobytes(), case.tokens.tobytes(), case.age_scaled, case.sex_binary)
        prior_label = observed.setdefault(key, case.label)
        conflicts += int(prior_label != case.label)
    return conflicts


class ShiftGauntlet:
    """Apply the preregistered shift families with deterministic random streams."""

    families: tuple[str, ...] = (
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
    )

    def apply(self, cases: Sequence[Case], spec: ShiftSpec) -> ShiftedCases:
        if spec.family not in self.families:
            raise ValueError(f"unknown shift family: {spec.family}")
        if not cases:
            raise ValueError("cases must not be empty")
        source = tuple(_copy_case(case) for case in cases)
        if spec.severity == 0.0:
            shifted = source
        else:
            rng = np.random.default_rng(spec.seed)
            transform = getattr(self, f"_apply_{spec.family}")
            shifted = tuple(transform(source, spec, rng))
        source_by_id = {case.case_id: case for case in source}
        aligned = [
            (source_by_id[case.case_id], case) for case in shifted if case.case_id in source_by_id
        ]
        audit = {
            "spec": asdict(spec),
            "input_cases": len(source),
            "output_cases": len(shifted),
            "changed_aligned_cases": sum(_changed(before, after) for before, after in aligned),
            "mean_input_length": float(np.mean([len(case.tokens) for case in source])),
            "mean_output_length": float(np.mean([len(case.tokens) for case in shifted])),
            "input_class_frequencies": _class_frequencies(source),
            "output_class_frequencies": _class_frequencies(shifted),
            "conflicting_observable_collisions": _observable_collision_count(shifted),
        }
        return ShiftedCases(shifted, audit)

    @staticmethod
    def _apply_nuisance_variable(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        del rng
        return [
            _copy_case(
                case,
                age_scaled=float(
                    np.clip(
                        (1.0 - spec.severity) * case.age_scaled
                        + spec.severity * (1.0 - case.age_scaled),
                        0.0,
                        1.0,
                    )
                ),
                sex_binary=float(1.0 - case.sex_binary)
                if spec.severity >= 0.5
                else case.sex_binary,
            )
            for case in cases
        ]

    @staticmethod
    def _apply_prevalence(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        labels = np.asarray([case.label for case in cases])
        class_log_weights = np.linspace(-2.0, 2.0, NeuroWorld.num_diagnoses)
        weights = np.exp(spec.severity * class_log_weights[labels])
        # Weighted sampling without replacement avoids introducing train/test duplicates.
        gumbel = -np.log(-np.log(rng.uniform(1e-12, 1.0 - 1e-12, size=len(cases))))
        count = max(NeuroWorld.num_diagnoses, round(len(cases) * (1.0 - 0.35 * spec.severity)))
        selected = np.argsort(np.log(weights) + gumbel)[-count:]
        return [_copy_case(cases[index]) for index in sorted(selected.tolist())]

    @staticmethod
    def _apply_conditional_feature(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        shifted: list[Case] = []
        for case in cases:
            tokens = case.tokens.copy()
            target_modulus = case.label % 5
            eligible = np.flatnonzero((tokens % NeuroWorld.num_findings) % 5 == target_modulus)
            flip = eligible[rng.random(eligible.size) < 0.45 * spec.severity]
            tokens[flip] = (tokens[flip] + NeuroWorld.num_findings) % NeuroWorld.num_tokens
            shifted.append(_copy_case(case, tokens=tokens, evidence=_rebuild_evidence(tokens)))
        return shifted

    @staticmethod
    def _apply_spurious_correlation_inversion(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        shifted: list[Case] = []
        direction = 1 if spec.split == "train" else -1
        for case in cases:
            high = bool((case.label % 2 == 0) == (direction > 0))
            target_age = 0.85 if high else 0.15
            target_sex = 1.0 if high else 0.0
            jitter = float(rng.normal(0.0, 0.015 * spec.severity))
            shifted.append(
                _copy_case(
                    case,
                    age_scaled=float(
                        np.clip(
                            (1.0 - spec.severity) * case.age_scaled
                            + spec.severity * target_age
                            + jitter,
                            0.0,
                            1.0,
                        )
                    ),
                    sex_binary=(target_sex if rng.random() < spec.severity else case.sex_binary),
                )
            )
        return shifted

    @staticmethod
    def _apply_missingness(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        mode = spec.mode or "mcar"
        if mode not in {"mcar", "mar", "mnar_like"}:
            raise ValueError(f"unknown missingness mode: {mode}")
        shifted: list[Case] = []
        for case in cases:
            base = 0.72 * spec.severity
            if mode == "mcar":
                deletion = np.full(len(case.tokens), base)
            elif mode == "mar":
                deletion = np.full(len(case.tokens), base * (0.35 + 0.65 * case.age_scaled))
            else:
                negative = case.tokens >= NeuroWorld.num_findings
                deletion = base * np.where(negative, 1.0, 0.30)
            kept = _ensure_one_token(
                case.tokens, case.tokens[rng.random(len(case.tokens)) >= deletion]
            )
            shifted.append(_copy_case(case, tokens=kept, evidence=_rebuild_evidence(kept)))
        return shifted

    @staticmethod
    def _apply_observation_noise(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        shifted: list[Case] = []
        for case in cases:
            tokens = case.tokens.copy()
            flip = rng.random(len(tokens)) < 0.45 * spec.severity
            tokens[flip] = (tokens[flip] + NeuroWorld.num_findings) % NeuroWorld.num_tokens
            shifted.append(_copy_case(case, tokens=tokens, evidence=_rebuild_evidence(tokens)))
        return shifted

    @staticmethod
    def _apply_evidence_order(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        mode = spec.mode or "random"
        if mode not in {"canonical", "random", "partially_corrupted", "adversarial", "reversed"}:
            raise ValueError(f"unknown evidence-order mode: {mode}")
        shifted: list[Case] = []
        for case in cases:
            tokens = case.tokens.copy()
            if mode == "canonical":
                pass
            elif mode == "random":
                if rng.random() < spec.severity:
                    tokens = tokens[rng.permutation(len(tokens))]
            elif mode == "partially_corrupted":
                for position in range(max(0, len(tokens) - 1)):
                    if rng.random() < 0.35 * spec.severity:
                        tokens[position], tokens[position + 1] = (
                            tokens[position + 1],
                            tokens[position],
                        )
            elif mode == "reversed":
                if rng.random() < spec.severity:
                    tokens = tokens[::-1].copy()
            elif case.is_order_dependent and rng.random() < spec.severity:
                marker_a = 2 * (case.label // 2)
                marker_b = marker_a + 1
                positions = [
                    np.flatnonzero(tokens % NeuroWorld.num_findings == marker)
                    for marker in (marker_a, marker_b)
                ]
                if all(position.size for position in positions):
                    first, second = int(positions[0][0]), int(positions[1][0])
                    if first < second:
                        tokens[first], tokens[second] = tokens[second], tokens[first]
            shifted.append(_copy_case(case, tokens=tokens))
        return shifted

    @staticmethod
    def _apply_distractor_evidence(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        shifted: list[Case] = []
        for case in cases:
            observed = set((case.tokens % NeuroWorld.num_findings).tolist())
            available = np.asarray(sorted(set(range(NeuroWorld.num_findings)) - observed))
            count = min(len(available), round(spec.severity * max(1, len(case.tokens) / 2)))
            additions = rng.choice(available, size=count, replace=False) if count else np.array([])
            tokens = case.tokens.tolist()
            for finding in additions.astype(int):
                token = _token_with_sign(finding, bool(rng.integers(0, 2)))
                tokens.insert(int(rng.integers(0, len(tokens) + 1)), token)
            token_array = np.asarray(tokens, dtype=np.int64)
            shifted.append(
                _copy_case(case, tokens=token_array, evidence=_rebuild_evidence(token_array))
            )
        return shifted

    @staticmethod
    def _apply_contradictory_evidence(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        shifted: list[Case] = []
        for case in cases:
            tokens = case.tokens.tolist()
            count = min(len(tokens), round(spec.severity * max(1, len(tokens) / 4)))
            positions = rng.choice(len(tokens), size=count, replace=False) if count else []
            for position in np.sort(positions):
                token = int(case.tokens[int(position)])
                tokens.append((token + NeuroWorld.num_findings) % NeuroWorld.num_tokens)
            token_array = np.asarray(tokens, dtype=np.int64)
            shifted.append(
                _copy_case(case, tokens=token_array, evidence=_rebuild_evidence(token_array))
            )
        return shifted

    @staticmethod
    def _apply_delayed_decisive_evidence(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        del rng
        shifted: list[Case] = []
        for case in cases:
            findings = case.tokens % NeuroWorld.num_findings
            if case.label < 8:
                decisive = np.isin(findings, [2 * (case.label // 2), 2 * (case.label // 2) + 1])
            else:
                local = case.label - 8
                decisive_ids = {
                    8 + 2 * (local % 5),
                    8 + 2 * (local % 5) + 1,
                    18 + 2 * ((local // 3) % 4),
                    18 + 2 * ((local // 3) % 4) + 1,
                }
                decisive = np.isin(findings, list(decisive_ids))
            move = decisive & (
                np.arange(len(case.tokens)) < round(spec.severity * len(case.tokens))
            )
            tokens = np.concatenate([case.tokens[~move], case.tokens[move]])
            shifted.append(_copy_case(case, tokens=tokens))
        return shifted

    @staticmethod
    def _map_findings(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator, *, by_label: bool
    ) -> list[Case]:
        groups = [np.arange(8, 18), np.arange(18, 26), np.arange(26, 32), np.arange(32, 40)]
        global_maps: dict[tuple[int, int], int] = {}
        for group_index, group in enumerate(groups):
            permutation = rng.permutation(group)
            global_maps.update(
                {
                    (group_index, int(source)): int(target)
                    for source, target in zip(group, permutation)
                }
            )
        shifted: list[Case] = []
        for case in cases:
            tokens = case.tokens.copy()
            for position, token in enumerate(tokens):
                finding = int(token % NeuroWorld.num_findings)
                for group_index, group in enumerate(groups):
                    if finding in group and rng.random() < spec.severity:
                        target = global_maps[(group_index, finding)]
                        if by_label:
                            offset = case.label % len(group)
                            target = int(
                                group[(np.flatnonzero(group == target)[0] + offset) % len(group)]
                            )
                        tokens[position] = _token_with_sign(
                            target, bool(token >= NeuroWorld.num_findings)
                        )
                        break
            shifted.append(_copy_case(case, tokens=tokens, evidence=_rebuild_evidence(tokens)))
        return shifted

    @classmethod
    def _apply_unseen_factor_combinations(
        cls, cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        return cls._map_findings(cases, spec, rng, by_label=True)

    @classmethod
    def _apply_unseen_world_mechanisms(
        cls, cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        return cls._map_findings(cases, spec, rng, by_label=False)

    @staticmethod
    def _apply_evidence_deletion(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        shifted: list[Case] = []
        for case in cases:
            kept = _ensure_one_token(
                case.tokens,
                case.tokens[rng.random(len(case.tokens)) >= 0.85 * spec.severity],
            )
            shifted.append(_copy_case(case, tokens=kept, evidence=_rebuild_evidence(kept)))
        return shifted

    @staticmethod
    def _apply_evidence_duplication(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        shifted: list[Case] = []
        for case in cases:
            tokens = case.tokens.tolist()
            count = min(len(tokens), round(spec.severity * len(tokens) / 2))
            positions = rng.choice(len(tokens), size=count, replace=False) if count else []
            for position in np.sort(positions)[::-1]:
                tokens.insert(int(position) + 1, tokens[int(position)])
            shifted.append(_copy_case(case, tokens=np.asarray(tokens, dtype=np.int64)))
        return shifted

    @staticmethod
    def _apply_temporal_dependency_change(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        shifted: list[Case] = []
        for case in cases:
            tokens = case.tokens.copy()
            if rng.random() < spec.severity:
                signs = tokens >= NeuroWorld.num_findings
                tie_break = rng.random(len(tokens))
                tokens = tokens[np.lexsort((tie_break, signs))]
            shifted.append(_copy_case(case, tokens=tokens))
        return shifted

    @staticmethod
    def _apply_class_expansion(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        del rng
        held_out = max(1, round(4 * spec.severity))
        if spec.split == "train":
            maximum_seen = NeuroWorld.num_diagnoses - held_out
            return [_copy_case(case) for case in cases if case.label < maximum_seen]
        return [_copy_case(case) for case in cases]

    @staticmethod
    def _apply_irreducible_ambiguity(
        cases: Sequence[Case], spec: ShiftSpec, rng: np.random.Generator
    ) -> list[Case]:
        shifted = [_copy_case(case) for case in cases]
        order = rng.permutation(len(shifted))
        pair_count = round(spec.severity * len(shifted) / 2)
        for pair in range(pair_count):
            first_index = int(order[2 * pair])
            second_index = int(order[2 * pair + 1])
            first = shifted[first_index]
            second = shifted[second_index]
            if first.label == second.label:
                continue
            shifted[second_index] = _copy_case(
                second,
                evidence=first.evidence.copy(),
                tokens=first.tokens.copy(),
                age_scaled=first.age_scaled,
                sex_binary=first.sex_binary,
            )
        return shifted


def map_shift_grid(
    cases: Sequence[Case], specs: Sequence[ShiftSpec], function: Callable[[ShiftedCases], Any]
) -> dict[str, Any]:
    """Apply and evaluate a declared shift grid without duplicating runner logic."""

    gauntlet = ShiftGauntlet()
    return {
        f"{spec.family}:{spec.mode or 'default'}:{spec.severity:g}": function(
            gauntlet.apply(cases, spec)
        )
        for spec in specs
    }
