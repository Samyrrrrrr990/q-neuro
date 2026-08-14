"""Adversarial validity checks for NeuroWorld shortcut and leakage analysis."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from neuroworld.simulator import Case, NeuroWorld


@dataclass(frozen=True)
class ShortcutScore:
    name: str
    validation_accuracy: float
    test_accuracy: float
    selected_feature: str | None = None


def _majority_label(labels: Sequence[int]) -> int:
    return Counter(labels).most_common(1)[0][0]


def _lookup_score(
    train: Sequence[Case],
    evaluation: Sequence[Case],
    key: Callable[[Case], object],
) -> float:
    fallback = _majority_label([case.label for case in train])
    grouped: dict[object, list[int]] = defaultdict(list)
    for case in train:
        grouped[key(case)].append(case.label)
    table = {value: _majority_label(labels) for value, labels in grouped.items()}
    return float(np.mean([table.get(key(case), fallback) == case.label for case in evaluation]))


def _select_lookup(
    name: str,
    train: Sequence[Case],
    validation: Sequence[Case],
    test: Sequence[Case],
    candidates: Sequence[tuple[str, Callable[[Case], object]]],
) -> ShortcutScore:
    scored = [(feature, key, _lookup_score(train, validation, key)) for feature, key in candidates]
    feature, key, validation_accuracy = max(scored, key=lambda value: (value[2], value[0]))
    return ShortcutScore(
        name=name,
        validation_accuracy=validation_accuracy,
        test_accuracy=_lookup_score(train, test, key),
        selected_feature=feature,
    )


def _case_fingerprint(case: Case) -> str:
    digest = hashlib.sha256()
    digest.update(case.evidence.tobytes())
    digest.update(case.tokens.tobytes())
    digest.update(np.asarray([case.age_scaled, case.sex_binary], dtype=np.float64).tobytes())
    return digest.hexdigest()


def duplicate_rate(first: Sequence[Case], second: Sequence[Case]) -> float:
    known = {_case_fingerprint(case) for case in first}
    return float(np.mean([_case_fingerprint(case) in known for case in second]))


def case_consistency_errors(cases: Sequence[Case]) -> list[str]:
    errors: list[str] = []
    for case in cases:
        observed = np.flatnonzero(case.evidence)
        token_findings = case.tokens % NeuroWorld.num_findings
        if len(np.unique(token_findings)) != len(token_findings):
            errors.append(f"case {case.case_id}: duplicate finding token")
        if sorted(observed.tolist()) != sorted(token_findings.tolist()):
            errors.append(f"case {case.case_id}: tokens and observed evidence disagree")
        for token in case.tokens:
            finding = int(token % NeuroWorld.num_findings)
            expected_negative = case.evidence[finding] < 0
            if bool(token >= NeuroWorld.num_findings) != bool(expected_negative):
                errors.append(f"case {case.case_id}: token sign disagrees with evidence")
    return errors


def nearest_neighbor_accuracy(train: Sequence[Case], test: Sequence[Case]) -> float:
    train_vectors = np.stack([NeuroWorld.vector_features(case) for case in train])
    train_labels = np.asarray([case.label for case in train])
    test_vectors = np.stack([NeuroWorld.vector_features(case) for case in test])
    predictions: list[np.ndarray] = []
    for start in range(0, len(test), 256):
        query = test_vectors[start : start + 256]
        distance = np.square(query[:, None, :] - train_vectors[None, :, :]).sum(axis=-1)
        predictions.append(train_labels[np.argmin(distance, axis=1)])
    return float(np.mean(np.concatenate(predictions) == [case.label for case in test]))


def shortcut_scores(
    train: Sequence[Case], validation: Sequence[Case], test: Sequence[Case]
) -> list[ShortcutScore]:
    prior = _majority_label([case.label for case in train])
    scores = [
        ShortcutScore(
            "class_prior",
            float(np.mean([case.label == prior for case in validation])),
            float(np.mean([case.label == prior for case in test])),
        ),
        ShortcutScore(
            "metadata_only",
            _lookup_score(
                train, validation, lambda case: (min(9, int(case.age_scaled * 10)), case.sex_binary)
            ),
            _lookup_score(
                train, test, lambda case: (min(9, int(case.age_scaled * 10)), case.sex_binary)
            ),
        ),
        ShortcutScore(
            "sequence_length_only",
            _lookup_score(train, validation, lambda case: len(case.tokens)),
            _lookup_score(train, test, lambda case: len(case.tokens)),
        ),
        ShortcutScore(
            "positive_negative_count",
            _lookup_score(
                train,
                validation,
                lambda case: (int(np.sum(case.evidence > 0)), int(np.sum(case.evidence < 0))),
            ),
            _lookup_score(
                train,
                test,
                lambda case: (int(np.sum(case.evidence > 0)), int(np.sum(case.evidence < 0))),
            ),
        ),
    ]
    order_candidates = [
        ("first_token", lambda case: int(case.tokens[0])),
        ("last_token", lambda case: int(case.tokens[-1])),
        (
            "first_last_token",
            lambda case: (int(case.tokens[0]), int(case.tokens[-1])),
        ),
    ]
    scores.append(_select_lookup("order_only", train, validation, test, order_candidates))
    single_candidates = [
        (f"finding_{index}", lambda case, index=index: int(case.evidence[index]))
        for index in range(NeuroWorld.num_findings)
    ]
    single = _select_lookup("single_feature", train, validation, test, single_candidates)
    scores.append(single)
    ranked = sorted(
        single_candidates,
        key=lambda item: _lookup_score(train, validation, item[1]),
        reverse=True,
    )[:8]
    pair_candidates = [
        (
            f"{first_name}+{second_name}",
            lambda case, first=first, second=second: (first(case), second(case)),
        )
        for first_index, (first_name, first) in enumerate(ranked)
        for second_name, second in ranked[first_index + 1 :]
    ]
    scores.append(_select_lookup("depth_two_lookup", train, validation, test, pair_candidates))
    scores.append(
        ShortcutScore(
            "nearest_neighbor",
            nearest_neighbor_accuracy(train, validation),
            nearest_neighbor_accuracy(train, test),
        )
    )
    return scores


def audit_dataset(
    train: Sequence[Case], validation: Sequence[Case], test: Sequence[Case]
) -> dict[str, object]:
    labels = np.asarray([case.label for case in (*train, *validation, *test)])
    counts = np.bincount(labels, minlength=NeuroWorld.num_diagnoses)
    expected = labels.size / NeuroWorld.num_diagnoses
    return {
        "case_consistency_errors": case_consistency_errors((*train, *validation, *test)),
        "train_validation_duplicate_rate": duplicate_rate(train, validation),
        "train_test_duplicate_rate": duplicate_rate(train, test),
        "maximum_class_prevalence_deviation": float(
            np.max(np.abs(counts - expected)) / labels.size
        ),
        "shortcuts": [score.__dict__ for score in shortcut_scores(train, validation, test)],
    }
