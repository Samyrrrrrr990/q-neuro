"""Active evidence-acquisition policies and evidence-efficiency metrics."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from neuroworld import Case, NeuroWorld
from qneuro.data import collate_cases


@dataclass(frozen=True)
class ActiveStep:
    query: int
    prediction: int
    confidence: float
    target_probability: float
    entropy: float


def canonicalize_case(case: Case) -> Case:
    """Remove temporal-order information while retaining signed evidence and demographics."""

    findings = np.flatnonzero(case.evidence)
    tokens = findings + (case.evidence[findings] < 0) * NeuroWorld.num_findings
    return replace(case, tokens=tokens.astype(np.int64), is_order_dependent=False)


def partial_case(case: Case, revealed: dict[int, int]) -> Case:
    """Construct a canonical partial observation from queried finding outcomes."""

    evidence = np.zeros(NeuroWorld.num_findings, dtype=np.int8)
    for finding, value in revealed.items():
        if not 0 <= finding < NeuroWorld.num_findings:
            raise ValueError(f"finding index out of range: {finding}")
        if value not in (-1, 1):
            raise ValueError("queried evidence must be positive or negative")
        evidence[finding] = value
    findings = np.flatnonzero(evidence)
    tokens = findings + (evidence[findings] < 0) * NeuroWorld.num_findings
    return replace(
        case,
        evidence=evidence,
        tokens=tokens.astype(np.int64),
        is_order_dependent=False,
        order_evidence_complete=True,
    )


def estimate_positive_likelihoods(cases: list[Case], smoothing: float = 1.0) -> torch.Tensor:
    """Estimate P(finding positive | diagnosis) from observed training evidence only."""

    positive = torch.full(
        (NeuroWorld.num_diagnoses, NeuroWorld.num_findings), smoothing, dtype=torch.float64
    )
    total = torch.full_like(positive, 2.0 * smoothing)
    for case in cases:
        observed = np.flatnonzero(case.evidence)
        total[case.label, observed] += 1.0
        positive_findings = observed[case.evidence[observed] > 0]
        positive[case.label, positive_findings] += 1.0
    return (positive / total).float()


def global_information_order(cases: list[Case], smoothing: float = 0.5) -> list[int]:
    """Rank binary findings by plug-in mutual information with diagnosis."""

    information: list[float] = []
    for finding in range(NeuroWorld.num_findings):
        counts = np.full((NeuroWorld.num_diagnoses, 2), smoothing, dtype=np.float64)
        for case in cases:
            value = int(case.evidence[finding])
            if value:
                counts[case.label, int(value > 0)] += 1.0
        joint = counts / counts.sum()
        label_marginal = joint.sum(axis=1, keepdims=True)
        value_marginal = joint.sum(axis=0, keepdims=True)
        mutual_information = np.sum(
            joint * np.log(joint / np.clip(label_marginal * value_marginal, 1e-12, None))
        )
        information.append(float(mutual_information))
    return sorted(range(NeuroWorld.num_findings), key=information.__getitem__, reverse=True)


@torch.no_grad()
def predict_cases(model: torch.nn.Module, cases: list[Case], device: torch.device) -> torch.Tensor:
    batch = {key: value.to(device) for key, value in collate_cases(cases).items()}
    return model(**batch).detach().float().cpu()


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)


@torch.no_grad()
def expected_information_gain_query(
    model: torch.nn.Module,
    full_case: Case,
    revealed: dict[int, int],
    current_probabilities: torch.Tensor,
    positive_likelihoods: torch.Tensor,
    device: torch.device,
) -> int:
    """Choose the query minimizing model entropy under positive/negative counterfactuals."""

    candidates = [finding for finding in range(NeuroWorld.num_findings) if finding not in revealed]
    variants: list[Case] = []
    for finding in candidates:
        variants.append(partial_case(full_case, {**revealed, finding: 1}))
        variants.append(partial_case(full_case, {**revealed, finding: -1}))
    probabilities = torch.softmax(predict_cases(model, variants, device), dim=-1).reshape(
        len(candidates), 2, -1
    )
    outcome_positive = current_probabilities @ positive_likelihoods[:, candidates]
    expected_entropy = outcome_positive * _entropy(probabilities[:, 0]) + (
        1.0 - outcome_positive
    ) * _entropy(probabilities[:, 1])
    return candidates[int(expected_entropy.argmin())]


def active_trajectory(
    model: torch.nn.Module,
    full_case: Case,
    strategy: str,
    max_queries: int,
    fixed_order: list[int],
    positive_likelihoods: torch.Tensor,
    random_seed: int,
    device: torch.device,
) -> list[ActiveStep]:
    """Reveal one binary finding per step and record the actual predictive trajectory."""

    if not 0 < max_queries <= NeuroWorld.num_findings:
        raise ValueError("max_queries must be in [1, num_findings]")
    if strategy not in {"random", "fixed_information", "expected_information_gain"}:
        raise ValueError(f"unknown strategy: {strategy}")
    rng = np.random.default_rng(random_seed)
    random_order = rng.permutation(NeuroWorld.num_findings).tolist()
    revealed: dict[int, int] = {}
    steps: list[ActiveStep] = []
    current_probabilities: torch.Tensor | None = None
    for step_index in range(max_queries):
        if strategy == "random":
            query = random_order[step_index]
        elif strategy == "fixed_information" or step_index == 0:
            query = next(finding for finding in fixed_order if finding not in revealed)
        else:
            if current_probabilities is None:
                raise RuntimeError("information-gain policy requires a current state")
            query = expected_information_gain_query(
                model,
                full_case,
                revealed,
                current_probabilities,
                positive_likelihoods,
                device,
            )
        revealed[query] = int(full_case.evidence[query])
        logits = predict_cases(model, [partial_case(full_case, revealed)], device)[0]
        current_probabilities = torch.softmax(logits, dim=-1)
        confidence, prediction = current_probabilities.max(dim=-1)
        steps.append(
            ActiveStep(
                query=int(query),
                prediction=int(prediction),
                confidence=float(confidence),
                target_probability=float(current_probabilities[full_case.label]),
                entropy=float(_entropy(current_probabilities[None])[0]),
            )
        )
    return steps


def aggregate_active_trajectories(
    trajectories: list[tuple[int, list[ActiveStep]]], confidence_threshold: float
) -> dict[str, object]:
    """Aggregate accuracy curves and a penalized confident-resolution time."""

    if not trajectories:
        raise ValueError("trajectories must not be empty")
    query_count = len(trajectories[0][1])
    if any(len(steps) != query_count for _, steps in trajectories):
        raise ValueError("all trajectories must have the same query budget")
    curves: list[dict[str, float]] = []
    query_frequency = np.zeros(NeuroWorld.num_findings, dtype=np.int64)
    resolution_times: list[int] = []
    for target, steps in trajectories:
        resolved = next(
            (
                index + 1
                for index, step in enumerate(steps)
                if step.prediction == target and step.confidence >= confidence_threshold
            ),
            query_count + 1,
        )
        resolution_times.append(resolved)
        for step in steps:
            query_frequency[step.query] += 1
    for index in range(query_count):
        step_values = [steps[index] for _, steps in trajectories]
        targets = [target for target, _ in trajectories]
        curves.append(
            {
                "queries": float(index + 1),
                "evidence_fraction": float((index + 1) / NeuroWorld.num_findings),
                "accuracy": float(
                    np.mean(
                        [step.prediction == target for step, target in zip(step_values, targets)]
                    )
                ),
                "nll": float(
                    np.mean([-np.log(max(step.target_probability, 1e-12)) for step in step_values])
                ),
                "mean_confidence": float(np.mean([step.confidence for step in step_values])),
                "mean_entropy": float(np.mean([step.entropy for step in step_values])),
            }
        )
    resolved = np.asarray(resolution_times) <= query_count
    return {
        "curve": curves,
        "accuracy_auc": float(np.mean([point["accuracy"] for point in curves])),
        "final_accuracy": curves[-1]["accuracy"],
        "final_nll": curves[-1]["nll"],
        "resolution_rate": float(resolved.mean()),
        "mean_queries_to_resolution_penalized": float(np.mean(resolution_times)),
        "median_queries_to_resolution_penalized": float(np.median(resolution_times)),
        "query_frequency": query_frequency.tolist(),
    }
