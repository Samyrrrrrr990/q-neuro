"""DISCOVERY-002 stages 3-4: the smallest experiment that can falsify basin coherence.

Lane B. Nothing here is a claim. See `docs/LANE_POLICY.md`.

The hypothesis under test is that a trajectory-aware, representation-invariant basin quantity
measured **before training finishes** predicts final functional agreement among perturbed learners
better than loss statistics. Falsifier 1 of the preregistration kills it if loss does as well, so the
experiment is built around that comparison and reports it whichever way it comes out.

Phenotype and metric are the ones declared in `docs/DISCOVERY_002_PREREGISTRATION.md` section 2:
logits on a frozen audit batch, compared in max-norm.

Two perturbation families are run separately, because they make the raw-parameter baseline mean very
different things:

* ``symmetry`` — hidden-unit permutations. Raw parameter distance is enormous by construction while
  functional distance is ~0, so a raw-parameter baseline is uninformative almost tautologically.
* ``noise`` — small parameter perturbations and reordered minibatches. Here raw parameter distance
  is a genuine competitor, and this is the honest test of whether the quotient view earns its place.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from qneuro.equivalence.maps import HiddenUnitPermutationMap
from qneuro.equivalence.microcosms import HomogeneousMLP, batch_stream, fixed_batch

#: Declared in advance: the tolerance at which two phenotypes count as the same functional region.
#: Sensitivity is reported across this whole list; no single value is privileged in Lane B.
EPSILON_GRID = (0.05, 0.1, 0.25, 0.5)


def _audit_batch(seed: int, features: int, classes: int) -> torch.Tensor:
    return fixed_batch(seed=seed + 31337, features=features, batch_size=64, classes=classes)["x"]


def _phenotype(model: torch.nn.Module, audit: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(audit).detach().clone()


def _pairwise_spread(phenotypes: list[torch.Tensor]) -> float:
    """Median pairwise max-norm logit distance. Median, not mean, so one outlier cannot carry it."""

    distances = [
        float((phenotypes[i] - phenotypes[j]).abs().max())
        for i in range(len(phenotypes))
        for j in range(i + 1, len(phenotypes))
    ]
    return sorted(distances)[len(distances) // 2] if distances else 0.0


def _coherence(phenotypes: list[torch.Tensor], epsilon: float) -> float:
    """Fraction of pairs whose phenotypes agree within ``epsilon``."""

    total = same = 0
    for i in range(len(phenotypes)):
        for j in range(i + 1, len(phenotypes)):
            total += 1
            same += float((phenotypes[i] - phenotypes[j]).abs().max()) < epsilon
    return same / total if total else 1.0


def _parameter_spread(models: list[torch.nn.Module]) -> float:
    vectors = [torch.cat([p.detach().flatten() for p in m.parameters()]) for m in models]
    distances = [
        float(torch.linalg.vector_norm(vectors[i] - vectors[j]))
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    return sorted(distances)[len(distances) // 2] if distances else 0.0


def _make_replicas(
    base: HomogeneousMLP, family: str, count: int, seed: int, noise: float
) -> list[HomogeneousMLP]:
    """Perturbed representatives of the same developmental starting condition."""

    replicas: list[HomogeneousMLP] = []
    for index in range(count):
        replica = copy.deepcopy(base)
        if index == 0:
            replicas.append(replica)
            continue
        if family == "symmetry":
            mapping = HiddenUnitPermutationMap.random(base, seed=seed * 100 + index)
            replica = mapping.build_target(base)
        else:
            generator = torch.Generator().manual_seed(seed * 100 + index)
            with torch.no_grad():
                for parameter in replica.parameters():
                    parameter.add_(
                        torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
                        * noise
                    )
        replicas.append(replica)
    return replicas


def run_cell(
    width: int,
    seed: int,
    scale: float,
    learning_rate: float,
    family: str,
    replicas: int = 12,
    steps: int = 300,
    early: int = 75,
    window: int = 25,
    noise: float = 1e-3,
    features: int = 6,
    classes: int = 4,
) -> dict[str, Any]:
    """One configuration: many perturbed replicas, measured early and at the end."""

    torch.manual_seed(seed)
    base = HomogeneousMLP(features, width, classes).double()
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.mul_(scale)

    audit = _audit_batch(seed, features, classes).double()
    models = _make_replicas(base, family, replicas, seed, noise)
    optimizers = [torch.optim.SGD(m.parameters(), lr=learning_rate) for m in models]

    # Each replica gets its own minibatch order: a real source of developmental variation.
    streams = [
        batch_stream(seed * 7 + index, features, steps, batch_size=64, classes=classes)
        for index in range(len(models))
    ]

    losses_start: list[float] = []
    losses_early: list[float] = []
    spread_window = 0.0
    coherence_early: dict[float, float] = {}
    spread_early = 0.0
    parameter_spread_early = 0.0

    for step in range(steps):
        if step == window:
            spread_window = _pairwise_spread([_phenotype(m, audit) for m in models])
        if step == early:
            phenotypes = [_phenotype(m, audit) for m in models]
            spread_early = _pairwise_spread(phenotypes)
            coherence_early = {e: _coherence(phenotypes, e) for e in EPSILON_GRID}
            parameter_spread_early = _parameter_spread(models)

        for index, (model, optimizer) in enumerate(zip(models, optimizers, strict=True)):
            batch = streams[index][step]
            batch = {"x": batch["x"].double(), "y": batch["y"]}
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(batch["x"]), batch["y"])
            if not torch.isfinite(loss):
                return {"failed": True}
            if step == 0:
                losses_start.append(float(loss.detach()))
            if step == early:
                losses_early.append(float(loss.detach()))
            loss.backward()
            optimizer.step()

    final = [_phenotype(m, audit) for m in models]
    if not all(torch.isfinite(p).all() for p in final):
        return {"failed": True}

    final_spread = _pairwise_spread(final)
    mean_loss_early = sum(losses_early) / len(losses_early) if losses_early else float("nan")
    mean_loss_start = sum(losses_start) / len(losses_start) if losses_start else float("nan")

    return {
        "failed": False,
        "width": width,
        "seed": seed,
        "scale": scale,
        "learning_rate": learning_rate,
        "family": family,
        # --- target ---
        "final_functional_spread": final_spread,
        "final_coherence": {e: _coherence(final, e) for e in EPSILON_GRID},
        # --- candidate early predictors (trajectory-aware, quotient-side) ---
        "basin_coherence_early": coherence_early,
        "functional_spread_early": spread_early,
        "functional_contraction": (
            spread_early / spread_window if spread_window > 0 else float("nan")
        ),
        # --- baselines ---
        "loss_early": mean_loss_early,
        "loss_start": mean_loss_start,
        "loss_decrease": mean_loss_start - mean_loss_early,
        "parameter_spread_early": parameter_spread_early,
        "learning_rate_baseline": learning_rate,
    }
