import torch

from neuroworld import NeuroWorld
from qneuro.data import collate_cases
from qneuro.model_factory import build_model, parameter_count
from qneuro.models import DiagnosticDensityDynamics, EnergyAttractorState

ADVANCED_MODELS = (
    "logistic",
    "complex_mlp",
    "real_accumulator",
    "complex_accumulator",
    "state_space",
    "hopfield",
    "graph_network",
    "coupled_tensor",
    "energy_attractor",
    "adaptive_attractor",
    "hamiltonian",
    "dissipative",
    "hybrid_dynamics",
    "density_dynamics",
    "complex_magnitude_readout",
    "complex_no_negative",
    "density_rank1",
    "density_rank4",
)


def test_advanced_models_have_finite_outputs_and_representations() -> None:
    batch = collate_cases(NeuroWorld().generate(6, seed=51))
    for name in ADVANCED_MODELS:
        model, metadata = build_model(name, 20_000, rank=2, max_length=40, step_size=0.25)
        logits = model(**batch)
        representation = model.encode(**batch)
        assert logits.shape == (6, 20), name
        assert representation.shape[0] == 6, name
        assert bool(torch.isfinite(logits).all()), name
        assert bool(torch.isfinite(representation).all()), name
        assert metadata["parameter_count"] == parameter_count(model)


def test_density_invariants_and_attractor_trajectory() -> None:
    batch = collate_cases(NeuroWorld().generate(5, seed=52))
    density_model, _ = build_model("density_dynamics", 20_000, rank=2, max_length=40, step_size=0.2)
    assert isinstance(density_model, DiagnosticDensityDynamics)
    factor = density_model.evolve_factor(batch["tokens"], batch["mask"], batch["vector"])
    density = density_model.density_matrix(factor)
    assert torch.allclose(density, density.conj().transpose(-2, -1), atol=1e-5)
    assert torch.allclose(
        density.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1), torch.ones(5), atol=1e-5
    )
    eigenvalues = torch.linalg.eigvalsh(density)
    assert bool((eigenvalues >= -1e-5).all())

    attractor, _ = build_model("adaptive_attractor", 20_000, rank=2, max_length=40, step_size=0.25)
    assert isinstance(attractor, EnergyAttractorState)
    diagnostics = attractor.trajectory_diagnostics(batch["tokens"], batch["mask"], batch["vector"])
    assert diagnostics["velocity"].shape == (5, attractor.steps)
    assert diagnostics["entropy"].shape == (5, attractor.steps)


def test_hard_halt_matches_final_state_when_threshold_is_unreachable() -> None:
    batch = collate_cases(NeuroWorld().generate(5, seed=54))
    model = EnergyAttractorState(80, 12, 20, steps=5, adaptive=True)
    _, final_logits = model.trajectory(batch["tokens"], batch["mask"], batch["vector"])
    hard_logits, steps = model.hard_halt(
        batch["tokens"], batch["mask"], batch["vector"], velocity_threshold=-1.0
    )
    assert torch.allclose(hard_logits, final_logits[-1])
    assert bool(steps.eq(5).all())


def test_hard_halt_executes_minimum_steps_at_large_threshold() -> None:
    batch = collate_cases(NeuroWorld().generate(5, seed=55))
    model = EnergyAttractorState(80, 12, 20, steps=5, adaptive=True)
    logits, steps = model.hard_halt(
        batch["tokens"],
        batch["mask"],
        batch["vector"],
        velocity_threshold=1e6,
        min_steps=2,
    )
    assert logits.shape == (5, 20)
    assert bool(steps.eq(2).all())


def test_advanced_model_gradients_are_finite() -> None:
    batch = collate_cases(NeuroWorld().generate(4, seed=53))
    for name in ADVANCED_MODELS:
        model, _ = build_model(name, 8_000, rank=1, max_length=40, step_size=0.2)
        loss = torch.nn.functional.cross_entropy(model(**batch), batch["label"])
        auxiliary_loss = getattr(model, "auxiliary_loss", None)
        if auxiliary_loss is not None:
            loss = loss + auxiliary_loss()
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        assert gradients, name
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients), name
