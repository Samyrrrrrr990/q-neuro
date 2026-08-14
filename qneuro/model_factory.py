"""Construct models with transparent real-scalar parameter budgeting."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from neuroworld import NeuroWorld
from qneuro.models import (
    AmbiguityAwareComplexOperator,
    AmbiguityAwareRealOperator,
    CausalTransformer,
    CommutatorPenaltyComplexOperator,
    CommutingComplexOperatorState,
    ComplexEvidenceAccumulator,
    ComplexEvidenceMLP,
    ComplexMagnitudeReadoutOperator,
    ComplexNoNegativeEvidenceOperator,
    ComplexOperatorState,
    CoupledTensorState,
    DenseRealMatrixRecurrence,
    DiagnosticDensityDynamics,
    DiagonalStateSpace,
    EnergyAttractorState,
    EvidenceGraphNetwork,
    EvidenceMLP,
    ExactRealBlockOperatorState,
    FixedRandomComplexOperator,
    FixedTwoStateAttractor,
    FrozenReadoutComplexOperator,
    HamiltonianDissipativeState,
    LogisticEvidence,
    MagnitudeDestroyedOperator,
    ModernHopfieldMemory,
    NoConjugationComplexOperator,
    NoncommutativeRealOperator,
    OrthogonalRealRecurrence,
    PhaseDestroyedTrainingOperator,
    RealEvidenceAccumulator,
    RealOperatorState,
    RealPolarOperatorState,
    RealRotationBlockOperator,
    ResidualGatedRecurrent,
    TinyGRU,
    TinyLSTM,
    TinyRNN,
    TinyTransformer,
    TwoChannelRealOperatorState,
)


def parameter_count(model: nn.Module) -> int:
    """Count stored real scalars; complex entries count as two.

    Frozen mechanism controls separately report ``trainable_parameter_count``.
    """

    count = 0
    for parameter in model.parameters():
        multiplier = 2 if parameter.is_complex() else 1
        count += multiplier * parameter.numel()
    return count


def _nearest_width(
    builder: Callable[[int], nn.Module], candidates: list[int], budget: int
) -> tuple[nn.Module, int]:
    models = [(builder(width), width) for width in candidates]
    return min(models, key=lambda item: abs(parameter_count(item[0]) - budget))


def build_model(
    name: str,
    parameter_budget: int,
    rank: int,
    max_length: int,
    step_size: float,
) -> tuple[nn.Module, dict[str, int | float | str]]:
    if name == "mlp":
        model, width = _nearest_width(
            lambda value: EvidenceMLP(82, value, NeuroWorld.num_diagnoses),
            list(range(8, 513)),
            parameter_budget,
        )
    elif name == "logistic":
        model = LogisticEvidence(82, NeuroWorld.num_diagnoses)
        width = 0
    elif name == "complex_mlp":
        model, width = _nearest_width(
            lambda value: ComplexEvidenceMLP(82, value, NeuroWorld.num_diagnoses),
            list(range(4, 257)),
            parameter_budget,
        )
    elif name == "real_accumulator":
        model, width = _nearest_width(
            lambda value: RealEvidenceAccumulator(
                NeuroWorld.num_tokens, value, NeuroWorld.num_diagnoses
            ),
            list(range(4, 257)),
            parameter_budget,
        )
    elif name == "complex_accumulator":
        model, width = _nearest_width(
            lambda value: ComplexEvidenceAccumulator(
                NeuroWorld.num_tokens, value, NeuroWorld.num_diagnoses
            ),
            list(range(4, 161)),
            parameter_budget,
        )
    elif name == "transformer":
        model, width = _nearest_width(
            lambda value: TinyTransformer(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                NeuroWorld.num_diagnoses,
                value,
                max_length,
                feedforward_dim=max(8, round(2.4 * value)),
            ),
            list(range(8, 129, 4)),
            parameter_budget,
        )
    elif name == "causal_transformer":
        model, width = _nearest_width(
            lambda value: CausalTransformer(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                NeuroWorld.num_diagnoses,
                value,
                max_length,
                feedforward_dim=max(8, round(2.4 * value)),
            ),
            list(range(8, 129, 4)),
            parameter_budget,
        )
    elif name == "gru":
        model, width = _nearest_width(
            lambda value: TinyGRU(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                NeuroWorld.num_diagnoses,
                hidden_dim=value,
                embedding_dim=max(8, value // 2),
            ),
            list(range(8, 193)),
            parameter_budget,
        )
    elif name in {"vanilla_rnn", "lstm", "residual_gated_recurrence"}:
        model_class = {
            "vanilla_rnn": TinyRNN,
            "lstm": TinyLSTM,
            "residual_gated_recurrence": ResidualGatedRecurrent,
        }[name]
        model, width = _nearest_width(
            lambda value: model_class(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                NeuroWorld.num_diagnoses,
                hidden_dim=value,
                embedding_dim=max(8, value // 2),
            ),
            list(range(8, 257)),
            parameter_budget,
        )
    elif name == "dense_real_recurrence":
        model, width = _nearest_width(
            lambda value: DenseRealMatrixRecurrence(
                NeuroWorld.num_tokens,
                NeuroWorld.num_diagnoses,
                value,
                step_size,
                readout_dim=max(20, 2 * value - 1),
            ),
            list(range(4, 65)),
            parameter_budget,
        )
    elif name == "orthogonal_real_recurrence":
        model, width = _nearest_width(
            lambda value: OrthogonalRealRecurrence(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                NeuroWorld.num_diagnoses,
                value,
            ),
            list(range(4, 257)),
            parameter_budget,
        )
    elif name == "state_space":
        model, width = _nearest_width(
            lambda value: DiagonalStateSpace(
                NeuroWorld.num_tokens, value, NeuroWorld.num_diagnoses
            ),
            list(range(4, 257)),
            parameter_budget,
        )
    elif name == "commuting_operator":
        model, width = _nearest_width(
            lambda value: CommutingComplexOperatorState(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                value,
                NeuroWorld.num_diagnoses,
                step_size,
            ),
            list(range(4, 257)),
            parameter_budget,
        )
    elif name == "hopfield":
        model, width = _nearest_width(
            lambda value: ModernHopfieldMemory(
                NeuroWorld.num_tokens, value, NeuroWorld.num_diagnoses
            ),
            list(range(4, 129, 2)),
            parameter_budget,
        )
    elif name == "graph_network":
        model, width = _nearest_width(
            lambda value: EvidenceGraphNetwork(value, NeuroWorld.num_diagnoses),
            list(range(4, 161, 2)),
            parameter_budget,
        )
    elif name == "coupled_tensor":
        model, width = _nearest_width(
            lambda value: CoupledTensorState(82, value, NeuroWorld.num_diagnoses),
            list(range(4, 161)),
            parameter_budget,
        )
    elif name == "real_operator":
        model, width = _nearest_width(
            lambda value: RealOperatorState(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                value,
                rank,
                NeuroWorld.num_diagnoses,
                step_size,
            ),
            list(range(4, 257)),
            parameter_budget,
        )
    elif name == "complex_operator":
        model, width = _nearest_width(
            lambda value: ComplexOperatorState(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                value,
                rank,
                NeuroWorld.num_diagnoses,
                step_size,
            ),
            list(range(4, 129)),
            parameter_budget,
        )
    elif name in {
        "commutator_penalty_operator",
        "phase_destroyed_training",
        "magnitude_destroyed",
        "no_conjugation_operator",
        "fixed_random_complex_operator",
        "frozen_dynamics",
        "frozen_readout",
        "ambiguity_aware_complex",
    }:
        mechanism_class = {
            "commutator_penalty_operator": CommutatorPenaltyComplexOperator,
            "phase_destroyed_training": PhaseDestroyedTrainingOperator,
            "magnitude_destroyed": MagnitudeDestroyedOperator,
            "no_conjugation_operator": NoConjugationComplexOperator,
            "fixed_random_complex_operator": FixedRandomComplexOperator,
            "frozen_dynamics": FixedRandomComplexOperator,
            "frozen_readout": FrozenReadoutComplexOperator,
            "ambiguity_aware_complex": AmbiguityAwareComplexOperator,
        }[name]
        model, width = _nearest_width(
            lambda value: mechanism_class(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                value,
                rank,
                NeuroWorld.num_diagnoses,
                step_size,
            ),
            list(range(4, 129)),
            parameter_budget,
        )
    elif name == "exact_real_block_operator":
        model, width = _nearest_width(
            lambda value: ExactRealBlockOperatorState(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                value,
                rank,
                NeuroWorld.num_diagnoses,
                step_size,
            ),
            list(range(4, 129)),
            parameter_budget,
        )
    elif name == "real_polar_operator":
        model, width = _nearest_width(
            lambda value: RealPolarOperatorState(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                value,
                rank,
                NeuroWorld.num_diagnoses,
                step_size,
            ),
            list(range(4, 129)),
            parameter_budget,
        )
    elif name == "real_rotation_block_operator":
        model, width = _nearest_width(
            lambda value: RealRotationBlockOperator(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                value,
                NeuroWorld.num_diagnoses,
                step_size,
            ),
            list(range(4, 257)),
            parameter_budget,
        )
    elif name in {"complex_magnitude_readout", "complex_no_negative"}:
        model_class = (
            ComplexMagnitudeReadoutOperator
            if name == "complex_magnitude_readout"
            else ComplexNoNegativeEvidenceOperator
        )
        model, width = _nearest_width(
            lambda value: model_class(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                value,
                rank,
                NeuroWorld.num_diagnoses,
                step_size,
            ),
            list(range(4, 129)),
            parameter_budget,
        )
    elif name in {"two_channel_operator", "unrestricted_paired_real_operator"}:
        model, width = _nearest_width(
            lambda value: TwoChannelRealOperatorState(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                value,
                rank,
                NeuroWorld.num_diagnoses,
                step_size,
            ),
            list(range(4, 257)) if name == "two_channel_operator" else list(range(4, 257, 2)),
            parameter_budget,
        )
    elif name in {"noncommutative_real_operator", "ambiguity_aware_real"}:
        mechanism_class = (
            NoncommutativeRealOperator
            if name == "noncommutative_real_operator"
            else AmbiguityAwareRealOperator
        )
        if name == "noncommutative_real_operator":
            model, width = _nearest_width(
                lambda value: mechanism_class(
                    NeuroWorld.num_tokens,
                    NeuroWorld.num_diagnoses,
                    value,
                    step_size,
                    readout_dim=max(20, 2 * value - 1),
                ),
                list(range(4, 65)),
                parameter_budget,
            )
        else:
            model, width = _nearest_width(
                lambda value: mechanism_class(
                    NeuroWorld.num_tokens,
                    NeuroWorld.pad_token,
                    value,
                    rank,
                    NeuroWorld.num_diagnoses,
                    step_size,
                ),
                list(range(4, 257)),
                parameter_budget,
            )
    elif name == "fixed_two_state_attractor":
        model, width = _nearest_width(
            lambda value: FixedTwoStateAttractor(
                NeuroWorld.num_tokens,
                value,
                NeuroWorld.num_diagnoses,
                step_size,
            ),
            list(range(4, 257)),
            parameter_budget,
        )
    elif name in {"energy_attractor", "adaptive_attractor"}:
        model, width = _nearest_width(
            lambda value: EnergyAttractorState(
                NeuroWorld.num_tokens,
                value,
                NeuroWorld.num_diagnoses,
                steps=8 if name == "adaptive_attractor" else 6,
                step_size=step_size,
                adaptive=name == "adaptive_attractor",
            ),
            list(range(4, 145, 2)),
            parameter_budget,
        )
    elif name in {"hamiltonian", "dissipative", "hybrid_dynamics"}:
        model, width = _nearest_width(
            lambda value: HamiltonianDissipativeState(
                NeuroWorld.num_tokens,
                value,
                rank,
                NeuroWorld.num_diagnoses,
                step_size,
                coherent=name != "dissipative",
                dissipative=name != "hamiltonian",
            ),
            list(range(4, 129)),
            parameter_budget,
        )
    elif name in {"density_dynamics", "density_rank1", "density_rank2", "density_rank4"}:
        factor_rank = {
            "density_rank1": 1,
            "density_rank2": 2,
            "density_rank4": 4,
        }.get(name, 2)
        model = DiagnosticDensityDynamics(
            NeuroWorld.num_tokens,
            NeuroWorld.num_diagnoses,
            factor_rank=factor_rank,
            operator_rank=rank,
            step_size=step_size,
        )
        width = NeuroWorld.num_diagnoses
    else:
        raise ValueError(f"unknown model: {name}")
    if name in {
        "ambiguity_aware_complex",
        "commuting_operator",
        "commutator_penalty_operator",
        "complex_magnitude_readout",
        "complex_no_negative",
        "complex_operator",
        "exact_real_block_operator",
        "fixed_random_complex_operator",
        "fixed_two_state_attractor",
        "frozen_dynamics",
        "frozen_readout",
        "magnitude_destroyed",
        "no_conjugation_operator",
        "phase_destroyed_training",
        "real_polar_operator",
        "real_rotation_block_operator",
    }:
        state_real_dof = 2 * int(width)
    else:
        state_real_dof = int(width)
    return model, {
        "name": name,
        "width": width,
        "rank": rank
        if any(value in name for value in ("operator", "hamiltonian", "dynamics"))
        else 0,
        "parameter_count": parameter_count(model),
        "trainable_parameter_count": sum(
            (2 if parameter.is_complex() else 1) * parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "parameter_budget": parameter_budget,
        "state_real_dof": state_real_dof,
    }
