"""Model families used in controlled Q-Neuro comparisons."""

from qneuro.models.advanced import (
    ComplexEvidenceAccumulator,
    ComplexEvidenceMLP,
    CoupledTensorState,
    DiagnosticDensityDynamics,
    DiagonalStateSpace,
    EnergyAttractorState,
    EvidenceGraphNetwork,
    HamiltonianDissipativeState,
    LogisticEvidence,
    ModernHopfieldMemory,
    RealEvidenceAccumulator,
)
from qneuro.models.baselines import (
    CausalTransformer,
    DenseRealMatrixRecurrence,
    EvidenceMLP,
    OrthogonalRealRecurrence,
    ResidualGatedRecurrent,
    TinyGRU,
    TinyLSTM,
    TinyRNN,
    TinyTransformer,
)
from qneuro.models.equivalent import (
    ExactRealBlockOperatorState,
    RealPolarOperatorState,
    RealRotationBlockOperator,
)
from qneuro.models.operators import (
    ComplexMagnitudeReadoutOperator,
    ComplexNoNegativeEvidenceOperator,
    ComplexOperatorState,
    RealOperatorState,
    TwoChannelRealOperatorState,
)

__all__ = [
    "CausalTransformer",
    "ComplexEvidenceAccumulator",
    "ComplexEvidenceMLP",
    "ComplexMagnitudeReadoutOperator",
    "ComplexNoNegativeEvidenceOperator",
    "ComplexOperatorState",
    "CoupledTensorState",
    "DenseRealMatrixRecurrence",
    "DiagnosticDensityDynamics",
    "DiagonalStateSpace",
    "EnergyAttractorState",
    "EvidenceGraphNetwork",
    "EvidenceMLP",
    "ExactRealBlockOperatorState",
    "HamiltonianDissipativeState",
    "LogisticEvidence",
    "ModernHopfieldMemory",
    "OrthogonalRealRecurrence",
    "RealEvidenceAccumulator",
    "RealOperatorState",
    "RealPolarOperatorState",
    "RealRotationBlockOperator",
    "ResidualGatedRecurrent",
    "TinyGRU",
    "TinyLSTM",
    "TinyRNN",
    "TinyTransformer",
    "TwoChannelRealOperatorState",
]
