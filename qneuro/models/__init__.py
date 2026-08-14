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
from qneuro.models.baselines import EvidenceMLP, TinyGRU, TinyTransformer
from qneuro.models.equivalent import ExactRealBlockOperatorState
from qneuro.models.operators import (
    ComplexMagnitudeReadoutOperator,
    ComplexNoNegativeEvidenceOperator,
    ComplexOperatorState,
    RealOperatorState,
    TwoChannelRealOperatorState,
)

__all__ = [
    "ComplexEvidenceAccumulator",
    "ComplexEvidenceMLP",
    "ComplexMagnitudeReadoutOperator",
    "ComplexNoNegativeEvidenceOperator",
    "ComplexOperatorState",
    "CoupledTensorState",
    "DiagnosticDensityDynamics",
    "DiagonalStateSpace",
    "EnergyAttractorState",
    "EvidenceGraphNetwork",
    "EvidenceMLP",
    "ExactRealBlockOperatorState",
    "HamiltonianDissipativeState",
    "LogisticEvidence",
    "ModernHopfieldMemory",
    "RealEvidenceAccumulator",
    "RealOperatorState",
    "TinyGRU",
    "TinyTransformer",
    "TwoChannelRealOperatorState",
]
