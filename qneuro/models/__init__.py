"""Model families used in controlled Q-Neuro comparisons."""

from qneuro.models.advanced import (
    ComplexEvidenceMLP,
    CoupledTensorState,
    DiagnosticDensityDynamics,
    DiagonalStateSpace,
    EnergyAttractorState,
    EvidenceGraphNetwork,
    HamiltonianDissipativeState,
    LogisticEvidence,
    ModernHopfieldMemory,
)
from qneuro.models.baselines import EvidenceMLP, TinyGRU, TinyTransformer
from qneuro.models.operators import (
    ComplexOperatorState,
    RealOperatorState,
    TwoChannelRealOperatorState,
)

__all__ = [
    "ComplexEvidenceMLP",
    "ComplexOperatorState",
    "CoupledTensorState",
    "DiagnosticDensityDynamics",
    "DiagonalStateSpace",
    "EnergyAttractorState",
    "EvidenceGraphNetwork",
    "EvidenceMLP",
    "HamiltonianDissipativeState",
    "LogisticEvidence",
    "ModernHopfieldMemory",
    "RealOperatorState",
    "TinyGRU",
    "TinyTransformer",
    "TwoChannelRealOperatorState",
]
