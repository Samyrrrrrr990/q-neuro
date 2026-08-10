"""Model families used in controlled Q-Neuro comparisons."""

from qneuro.models.baselines import EvidenceMLP, TinyGRU, TinyTransformer
from qneuro.models.operators import (
    ComplexOperatorState,
    RealOperatorState,
    TwoChannelRealOperatorState,
)

__all__ = [
    "ComplexOperatorState",
    "EvidenceMLP",
    "RealOperatorState",
    "TinyGRU",
    "TinyTransformer",
    "TwoChannelRealOperatorState",
]
