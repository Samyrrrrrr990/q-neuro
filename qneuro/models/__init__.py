"""Model families used in controlled Q-Neuro comparisons."""

from qneuro.models.baselines import EvidenceMLP, TinyTransformer
from qneuro.models.operators import ComplexOperatorState, RealOperatorState

__all__ = ["ComplexOperatorState", "EvidenceMLP", "RealOperatorState", "TinyTransformer"]
