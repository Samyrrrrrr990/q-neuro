"""Equivalence compiler core.

Frozen interface for `docs/ML2_PREREGISTRATION_001.md`. The package exists to make one class of
error impossible to repeat: registering a control that shares the candidate's parameter
coordinates and then reading agreement with it as architectural evidence.
"""

from qneuro.equivalence.certificate import Certificate
from qneuro.equivalence.complex_real import ComplexToExactRealMap
from qneuro.equivalence.defects import predictive_divergence, residual_summary
from qneuro.equivalence.factorization import FactorizedToDenseMap
from qneuro.equivalence.maps import (
    HiddenUnitPermutationMap,
    IdentityMap,
    IndexShuffleMap,
    ParameterMap,
)
from qneuro.equivalence.native_complex import ComplexRealificationMap
from qneuro.equivalence.scaling import DiagonalScalingMap, HomogeneousScalingMap
from qneuro.equivalence.spec import (
    DomainRestriction,
    EquivalenceLevel,
    MapSpec,
    TransportLevel,
)
from qneuro.equivalence.transport import paired_training_divergence

__all__ = [
    "Certificate",
    "ComplexRealificationMap",
    "ComplexToExactRealMap",
    "DiagonalScalingMap",
    "DomainRestriction",
    "EquivalenceLevel",
    "FactorizedToDenseMap",
    "HiddenUnitPermutationMap",
    "HomogeneousScalingMap",
    "IdentityMap",
    "IndexShuffleMap",
    "MapSpec",
    "ParameterMap",
    "TransportLevel",
    "paired_training_divergence",
    "predictive_divergence",
    "residual_summary",
]
