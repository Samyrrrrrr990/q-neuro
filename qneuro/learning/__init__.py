"""Experimental learning laws for Q-Neuro states."""

from qneuro.learning.rules import (
    AuxiliaryTrainingModel,
    apply_pcgrad,
    apply_phase_gradient,
    fit_centroid_readout,
    local_plasticity_epoch,
    multi_objective_losses,
)

__all__ = [
    "AuxiliaryTrainingModel",
    "apply_pcgrad",
    "apply_phase_gradient",
    "fit_centroid_readout",
    "local_plasticity_epoch",
    "multi_objective_losses",
]
