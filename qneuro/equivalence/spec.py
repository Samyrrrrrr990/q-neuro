"""Equivalence levels, transport levels, domain restrictions, and map specifications.

Frozen by `docs/ML2_PREREGISTRATION_001.md` section 2. A claim that does not state its level and
its declared domain is not a claim this package will serialize.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

# E0 is strongest. Lower rank means a stronger guarantee.
_LEVEL_RANK = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 4, "FAILED": 5}

#: Levels that assert exactness over every representable input, so they cannot coexist with an
#: excluded region. See EQUIVALENCE_SCIENCE_AMENDMENT_001 section 3.4.
GLOBALLY_EXACT_LEVELS = frozenset({"E0", "E1"})


class EquivalenceLevel(str, Enum):
    """Forward-equivalence strength, strongest first."""

    E0 = "E0"
    """Algebraic equivalence by symbolic identity."""

    E1 = "E1"
    """Exact finite-precision forward equivalence for all representable inputs."""

    E2 = "E2"
    """Tested forward equivalence on a deterministic adversarial audit suite."""

    E3 = "E3"
    """Distributional predictive equivalence within a prespecified tolerance."""

    E4 = "E4"
    """Task-metric equivalence only."""

    FAILED = "FAILED"
    """Certification attempted and refused."""

    def is_at_least(self, other: EquivalenceLevel) -> bool:
        """True when this level is at least as strong as ``other``."""

        if self is EquivalenceLevel.FAILED:
            return False
        return _LEVEL_RANK[self.value] <= _LEVEL_RANK[other.value]

    def is_at_least_e3(self) -> bool:
        return self.is_at_least(EquivalenceLevel.E3)

    @property
    def is_globally_exact(self) -> bool:
        return self.value in GLOBALLY_EXACT_LEVELS


class TransportLevel(IntEnum):
    """How much of the training system has been transported across the map."""

    T0 = 0
    """No coupling beyond nominal hyperparameters."""

    T1 = 1
    """Matched data order and random seed."""

    T2 = 2
    """Mapped initialization and matched minibatches."""

    T3 = 3
    """Mapped initialization, gradients, and regularization."""

    T4 = 4
    """Conjugate discrete update maps including optimizer state."""

    T5 = 5
    """Matched stopping, selection, search, precision, and resource contracts."""


@dataclass(frozen=True)
class DomainRestriction:
    """An explicitly excluded region of the input or parameter domain.

    Recording this is mandatory whenever exactness fails somewhere. Gate A of the preregistration
    requires downgrades to be declared, never hidden.
    """

    description: str
    excluded: str
    radius: float | None = None
    dtype: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "description": self.description,
            "excluded": self.excluded,
            "radius": self.radius,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class MapSpec:
    """Static declaration of what a map claims, checked at construction."""

    name: str
    family: str
    declared_level: EquivalenceLevel
    invertible: bool
    domain: DomainRestriction | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.domain is not None and self.declared_level.is_globally_exact:
            raise ValueError(
                f"{self.name!r} declares {self.declared_level.value}, which asserts exactness for "
                "every representable input, but it also carries a domain restriction. Declare E2 "
                "or weaker, or remove the domain restriction."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "declared_level": self.declared_level.value,
            "invertible": self.invertible,
            "domain": None if self.domain is None else self.domain.as_dict(),
            "notes": self.notes,
        }
