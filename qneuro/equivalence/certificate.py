"""Machine-readable equivalence certificates.

A certificate is the authoritative statement of what a map guarantees. Class names are not
authoritative: `ExactRealBlockOperatorState` is named for its algebra, not for a proven level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from qneuro.equivalence.spec import DomainRestriction, EquivalenceLevel, TransportLevel


@dataclass(frozen=True)
class Certificate:
    """The output of an equivalence audit."""

    SCHEMA_VERSION = "1.0.0"

    source: str
    target: str
    map_name: str
    declared_level: EquivalenceLevel
    transport_level: TransportLevel
    transport_degenerate: bool
    dtype: str
    device: str
    residuals: dict[str, float]
    domain: DomainRestriction | None = None
    downgrades: tuple[tuple[str, str, str], ...] = ()
    known_failure_modes: tuple[str, ...] = ()
    notes: str = ""
    environment: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.declared_level, EquivalenceLevel):
            raise TypeError(
                "declared_level must be an EquivalenceLevel; a certificate without a declared "
                "level cannot be serialized."
            )
        if not isinstance(self.transport_level, TransportLevel):
            raise TypeError("transport_level must be a TransportLevel")
        if self.domain is not None and self.declared_level.is_globally_exact:
            raise ValueError(
                f"cannot certify {self.declared_level.value} alongside a domain restriction"
            )

    def downgrade(self, level: EquivalenceLevel, *, reason: str) -> Certificate:
        """Record a weakening of the claim. Downgrades accumulate and are never dropped."""

        if level.is_at_least(self.declared_level) and level is not self.declared_level:
            raise ValueError(
                f"{level.value} is stronger than the current {self.declared_level.value}; "
                "downgrade() may only weaken a claim."
            )
        return replace(
            self,
            declared_level=level,
            downgrades=(*self.downgrades, (self.declared_level.value, level.value, reason)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "source": self.source,
            "target": self.target,
            "map_name": self.map_name,
            "declared_level": self.declared_level.value,
            "transport_level": int(self.transport_level),
            "transport_degenerate": self.transport_degenerate,
            "dtype": self.dtype,
            "device": self.device,
            "residuals": dict(self.residuals),
            "domain": None if self.domain is None else self.domain.as_dict(),
            "downgrades": [list(entry) for entry in self.downgrades],
            "known_failure_modes": list(self.known_failure_modes),
            "notes": self.notes,
            "environment": dict(self.environment),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> Certificate:
        data = json.loads(payload)
        if data.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported certificate schema_version: {data.get('schema_version')}"
            )
        domain = data.get("domain")
        return cls(
            source=data["source"],
            target=data["target"],
            map_name=data["map_name"],
            declared_level=EquivalenceLevel(data["declared_level"]),
            transport_level=TransportLevel(int(data["transport_level"])),
            transport_degenerate=bool(data["transport_degenerate"]),
            dtype=data["dtype"],
            device=data["device"],
            residuals=dict(data["residuals"]),
            domain=None if domain is None else DomainRestriction(**domain),
            downgrades=tuple(tuple(entry) for entry in data.get("downgrades", ())),
            known_failure_modes=tuple(data.get("known_failure_modes", ())),
            notes=data.get("notes", ""),
            environment=dict(data.get("environment", {})),
        )
