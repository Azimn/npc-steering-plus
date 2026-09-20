"""Persistent, model-independent organism state."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class AxisSpec:
    name: str
    lo: float
    hi: float
    baseline: float
    tau_s: float
    drift_per_s: float = 0.0

    def __post_init__(self) -> None:
        if self.lo >= self.hi:
            raise ValueError(f"Axis {self.name!r}: lo must be less than hi")
        if not self.lo <= self.baseline <= self.hi:
            raise ValueError(f"Axis {self.name!r}: baseline outside bounds")
        if self.tau_s <= 0:
            raise ValueError(f"Axis {self.name!r}: tau_s must be positive")

    def clamp(self, value: float) -> float:
        return max(self.lo, min(self.hi, float(value)))


DEFAULT_AXIS_SPECS: dict[str, AxisSpec] = {
    "valence": AxisSpec("valence", -1.0, 1.0, 0.0, 90.0),
    "arousal": AxisSpec("arousal", -1.0, 1.0, 0.0, 60.0),
    "dominance": AxisSpec("dominance", -1.0, 1.0, 0.0, 90.0),
    "stress": AxisSpec("stress", 0.0, 1.0, 0.0, 300.0),
    "fatigue": AxisSpec("fatigue", 0.0, 1.0, 0.05, 21_600.0, 0.000035),
    "hunger": AxisSpec("hunger", 0.0, 1.0, 0.10, 28_800.0, 0.000025),
    "pain": AxisSpec("pain", 0.0, 1.0, 0.0, 900.0),
    "threat": AxisSpec("threat", 0.0, 1.0, 0.0, 45.0),
    "affiliation_need": AxisSpec("affiliation_need", 0.0, 1.0, 0.20, 14_400.0, 0.000010),
    "curiosity": AxisSpec("curiosity", 0.0, 1.0, 0.55, 3_600.0),
    "competence_need": AxisSpec("competence_need", 0.0, 1.0, 0.25, 7_200.0),
}


@dataclass
class OrganismState:
    organism_id: str
    values: dict[str, float] = field(default_factory=dict)
    age_s: float = 0.0
    revision: int = 0
    schema_version: int = 1
    specs: dict[str, AxisSpec] = field(default_factory=lambda: dict(DEFAULT_AXIS_SPECS), repr=False)

    def __post_init__(self) -> None:
        if not self.organism_id.strip():
            raise ValueError("organism_id must be non-empty")
        merged: dict[str, float] = {}
        for name, spec in self.specs.items():
            merged[name] = spec.clamp(self.values.get(name, spec.baseline))
        unknown = set(self.values) - set(self.specs)
        if unknown:
            raise KeyError(f"Unknown organism axes: {sorted(unknown)}")
        self.values = merged
        self.age_s = max(0.0, float(self.age_s))
        self.revision = max(0, int(self.revision))

    def get(self, axis: str) -> float:
        return self.values[axis]

    def set(self, axis: str, value: float) -> None:
        self.values[axis] = self.specs[axis].clamp(value)
        self.revision += 1

    def apply(self, delta: Mapping[str, float]) -> None:
        for axis, amount in delta.items():
            if axis not in self.specs:
                raise KeyError(f"Unknown organism axis: {axis}")
            self.values[axis] = self.specs[axis].clamp(self.values[axis] + float(amount))
        if delta:
            self.revision += 1

    def step(self, dt_s: float) -> None:
        dt_s = float(dt_s)
        if dt_s < 0:
            raise ValueError("dt_s must be non-negative")
        if dt_s == 0:
            return
        for name, spec in self.specs.items():
            x = self.values[name]
            recovered = spec.baseline + (x - spec.baseline) * math.exp(-dt_s / spec.tau_s)
            self.values[name] = spec.clamp(recovered + spec.drift_per_s * dt_s)
        self.age_s += dt_s
        self.revision += 1

    def snapshot(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "organism_id": self.organism_id,
            "age_s": self.age_s,
            "revision": self.revision,
            "values": dict(self.values),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True) + "\n")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path, *, specs: Mapping[str, AxisSpec] | None = None) -> "OrganismState":
        blob = json.loads(Path(path).read_text())
        if blob.get("schema_version") != 1:
            raise ValueError(f"Unsupported organism schema_version: {blob.get('schema_version')!r}")
        return cls(
            organism_id=blob["organism_id"],
            values=blob.get("values", {}),
            age_s=blob.get("age_s", 0.0),
            revision=blob.get("revision", 0),
            schema_version=1,
            specs=dict(specs or DEFAULT_AXIS_SPECS),
        )

    def describe(self) -> str:
        ordered = ", ".join(f"{k}={self.values[k]:+.3f}" for k in sorted(self.values))
        return f"{self.organism_id} age={self.age_s:.1f}s rev={self.revision}: {ordered}"
