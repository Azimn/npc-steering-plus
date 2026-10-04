"""Interoceptive observation layer.

Separates authoritative physiological state from the organism's observation of
that state. The default sensor is exact; quantization and reproducible noise
can be enabled for sensitivity experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .state import PHYSIOLOGICAL_AXES, OrganismState


@dataclass(frozen=True)
class InteroceptiveObservation:
    values: Mapping[str, float]
    source_revision: int


@dataclass
class InteroceptiveSensor:
    quantization: float = 0.0
    noise_sd: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.quantization < 0.0:
            raise ValueError("quantization must be non-negative")
        if self.noise_sd < 0.0:
            raise ValueError("noise_sd must be non-negative")
        self._rng = np.random.default_rng(self.seed)

    def observe(self, state: OrganismState) -> InteroceptiveObservation:
        values: dict[str, float] = {}
        for axis in PHYSIOLOGICAL_AXES:
            value = state.get(axis)
            if self.noise_sd > 0.0:
                value += float(self._rng.normal(0.0, self.noise_sd))
            spec = state.specs[axis]
            value = spec.clamp(value)
            if self.quantization > 0.0:
                value = round(value / self.quantization) * self.quantization
                value = spec.clamp(value)
            values[axis] = float(value)
        return InteroceptiveObservation(values=values, source_revision=state.revision)
