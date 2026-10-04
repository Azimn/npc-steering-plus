"""Translation from model-independent organism state to model-specific axes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .state import PHYSIOLOGICAL_AXES, OrganismState


@dataclass(frozen=True)
class SteeringBinding:
    source_axis: str
    probe_axis: str
    gain: float
    center: float = 0.0
    deadband: float = 0.0
    max_abs_alpha: float | None = None
    enabled: bool = True
    feedback_gain: float = 0.0
    readback_center: float = 0.0
    readback_scale: float = 1.0

    def alpha_from_value(self, source_value: float) -> float:
        if not self.enabled:
            return 0.0
        offset = float(source_value) - self.center
        if abs(offset) <= self.deadband:
            return 0.0
        value = self.gain * offset
        if self.max_abs_alpha is not None:
            cap = abs(float(self.max_abs_alpha))
            value = max(-cap, min(cap, value))
        return float(value)

    def alpha(self, state: OrganismState) -> float:
        return self.alpha_from_value(state.get(self.source_axis))

    def readback_target(self, readback_value: float) -> float:
        return self.readback_center + self.readback_scale * float(readback_value)


@dataclass(frozen=True)
class SteeringProfile:
    model_id: str
    bindings: tuple[SteeringBinding, ...]
    profile_version: int = 2

    def alphas(
        self,
        state: OrganismState,
        *,
        observed_values: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for binding in self.bindings:
            if not binding.enabled:
                continue
            if binding.source_axis not in state.values:
                raise KeyError(f"Profile references unknown state axis {binding.source_axis!r}")

            if binding.source_axis in PHYSIOLOGICAL_AXES and observed_values is not None:
                if binding.source_axis not in observed_values:
                    raise KeyError(
                        f"Interoceptive observation missing physiological axis "
                        f"{binding.source_axis!r}"
                    )
                source_value = observed_values[binding.source_axis]
            else:
                source_value = state.get(binding.source_axis)

            value = binding.alpha_from_value(source_value)
            out[binding.probe_axis] = out.get(binding.probe_axis, 0.0) + value
        return out

    def feedback_delta(
        self,
        state: OrganismState,
        readback: Mapping[str, float],
    ) -> dict[str, float]:
        """Return bounded cognitive-state feedback.

        Primary physiology is world/body authoritative and may not be rewritten
        from model readback. This blocks loops where the LLM becomes less thirsty
        merely because its generated text looks less thirsty.
        """

        delta: dict[str, float] = {}
        for binding in self.bindings:
            if not binding.enabled or binding.feedback_gain <= 0.0:
                continue
            if binding.source_axis in PHYSIOLOGICAL_AXES:
                raise ValueError(
                    f"Model readback may not directly update physiological axis "
                    f"{binding.source_axis!r}"
                )
            if binding.probe_axis not in readback:
                continue
            target = binding.readback_target(readback[binding.probe_axis])
            current = state.get(binding.source_axis)
            delta[binding.source_axis] = (
                delta.get(binding.source_axis, 0.0)
                + binding.feedback_gain * (target - current)
            )
        return delta

    @classmethod
    def from_dict(cls, blob: dict) -> "SteeringProfile":
        version = int(blob.get("profile_version", 1))
        if version not in (1, 2):
            raise ValueError(f"Unsupported steering profile version: {version}")
        return cls(
            model_id=str(blob["model_id"]),
            profile_version=2,
            bindings=tuple(SteeringBinding(**row) for row in blob.get("bindings", [])),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SteeringProfile":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        return {
            "profile_version": 2,
            "model_id": self.model_id,
            "bindings": [binding.__dict__.copy() for binding in self.bindings],
        }
