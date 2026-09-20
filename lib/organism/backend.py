"""Backend contract for any LLM that can accept activation steering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class CognitiveResponse:
    text: str
    readback: Mapping[str, float] = field(default_factory=dict)
    raw_projection: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class CognitiveBackend(Protocol):
    model_id: str

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        steering_alphas: Mapping[str, float],
    ) -> CognitiveResponse:
        ...
