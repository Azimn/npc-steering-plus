"""Backend contract for any LLM that can accept activation steering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class CognitiveResponse:
    text: str
    # readback is reserved for model-internal or cognitive signals eligible for
    # bounded feedback. It must not be populated from post-hoc text projection
    # when that would create a circular validation loop.
    readback: Mapping[str, float] = field(default_factory=dict)
    # Online activation measurements captured during steered generation.
    online_activation: Mapping[str, float] = field(default_factory=dict)
    # Projection obtained by re-encoding generated language after generation.
    # This is a phenotype measure, not independent evidence of latent persistence.
    phenotype_projection: Mapping[str, float] = field(default_factory=dict)
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
