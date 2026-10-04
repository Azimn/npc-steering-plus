"""Persistent organism controller and closed-loop substrate handoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .backend import CognitiveBackend, CognitiveResponse
from .body import BodyAction, BodyEnvironment, RegulatoryStep, regulate_body
from .interoception import InteroceptiveObservation, InteroceptiveSensor
from .motives import Motive, selected_motive
from .state import OrganismState
from .steering import SteeringProfile


@dataclass
class OrganismTurn:
    response: CognitiveResponse
    steering_alphas: dict[str, float]
    selected_motive: Motive
    interoceptive_values: dict[str, float]
    pre_state: dict
    post_state: dict


class PersistentOrganism:
    def __init__(
        self,
        state: OrganismState,
        profile: SteeringProfile,
        *,
        sensor: InteroceptiveSensor | None = None,
    ) -> None:
        self.state = state
        self.profile = profile
        self.sensor = sensor or InteroceptiveSensor()

    def tick(self, dt_s: float) -> None:
        self.state.step(dt_s)

    def perturb(self, delta: Mapping[str, float]) -> None:
        self.state.apply(delta)

    def observe_body(self) -> InteroceptiveObservation:
        return self.sensor.observe(self.state)

    def steering_alphas(self) -> dict[str, float]:
        observation = self.observe_body()
        return self.profile.alphas(
            self.state,
            observed_values=observation.values,
        )

    def regulate(
        self,
        environment: BodyEnvironment,
        dt_s: float,
        *,
        action: BodyAction | None = None,
    ) -> RegulatoryStep:
        return regulate_body(
            self.state,
            environment,
            dt_s,
            action=action,
        )

    def bind_profile(self, profile: SteeringProfile) -> None:
        self.profile = profile

    def turn(
        self,
        backend: CognitiveBackend,
        messages: Sequence[Mapping[str, str]],
    ) -> OrganismTurn:
        if backend.model_id != self.profile.model_id:
            raise ValueError(
                f"Backend/profile mismatch: backend={backend.model_id!r}, "
                f"profile={self.profile.model_id!r}"
            )

        pre = self.state.snapshot()
        motive = selected_motive(self.state)
        observation = self.observe_body()
        alphas = self.profile.alphas(
            self.state,
            observed_values=observation.values,
        )
        response = backend.generate(messages, alphas)

        feedback = self.profile.feedback_delta(self.state, dict(response.readback))
        if feedback:
            self.state.apply(feedback)

        return OrganismTurn(
            response=response,
            steering_alphas=alphas,
            selected_motive=motive,
            interoceptive_values=dict(observation.values),
            pre_state=pre,
            post_state=self.state.snapshot(),
        )

    def save(self, path: str | Path) -> None:
        self.state.save(path)
