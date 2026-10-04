"""Closed-loop virtual-body dynamics.

The body is authoritative. Environment and actions change physiological state;
the language model never directly writes these variables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .motives import Motive, selected_motive
from .state import OrganismState


class BodyAction(str, Enum):
    NONE = "none"
    REST = "rest"
    EAT = "eat"
    DRINK = "drink"
    WARM = "warm"
    COOL = "cool"
    WORK = "work"


@dataclass(frozen=True)
class BodyEnvironment:
    """Minimal external conditions that can affect or regulate the body."""

    food_available: bool = False
    water_available: bool = False
    rest_quality: float = 0.0
    warmth_available: bool = False
    cooling_available: bool = False
    thermal_exposure: float = 0.0
    exertion: float = 0.0
    injury_load: float = 0.0

    def __post_init__(self) -> None:
        for name in ("rest_quality", "exertion", "injury_load"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not -1.0 <= float(self.thermal_exposure) <= 1.0:
            raise ValueError("thermal_exposure must be in [-1, 1]")


@dataclass(frozen=True)
class RegulatoryStep:
    motive: Motive
    action: BodyAction
    dt_s: float
    pre_state: dict
    post_state: dict


def choose_regulatory_action(state: OrganismState, env: BodyEnvironment) -> BodyAction:
    """Choose a deterministic action from the current winning motive.

    This is deliberately simple. It exists to make motive competition causally
    consequential while keeping the policy inspectable for experiments.
    """

    motive = selected_motive(state)
    if motive.name == "drink" and env.water_available:
        return BodyAction.DRINK
    if motive.name == "eat" and env.food_available:
        return BodyAction.EAT
    if motive.name == "warm" and env.warmth_available:
        return BodyAction.WARM
    if motive.name == "cool" and env.cooling_available:
        return BodyAction.COOL
    if motive.name == "rest" and env.rest_quality > 0.0:
        return BodyAction.REST
    if motive.name in {"explore", "master"}:
        return BodyAction.WORK
    return BodyAction.NONE


def regulate_body(
    state: OrganismState,
    env: BodyEnvironment,
    dt_s: float,
    *,
    action: BodyAction | None = None,
) -> RegulatoryStep:
    """Advance physiology under environment plus one body action.

    Endogenous drift/recovery is integrated by OrganismState.step. Additional
    environment/action effects are then applied. All coefficients are explicit
    engineering parameters, not claims about human physiology.
    """

    dt_s = float(dt_s)
    if dt_s < 0.0:
        raise ValueError("dt_s must be non-negative")

    pre = state.snapshot()
    motive = selected_motive(state)
    chosen = action if action is not None else choose_regulatory_action(state, env)

    # Exact endogenous integration first.
    state.step(dt_s)

    # Environmental load. WORK adds exertion on top of ambient exertion.
    exertion = min(1.0, env.exertion + (0.45 if chosen == BodyAction.WORK else 0.0))
    if dt_s > 0.0:
        state.apply(
            {
                "fatigue": 0.00008 * exertion * dt_s,
                "thirst": 0.00006 * exertion * dt_s,
                "hunger": 0.00002 * exertion * dt_s,
                "pain": 0.00100 * env.injury_load * dt_s,
            }
        )

        # Thermal body state follows ambient exposure unless actively regulated.
        thermal_target = float(env.thermal_exposure)
        thermal_tau_s = 180.0
        current_thermal = state.get("thermal")
        coupled = thermal_target + (current_thermal - thermal_target) * math.exp(-dt_s / thermal_tau_s)
        state.set("thermal", coupled)

        if chosen == BodyAction.REST and env.rest_quality > 0.0:
            _relieve_toward_baseline(state, "fatigue", dt_s, 720.0 / max(env.rest_quality, 0.05))
        elif chosen == BodyAction.EAT and env.food_available:
            _relieve_toward_baseline(state, "hunger", dt_s, 300.0)
        elif chosen == BodyAction.DRINK and env.water_available:
            _relieve_toward_baseline(state, "thirst", dt_s, 120.0)
        elif chosen == BodyAction.WARM and env.warmth_available and state.get("thermal") < 0.0:
            _relieve_toward_value(state, "thermal", 0.0, dt_s, 90.0)
        elif chosen == BodyAction.COOL and env.cooling_available and state.get("thermal") > 0.0:
            _relieve_toward_value(state, "thermal", 0.0, dt_s, 90.0)

    return RegulatoryStep(
        motive=motive,
        action=chosen,
        dt_s=dt_s,
        pre_state=pre,
        post_state=state.snapshot(),
    )


def _relieve_toward_baseline(state: OrganismState, axis: str, dt_s: float, tau_s: float) -> None:
    target = state.specs[axis].baseline
    _relieve_toward_value(state, axis, target, dt_s, tau_s)


def _relieve_toward_value(
    state: OrganismState,
    axis: str,
    target: float,
    dt_s: float,
    tau_s: float,
) -> None:
    current = state.get(axis)
    value = target + (current - target) * math.exp(-float(dt_s) / float(tau_s))
    state.set(axis, value)
