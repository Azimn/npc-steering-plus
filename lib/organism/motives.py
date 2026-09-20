"""Derived motive competition for the persistent organism."""

from __future__ import annotations

from dataclasses import dataclass

from .state import OrganismState


@dataclass(frozen=True)
class Motive:
    name: str
    strength: float
    source: str


def motives(state: OrganismState) -> tuple[Motive, ...]:
    v = state.values
    threat_load = max(v["threat"], 0.65 * v["pain"], 0.45 * v["stress"])
    fatigue_load = max(v["fatigue"], 0.40 * v["stress"])
    candidates = (
        Motive("seek_safety", _clip01(threat_load), "threat/pain/stress"),
        Motive("rest", _clip01(fatigue_load), "fatigue/stress"),
        Motive("eat", _clip01(v["hunger"]), "hunger"),
        Motive("connect", _clip01(v["affiliation_need"] * (1.0 - 0.55 * threat_load)), "affiliation_need"),
        Motive("explore", _clip01(v["curiosity"] * (1.0 - 0.75 * threat_load) * (1.0 - 0.45 * fatigue_load)), "curiosity"),
        Motive("master", _clip01(v["competence_need"] * (1.0 - 0.60 * threat_load)), "competence_need"),
    )
    return tuple(sorted(candidates, key=lambda m: (-m.strength, m.name)))


def selected_motive(state: OrganismState) -> Motive:
    return motives(state)[0]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
